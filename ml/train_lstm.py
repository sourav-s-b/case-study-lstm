"""Small 4-state LSTM: (B, 2048, 6) -> {healthy, inner-only, outer-only, both}.

Why 4-state softmax instead of 2-head BCE (evidence, 2026-10-04):
  - 2-head BCE trained 27 epochs: inner-F1 0.82, outer-F1 0.000. The outer head
    died (P(outer)~0.01 on every state) — the shared trunk converged to features
    where all fault evidence routes through the inner head, and the dead head
    never recovers (sigmoid saturation + no competitive pressure).
  - The earlier 3-class softmax run had outer perfect (KA15 104/104), proving the
    signature IS learnable when heads compete.
  - Joint-state softmax is exact here (every window is in exactly one joint
    state); marginals P(inner)=P(inner-only)+P(both) give multi-label-style
    outputs for the UI. KB23/KB27 train honestly as "both".

Usage:
  uv run python -m ml.train_lstm --smoke
  uv run python -m ml.train_lstm --epochs 12 --max-recs 120 --windows-per-rec 8
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score

from ml.data_loader import load_metadata, recordings, read_signal
from ml.preprocess import synthesize_aux

WIN = 2048
STATES = ["healthy", "inner", "outer", "both"]  # inner/outer = *-only; both = inner+outer
STATE_TO_IDX = {(0, 0): 0, (1, 0): 1, (0, 1): 2, (1, 1): 3}


class SmallLSTM(nn.Module):
    """Shared trunk, 4-way joint-state head. ~54k params."""

    def __init__(self, n_feat: int = 6, hidden: int = 64, layers: int = 2,
                 dropout: float = 0.3, n_states: int = 4):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, layers, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(32, n_states))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        return self.head(self.norm(h[-1]))


def decimate(x: np.ndarray, factor: int = 5) -> np.ndarray:
    # 64kHz -> 12.8kHz plain slicing. (Ablation 2026-10-04: FIR anti-alias
    # decimation changed input scale, destabilized Adam at lr=1e-3 and collapsed
    # healthy recall to 0. Revisit with retuned LR if spectral purity matters.)
    n = len(x) - (len(x) % factor)
    return x[:n:factor].astype(np.float32)


def recording_to_windows(rec: dict, windows_per_rec: int = 4, seed: int = 0) -> np.ndarray:
    """One recording -> (W, WIN, 6). Reads 3x64kHz lazily, decimates, samples W windows."""
    rng = np.random.default_rng(seed)
    rid = f"{rec['operating_condition']}_{rec['bearing_id']}_{rec['repetition']}"
    vib = decimate(read_signal(f"{rid}/vibration"))
    i1 = decimate(read_signal(f"{rid}/phase_current_1"))
    i2 = decimate(read_signal(f"{rid}/phase_current_2"))
    n = min(len(vib), len(i1), len(i2))
    vib, i1, i2 = vib[:n], i1[:n], i2[:n]
    max_start = n - WIN
    if max_start <= 0:
        raise ValueError(f"recording too short: {rid} n={n}")
    starts = np.linspace(0, max_start, windows_per_rec).astype(int)
    W = np.stack([np.stack([vib[s:s + WIN], i1[s:s + WIN], i2[s:s + WIN]], axis=-1)
                  for s in starts]).astype(np.float32)  # (W,WIN,3)
    rms = np.sqrt((W[:, :, 0] ** 2).mean(axis=1))
    aux = np.zeros((len(starts), WIN, 3), dtype=np.float32)
    for k in range(len(starts)):
        s = synthesize_aux(float(rms[k]), rec["operating_condition"], rec["cls"], WIN, rng)
        aux[k, :, 0], aux[k, :, 1], aux[k, :, 2] = s["temperature"], s["pressure"], s["rpm"]
    return np.concatenate([W, aux], axis=-1)


def build_dataset(max_recs: int, windows_per_rec: int, held_out_cond: str,
                  test_bearings: list, seed: int = 0, both_repeats: int = 4):
    df = recordings(load_metadata())
    test_mask = df["bearing_id"].isin(test_bearings)
    rest = df[~test_mask]

    def cap(frame, n):
        if len(frame) > n:
            return frame.sample(n, random_state=seed)
        return frame

    # leave-one-condition-out: val = held_out condition
    val = rest[rest.operating_condition == held_out_cond]
    train_pool = rest[rest.operating_condition != held_out_cond]
    train = cap(train_pool, max_recs)
    val = cap(val, max(8, max_recs // 3))
    test = cap(df[test_mask], max(8, max_recs // 3))

    def stack(frame):
        Xs, ys, conds, bids = [], [], [], []
        for _, rec in frame.iterrows():
            # oversample the rare "both" state (only 2 bearings dataset-wide):
            # same recording, different window offsets per repeat.
            is_both = bool(rec["ml_inner"] and rec["ml_outer"])
            reps = both_repeats if is_both else 1
            for r in range(reps):
                try:
                    X = recording_to_windows(rec.to_dict(), windows_per_rec,
                                             seed=int(rec["repetition"]) + 100 * r)
                except Exception as e:
                    print(f"skip {rec['bearing_id']} {rec['operating_condition']} "
                          f"r{rec['repetition']}: {e}", flush=True)
                    continue
                Xs.append(X)
                ys += [STATE_TO_IDX[(rec["ml_inner"], rec["ml_outer"])]] * len(X)
                conds += [rec["operating_condition"]] * len(X)
                bids += [rec["bearing_id"]] * len(X)
        if not Xs:
            raise RuntimeError("no windows built — check data path")
        return np.concatenate(Xs), np.array(ys), conds, bids
    return (stack(train), stack(val), stack(test))


def normalize_fit(Xtr):
    mu, sd = Xtr.reshape(-1, Xtr.shape[-1]).mean(0), Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def predict_proba(model, X):
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(X)), -1).numpy()


def marginals(probs):
    """4-state probs -> multi-label-style {healthy, inner, outer, anomaly, both}."""
    ph, pi, po, pb = probs[:, 0], probs[:, 1], probs[:, 2], probs[:, 3]
    return {"healthy": ph, "inner": pi + pb, "outer": po + pb,
            "anomaly": 1 - ph, "both": pb}


def report(name, y_true, probs, conds=None):
    pred = probs.argmax(1)
    exact = (pred == y_true).mean()
    m = marginals(probs)
    inner_true = ((y_true == 1) | (y_true == 3)).astype(int)
    outer_true = ((y_true == 2) | (y_true == 3)).astype(int)
    f_in = f1_score(inner_true, (m["inner"] > 0.5).astype(int), zero_division=0)
    f_out = f1_score(outer_true, (m["outer"] > 0.5).astype(int), zero_division=0)
    f_fault = f1_score((y_true > 0).astype(int), (pred > 0).astype(int), zero_division=0)
    print(f"[{name}] exact={exact:.3f} marginal inner-F1={f_in:.3f} "
          f"outer-F1={f_out:.3f} faulty-vs-healthy-F1={f_fault:.3f} n={len(y_true)}", flush=True)
    for s, sname in enumerate(STATES):
        idx = np.where(y_true == s)[0]
        if len(idx):
            print(f"   {sname}: recall={(pred[idx] == s).mean():.3f} n={len(idx)}", flush=True)
    if conds is not None:
        for c in sorted(set(conds)):
            idx = [i for i, x in enumerate(conds) if x == c]
            print(f"   @{c}: exact={(pred[idx] == y_true[idx]).mean():.3f} n={len(idx)}", flush=True)
    return {"exact": float(exact), "inner_f1": float(f_in),
            "outer_f1": float(f_out), "fault_f1": float(f_fault)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--max-recs", type=int, default=120)
    ap.add_argument("--windows-per-rec", type=int, default=8)
    ap.add_argument("--held-out", default="N09_M07_F10")
    ap.add_argument("--test-bearings", nargs="+", default=["K002", "KI14", "KA15", "KB27"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--both-repeats", type=int, default=4,
                    help="oversample repeats for the rare both state (2 bearings dataset-wide)")
    a = ap.parse_args()
    if a.smoke:
        a.epochs, a.max_recs, a.windows_per_rec = 2, 24, 4
    torch.manual_seed(a.seed)
    print(f"seed={a.seed} held_out={a.held_out} test={a.test_bearings} states={STATES}", flush=True)
    (Xtr, ytr, _, _), (Xva, yva, _, _), (Xte, yte, cte, bte) = build_dataset(
        a.max_recs, a.windows_per_rec, a.held_out, a.test_bearings, a.seed, a.both_repeats)
    mu, sd = normalize_fit(Xtr)
    Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    print(f"train {Xtr.shape} class_counts={np.bincount(ytr, minlength=4).tolist()}", flush=True)
    print(f"val   {Xva.shape} class_counts={np.bincount(yva, minlength=4).tolist()}", flush=True)
    print(f"test  {Xte.shape} class_counts={np.bincount(yte, minlength=4).tolist()}", flush=True)

    train_ds = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                          batch_size=64, shuffle=True)
    model = SmallLSTM()
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    counts = np.bincount(ytr, minlength=4)
    w = counts.sum() / (4 * np.maximum(counts, 1))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
    for ep in range(a.epochs):
        model.train()
        tot = 0.0
        for xb, yb in train_ds:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(xb)
        pv = predict_proba(model, Xva).argmax(1)
        print(f"ep{ep + 1}/{a.epochs} loss={tot / len(Xtr):.4f} "
              f"val_exact={(pv == yva).mean():.3f}", flush=True)

    print("--- validation (held-out condition) ---", flush=True)
    val_m = report("VAL", yva, predict_proba(model, Xva))
    print("--- test (held-out bearings) ---", flush=True)
    test_m = report("TEST", yte, predict_proba(model, Xte), conds=cte)
    print("--- test by bearing ---", flush=True)
    for b in sorted(set(bte)):
        idx = [i for i, x in enumerate(bte) if x == b]
        yt, probs = yte[idx], predict_proba(model, Xte[idx])
        print(f"   {b}: exact={(probs.argmax(1) == yt).mean():.3f} n={len(idx)} "
              f"true_state={STATES[int(yt[0])]}", flush=True)

    out = Path("models")
    out.mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(), "mu": mu, "sd": sd,
                "states": STATES, "joint_states": True}, out / "lstm_small.pt")
    json.dump({"joint_states": STATES, "val": val_m, "test": test_m},
              open(out / "metrics.json", "w"), indent=2)
    print("saved models/lstm_small.pt + metrics.json", flush=True)


if __name__ == "__main__":
    main()
