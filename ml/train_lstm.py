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
CACHE_VERSION = 4  # bump when windowing/synthesis/labels change
CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"
DATA_WORKERS = 8  # parquet reads are I/O-bound; threads overlap them ~5x faster


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
    # vectorized aux synthesis: one RNG draw per channel for all windows.
    # (residual encoding: pressure/rpm centered on condition nominals,
    # temp driven by measured rms + small fault term + heavy noise)
    sev = {"healthy": 0.0, "inner": 0.6, "outer": 1.0}.get(rec["cls"], 0.0)
    nw = len(starts)
    temp = (46.0 + 1.5 * sev + 1.5 * rms[:, None]
            + rng.normal(0, 0.8, (nw, WIN))).astype(np.float32)
    press = (0.2 * rms[:, None] * sev
             + rng.normal(0, 0.05, (nw, WIN))).astype(np.float32)
    rpm = rng.normal(0, 5, (nw, WIN)).astype(np.float32)
    aux = np.stack([temp, press, rpm], axis=-1)
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

    import time as _time
    from concurrent.futures import ThreadPoolExecutor

    def one(rec, rep):
        try:
            X = recording_to_windows(rec, windows_per_rec, seed=int(rec["repetition"]) + 100 * rep)
        except Exception as e:
            print(f"skip {rec['bearing_id']} {rec['operating_condition']} "
                  f"r{rec['repetition']}: {e}", flush=True)
            return None
        y = STATE_TO_IDX[(rec["ml_inner"], rec["ml_outer"])]
        return X, [y] * len(X), [rec["operating_condition"]] * len(X), [rec["bearing_id"]] * len(X)

    def stack(frame):
        jobs = []
        for _, rec in frame.iterrows():
            d = rec.to_dict()
            is_both = bool(d["ml_inner"] and d["ml_outer"])
            for r in range(both_repeats if is_both else 1):
                jobs.append((d, r))
        Xs, ys, conds, bids = [], [], [], []
        t0 = _time.time()
        with ThreadPoolExecutor(max_workers=DATA_WORKERS) as ex:
            for i, res in enumerate(ex.map(lambda j: one(*j), jobs)):
                if res is None:
                    continue
                X, y, c, b = res
                Xs.append(X)
                ys += y
                conds += c
                bids += b
                if (i + 1) % 50 == 0 or (i + 1) == len(jobs):
                    print(f"  ...{i + 1}/{len(jobs)} recs [{_time.time() - t0:.0f}s]", flush=True)
        if not Xs:
            raise RuntimeError("no windows built — check data path")
        return np.concatenate(Xs), np.array(ys), conds, bids
    return (stack(train), stack(val), stack(test))


def normalize_fit(Xtr):
    mu, sd = Xtr.reshape(-1, Xtr.shape[-1]).mean(0), Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def cache_key(dataset, max_recs, windows_per_rec, held_out, test_bearings, seed, both_repeats):
    import hashlib
    raw = f"v{CACHE_VERSION}|{dataset}|{max_recs}|{windows_per_rec}|{held_out}|" \
          f"{','.join(sorted(test_bearings))}|{seed}|{both_repeats}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def load_or_build(dataset, max_recs, windows_per_rec, held_out, test_bearings,
                  seed, both_repeats):
    """Window cache: parquet reads + synthesis run once, reused across runs."""
    import time
    key = cache_key(dataset, max_recs, windows_per_rec, held_out,
                    test_bearings, seed, both_repeats)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"windows_{key}.npz"
    if path.exists():
        t0 = time.time()
        z = np.load(path, allow_pickle=True)
        print(f"cache HIT {path.name} ({time.time() - t0:.1f}s)", flush=True)
        return ((z["Xtr"], z["ytr"], None, None), (z["Xva"], z["yva"], None, None),
                (z["Xte"], z["yte"], z["cte"].tolist(), z["bte"].tolist()))
    t0 = time.time()
    out = build_dataset(max_recs, windows_per_rec, held_out, test_bearings, seed, both_repeats)
    (Xtr, ytr, _, _), (Xva, yva, _, _), (Xte, yte, cte, bte) = out
    np.savez_compressed(path, Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva,
                        Xte=Xte, yte=yte,
                        cte=np.array(cte, dtype=object), bte=np.array(bte, dtype=object))
    print(f"cache MISS — built + saved {path.name} ({time.time() - t0:.1f}s)", flush=True)
    return out


def predict_proba(model, X, device="cpu", batch=256):
    """Batched softmax inference — never OOMs on full val/test sets."""
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).to(device)
            outs.append(torch.softmax(model(xb), -1).cpu().numpy())
    return np.concatenate(outs)


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
    ap.add_argument("--batch-size", type=int, default=64,
                    help="64 + full precision is the quality-validated setting")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="auto = RTX GPU if available (10-30x faster than CPU)")
    ap.add_argument("--no-cache", action="store_true", help="rebuild windows from parquet")
    ap.add_argument("--early-stop", type=int, default=0,
                    help="stop if val_exact stalls N epochs (0 = off)")
    ap.add_argument("--both-repeats", type=int, default=4,
                    help="oversample repeats for the rare both state (2 bearings dataset-wide)")
    ap.add_argument("--amp", action="store_true",
                    help="mixed precision (faster, but ablation showed outer->both collapse)")
    a = ap.parse_args()
    if a.smoke:
        a.epochs, a.max_recs, a.windows_per_rec = 2, 24, 4
    torch.manual_seed(a.seed)
    import os, time
    device = (a.device if a.device != "auto"
              else ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = os.environ.get("DATASET", "paderborn_small")
    print(f"seed={a.seed} held_out={a.held_out} test={a.test_bearings} states={STATES}", flush=True)
    print(f"device={device} batch={a.batch_size} dataset={dataset}", flush=True)
    import logging
    logpath = Path("models") / "train.log"
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(logpath), logging.StreamHandler()], force=True)
    log = logging.getLogger("train")
    log.info(f"seed={a.seed} held_out={a.held_out} test={a.test_bearings} states={STATES}")
    log.info(f"device={device} batch={a.batch_size} dataset={dataset} both_repeats={a.both_repeats}")
    if device == "cuda":
        log.info(f"gpu={torch.cuda.get_device_name(0)} "
                 f"mem={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    t0 = time.time()
    if a.no_cache:
        out = build_dataset(a.max_recs, a.windows_per_rec, a.held_out,
                            a.test_bearings, a.seed, a.both_repeats)
    else:
        out = load_or_build(dataset, a.max_recs, a.windows_per_rec, a.held_out,
                            a.test_bearings, a.seed, a.both_repeats)
    (Xtr, ytr, _, _), (Xva, yva, _, _), (Xte, yte, cte, bte) = out
    print(f"data ready in {time.time() - t0:.1f}s", flush=True)
    mu, sd = normalize_fit(Xtr)
    Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    print(f"train {Xtr.shape} class_counts={np.bincount(ytr, minlength=4).tolist()}", flush=True)
    print(f"val   {Xva.shape} class_counts={np.bincount(yva, minlength=4).tolist()}", flush=True)
    print(f"test  {Xte.shape} class_counts={np.bincount(yte, minlength=4).tolist()}", flush=True)

    train_ds = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                          batch_size=a.batch_size, shuffle=True)
    model = SmallLSTM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    counts = np.bincount(ytr, minlength=4)
    w = counts.sum() / (4 * np.maximum(counts, 1))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32).to(device))
    use_amp = (device == "cuda" and a.amp)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    best, stale = -1.0, 0
    t0 = time.time()
    from tqdm import tqdm
    for ep in range(a.epochs):
        model.train()
        tot = 0.0
        bar = tqdm(train_ds, desc=f"ep{ep + 1}/{a.epochs} train", leave=False,
                   bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
        for xb, yb in bar:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            if use_amp:
                with torch.autocast("cuda"):
                    loss = loss_fn(model(xb), yb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss = loss_fn(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tot += loss.item() * len(xb)
            bar.set_postfix(loss=f"{loss.item():.4f}")
        pv = predict_proba(model, Xva, device).argmax(1)
        vacc = (pv == yva).mean()
        log.info(f"ep{ep + 1}/{a.epochs} loss={tot / len(Xtr):.4f} "
                 f"val_exact={vacc:.3f} [{time.time() - t0:.0f}s]")
        if a.early_stop:
            if vacc > best + 1e-4:
                best, stale = vacc, 0
            else:
                stale += 1
                if stale >= a.early_stop:
                    print(f"early stop at ep{ep + 1} (best val_exact={best:.3f})", flush=True)
                    break

    print("--- validation (held-out condition) ---", flush=True)
    val_m = report("VAL", yva, predict_proba(model, Xva, device))
    print("--- test (held-out bearings) ---", flush=True)
    test_m = report("TEST", yte, predict_proba(model, Xte, device), conds=cte)
    print("--- test by bearing ---", flush=True)
    for b in sorted(set(bte)):
        idx = [i for i, x in enumerate(bte) if x == b]
        yt, probs = yte[idx], predict_proba(model, Xte[idx], device)
        print(f"   {b}: exact={(probs.argmax(1) == yt).mean():.3f} n={len(idx)} "
              f"true_state={STATES[int(yt[0])]}", flush=True)

    out = Path("models")
    out.mkdir(exist_ok=True)
    # Smoke runs must NEVER overwrite the production weights (previous
    # incidents: 2-epoch smoke clobbered the good full-data model twice).
    tag = "smoke_" if a.smoke else ""
    torch.save({"model": model.state_dict(), "mu": mu, "sd": sd,
                "states": STATES, "joint_states": True}, out / f"{tag}lstm_small.pt")
    json.dump({"joint_states": STATES, "val": val_m, "test": test_m},
              open(out / f"{tag}metrics.json", "w"), indent=2)
    log.info(f"FINAL val={val_m} test={test_m}")
    print(f"saved {out / (tag + 'lstm_small.pt')} + {tag}metrics.json "
          f"(full log: {logpath})", flush=True)


if __name__ == "__main__":
    main()
