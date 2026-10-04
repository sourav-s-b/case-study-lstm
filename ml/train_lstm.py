"""Small LSTM: (B, 2048, 6) -> Healthy/IR/OR. CPU-trainable.

Usage:
  uv run python ml/train_lstm.py --smoke            # 24 recs, 2 epochs, ~2min
  uv run python ml/train_lstm.py --epochs 10 --max-recs 120 --windows-per-rec 8
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, confusion_matrix

from ml.data_loader import load_metadata, recordings, read_signal, CLASS_TO_IDX
from ml.preprocess import synthesize_aux, COND_BASE

WIN = 2048


class SmallLSTM(nn.Module):
    def __init__(self, n_feat: int = 6, hidden: int = 64, layers: int = 2,
                 dropout: float = 0.3, n_class: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, layers, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(32, n_class))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        return self.head(self.norm(h[-1]))


def decimate(x: np.ndarray, factor: int = 5) -> np.ndarray:
    # 64kHz -> 12.8kHz fast path for smoke/full training (no FIR, fine for prototype).
    # Center-crop to multiple of factor first for stable lengths.
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
    # evenly spaced window starts across the recording
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
                  test_bearings: list, seed: int = 0):
    df = recordings(load_metadata())
    rng = np.random.default_rng(seed)
    # bearing-independent test split first
    test_mask = df["bearing_id"].isin(test_bearings)
    rest = df[~test_mask]
    # cap recs per split for memory safety, stratified by cls
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
        Xs, ys, conds = [], [], []
        for _, rec in frame.iterrows():
            try:
                X = recording_to_windows(rec.to_dict(), windows_per_rec, seed=int(rec["repetition"]))
            except Exception as e:
                print(f"skip {rec['bearing_id']} {rec['operating_condition']} r{rec['repetition']}: {e}")
                continue
            Xs.append(X)
            ys += [CLASS_TO_IDX[rec["cls"]]] * len(X)
            conds += [rec["operating_condition"]] * len(X)
        if not Xs:
            raise RuntimeError("no windows built — check data path")
        return np.concatenate(Xs), np.array(ys), conds
    return (stack(train), stack(val), stack(test))


def normalize_fit(Xtr):
    mu, sd = Xtr.reshape(-1, Xtr.shape[-1]).mean(0), Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max-recs", type=int, default=120)
    ap.add_argument("--windows-per-rec", type=int, default=8)
    ap.add_argument("--held-out", default="N09_M07_F10")
    ap.add_argument("--test-bearings", nargs="+", default=["K002", "KB27"])
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.smoke:
        a.epochs, a.max_recs, a.windows_per_rec = 2, 24, 4
    torch.manual_seed(a.seed)
    (Xtr, ytr, _), (Xva, yva, cva), (Xte, yte, cte) = build_dataset(
        a.max_recs, a.windows_per_rec, a.held_out, a.test_bearings, a.seed)
    mu, sd = normalize_fit(Xtr)
    Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    print(f"train {Xtr.shape} val {Xva.shape} test {Xte.shape}")

    train_ds = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                          batch_size=64, shuffle=True)
    model = SmallLSTM()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    counts = np.bincount(ytr, minlength=3)
    w = (counts.sum() / (3 * np.maximum(counts, 1)))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
    for ep in range(a.epochs):
        model.train()
        tot = 0.0
        for xb, yb in train_ds:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            pv = model(torch.from_numpy(Xva)).argmax(1).numpy()
        print(f"ep{ep+1}/{a.epochs} loss={tot/len(Xtr):.4f} valF1={f1_score(yva, pv, average='macro'):.3f}")

    model.eval()
    with torch.no_grad():
        pt = model(torch.from_numpy(Xte)).argmax(1).numpy()
        pv = model(torch.from_numpy(Xva)).argmax(1).numpy()
    print("VAL cm:\n", confusion_matrix(yva, pv))
    print("TEST cm:\n", confusion_matrix(yte, pt))
    for c in sorted(set(cte)):
        idx = [i for i, x in enumerate(cte) if x == c]
        print(f"test@{c}: F1={f1_score(yte[idx], pt[idx], average='macro'):.3f} n={len(idx)}")
    out = Path("models")
    out.mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(), "mu": mu, "sd": sd,
                "classes": ["healthy", "inner", "outer"]}, out / "lstm_small.pt")
    json.dump({"val_f1": float(f1_score(yva, pv, average="macro")),
               "test_f1": float(f1_score(yte, pt, average="macro"))},
              open(out / "metrics.json", "w"), indent=2)
    print("saved models/lstm_small.pt + metrics.json")


if __name__ == "__main__":
    main()
