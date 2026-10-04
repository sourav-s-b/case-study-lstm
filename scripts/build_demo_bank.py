"""Build models/demo_bank.npz: real LSTM windows per class for the live demo.

The UI health slider morphs between a real healthy window and a real faulty
window, so orange->red transitions come from genuine Paderborn vibration
texture instead of fabricated pseudo-windows (which are OOD for the LSTM).

Bank: {condition}_{cls} -> (4, 2048, 6) float32. ~600KB total.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from ml.data_loader import load_metadata, recordings
from ml.train_lstm import recording_to_windows

CONDITIONS = ["N15_M07_F10", "N09_M07_F10"]
PICK = {"healthy": "K001", "inner": "KI01", "outer": "KA01"}  # pure, unambiguous bearings
OUT = Path(__file__).resolve().parents[1] / "models" / "demo_bank.npz"


def main():
    df = recordings(load_metadata())
    bank = {}
    for cond in CONDITIONS:
        for cls, bid in PICK.items():
            recs = df[(df.bearing_id == bid) & (df.operating_condition == cond)]
            if len(recs) == 0:
                raise RuntimeError(f"no recording for {bid} {cond}")
            rec = recs.iloc[0].to_dict()
            bank[f"{cond}_{cls}"] = recording_to_windows(rec, windows_per_rec=4, seed=7)
            print(f"{cond} {cls}: {bank[f'{cond}_{cls}'].shape}")
    np.savez_compressed(OUT, **bank)
    print("saved", OUT, f"{OUT.stat().st_size / 1024:.0f}KB")


if __name__ == "__main__":
    main()
