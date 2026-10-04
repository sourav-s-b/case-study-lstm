"""Build models/demo_bank.npz: severity-ladder windows for the live demo.

v1 morphed healthy<->fault windows along a straight line; the 4-state model
reads those midpoints as the WRONG race (healthy->inner->outer regardless of
the selected fault). v2 uses a real severity ladder per fault class:

  health>66%  -> healthy rung  (K001)
  33-66%      -> mild rung     (sev-1 bearing of the selected fault)
  <33%        -> severe rung   (sev-2/3 bearing of the selected fault)

Every slider stop is a genuine Paderborn window, so the model reads the
selected fault faithfully. Ladders (mild, severe):
  inner: KI01 (art EDM, sev1), KI18 (art engraver, sev2)
  outer: KA01 (art EDM, sev1), KA16 (real pitting, sev2)
  both:  KB27 (real indentation, sev1), KB23 (real pitting, sev2)

Bank keys: {cond}_{cls} for healthy, {cond}_{cls}_mild/_severe otherwise.
Each -> (4, 2048, 6) float32. NOTE: demo asset only, never an eval set.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from ml.data_loader import load_metadata, recordings
from ml.train_lstm import recording_to_windows

CONDITIONS = ["N15_M07_F10", "N09_M07_F10"]
LADDER = {
    "healthy": ["K001"],
    "inner": ["KI01", "KI18"],
    "outer": ["KA01", "KA16"],
    "both": ["KB27", "KB23"],
}
OUT = Path(__file__).resolve().parents[1] / "models" / "demo_bank.npz"


def main():
    df = recordings(load_metadata())
    bank = {}
    for cond in CONDITIONS:
        for cls, bids in LADDER.items():
            for i, bid in enumerate(bids):
                recs = df[(df.bearing_id == bid) & (df.operating_condition == cond)]
                if len(recs) == 0:
                    raise RuntimeError(f"no recording for {bid} {cond}")
                rec = recs.iloc[0].to_dict()
                rung = "" if cls == "healthy" else ("_mild" if i == 0 else "_severe")
                bank[f"{cond}_{cls}{rung}"] = recording_to_windows(rec, windows_per_rec=4, seed=7)
                print(f"{cond}_{cls}{rung} [{bid}]: {bank[f'{cond}_{cls}{rung}'].shape}")
    np.savez_compressed(OUT, **bank)
    print("saved", OUT, f"{OUT.stat().st_size / 1024:.0f}KB")


if __name__ == "__main__":
    main()
