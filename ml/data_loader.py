"""Load Paderborn smart-subset built by `bearing-datasets` into numpy windows.

Expected on-disk layout (built by bearing-datasets):
  data/paderborn_small/metadata.parquet   # one row per signal (4480 rows)
  data/paderborn_small/signals/*.parquet  # signal_id + signal list<double>

Verified 2026-10-04: 8 bearings x 4 conditions x 20 reps = 640 recordings
x 7 channels. Channels: vibration, phase_current_1/2 @64kHz; force, speed,
torque @4kHz; temperature @1Hz.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
# DATASET=paderborn selects the full 32-bearing build; default is the 8-bearing
# smart subset. Both share the bearing-datasets schema. Demo bank + scripts
# honor the same variable.
import os as _os
SUBSET = DATA_ROOT / _os.environ.get("DATASET", "paderborn_small")

# fault_type values in metadata: normal / inner / outer / inner+outer (KB23, KB27).
# Single-label mapping kept for compat; multi-label is the real target:
# healthy=[0,0], inner=[1,0], outer=[0,1], combined=[1,1].
FAULT_TO_CLASS = {"normal": "healthy", "inner": "inner",
                  "outer": "outer", "inner+outer": "outer"}
FAULT_TO_MULTILABEL = {"normal": (0, 0), "inner": (1, 0),
                       "outer": (0, 1), "inner+outer": (1, 1)}
CLASS_TO_IDX = {"healthy": 0, "inner": 1, "outer": 2}
LABELS = ["inner", "outer"]

CONDITIONS = ["N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04"]


def load_metadata(root: Path = SUBSET) -> pd.DataFrame:
    meta = root / "metadata.parquet"
    if not meta.exists():
        raise FileNotFoundError(
            f"No {meta}. Run bash scripts/download_subset.sh first.")
    df = pd.read_parquet(meta)
    df["cls"] = df["fault_type"].map(FAULT_TO_CLASS).fillna("unknown")
    df["is_combined"] = df["fault_type"] == "inner+outer"
    df["label"] = df["cls"].map(CLASS_TO_IDX)
    df[["ml_inner", "ml_outer"]] = pd.DataFrame(
        df["fault_type"].map(FAULT_TO_MULTILABEL).tolist(), index=df.index)
    return df


def recordings(df: pd.DataFrame) -> pd.DataFrame:
    """One row per recording (bearing x condition x repetition)."""
    cols = ["bearing_id", "operating_condition", "repetition",
            "cls", "label", "ml_inner", "ml_outer",
            "is_combined", "fault_origin", "damage"]
    recs = df.drop_duplicates(subset=["bearing_id", "operating_condition", "repetition"]).copy()
    return recs[[c for c in cols if c in recs.columns]]


def read_signal(signal_id: str, root: Path = SUBSET):
    """Lazy single-signal read via pyarrow filter — never loads the 1GB file fully."""
    import pyarrow.parquet as pq
    import numpy as np
    sig_dir = root / "signals"
    for part in sorted(sig_dir.glob("*.parquet")):
        tbl = pq.read_table(part, filters=[("signal_id", "==", signal_id)])
        if tbl.num_rows:
            return np.asarray(tbl.column("signal")[0].as_py(), dtype=np.float64)
    raise KeyError(f"signal_id not found: {signal_id}")


def recording_id_of(bearing_row) -> str:
    # signal_id = RECORDING/channel, e.g. N09_M07_F10_K001_1/vibration
    return f"{bearing_row['operating_condition']}_{bearing_row['bearing_id']}_{bearing_row['repetition']}"


def describe_subset(df: pd.DataFrame) -> pd.DataFrame:
    """One-row-per-bearing summary for the UI Dataset Explorer."""
    return (df.groupby(["bearing_id", "cls", "fault_origin", "damage",
                        "operating_condition"], dropna=False)
              .size().reset_index(name="n_signals"))
