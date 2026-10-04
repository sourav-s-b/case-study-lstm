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
SUBSET = DATA_ROOT / "paderborn_small"

# fault_type values in metadata: normal / inner / outer / inner+outer (KB23, KB27).
# For the 3-class LSTM, combined maps to outer (flagged via is_combined).
FAULT_TO_CLASS = {"normal": "healthy", "inner": "inner",
                  "outer": "outer", "inner+outer": "outer"}
CLASS_TO_IDX = {"healthy": 0, "inner": 1, "outer": 2}

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
    return df


def recordings(df: pd.DataFrame) -> pd.DataFrame:
    """One row per recording (bearing x condition x repetition)."""
    cols = ["bearing_id", "operating_condition", "repetition",
            "cls", "label", "is_combined", "fault_origin", "damage"]
    return df.drop_duplicates(subset=["bearing_id", "operating_condition", "repetition"]).copy()


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
