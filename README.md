# db_case — Paderborn LSTM Anomaly Detector (prototype)

Simulates temperature / vibration / pressure / motor-current from industrial
bearings, classifies `Healthy / Inner-Race / Outer-Race` with a small LSTM,
robust across all 4 Paderborn operating conditions, served via FastAPI +
React cockpit.

## Quickstart (uv)

```bash
uv sync                  # create .venv + install deps
uv run python -c "import torch; print(torch.__version__)"

# 1. Download smart subset (~8 bearings, ~800MB, resumable)
bash scripts/download_subset.sh

# 2. EDA
uv run jupyter notebook notebooks/01_EDA.ipynb

# 3. Train
uv run python ml/train_lstm.py --config ml/config.yaml

# 4. Serve API
uv run uvicorn backend.main:app --reload --port 8000

# 5. UI (in frontend/)
npm install && npm run dev
```

## Data — Paderborn University Bearing DataCenter
Lessmeier et al., PHM 2016. 32x 6203 bearings: 6 healthy (K001-K006),
12 artificial, 14 real pitting. Vibration + 2x motor current @64kHz, 4s each,
20 repeats x 4 conditions (N15_M07_F10 baseline, N09_M07_F10, N15_M01_F10,
N15_M07_F04). Temp/pressure are **simulated** on top (see `ml/preprocess.py`).

Cite: Lessmeier, Kimotho, Zimmer & Sextro, *Condition Monitoring of Bearing
Damage in Electromechanical Drive Systems...*, PHM Society 2016.
