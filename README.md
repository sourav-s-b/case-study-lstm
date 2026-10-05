# Bearing Anomaly Cockpit — Paderborn LSTM (prototype)

A prototype that watches industrial bearings and raises an alarm before they
fail. It learns from real vibration and motor-current recordings (Paderborn
University dataset), classifies each sample as **healthy / inner-ring fault /
outer-ring fault / both**, and shows it all on a live React dashboard with a
machine-health slider for demonstration.

## Quickstart

```bash
uv sync                  # create .venv + install deps

# 1. Data. Smart subset (~8 bearings, ~1.3GB, resumable) for iteration...
bash scripts/download_subset.sh
# ...or the full 32-bearing set (~5.4GB) for real training:
uv run bearing-datasets build paderborn

# 2. EDA
uv run jupyter notebook notebooks/01_EDA.ipynb

# 3. Train (auto-uses GPU if present; windows are cached after first build)
DATASET=paderborn uv run python -m ml.train_lstm --epochs 12 --max-recs 480

# 4. Serve API
uv run uvicorn backend.main:app --port 8000

# 5. UI (in frontend/)
npm install && npm run dev   # -> http://localhost:5173
```

`DATASET` selects the dataset: `paderborn_small` (default, 8 bearings) or
`paderborn` (full, 32 bearings). `data/` is local-only and never committed.

## Folder structure

```
ml/                  machine learning
  data_loader.py     reads the dataset (lazy, one signal at a time)
  preprocess.py      windowing + downsampling + simulated sensors
  train_lstm.py      model definition + training + evaluation
backend/             FastAPI: /predict, /bearings, live WebSocket stream
frontend/            React cockpit (Vite): live charts, brain view, dataset explorer
models/              lstm_small.pt (trained weights), demo_bank.npz, metrics.json
notebooks/           01_EDA.ipynb (exploratory plots)
scripts/             download_subset.sh, build_demo_bank.py
data/                dataset itself (local only, gitignored — 8+ GB)
```

## Model — SmallLSTM (54,052 params, 217 KB)

1. **Input:** one window of **2048 time steps × 6 sensor channels**
   (~160 ms of machine data)
2. **LSTM trunk:** 2 layers × 64 hidden units, dropout 0.3 — reads the window
   in order and summarizes it into its final hidden state
3. **LayerNorm** + **head:** Dense 64→32 (ReLU, dropout), Dense 32→4 with
   **softmax** over the 4 joint states: healthy / inner-only / outer-only / both

Why an LSTM: bearing faults appear as repeating impact patterns (a damaged
spot strikes once per revolution) — exactly what a recurrent network hears.
Why small: trains in under a minute on an RTX 3050 and infers live on CPU.

Design note: an earlier 2-head multi-label (BCE) version failed — the outer
head flatlined at F1 0.0 over 27 epochs. Joint-state softmax keeps the heads
competing and learns all states. See code comments for the full ablation log.

## Dataset — Paderborn University Bearing DataCenter

Lessmeier et al., PHM Society European Conference 2016. 32 ball bearings of
type 6203 on a motor test rig:

- **6 healthy** (K001–K006), **12 artificially damaged** (drilling, engraving,
  EDM), **14 really damaged** (pitting from accelerated lifetime tests)
- **4 operating conditions:** 1500/900 rpm × 0.7/0.1 Nm × 1000/400 N radial
  force (`N15_M07_F10` baseline, `N09_M07_F10`, `N15_M01_F10`, `N15_M07_F04`);
  20 recordings × 4 s each → **2,560 recordings**
- **Real channels:** housing vibration @64 kHz, 2× motor phase currents @64 kHz,
  plus slow force/speed/torque @4 kHz and temperature @1 Hz
- **6 model inputs:** vibration + 2 currents (real, downsampled to 12.8 kHz) +
  temperature/pressure/rpm (**simulated** per the case-study spec, encoded as
  residuals around the operating-condition nominals so they can't leak the
  condition or the label — see `ml/preprocess.py`)

## Training procedure

1. **Honest splits:** test bearings fully held out (K006, KI16, KA22, KB24);
   validation is a fully held-out *operating condition* (900 rpm).
2. **Windowing:** 8 evenly spaced 2048-sample windows per recording; the rare
   "both rings" state (2 bearings dataset-wide) repeated 4× (`--both-repeats`).
3. **Optimization:** weighted cross-entropy, Adam 1e-3, grad clipping,
   batch 64, full precision, 12 epochs. Full run: ~40 s on GPU after a
   one-time ~3 min cached window build (`data/cache/`, `--no-cache` to redo).
4. **Speed engineering:** RTX auto-detect, mixed precision opt-in only
   (ablation: AMP+large batch collapsed outer→both), 8-thread parquet reads
   with live counters, `models/train.log` file logging.
5. **Smoke guard:** `--smoke` writes `smoke_*` files and can never overwrite
   production weights (burned twice before this rule existed).

## Results (weights on `main`)

| Split | Exact | Fault found? | Inner | Outer | Both |
|---|---|---|---|---|---|
| Val (unseen 900 rpm) | 0.76 | 0.99 | 0.69 | 0.999 | 0.00 |
| Test (unseen bearings) | **0.87** | **1.00** | 0.92 | 0.98 | 0.78 |

Anomaly detection is perfect and single-ring faults classify near-perfectly —
including real pitting on bearings never seen in training. The known weakness
is "both rings at once" on unseen speeds: only 2 such bearings exist in the
dataset. That's a data limit, not a modeling bug.

## Demo

The health slider walks a **severity ladder of real recordings**
(`scripts/build_demo_bank.py` → `models/demo_bank.npz`): healthy K001 →
mild (sev-1) → severe (sev-2) bearing of the selected fault. Every slider stop
is genuine Paderborn data, so the screen shows what the model truly thinks —
including its mistakes (e.g. mild both-rings reads as outer, matching eval).

## Citation

Lessmeier, C.; Kimotho, J. K.; Zimmer, D.; Sextro, W.: *Condition Monitoring
of Bearing Damage in Electromechanical Drive Systems by Using Motor Current
Signals of Electric Motors: A Benchmark Data Set for Data-Driven
Classification*, European Conference of the Prognostics and Health Management
Society, Bilbao, 2016.
