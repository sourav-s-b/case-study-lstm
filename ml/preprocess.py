"""Windowing + resampling + synthetic temp/pressure for the small LSTM.

Real channels (Paderborn @64kHz -> downsample 12kHz):
  vibration_1, current_1, current_2
Simulated channels (conditioned on class + vibration RMS):
  temperature, pressure, rpm_jitter
Window: 2048 samples (~170ms @12kHz), stride 512. Per-condition z-score.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import resample_poly

FS_ORIG = 64_000
FS_TARGET = 12_000
WIN, STRIDE = 2048, 512

COND_BASE = {  # rpm, torque, radial N  (for synthetic channels)
    "N15_M07_F10": (1500, 0.7, 1000),
    "N09_M07_F10": (900, 0.7, 1000),
    "N15_M01_F10": (1500, 0.1, 1000),
    "N15_M07_F04": (1500, 0.7, 400),
}


def downsample(x: np.ndarray) -> np.ndarray:
    # 64000 -> 12000 = up 3 / down 16
    return resample_poly(x, 3, 16).astype(np.float32)


def sliding_windows(x: np.ndarray, win: int = WIN, stride: int = STRIDE) -> np.ndarray:
    n = (len(x) - win) // stride + 1
    return np.stack([x[i * stride:i * stride + win] for i in range(n)]).astype(np.float32)


def synthesize_aux(vib_rms: float, condition: str, fault: str, n: int, rng: np.random.Generator) -> dict:
    """Plausible temp/pressure anchored to physics; label as simulated in UI."""
    rpm, _, radial = COND_BASE.get(condition, (1500, 0.7, 1000))
    sev = {"healthy": 0.0, "inner": 0.6, "outer": 1.0}.get(fault, 0.0)
    temp = 46.0 + 4.0 * sev + 1.5 * vib_rms + rng.normal(0, 0.3, n)
    pressure = radial / 1000.0 + 0.2 * vib_rms * sev + rng.normal(0, 0.02, n)
    rpm_sig = rpm + rng.normal(0, 5, n)
    return {"temperature": temp.astype(np.float32),
            "pressure": pressure.astype(np.float32),
            "rpm": rpm_sig.astype(np.float32)}


def make_sample(vib: np.ndarray, i1: np.ndarray, i2: np.ndarray,
                condition: str, fault: str, seed: int = 0) -> np.ndarray:
    """-> (n_windows, WIN, 6) with channels [vib, I1, I2, temp, press, rpm]."""
    rng = np.random.default_rng(seed)
    vib_d, i1_d, i2_d = downsample(vib), downsample(i1), downsample(i2)
    W = np.stack([sliding_windows(c) for c in (vib_d, i1_d, i2_d)], axis=-1)  # (N,WIN,3)
    N = W.shape[0]
    rms = np.sqrt((W[:, :, 0] ** 2).mean(axis=1))
    aux = np.zeros((N, WIN, 3), dtype=np.float32)
    for n in range(N):
        s = synthesize_aux(float(rms[n]), condition, fault, WIN, rng)
        aux[n, :, 0] = s["temperature"]
        aux[n, :, 1] = s["pressure"]
        aux[n, :, 2] = s["rpm"]
    return np.concatenate([W, aux], axis=-1)
