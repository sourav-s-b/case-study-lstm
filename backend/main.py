"""FastAPI: real LSTM inference (models/lstm_small.pt) + simulated stream.

Endpoints:
  GET  /model/info
  GET  /bearings              # 8-bearing smart subset catalogue
  POST /predict               # {features} -> class probs via LSTM (or heuristic fallback)
  WS   /simulate/stream       # client sends {condition, health_pct, fault} then receives 10Hz ticks
"""
from __future__ import annotations
import asyncio
import json
import math
import random
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

MODELS = Path(__file__).resolve().parents[1] / "models"
CLASSES = ["healthy", "inner", "outer"]
CONDITIONS = ["N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04"]
CATALOGUE = [
    {"code": "K001", "cls": "healthy", "origin": "none", "damage": "none"},
    {"code": "K002", "cls": "healthy", "origin": "none", "damage": "none"},
    {"code": "KA01", "cls": "outer", "origin": "artificial", "damage": "EDM"},
    {"code": "KA15", "cls": "outer", "origin": "real", "damage": "indentation"},
    {"code": "KB23", "cls": "outer", "origin": "real", "damage": "pitting (inner+outer)"},
    {"code": "KB27", "cls": "outer", "origin": "real", "damage": "indentation (inner+outer)"},
    {"code": "KI01", "cls": "inner", "origin": "artificial", "damage": "EDM"},
    {"code": "KI14", "cls": "inner", "origin": "real", "damage": "pitting"},
]

_model, _mu, _sd, _bank = None, None, None, None


def get_bank():
    """Real demo windows: {condition_cls: (4, 2048, 6)}. Built by scripts/build_demo_bank.py."""
    global _bank
    if _bank is None:
        import numpy as np
        _bank = dict(np.load(MODELS / "demo_bank.npz"))
    return _bank


def get_model():
    global _model, _mu, _sd
    if _model is not None:
        return _model
    pt = MODELS / "lstm_small.pt"
    if not pt.exists():
        return None
    import torch
    from ml.train_lstm import SmallLSTM
    ckpt = torch.load(pt, map_location="cpu", weights_only=False)  # trusted local file
    m = SmallLSTM()
    m.load_state_dict(ckpt["model"])
    m.eval()
    _model, _mu, _sd = m, ckpt["mu"], ckpt["sd"]
    return m


def heuristic_probs(sev: float, fault: str) -> dict:
    sev = max(0.0, min(1.0, sev))
    if fault == "inner":
        return {"healthy": 1 - sev, "inner": 0.75 * sev, "outer": 0.25 * sev, "anomaly": sev}
    if fault == "outer":
        return {"healthy": 1 - sev, "inner": 0.25 * sev, "outer": 0.75 * sev, "anomaly": sev}
    return {"healthy": 1 - sev, "inner": 0.5 * sev, "outer": 0.5 * sev, "anomaly": sev}


def infer_window(window) -> dict:
    """window: (2048, 6) array-like. 4-state softmax -> multi-label-style marginals."""
    m = get_model()
    if m is None:
        raise RuntimeError("no model")
    import torch
    import numpy as np
    x = np.asarray(window, dtype=np.float32)
    x = (x - _mu) / _sd
    with torch.no_grad():
        logits = m(torch.from_numpy(x[None]))
        p = torch.softmax(logits, -1).numpy()[0]
    ph, pi, po, pb = (float(v) for v in p)
    inner, outer = pi + pb, po + pb
    return {"inner": inner, "outer": outer, "healthy": ph,
            "anomaly": float(1 - ph), "both": bool(pb > 0.5), "p_both": pb}


def synth_tick(t: float, condition: str, health_pct: float, fault: str) -> dict:
    sev = 1.0 - health_pct / 100.0
    # vib_rms scaled to measured Paderborn range (real windows: ~0.1 healthy,
    # up to ~0.4 faulty) so charts and inference stay consistent.
    vib_rms = 0.12 + 0.28 * sev + 0.03 * math.sin(t) + random.gauss(0, 0.015)
    current = 2.3 + 0.5 * sev + random.gauss(0, 0.05)
    temp = 46 + 4 * sev + random.gauss(0, 0.2)
    pressure = (1.0 if "F10" in condition else 0.4) + 0.2 * sev + random.gauss(0, 0.02)
    rpm = (1500 if "N15" in condition else 900) + random.gauss(0, 5)
    return {"t": round(t, 1), "vib_rms": round(vib_rms, 3), "current": round(current, 3),
            "temp": round(temp, 2), "pressure": round(pressure, 3), "rpm": round(rpm, 1),
            "condition": condition, "health_pct": health_pct, "fault": fault,
            "severity": round(sev, 3)}


app = FastAPI(title="db-case bearing API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/model/info")
def info():
    metrics = {}
    mp = MODELS / "metrics.json"
    if mp.exists():
        metrics = json.loads(mp.read_text())
    return {"model": "SmallLSTM-64x2", "classes": CLASSES, "conditions": CONDITIONS,
            "weights_loaded": (MODELS / "lstm_small.pt").exists(), "metrics": metrics,
            "note": "temp/pressure are simulated; vibration+current anchored to Paderborn"}


@app.get("/bearings")
def bearings():
    return CATALOGUE


@app.post("/predict")
def predict(payload: dict):
    fault = str(payload.get("fault", "outer"))
    sev = float(payload.get("severity", payload.get("anomaly", 0.0)))
    window = payload.get("window")
    if window is not None:
        try:
            return infer_window(window)
        except Exception:
            pass
    # feature-based path: severity LADDER of real demo windows. v1 morphed
    # healthy<->fault along a line and the model read midpoints as the wrong
    # race; v2 picks a genuine window from the matching severity rung, so the
    # selected fault reads faithfully at every slider stop.
    try:
        import numpy as np
        m = get_model()
        bank = get_bank()
        if m is not None and bank is not None:
            cond = str(payload.get("condition", "N15_M07_F10"))
            if cond not in CONDITIONS:
                cond = "N15_M07_F10"
            key = fault if fault in ("healthy", "inner", "outer", "combined") else "outer"
            health = 100 * (1 - sev)
            if key == "healthy":
                # no fault injected: the machine stays healthy all along the slider
                rung = bank[f"{cond}_healthy"]
            elif health > 66:
                rung = bank[f"{cond}_healthy"]
            elif health > 33:
                rung = bank[f"{cond}_{'both' if key == 'combined' else key}_mild"]
            else:
                rung = bank[f"{cond}_{'both' if key == 'combined' else key}_severe"]
            win = rung[np.random.randint(len(rung))].astype(np.float32)
            win += np.random.normal(0, 1, win.shape).astype(np.float32) * 0.005
            return infer_window(win)
    except Exception:
        pass
    return heuristic_probs(sev, fault)


@app.websocket("/simulate/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    condition, health_pct, fault = "N15_M07_F10", 100.0, "outer"
    try:
        # first message (if any) carries config; 0.5s window to avoid blocking
        cfg = await asyncio.wait_for(ws.receive_json(), timeout=0.5)
        condition = str(cfg.get("condition", condition))
        health_pct = float(cfg.get("health_pct", health_pct))
        fault = str(cfg.get("fault", fault))
    except Exception:
        pass
    t, tick = 0.0, 0
    while True:
        try:
            # non-blocking config updates from client
            try:
                cfg = await asyncio.wait_for(ws.receive_json(), timeout=0.1)
                condition = str(cfg.get("condition", condition))
                health_pct = float(cfg.get("health_pct", health_pct))
                fault = str(cfg.get("fault", fault))
            except asyncio.TimeoutError:
                pass
            s = synth_tick(t, condition, health_pct, fault)
            # live inference every 5th tick (2Hz) to keep CPU light
            if tick % 5 == 0:
                try:
                    probs = predict({"fault": fault, "severity": s["severity"],
                                     "condition": s["condition"],
                                     "vib_rms": s["vib_rms"], "current": s["current"],
                                     "temp": s["temp"], "pressure": s["pressure"], "rpm": s["rpm"]})
                except Exception:
                    probs = heuristic_probs(s["severity"], fault)
                s["probs"] = probs
                s["pred"] = ("combined" if probs.get("both")
                             else max(("healthy", "inner", "outer"), key=lambda k: probs[k]))
            await ws.send_json(s)
            t += 0.1
            tick += 1
        except WebSocketDisconnect:
            break
        except Exception:
            await asyncio.sleep(0.1)
