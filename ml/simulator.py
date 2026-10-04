"""Fault-ramp simulator feeding the React cockpit over WebSocket @10Hz."""
import time, math, random


def stream(condition="N15_M07_F10", health_pct=100.0):
    t = 0.0
    sev = 1.0 - health_pct / 100.0
    while True:
        vib_rms = 0.5 + 2.0 * sev + 0.1 * math.sin(t) + random.gauss(0, 0.05)
        yield {"t": t, "vib_rms": vib_rms,
               "current": 2.3 + 0.5 * sev + random.gauss(0, 0.05),
               "temp": 46 + 4 * sev + random.gauss(0, 0.2),
               "pressure": 1.0 + 0.2 * sev + random.gauss(0, 0.02),
               "anomaly": min(1.0, sev + 0.05 * random.gauss(0, 1)),
               "condition": condition}
        t += 0.1
        time.sleep(0.1)
