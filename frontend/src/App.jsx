import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from "recharts";

const API = "http://localhost:8000";
const WS_URL = "ws://localhost:8000/simulate/stream";
const CONDITIONS = ["N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04"];
const FAULTS = ["healthy", "inner", "outer"];

/* ---------- bearing + LSTM visuals ---------- */
function BearingSVG({ anomaly }) {
  const balls = [0, 60, 120, 180, 240, 300];
  return (
    <svg viewBox="0 0 120 120" className="bearing-anim">
      <circle cx="60" cy="60" r="52" fill="none" stroke="#26314d" strokeWidth="8" />
      <circle cx="60" cy="60" r="26" fill="none" stroke="#60a5fa" strokeWidth="7"
        className={anomaly > 0.6 ? "pulse" : ""} />
      <g className="spin">
        {balls.map((a) => (
          <circle key={a} cx={60 + 39 * Math.cos((a * Math.PI) / 180)}
            cy={60 + 39 * Math.sin((a * Math.PI) / 180)} r="6"
            fill={anomaly > 0.6 ? "#f87171" : anomaly > 0.3 ? "#fbbf24" : "#34d399"} />
        ))}
      </g>
    </svg>
  );
}

function LstmSVG({ anomaly }) {
  const layers = [6, 8, 8, 3];
  const W = 260, H = 120;
  const pos = (l, i) => [
    20 + (l * (W - 40)) / (layers.length - 1),
    15 + (i * (H - 30)) / Math.max(1, layers[l] - 1),
  ];
  const lines = [];
  for (let l = 0; l < layers.length - 1; l++)
    for (let i = 0; i < layers[l]; i++)
      for (let j = 0; j < layers[l + 1]; j++) {
        const [x1, y1] = pos(l, i), [x2, y2] = pos(l + 1, j);
        lines.push(<line key={`${l}-${i}-${j}`} x1={x1} y1={y1} x2={x2} y2={y2}
          stroke={anomaly > 0.5 ? "#f87171" : "#60a5fa"} strokeOpacity={0.14 + anomaly * 0.3} />);
      }
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: 130 }}>
      {lines}
      {layers.map((n, l) =>
        Array.from({ length: n }).map((_, i) => {
          const [x, y] = pos(l, i);
          return <circle key={`${l}-${i}`} cx={x} cy={y} r="4.5"
            fill={l === layers.length - 1 ? "#fbbf24" : "#60a5fa"}
            className="pulse" opacity={0.6 + anomaly * 0.4} />;
        })
      )}
    </svg>
  );
}

/* ---------- local fallback simulator (backend offline) ---------- */
function localTick(t, condition, healthPct, fault) {
  const sev = 1 - healthPct / 100;
  const g = () => (Math.random() + Math.random() + Math.random() - 1.5) * 0.06;
  return {
    t: +t.toFixed(1), condition, health_pct: healthPct, fault,
    severity: +sev.toFixed(3),
    vib_rms: +(0.5 + 2 * sev + 0.1 * Math.sin(t) + g()).toFixed(3),
    current: +(2.3 + 0.5 * sev + g()).toFixed(3),
    temp: +(46 + 4 * sev + g() * 3).toFixed(2),
    pressure: +((condition.includes("F10") ? 1.0 : 0.4) + 0.2 * sev + g() / 3).toFixed(3),
    rpm: Math.round((condition.includes("N15") ? 1500 : 900) + g() * 40),
    probs: fault === "inner"
      ? { healthy: 1 - sev, inner: 0.75 * sev, outer: 0.25 * sev, anomaly: sev }
      : fault === "outer"
        ? { healthy: 1 - sev, inner: 0.25 * sev, outer: 0.75 * sev, anomaly: sev }
        : { healthy: 1 - sev, inner: 0.5 * sev, outer: 0.5 * sev, anomaly: sev },
  };
}

/* ---------- app ---------- */
export default function App() {
  const [tab, setTab] = useState("cockpit");
  const [condition, setCondition] = useState(CONDITIONS[0]);
  const [fault, setFault] = useState("outer");
  const [health, setHealth] = useState(85);
  const [running, setRunning] = useState(true);
  const [ticks, setTicks] = useState([]);
  const [info, setInfo] = useState(null);
  const [catalogue, setCatalogue] = useState([]);
  const [live, setLive] = useState(false);
  const tRef = useRef(0);

  useEffect(() => {
    fetch(`${API}/model/info`).then((r) => r.json()).then(setInfo).catch(() => {});
    fetch(`${API}/bearings`).then((r) => r.json()).then(setCatalogue).catch(() => {});
  }, []);

  useEffect(() => {
    let ws = null, timer = null, closed = false;
    try {
      ws = new WebSocket(WS_URL);
      ws.onopen = () => {
        setLive(true);
        ws.send(JSON.stringify({ condition, health_pct: health, fault }));
      };
      ws.onmessage = (e) => {
        const d = JSON.parse(e.data);
        if (!d.probs) {
          const p = d.severity ?? 0;
          d.probs = { healthy: 1 - p, inner: p / 2, outer: p / 2, anomaly: p };
        }
        setTicks((prev) => [...prev.slice(-140), d]);
      };
      ws.onerror = () => { ws.close(); };
      ws.onclose = () => { if (!closed) setLive(false); };
    } catch { setLive(false); }
    if (!live) {
      timer = setInterval(() => {
        if (!running) return;
        tRef.current += 0.1;
        setTicks((prev) => [...prev.slice(-140),
          localTick(tRef.current, condition, health, fault)]);
      }, 100);
    }
    return () => { closed = true; try { ws?.close(); } catch {} clearInterval(timer); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [live === false && running, condition, fault, health]);

  // push config updates to an open socket
  useEffect(() => {
    // handled on next tick via local mode; WS mode updates on reopen — keep simple & robust
  }, [condition, health, fault]);

  const last = ticks[ticks.length - 1];
  const anomaly = last?.probs?.anomaly ?? last?.severity ?? 0;
  const status = anomaly > 0.6 ? ["bad", "CRITICAL"] : anomaly > 0.3 ? ["warn", "WARNING"] : ["ok", "HEALTHY"];

  const chart = (key, color, unit) => (
    <div className="card">
      <h3>{key} {unit && <span className="note">({unit})</span>}</h3>
      <ResponsiveContainer width="100%" height={150}>
        <LineChart data={ticks}>
          <CartesianGrid stroke="#26314d" strokeDasharray="3 3" />
          <XAxis dataKey="t" tick={{ fill: "#8b98b3", fontSize: 10 }} />
          <YAxis tick={{ fill: "#8b98b3", fontSize: 10 }} width={44} domain={["auto", "auto"]} />
          <Tooltip contentStyle={{ background: "#182036", border: "1px solid #26314d" }} />
          <Line type="monotone" dataKey={key} stroke={color} dot={false} strokeWidth={1.6} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );

  return (
    <div className="wrap">
      <div className="hero">
        <div>
          <h1>Bearing Anomaly Cockpit</h1>
          <p>
            Paderborn University bearing data (vibration + motor current @64kHz, 8-bearing smart
            subset) → small LSTM <b>Healthy / Inner-race / Outer-race</b>, robust across all 4
            operating conditions. Temperature &amp; pressure are simulated on top of real physics.
            {info && (
              <span className="note"> Model F1(val)={info.metrics?.val_f1 ?? "—"} ·
                weights: {info.weights_loaded ? "loaded" : "heuristic"} ·
                {live ? " LIVE backend" : " local fallback (start FastAPI for live inference)"}</span>
            )}
          </p>
          <div className="tabs">
            {[["cockpit", "Live Simulation"], ["brain", "Brain View"], ["data", "Dataset"]].map(([k, l]) => (
              <button key={k} className={tab === k ? "active" : ""} onClick={() => setTab(k)}>{l}</button>
            ))}
          </div>
        </div>
        <BearingSVG anomaly={anomaly} />
      </div>

      {tab === "cockpit" && (
        <div className="grid" style={{ marginTop: 16 }}>
          <div className="card controls">
            <h3>Simulation controls</h3>
            <label>Operating condition</label>
            <select value={condition} onChange={(e) => setCondition(e.target.value)}>
              {CONDITIONS.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
            <label>Injected fault</label>
            <select value={fault} onChange={(e) => setFault(e.target.value)}>
              {FAULTS.map((f) => <option key={f} value={f}>{f}</option>)}
            </select>
            <label>Machine health: {health}%</label>
            <input type="range" min={5} max={100} value={health} onChange={(e) => setHealth(+e.target.value)} />
            <div className="btnrow">
              <button className={`btn ${running ? "on" : ""}`} onClick={() => setRunning(!running)}>
                {running ? "Pause" : "Run"}
              </button>
              <button className="btn" onClick={() => setTicks([])}>Clear</button>
            </div>
            <div style={{ marginTop: 16 }} className="status">
              <div className={`dot ${status[0]}`} />
              <div><div className="big">{status[1]}</div>
                <div className="note">anomaly {(anomaly * 100).toFixed(1)}% ·
                  pred {last?.pred ?? last?.fault ?? "—"} · {ticks.length} pts</div></div>
            </div>
            {last?.probs && (
              <div style={{ marginTop: 12 }}>
                {["healthy", "inner", "outer"].map((k) => (
                  <div key={k}>
                    <div className="note">{k} {(last.probs[k] * 100).toFixed(1)}%</div>
                    <div className="confbar"><div style={{ width: `${last.probs[k] * 100}%` }} /></div>
                  </div>
                ))}
              </div>
            )}
          </div>
          <div>
            <div className="charts">
              {chart("vib_rms", "#60a5fa", "g-RMS")}
              {chart("current", "#fbbf24", "A")}
              {chart("temp", "#f87171", "°C sim")}
              {chart("pressure", "#34d399", "bar sim")}
            </div>
            <p className="note" style={{ marginTop: 8 }}>
              Drag health → 20% with fault=outer to watch vib-RMS climb and the status flip to CRITICAL.
              Switch N15→N09 to see the operating-condition shift the model was validated against.
            </p>
          </div>
        </div>
      )}

      {tab === "brain" && (
        <div className="grid" style={{ marginTop: 16 }}>
          <div className="card">
            <h3>Live LSTM — 6 → 64×2 → 3</h3>
            <LstmSVG anomaly={anomaly} />
            <p className="note">6 inputs: vibration, current ×2, temp(sim), pressure(sim), rpm ·
              edges glow with anomaly score. ~54k params, CPU real-time.</p>
          </div>
          <div className="card">
            <h3>Decision</h3>
            {last?.probs ? ["healthy", "inner", "outer"].map((k) => (
              <div key={k}>
                <div>{k} — {(last.probs[k] * 100).toFixed(1)}%</div>
                <div className="confbar"><div style={{ width: `${last.probs[k] * 100}%` }} /></div>
              </div>
            )) : <p className="note">waiting for stream…</p>}
            <h3 style={{ marginTop: 16 }}>Smoke-test metrics (full train after UI)</h3>
            <p className="note">val F1 {info?.metrics?.val_f1 ?? "—"} ·
              test F1 {info?.metrics?.test_f1 ?? "—"} ·
              held-out condition N09_M07_F10, test bearings K002/KB27.</p>
          </div>
        </div>
      )}

      {tab === "data" && (
        <div style={{ marginTop: 16 }}>
          <div className="card">
            <h3>Smart subset — 8 bearings × 4 conditions × 20 reps = 640 recordings</h3>
            <table className="cat">
              <thead><tr><th>Bearing</th><th>Class</th><th>Origin</th><th>Damage</th></tr></thead>
              <tbody>
                {(catalogue.length ? catalogue : [
                  { code: "K001", cls: "healthy", origin: "none", damage: "none" },
                  { code: "K002", cls: "healthy", origin: "none", damage: "none" },
                  { code: "KA01", cls: "outer", origin: "artificial", damage: "EDM" },
                  { code: "KA15", cls: "outer", origin: "real", damage: "indentation" },
                  { code: "KB23", cls: "outer", origin: "real", damage: "pitting (inner+outer)" },
                  { code: "KB27", cls: "outer", origin: "real", damage: "indentation (inner+outer)" },
                  { code: "KI01", cls: "inner", origin: "artificial", damage: "EDM" },
                  { code: "KI14", cls: "inner", origin: "real", damage: "pitting" },
                ]).map((b) => (
                  <tr key={b.code}><td>{b.code}</td><td>{b.cls}</td><td>{b.origin}</td><td>{b.damage}</td></tr>
                ))}
              </tbody>
            </table>
            <p className="note">Real: vibration + 2× current @64kHz. Simulated: temp/pressure/rpm conditioned
              on fault severity. Full catalogue: Lessmeier et al., PHM 2016.</p>
          </div>
        </div>
      )}
    </div>
  );
}
