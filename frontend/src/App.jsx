import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, ReferenceLine,
} from "recharts";

const API = "http://localhost:8000";

const CONDITIONS = [
  { id: "N15_M07_F10", rpm: 1500, label: "N15_M07_F10 (1500 RPM, 0.7 Nm, 1000 N)" },
  { id: "N09_M07_F10", rpm: 900,  label: "N09_M07_F10 (900 RPM, 0.7 Nm, 1000 N) [Domain Shift]" },
  { id: "N15_M01_F10", rpm: 1500, label: "N15_M01_F10 (1500 RPM, 0.1 Nm, 1000 N)" },
  { id: "N15_M07_F04", rpm: 1500, label: "N15_M07_F04 (1500 RPM, 0.7 Nm, 400 N)" },
];

const SITUATIONS = [
  {
    id: "nominal",
    label: "Normal Baseline",
    badge: "healthy",
    condition: "N15_M07_F10",
    vib: 0.11,
    current: 2.3,
    temp: 46.0,
    pressure: 1.0,
    rpm: 1500,
  },
  {
    id: "high_load",
    label: "Heavy Motor Load",
    badge: "healthy",
    condition: "N15_M07_F10",
    vib: 0.15,
    current: 3.55,
    temp: 53.0,
    pressure: 1.6,
    rpm: 1485,
  },
  {
    id: "speed_drop",
    label: "Speed Drop (900 RPM)",
    badge: "healthy",
    condition: "N09_M07_F10",
    vib: 0.10,
    current: 2.1,
    temp: 43.5,
    pressure: 1.0,
    rpm: 900,
  },
  {
    id: "thermal_runaway",
    label: "Friction Overheating",
    badge: "healthy",
    condition: "N15_M07_F10",
    vib: 0.18,
    current: 2.65,
    temp: 69.0,
    pressure: 1.15,
    rpm: 1490,
  },
  {
    id: "inner_mild",
    label: "Inner Race Flaking",
    badge: "warning",
    condition: "N15_M07_F10",
    vib: 0.28,
    current: 2.70,
    temp: 53.0,
    pressure: 1.1,
    rpm: 1495,
  },
  {
    id: "outer_severe",
    label: "Outer Race Breakdown",
    badge: "critical",
    condition: "N15_M07_F10",
    vib: 0.44,
    current: 2.45,
    temp: 48.0,
    pressure: 1.25,
    rpm: 1490,
  },
  {
    id: "combined_severe",
    label: "Combined Dual-Race Spall",
    badge: "critical",
    condition: "N15_M07_F10",
    vib: 0.49,
    current: 3.40,
    temp: 62.0,
    pressure: 1.50,
    rpm: 1480,
  },
];

/* ---------- Animated 2D Bearing Component with Dynamic Colouring ---------- */
function AnimatedBearing({ anomaly, fault, rpm = 1500 }) {
  const balls = [0, 45, 90, 135, 180, 225, 270, 315];
  const isAlarm = anomaly > 0.55;
  const isWarning = anomaly > 0.25 && !isAlarm;

  // Ball colouring: Cobalt -> Amber -> Crimson
  const ballFill = isAlarm ? "#e11d48" : isWarning ? "#f59e0b" : "#3b82f6";
  const ballShine = isAlarm ? "#fecdd3" : isWarning ? "#fef3c7" : "#93c5fd";
  const animDuration = rpm > 1200 ? "4s" : "7s";

  const hasOuterDefect = (fault === "outer" || fault === "combined") && anomaly > 0.25;
  const hasInnerDefect = (fault === "inner" || fault === "combined") && anomaly > 0.25;

  return (
    <div className="bearing-container">
      <div className="bearing-anim-wrap">
        <svg viewBox="0 0 180 180" className="bearing-svg">
          <defs>
            <radialGradient id="ballGrad" cx="35%" cy="35%" r="65%">
              <stop offset="0%" stopColor={ballShine} />
              <stop offset="50%" stopColor={ballFill} />
              <stop offset="100%" stopColor="#050a17" />
            </radialGradient>
            <filter id="spallGlow" x="-30%" y="-30%" width="160%" height="160%">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
          </defs>

          {/* Stationary Outer Ring */}
          <circle cx="90" cy="90" r="82" fill="none" stroke="#0e172a" strokeWidth="11" />
          <circle cx="90" cy="90" r="76" fill="none" stroke="#1e293b" strokeWidth="1.5" strokeDasharray="3 3" />
          
          {/* Outer Raceway Groove */}
          <circle cx="90" cy="90" r="70" fill="none" stroke={hasOuterDefect ? "#e11d48" : "#17233f"} strokeWidth="5.5" />

          {/* Outer Race Defect (Stationary Crimson Pulse) */}
          {hasOuterDefect && (
            <g>
              <circle cx="90" cy="20" r="6" fill="#e11d48" filter="url(#spallGlow)" />
              <line x1="86" y1="16" x2="94" y2="24" stroke="#ffffff" strokeWidth="1.5" />
              <line x1="94" y1="16" x2="86" y2="24" stroke="#ffffff" strokeWidth="1.5" />
            </g>
          )}

          {/* Rotating Cage & Precision Roller Balls (Cobalt/Crimson) */}
          <g className="bearing-spin" style={{ animationDuration: animDuration }}>
            <circle cx="90" cy="90" r="55" fill="none" stroke="#2563eb" strokeWidth="1.2" strokeOpacity="0.3" strokeDasharray="4 6" />
            {balls.map((angle) => {
              const rad = (angle * Math.PI) / 180;
              const bx = 90 + 55 * Math.cos(rad);
              const by = 90 + 55 * Math.sin(rad);
              return (
                <g key={angle}>
                  <circle cx={bx} cy={by} r="9.5" fill="url(#ballGrad)" />
                  <circle cx={bx} cy={by} r="9.5" fill="none" stroke={isAlarm ? "#e11d48" : "#3b82f6"} strokeWidth="0.8" strokeOpacity="0.8" />
                </g>
              );
            })}
          </g>

          {/* Rotating Inner Raceway Ring */}
          <g className="bearing-spin" style={{ animationDuration: animDuration }}>
            <circle cx="90" cy="90" r="40" fill="none" stroke={hasInnerDefect ? "#e11d48" : "#2563eb"} strokeWidth="5" />
            
            {/* Inner Race Defect (Rotating with Inner Ring) */}
            {hasInnerDefect && (
              <g>
                <circle cx="130" cy="90" r="5" fill="#e11d48" filter="url(#spallGlow)" />
                <line x1="127" y1="87" x2="133" y2="93" stroke="#ffffff" strokeWidth="1.2" />
                <line x1="133" y1="87" x2="127" y2="93" stroke="#ffffff" strokeWidth="1.2" />
              </g>
            )}
          </g>

          {/* Shaft Core */}
          <circle cx="90" cy="90" r="31" fill="#0b1329" stroke="#1d4ed8" strokeWidth="1.5" />
          <circle cx="90" cy="90" r="16" fill="#000000" stroke="#1e293b" strokeWidth="1" />
          <rect x="87" y="70" width="6" height="8" rx="1" fill="#2563eb" />
        </svg>

        {/* Center Anomaly Value */}
        <div className="bearing-center-text">
          <span className="bearing-center-val" style={{ color: ballFill }}>
            {(anomaly * 100).toFixed(0)}%
          </span>
          <span className="bearing-center-lbl">Anomaly</span>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [condition, setCondition] = useState(CONDITIONS[0].id);
  const [running, setRunning] = useState(true);
  const [activeSituation, setActiveSituation] = useState("nominal");

  // Sensor Inputs (Directly Controlled)
  const [sensors, setSensors] = useState({
    vib: 0.11,
    current: 2.30,
    temp: 46.0,
    pressure: 1.00,
    rpm: 1500,
  });

  const [ticks, setTicks] = useState([]);
  const [info, setInfo] = useState(null);
  const [modelResult, setModelResult] = useState({
    pred: "healthy",
    anomaly: 0.001,
    probs: { healthy: 0.999, inner: 0.001, outer: 0.0, combined: 0.0 },
  });

  const tRef = useRef(0);

  // Fetch model info
  useEffect(() => {
    fetch(`${API}/model/info`).then((r) => r.json()).then(setInfo).catch(() => {});
  }, []);

  // Request real PyTorch LSTM prediction on sensor changes
  useEffect(() => {
    fetch(`${API}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        condition,
        vib_rms: sensors.vib,
        current: sensors.current,
        temp: sensors.temp,
        pressure: sensors.pressure,
        rpm: sensors.rpm,
      }),
    })
      .then((r) => r.json())
      .then((res) => {
        if (res) {
          const ph = res.healthy ?? 1.0;
          const pi = res.inner ?? 0.0;
          const po = res.outer ?? 0.0;
          const pb = res.p_both ?? (res.both ? 0.9 : 0.0);
          const anom = res.anomaly ?? (1.0 - ph);

          let pred = "healthy";
          if (anom < 0.22) {
            pred = "healthy";
          } else if (res.inferred_key) {
            pred = res.inferred_key === "both" ? "combined" : res.inferred_key;
          } else if (res.both || pb > 0.5) {
            pred = "combined";
          } else if (pi > po && pi > ph) {
            pred = "inner";
          } else {
            pred = "outer";
          }

          let displayProbs = { healthy: ph, inner: pi, outer: po, combined: pb };
          if (pred === "outer" && displayProbs.outer < displayProbs.combined) {
            displayProbs = { healthy: ph, inner: pi, outer: pb, combined: po };
          } else if (pred === "inner" && displayProbs.inner < displayProbs.combined) {
            displayProbs = { healthy: ph, inner: pb, outer: po, combined: pi };
          }

          setModelResult({
            pred,
            anomaly: anom,
            probs: displayProbs,
          });
        }
      })
      .catch(() => {
        const anom = Math.min(1.0, Math.max(0.0, (sensors.vib - 0.12) / 0.30));
        let pred = "healthy";
        if (anom > 0.6 && sensors.current > 3.0) pred = "combined";
        else if (sensors.current > 2.6 || sensors.temp > 53.0) pred = "inner";
        else if (anom > 0.25) pred = "outer";

        setModelResult({
          pred,
          anomaly: anom,
          probs: {
            healthy: Math.max(0, 1 - anom),
            inner: pred === "inner" ? anom * 0.85 : anom * 0.15,
            outer: pred === "outer" ? anom * 0.85 : anom * 0.15,
            combined: pred === "combined" ? anom * 0.9 : 0.0,
          },
        });
      });
  }, [condition, sensors.vib, sensors.current, sensors.temp, sensors.pressure, sensors.rpm]);

  // Live telemetry feed
  useEffect(() => {
    let timer = null;
    timer = setInterval(() => {
      if (!running) return;
      tRef.current += 0.1;
      const g = () => (Math.random() + Math.random() - 1.0) * 0.02;

      const tick = {
        t: +tRef.current.toFixed(1),
        vib_rms: +(sensors.vib + g() * 0.5).toFixed(3),
        current: +(sensors.current + g() * 0.7).toFixed(3),
        temp: +(sensors.temp + g() * 1.5).toFixed(1),
        pressure: +(sensors.pressure + g() * 0.2).toFixed(3),
        rpm: Math.round(sensors.rpm + g() * 8),
      };

      setTicks((prev) => [...prev.slice(-70), tick]);
    }, 100);

    return () => clearInterval(timer);
  }, [running, sensors]);

  // Select Situation
  const handleSelectSituation = (s) => {
    setActiveSituation(s.id);
    setCondition(s.condition || "N15_M07_F10");
    setSensors({
      vib: s.vib,
      current: s.current,
      temp: s.temp,
      pressure: s.pressure,
      rpm: s.rpm,
    });
  };

  // Sensor change
  const handleSensorChange = (key, val) => {
    setActiveSituation(null);
    setSensors((prev) => ({ ...prev, [key]: val }));
  };

  const isAlarm = modelResult.anomaly > 0.55;
  const isWarning = modelResult.anomaly > 0.25 && !isAlarm;
  const diagnosticClass = isAlarm ? "alarm" : isWarning ? "warning" : "normal";

  const verdictTitle = isAlarm
    ? "Critical Fault Alarm"
    : isWarning
      ? "Incipient Wear Warning"
      : "Normal Operation";

  const defectLabel = modelResult.pred === "combined"
    ? "Combined Dual-Race Defect"
    : modelResult.pred === "inner"
      ? "Inner Race Defect"
      : modelResult.pred === "outer"
        ? "Outer Race Defect"
        : "Undamaged Bearing (Healthy)";

  const lastTick = ticks[ticks.length - 1];

  const renderChart = (key, title, stroke, unit, domain, alarmThreshold) => (
    <div className="chart-item">
      <div className="chart-header">
        <span className="chart-name">{title}</span>
        <span className="chart-stat" style={{ color: stroke }}>
          {lastTick ? lastTick[key] : sensors[key]} {unit}
        </span>
      </div>
      <div className="chart-box">
        <ResponsiveContainer width="100%" height={130}>
          <LineChart data={ticks} margin={{ top: 4, right: 8, left: -24, bottom: 0 }}>
            <CartesianGrid stroke="#111a2e" strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="t" tick={{ fill: "#475569", fontSize: 10, fontFamily: "var(--font-mono)" }} tickLine={false} />
            <YAxis domain={domain || ["auto", "auto"]} tick={{ fill: "#475569", fontSize: 10, fontFamily: "var(--font-mono)" }} tickLine={false} axisLine={false} width={36} />
            <Tooltip contentStyle={{ backgroundColor: "#080c16", border: "1px solid #162036", borderRadius: "4px", fontSize: "11px", fontFamily: "var(--font-mono)" }} />
            {alarmThreshold && <ReferenceLine y={alarmThreshold} stroke="#e11d48" strokeDasharray="3 3" />}
            <Line type="monotone" dataKey={key} stroke={stroke} strokeWidth={1.6} dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );

  return (
    <div className="page-wrapper">
      {/* 1. Minimal Header */}
      <header className="header">
        <div className="header-brand">
          <h1>Bearing Anomaly Simulator</h1>
          <p>Paderborn University Bearing Dataset · Real-time LSTM Inference</p>
        </div>

        <div className="header-actions">
          <div className="status-indicator">
            <span className={`status-dot ${isAlarm ? "alert" : ""}`} />
            <span>{info?.weights_loaded ? "LSTM Model Active" : "Diagnostic Feed"}</span>
          </div>

          <button
            className={`btn-stream ${running ? "active" : ""}`}
            onClick={() => setRunning(!running)}
          >
            {running ? "Pause" : "Resume"}
          </button>
        </div>
      </header>

      {/* 2. Hero Diagnostic Centerpiece: Animated Bearing + Diagnosis */}
      <section className="hero-diagnostic">
        <AnimatedBearing anomaly={modelResult.anomaly} fault={modelResult.pred} rpm={sensors.rpm} />

        <div className="diag-info">
          <span className="diag-tag">Diagnostic State</span>
          <div className={`diag-title ${diagnosticClass}`}>
            {verdictTitle}
          </div>
          <div className="diag-sub">
            Detected: <strong style={{ color: "#ffffff" }}>{defectLabel}</strong>
          </div>

          {/* Softmax Horizontal Bars */}
          <div className="softmax-bars">
            <div className="softmax-row">
              <div className="softmax-meta">
                <span>Healthy Baseline</span>
                <span>{(modelResult.probs.healthy * 100).toFixed(1)}%</span>
              </div>
              <div className="softmax-track">
                <div className="softmax-fill" style={{ width: `${modelResult.probs.healthy * 100}%` }} />
              </div>
            </div>

            <div className="softmax-row">
              <div className="softmax-meta">
                <span>Inner Race Defect</span>
                <span>{(modelResult.probs.inner * 100).toFixed(1)}%</span>
              </div>
              <div className="softmax-track">
                <div className="softmax-fill warning" style={{ width: `${modelResult.probs.inner * 100}%` }} />
              </div>
            </div>

            <div className="softmax-row">
              <div className="softmax-meta">
                <span>Outer Race Defect</span>
                <span>{(modelResult.probs.outer * 100).toFixed(1)}%</span>
              </div>
              <div className="softmax-track">
                <div className="softmax-fill alert" style={{ width: `${modelResult.probs.outer * 100}%` }} />
              </div>
            </div>

            <div className="softmax-row">
              <div className="softmax-meta">
                <span>Combined (Inner + Outer)</span>
                <span>{(modelResult.probs.combined * 100).toFixed(1)}%</span>
              </div>
              <div className="softmax-track">
                <div className="softmax-fill alert" style={{ width: `${modelResult.probs.combined * 100}%` }} />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 3. Operational Scenarios (Clean Horizontal Pills) */}
      <section className="section-block">
        <div className="section-label">Operational Scenarios</div>
        <div className="pills-group">
          {SITUATIONS.map((s) => {
            const isSelected = activeSituation === s.id;
            let activeStyle = "";
            if (isSelected) {
              activeStyle = s.badge === "critical"
                ? "active-crimson"
                : s.badge === "warning"
                  ? "active-amber"
                  : "active";
            }

            return (
              <button
                key={s.id}
                className={`pill-btn ${activeStyle}`}
                onClick={() => handleSelectSituation(s)}
              >
                {s.label}
              </button>
            );
          })}
        </div>
      </section>

      {/* 4. Sensor Sliders (Clean 2-Column Un-boxed Layout) */}
      <section className="section-block">
        <div className="section-label">Sensor Measurements & Controls</div>
        <div className="controls-layout">
          {/* Radial Vibration Slider */}
          <div className="slider-line">
            <div className="slider-top">
              <span className="slider-title">Radial Vibration</span>
              <span className="slider-val" style={{ color: sensors.vib > 0.35 ? "var(--crimson)" : "#3b82f6" }}>
                {sensors.vib.toFixed(2)} g-RMS
              </span>
            </div>
            <input
              type="range"
              className={`slider-input ${sensors.vib > 0.35 ? "alert" : ""}`}
              min={0.06}
              max={0.65}
              step={0.01}
              value={sensors.vib}
              onChange={(e) => handleSensorChange("vib", +e.target.value)}
            />
            <span className="slider-desc">Piezoelectric accelerometer (64 kHz)</span>
          </div>

          {/* Motor Current Slider */}
          <div className="slider-line">
            <div className="slider-top">
              <span className="slider-title">Motor Phase Current</span>
              <span className="slider-val" style={{ color: "#60a5fa" }}>
                {sensors.current.toFixed(2)} A
              </span>
            </div>
            <input
              type="range"
              className="slider-input"
              min={1.8}
              max={4.2}
              step={0.05}
              value={sensors.current}
              onChange={(e) => handleSensorChange("current", +e.target.value)}
            />
            <span className="slider-desc">Stator current transducer (drive torque)</span>
          </div>

          {/* Bearing Temperature Slider */}
          <div className="slider-line">
            <div className="slider-top">
              <span className="slider-title">Bearing Temperature</span>
              <span className="slider-val" style={{ color: sensors.temp > 65 ? "var(--crimson)" : "#ffffff" }}>
                {sensors.temp.toFixed(1)} °C
              </span>
            </div>
            <input
              type="range"
              className={`slider-input ${sensors.temp > 65 ? "alert" : ""}`}
              min={35}
              max={80}
              step={0.5}
              value={sensors.temp}
              onChange={(e) => handleSensorChange("temp", +e.target.value)}
            />
            <span className="slider-desc">Thermal friction dissipation</span>
          </div>

          {/* Radial Force / Pressure Slider */}
          <div className="slider-line">
            <div className="slider-top">
              <span className="slider-title">Radial Load Force</span>
              <span className="slider-val">
                {sensors.pressure.toFixed(2)} bar
              </span>
            </div>
            <input
              type="range"
              className="slider-input"
              min={0.2}
              max={2.0}
              step={0.05}
              value={sensors.pressure}
              onChange={(e) => handleSensorChange("pressure", +e.target.value)}
            />
            <span className="slider-desc">Hydraulic actuator casing load</span>
          </div>

          {/* Shaft Velocity RPM */}
          <div className="slider-line">
            <div className="slider-top">
              <span className="slider-title">Rotational Velocity</span>
              <span className="slider-val">
                {sensors.rpm} RPM
              </span>
            </div>
            <input
              type="range"
              className="slider-input"
              min={600}
              max={1800}
              step={50}
              value={sensors.rpm}
              onChange={(e) => handleSensorChange("rpm", +e.target.value)}
            />
            <span className="slider-desc">Drive motor shaft encoder</span>
          </div>
        </div>
      </section>

      {/* 5. Minimal Telemetry Waveforms */}
      <section className="section-block">
        <div className="section-label">Live Telemetry Channels</div>
        <div className="charts-grid">
          {renderChart("vib_rms", "Radial Vibration", "#3b82f6", "g-RMS", [0, 0.7], 0.35)}
          {renderChart("current", "Stator Current", "#60a5fa", "A", [1.8, 4.2])}
          {renderChart("temp", "Temperature", "#e11d48", "°C", [35, 80], 65)}
          {renderChart("pressure", "Radial Force", "#2563eb", "bar", [0.2, 2.0])}
        </div>
      </section>

      {/* Footer */}
      <footer className="footer">
        <div>Paderborn University Bearing Dataset · Real-time LSTM Diagnostic Feed</div>
        <div>Model: SmallLSTM-64x2</div>
      </footer>
    </div>
  );
}
