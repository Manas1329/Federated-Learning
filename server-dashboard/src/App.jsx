import { useState, useCallback, useEffect } from "react";
import "./index.css";

import Sidebar             from "./components/Sidebar";
import Header              from "./components/Header";
import StatCards           from "./components/StatCards";
import TrainingControl     from "./components/TrainingControl";
import PerformanceChart    from "./components/PerformanceChart";
import HospitalClientTable from "./components/HospitalClientTable";
import PerformanceMetrics  from "./components/PerformanceMetrics";
import UpdateStatistics    from "./components/UpdateStatistics";
import DropoutMonitor      from "./components/DropoutMonitor";
import RecentEvents        from "./components/RecentEvents";
import SecurityFooter      from "./components/SecurityFooter";
import ConnectionBanner    from "./components/ConnectionBanner";
import { TrustStatus }     from "./components/TrustStatus";

import { useLiveState }    from "./hooks/useLiveState";

// ── Theme helpers ─────────────────────────────────────────────────
function getStoredTheme() {
  return localStorage.getItem("fl-dashboard-theme") || "dark";
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("fl-dashboard-theme", theme);
}

// ──────────────────────────────────────────────────────────────────
// APP
// ──────────────────────────────────────────────────────────────────
export default function App() {
  const [activeNav, setActiveNav] = useState("System Overview");
  const [theme, setTheme]         = useState(getStoredTheme);

  // Live state from WebSocket / REST fallback
  const { liveState, isConnected, wsError, sendCommand } = useLiveState();

  // Apply theme on mount + on change
  useEffect(() => { applyTheme(theme); }, [theme]);

  const toggleTheme = () =>
    setTheme((t) => (t === "dark" ? "light" : "dark"));

  // ── Training control handlers ──────────────────────────────────
  const handleStart = async () => {
    const rounds  = liveState.total_rounds || 5;
    const clients = liveState.config?.target_clients || 3;
    await sendCommand("/api/training/start", {
      num_rounds:     rounds,
      target_clients: clients,
    });
  };

  const handleStop = async () => {
    await sendCommand("/api/training/stop");
  };

  const handleReset = async () => {
    await sendCommand("/api/training/reset");
  };

  // ── Derived values for components ─────────────────────────────
  const config   = liveState.config   || {};
  const dropout  = liveState.dropout_info || {};
  const metrics  = liveState.latest_metrics || {};
  const updateSt = liveState.update_stats  || {};
  const trustData = liveState.trust_data   || {};

  // Build hospital rows from live client_details
  const hospitals = buildHospitalRows(liveState);

  // KPI card data derived from live state
  const kpiData = buildKpiData(liveState);

  // Performance chart history
  const perfHistory = (liveState.performance_history || []).map((r) => ({
    round:    `R${r.round}`,
    accuracy: r.accuracy,
    loss:     r.loss,
  }));

  // Summary metrics for chart footer
  const perfSummary = buildPerfSummary(liveState.performance_history || []);

  return (
    <div className="app-layout">
      {/* Sidebar */}
      <Sidebar activeItem={activeNav} onNavClick={setActiveNav} />

      {/* Main */}
      <div className="main-area">
        {/* Header with theme toggle */}
        <Header
          theme={theme}
          onToggleTheme={toggleTheme}
          isConnected={isConnected}
        />

        {/* Connection status banner */}
        {wsError && <ConnectionBanner message={wsError} />}

        {/* Scrollable content */}
        <div className="page-content">

          {/* KPI Cards */}
          <StatCards kpiData={kpiData} />

          {/* Training Control Panel */}
          <TrainingControl
            status={liveState.training_status || "Ready"}
            clientsConnected={liveState.connected_clients?.length || 0}
            totalClients={config.target_clients || 3}
            totalRounds={liveState.total_rounds || 5}
            currentRound={liveState.current_round || 0}
            onStart={handleStart}
            onStop={handleStop}
            onReset={handleReset}
          />

          {/* Row 1: Performance Chart + Hospital Table */}
          <div className="main-grid">
            <PerformanceChart
              history={perfHistory}
              summary={perfSummary}
            />
            <HospitalClientTable hospitals={hospitals} />
          </div>

          {/* Row 2: Metrics + Update Stats + Dropout Monitor + Events */}
          <div className="main-grid-3">
            <PerformanceMetrics
              metrics={metrics}
              useQuantization={config.use_quantization}
              useDp={config.use_dp}
            />
            <UpdateStatistics
              stats={updateSt}
              useQuantization={config.use_quantization}
              useDp={config.use_dp}
            />
            <DropoutMonitor dropoutInfo={dropout} />
            <RecentEvents events={liveState.events || []} />
          </div>

          {/* Row 3: Client Trust / Tagging Status */}
          <TrustStatus trustData={trustData} />

        </div>

        {/* Security Footer */}
        <SecurityFooter
          useQuantization={config.use_quantization}
          useDp={config.use_dp}
          isOperational={isConnected}
        />
      </div>
    </div>
  );
}

// ──────────────────────────────────────────────────────────────────
// Data transformation helpers
// ──────────────────────────────────────────────────────────────────

function buildHospitalRows(liveState) {
  const details  = liveState.client_details || {};
  const names    = liveState.connected_clients || [];

  // If we have live client details, use them
  if (names.length > 0) {
    return names.map((name, i) => {
      const d = details[name] || {};
      return {
        id:         name,
        name:       formatClientName(name),
        location:   getHospitalLocation(name),
        status:     d.status || "Connected",
        accuracy:   d.accuracy ?? null,
        loss:       d.loss ?? null,
        updateSize: d.update_size || "—",
        lastUpdate: d.last_update || "—",
      };
    });
  }

  // No clients connected yet — show placeholder rows
  return [
    { id: "hospital-a", name: "Hospital A", location: "Mumbai",    status: "Waiting", accuracy: null, loss: null, updateSize: "—", lastUpdate: "—" },
    { id: "hospital-b", name: "Hospital B", location: "Delhi",     status: "Waiting", accuracy: null, loss: null, updateSize: "—", lastUpdate: "—" },
    { id: "hospital-c", name: "Hospital C", location: "Bangalore", status: "Waiting", accuracy: null, loss: null, updateSize: "—", lastUpdate: "—" },
  ];
}

function formatClientName(raw) {
  // "Hospital_A" → "Hospital A", "hospital_b" → "Hospital B"
  return raw.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function getHospitalLocation(name) {
  const lower = (name || "").toLowerCase();
  if (lower.includes("_a") || lower.endsWith("a")) return "Mumbai";
  if (lower.includes("_b") || lower.endsWith("b")) return "Delhi";
  if (lower.includes("_c") || lower.endsWith("c")) return "Bangalore";
  return "India";
}

function buildKpiData(liveState) {
  const config  = liveState.config || {};
  const dropout = liveState.dropout_info || {};
  const metrics = liveState.latest_metrics || {};
  const history = liveState.performance_history || [];
  const target  = config.target_clients || 3;
  const active  = liveState.connected_clients?.length || 0;

  // Best accuracy from history
  const accuracies = history.map((r) => r.accuracy);
  const bestAcc    = accuracies.length > 0 ? Math.max(...accuracies) : 0;
  const prevAcc    = accuracies.length >= 2 ? accuracies[accuracies.length - 2] : null;
  const accDelta   = prevAcc !== null ? (metrics.accuracy - prevAcc).toFixed(1) : null;

  return {
    activeHospitals: {
      current:          active,
      total:            target,
      participationPct: target > 0 ? Math.round((active / target) * 100) : 0,
    },
    globalAccuracy: {
      value: metrics.accuracy ?? 0,
      delta: accDelta,
    },
    currentRound: {
      current: liveState.current_round || 0,
      total:   liveState.total_rounds  || 5,
    },
    trustedClients: {
      value: active,
      pct:   target > 0 ? Math.round((active / target) * 100) : 0,
    },
    suspiciousClients: {
      value: 0,
      pct:   0,
      label: "Not Implemented",
    },
    unlearningRequests: {
      value: 0,
    },
  };
}

function buildPerfSummary(history) {
  if (!history || history.length === 0) {
    return {
      bestAccuracy: 0, bestAccuracyRound: "—",
      avgAccuracy: 0,
      bestLoss: 0, bestLossRound: "—",
      currentLoss: 0, lossChange: 0,
    };
  }
  const accs   = history.map((r) => r.accuracy);
  const losses = history.map((r) => r.loss);
  const maxAcc = Math.max(...accs);
  const minLoss = Math.min(...losses);
  const avgAcc = accs.reduce((a, b) => a + b, 0) / accs.length;
  const curr   = history[history.length - 1];
  const prev   = history.length >= 2 ? history[history.length - 2] : null;

  return {
    bestAccuracy:      parseFloat(maxAcc.toFixed(1)),
    bestAccuracyRound: `Round ${history[accs.indexOf(maxAcc)]?.round || "?"}`,
    avgAccuracy:       parseFloat(avgAcc.toFixed(1)),
    bestLoss:          parseFloat(minLoss.toFixed(4)),
    bestLossRound:     `Round ${history[losses.indexOf(minLoss)]?.round || "?"}`,
    currentLoss:       parseFloat(curr.loss.toFixed(4)),
    lossChange:        prev ? parseFloat((curr.loss - prev.loss).toFixed(4)) : 0,
  };
}
