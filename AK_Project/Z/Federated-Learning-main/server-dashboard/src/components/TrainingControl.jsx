import { Play, Square, RefreshCw, AlertCircle } from "lucide-react";

const STATUS_CONFIG = {
  Ready:     { color: "#22c55e", dotColor: "#22c55e" },
  Waiting:   { color: "#f59e0b", dotColor: "#f59e0b" },
  Training:  { color: "#3b82f6", dotColor: "#3b82f6" },
  Completed: { color: "#22c55e", dotColor: "#22c55e" },
  Stopped:   { color: "#ef4444", dotColor: "#ef4444" },
  Error:     { color: "#ef4444", dotColor: "#ef4444" },
};

export default function TrainingControl({
  status,
  clientsConnected,
  totalClients,
  totalRounds,
  currentRound,
  onStart,
  onStop,
  onReset,
}) {
  const cfg        = STATUS_CONFIG[status] || STATUS_CONFIG.Ready;
  const isTraining = status === "Training" || status === "Waiting";
  const isReady    = status === "Ready";
  const isDone     = status === "Completed" || status === "Stopped" || status === "Error";

  return (
    <div className="training-control">
      <span className="tc-label">Federated Training Control</span>

      <div className="tc-divider" />

      {/* Status */}
      <div className="tc-status" style={{ color: cfg.color }}>
        <span
          className="tc-dot"
          style={{
            background:  cfg.dotColor,
            boxShadow:   `0 0 6px ${cfg.dotColor}`,
            animation:   isTraining ? "pulse 1.5s ease-in-out infinite" : "none",
          }}
        />
        Status: <strong style={{ color: cfg.color, marginLeft: 4 }}>{status}</strong>
      </div>

      <div className="tc-divider" />

      {/* Clients connected */}
      <div className="tc-info">
        Clients Connected:{" "}
        <strong style={{ color: clientsConnected >= totalClients ? "var(--green)" : "var(--orange)" }}>
          {clientsConnected} / {totalClients}
        </strong>
      </div>

      <div className="tc-divider" />

      {/* Rounds */}
      <div className="tc-info">
        {isTraining
          ? <>Round: <strong style={{ color: "var(--blue)" }}>{currentRound} / {totalRounds}</strong></>
          : <>Rounds: <strong>{totalRounds}</strong></>
        }
      </div>

      {/* Buttons */}
      <div className="tc-buttons">
        {/* Start */}
        <button
          className="btn btn-start"
          onClick={onStart}
          disabled={isTraining}
          title="Start Training"
          id="btn-start-training"
        >
          <Play size={14} />
          Start Training
        </button>

        {/* Pause — disabled in Phase 1 (Flower doesn't support mid-round pause) */}
        <button
          className="btn btn-pause"
          disabled
          title="Pause not available during active Flower round"
          style={{ position: "relative" }}
          id="btn-pause-training"
        >
          <AlertCircle size={14} />
          Pause
        </button>

        {/* Stop */}
        <button
          className="btn btn-stop"
          onClick={onStop}
          disabled={!isTraining}
          title={isTraining ? "Stop training (terminates FL process)" : "Not running"}
          id="btn-stop-training"
        >
          <Square size={14} />
          Stop
        </button>

        {/* Reset */}
        <button
          className="btn btn-reset"
          onClick={onReset}
          disabled={isTraining}
          title={isTraining ? "Cannot reset while training" : "Reset dashboard state"}
          id="btn-reset-training"
        >
          <RefreshCw size={14} />
          Reset
        </button>
      </div>
    </div>
  );
}
