import { Building2 } from "lucide-react";

const STATUS_CLASSES = {
  "Connected":        "connected",
  "Training":         "training",
  "Update Submitted": "training",
  "Waiting":          "waiting",
  "Dropped Out":      "dropped",
  "Disconnected":     "disconnected",
  "Suspicious":       "suspicious",
  "Malicious":        "malicious",
};

const STATUS_DOTS = {
  "Connected":        "#22c55e",
  "Training":         "#3b82f6",
  "Update Submitted": "#a855f7",
  "Waiting":          "#f59e0b",
  "Dropped Out":      "#ef4444",
  "Disconnected":     "#ef4444",
  "Suspicious":       "#f97316",
  "Malicious":        "#ef4444",
};

function StatusBadge({ status }) {
  const cls     = STATUS_CLASSES[status] || "waiting";
  const dotClr  = STATUS_DOTS[status]    || "#64748b";
  return (
    <span className={`status-badge ${cls}`}>
      <span style={{ width: 7, height: 7, borderRadius: "50%", background: dotClr, display: "inline-block" }} />
      {status}
    </span>
  );
}

function LossBar({ value }) {
  if (value === null || value === undefined) {
    return <span style={{ color: "var(--text-muted)", fontSize: 11 }}>—</span>;
  }
  const pct = Math.min((value / 0.5) * 100, 100);
  return (
    <div className="loss-bar-wrap">
      <span style={{ fontWeight: 600, minWidth: 36 }}>{value.toFixed(2)}</span>
      <div className="loss-bar">
        <div className="loss-bar-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

const ICON_COLORS = ["#3b82f6", "#22c55e", "#a855f7"];
const ICON_BG     = ["#1e3a5f", "#1a3a2a", "#2e1a4a"];

export default function HospitalClientTable({ hospitals = [] }) {
  return (
    <div className="card">
      <div className="card-header">
        <span className="card-title">Hospital Client Status</span>
        <span className="card-link">View All</span>
      </div>

      <table className="hospital-table">
        <thead>
          <tr>
            <th>Hospital</th>
            <th>Status</th>
            <th>Accuracy</th>
            <th>Loss</th>
            <th>Update Size</th>
            <th>Last Update</th>
          </tr>
        </thead>
        <tbody>
          {hospitals.map((h, i) => (
            <tr key={h.id}>
              <td>
                <div className="hospital-name-cell">
                  <div className="hospital-icon" style={{ background: ICON_BG[i % ICON_BG.length] }}>
                    <Building2 size={15} color={ICON_COLORS[i % ICON_COLORS.length]} />
                  </div>
                  <div className="hospital-text">
                    <div className="h-name">{h.name}</div>
                    <div className="h-loc">{h.location}</div>
                  </div>
                </div>
              </td>
              <td><StatusBadge status={h.status} /></td>
              <td>
                {h.accuracy !== null
                  ? <span style={{ fontWeight: 600, color: "#4ade80" }}>{h.accuracy}%</span>
                  : <span style={{ color: "var(--text-muted)", fontSize: 11 }}>—</span>
                }
              </td>
              <td><LossBar value={h.loss} /></td>
              <td style={{ color: "var(--text-secondary)" }}>{h.updateSize}</td>
              <td style={{ color: "var(--text-muted)", fontSize: 11.5 }}>{h.lastUpdate}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
