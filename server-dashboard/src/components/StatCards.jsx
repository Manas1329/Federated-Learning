import {
  Users, TrendingUp, Database, ShieldCheck, AlertTriangle, Settings2
} from "lucide-react";

const CARD_DEFS = [
  {
    key:          "activeHospitals",
    label:        "Active Hospitals",
    icon:         Users,
    iconBg:       "#1e3a5f",
    iconColor:    "#60a5fa",
    accentColor:  "#3b82f6",
    render: (d) => ({
      value:         `${d.activeHospitals.current} / ${d.activeHospitals.total}`,
      sub:           `${d.activeHospitals.participationPct}% Participation`,
      subClass:      "green",
      progress:      d.activeHospitals.participationPct,
      progressColor: "#3b82f6",
    }),
  },
  {
    key:          "globalAccuracy",
    label:        "Global Accuracy",
    icon:         TrendingUp,
    iconBg:       "#1a3a2a",
    iconColor:    "#4ade80",
    accentColor:  "#22c55e",
    render: (d) => ({
      value:    d.globalAccuracy.value > 0 ? `${d.globalAccuracy.value.toFixed(1)}%` : "—",
      sub:      d.globalAccuracy.delta !== null
                  ? `↑ ${Math.abs(d.globalAccuracy.delta)}% (vs last round)`
                  : "Waiting for first round",
      subClass: d.globalAccuracy.value > 0 ? "green" : "",
    }),
  },
  {
    key:          "currentRound",
    label:        "Current Round",
    icon:         Database,
    iconBg:       "#312a1a",
    iconColor:    "#fb923c",
    accentColor:  "#f59e0b",
    render: (d) => ({
      value:         `${d.currentRound.current} / ${d.currentRound.total}`,
      sub:           d.currentRound.current > 0
                       ? `${d.currentRound.total - d.currentRound.current} rounds remaining`
                       : "Not started",
      progress:      d.currentRound.total > 0
                       ? (d.currentRound.current / d.currentRound.total) * 100
                       : 0,
      progressColor: "#f59e0b",
    }),
  },
  {
    key:          "trustedClients",
    label:        "Trusted Clients",
    icon:         ShieldCheck,
    iconBg:       "#1e3a2e",
    iconColor:    "#34d399",
    accentColor:  "#22c55e",
    render: (d) => ({
      value:    `${d.trustedClients.value}`,
      sub:      `${d.trustedClients.pct}%`,
      subClass: d.trustedClients.value > 0 ? "green" : "",
    }),
  },
  {
    key:          "suspiciousClients",
    label:        "Suspicious Clients",
    icon:         AlertTriangle,
    iconBg:       "#3a1a1a",
    iconColor:    "#f87171",
    accentColor:  "#ef4444",
    render: (d) => ({
      value:    `${d.suspiciousClients.value}`,
      sub:      d.suspiciousClients.label || `${d.suspiciousClients.pct}%`,
      subClass: d.suspiciousClients.value > 0 ? "red" : "",
    }),
  },
  {
    key:          "unlearningRequests",
    label:        "Unlearning Requests",
    icon:         Settings2,
    iconBg:       "#2a1a3a",
    iconColor:    "#c084fc",
    accentColor:  "#a855f7",
    render: (d) => ({
      value: `${d.unlearningRequests.value}`,
      sub:   "No pending requests",
    }),
  },
];

export default function StatCards({ kpiData }) {
  if (!kpiData) return null;

  return (
    <div className="kpi-grid">
      {CARD_DEFS.map((card) => {
        const Icon     = card.icon;
        const rendered = card.render(kpiData);

        return (
          <div className="kpi-card" key={card.key}>
            <div className="kpi-card-top">
              <div>
                <div className="kpi-label" style={{ marginBottom: 4 }}>
                  {card.label}
                </div>
                <div className="kpi-value" style={{ color: card.accentColor }}>
                  {rendered.value}
                </div>
              </div>
              <div className="kpi-icon" style={{ background: card.iconBg }}>
                <Icon size={18} color={card.iconColor} />
              </div>
            </div>

            <div className={`kpi-sub ${rendered.subClass || ""}`}>
              {rendered.sub}
            </div>

            {rendered.progress != null && (
              <div className="progress-bar-wrap">
                <div
                  className="progress-bar-fill"
                  style={{
                    width:      `${Math.min(rendered.progress, 100)}%`,
                    background: rendered.progressColor,
                  }}
                />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
