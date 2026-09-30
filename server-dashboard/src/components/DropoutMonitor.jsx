import { CheckCircle, AlertTriangle, XCircle } from "lucide-react";

const STATUS_CONFIG = {
  all_active: {
    className: "active",
    icon:      CheckCircle,
    title:     "All clients active",
    sub:       "Training can continue",
    iconColor: "#22c55e",
  },
  temporarily_disconnected: {
    className: "warning",
    icon:      AlertTriangle,
    title:     "Client temporarily disconnected",
    sub:       "Monitoring reconnection...",
    iconColor: "#f59e0b",
  },
  grace_period: {
    className: "warning",
    icon:      AlertTriangle,
    title:     "Grace period active",
    sub:       "Waiting for client to reconnect",
    iconColor: "#f59e0b",
  },
  dropped_out: {
    className: "warning",
    icon:      AlertTriangle,
    title:     "Client dropped out",
    sub:       "Training continues with remaining clients",
    iconColor: "#f59e0b",
  },
  minimum_reached: {
    className: "warning",
    icon:      AlertTriangle,
    title:     "Minimum clients reached",
    sub:       "Training continues with reduced clients",
    iconColor: "#f59e0b",
  },
  insufficient_clients: {
    className: "danger",
    icon:      XCircle,
    title:     "Insufficient clients",
    sub:       "Training paused — waiting for clients",
    iconColor: "#ef4444",
  },
};

function MetricBox({ label, value, valueColor }) {
  return (
    <div className="dropout-item">
      <div className="dropout-item-label">{label}</div>
      <div className="dropout-item-val" style={valueColor ? { color: valueColor } : {}}>
        {value}
      </div>
    </div>
  );
}

export default function DropoutMonitor({ dropoutInfo = {} }) {
  const {
    expected_clients  = 3,
    active_clients    = 0,
    dropped_clients   = 0,
    minimum_required  = 2,
    overall_status    = "all_active",
  } = dropoutInfo;

  const cfg  = STATUS_CONFIG[overall_status] || STATUS_CONFIG.all_active;
  const Icon = cfg.icon;

  return (
    <div className="card">
      <div className="card-header">
        <span className="card-title">Client Dropout Monitor</span>
      </div>

      <div className="dropout-grid">
        <MetricBox label="Expected Clients" value={expected_clients} />
        <MetricBox
          label="Active Clients"
          value={active_clients}
          valueColor={active_clients > 0 ? "#22c55e" : undefined}
        />
        <MetricBox
          label="Dropped Clients"
          value={dropped_clients}
          valueColor={dropped_clients > 0 ? "#ef4444" : undefined}
        />
        <MetricBox label="Minimum Required" value={minimum_required} />
      </div>

      <div className={`dropout-status-panel ${cfg.className}`}>
        <div
          className="dropout-status-icon"
          style={{ background: `${cfg.iconColor}22` }}
        >
          <Icon size={16} color={cfg.iconColor} />
        </div>
        <div className="dropout-status-text">
          <div className="dst-title">{cfg.title}</div>
          <div className="dst-sub">{cfg.sub}</div>
        </div>
      </div>
    </div>
  );
}
