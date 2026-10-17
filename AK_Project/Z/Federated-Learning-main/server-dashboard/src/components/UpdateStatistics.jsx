import {
  UploadCloud, Hash, HardDrive, Minimize2, Percent, Lock
} from "lucide-react";

function StatRow({ icon: Icon, label, value, color }) {
  return (
    <div className="update-stat-item">
      <div className="update-stat-left">
        <Icon color={color} />
        {label}
      </div>
      <span className="update-stat-val">{value}</span>
    </div>
  );
}

function fmt(val, decimals = 1, suffix = "") {
  if (!val && val !== 0) return "—";
  return `${Number(val).toFixed(decimals)}${suffix}`;
}

export default function UpdateStatistics({ stats = {}, useQuantization = true, useDp = false }) {
  const {
    total_updates_received   = 0,
    updates_this_round       = 0,
    avg_update_size_raw_mb   = 0,
    avg_update_size_quant_mb = 0,
    compression_ratio        = 0,
    dp_epsilon_avg           = 0,
  } = stats;

  return (
    <div className="card">
      <div className="card-header">
        <span className="card-title">Model Update Statistics</span>
      </div>

      <div className="update-stat-list">
        <StatRow
          icon={Hash}
          color="#60a5fa"
          label="Total Updates Received"
          value={total_updates_received || "0"}
        />
        <StatRow
          icon={UploadCloud}
          color="#4ade80"
          label="Updates This Round"
          value={updates_this_round || "0"}
        />
        <StatRow
          icon={HardDrive}
          color="#fb923c"
          label="Avg Update Size (Raw)"
          value={avg_update_size_raw_mb > 0 ? `${fmt(avg_update_size_raw_mb, 2)} MB` : "—"}
        />
        <StatRow
          icon={Minimize2}
          color="#c084fc"
          label="Avg Update Size (Quantized)"
          value={
            !useQuantization
              ? "Disabled"
              : avg_update_size_quant_mb > 0
                ? `${fmt(avg_update_size_quant_mb, 2)} MB`
                : "—"
          }
        />
        <StatRow
          icon={Percent}
          color="#34d399"
          label="Compression Ratio"
          value={
            !useQuantization
              ? "N/A"
              : compression_ratio > 0
                ? `${fmt(compression_ratio, 2)}x`
                : "—"
          }
        />
        <StatRow
          icon={Lock}
          color="#f472b6"
          label="DP Epsilon (Avg)"
          value={
            !useDp
              ? "Disabled"
              : dp_epsilon_avg > 0
                ? fmt(dp_epsilon_avg, 4)
                : "—"
          }
        />
      </div>
    </div>
  );
}
