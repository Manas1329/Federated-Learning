import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from "recharts";

const METRIC_DEFS = [
  { key: "accuracy",  label: "Accuracy",  color: "#3b82f6" },
  { key: "precision", label: "Precision", color: "#22c55e" },
  { key: "recall",    label: "Recall",    color: "#f59e0b" },
  { key: "f1",        label: "F1 Score",  color: "#a855f7" },
];

const CustomTooltip = ({ active, payload }) => {
  if (active && payload?.length) {
    return (
      <div style={{
        background: "var(--bg-card)", border: "1px solid var(--border-light)",
        borderRadius: 8, padding: "6px 12px", fontSize: 12,
      }}>
        <div style={{ color: payload[0].payload.color, fontWeight: 700 }}>
          {payload[0].name}: {payload[0].value.toFixed(1)}%
        </div>
      </div>
    );
  }
  return null;
};

function EmptyDoughnut() {
  return (
    <div style={{
      width: 160, height: 160, borderRadius: "50%",
      background: "var(--bg-card-alt)",
      border: "2px dashed var(--border)",
      display: "flex", alignItems: "center", justifyContent: "center",
      flexDirection: "column",
    }}>
      <span style={{ fontSize: 11, color: "var(--text-muted)", textAlign: "center", padding: "0 12px" }}>
        Waiting for first round
      </span>
    </div>
  );
}

export default function PerformanceMetrics({ metrics = {}, useQuantization, useDp }) {
  const hasData = metrics.accuracy > 0;

  const chartData = METRIC_DEFS.map((item) => ({
    name:  item.label,
    value: parseFloat(((metrics[item.key] || 0)).toFixed(1)),
    color: item.color,
  }));

  return (
    <div className="card" style={{ display: "flex", flexDirection: "column" }}>
      <div className="card-header">
        <span className="card-title">Performance Metrics (Global Model)</span>
      </div>

      <div className="perf-metrics-body">
        {/* Doughnut */}
        <div style={{ width: 160, height: 160, flexShrink: 0, position: "relative" }}>
          {!hasData ? (
            <EmptyDoughnut />
          ) : (
            <>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={chartData}
                    cx="50%"
                    cy="50%"
                    innerRadius={52}
                    outerRadius={72}
                    paddingAngle={3}
                    dataKey="value"
                    startAngle={90}
                    endAngle={-270}
                  >
                    {chartData.map((entry, i) => (
                      <Cell key={i} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip content={<CustomTooltip />} />
                </PieChart>
              </ResponsiveContainer>
              <div style={{
                position: "absolute", inset: 0,
                display: "flex", flexDirection: "column",
                alignItems: "center", justifyContent: "center",
                pointerEvents: "none",
              }}>
                <span style={{ fontSize: 20, fontWeight: 800, color: "var(--text-primary)" }}>
                  {(metrics.accuracy || 0).toFixed(1)}%
                </span>
                <span style={{ fontSize: 10.5, color: "var(--text-muted)" }}>Accuracy</span>
              </div>
            </>
          )}
        </div>

        {/* Legend */}
        <div className="perf-metrics-legend">
          {METRIC_DEFS.map((item) => (
            <div key={item.key} className="perf-legend-item">
              <div className="perf-legend-left">
                <span className="perf-dot" style={{ background: item.color }} />
                {item.label}
              </div>
              <span className="perf-legend-val">
                {hasData ? `${(metrics[item.key] || 0).toFixed(1)}%` : "—"}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Footer */}
      <div className="perf-metrics-footer">
        <div className="perf-footer-item">
          <div className="perf-footer-label">Loss (Current)</div>
          <div className="perf-footer-val">
            {hasData ? (metrics.loss || 0).toFixed(4) : "—"}
          </div>
        </div>
        <div style={{ width: 1, background: "var(--border)", margin: "0 4px" }} />
        <div className="perf-footer-item">
          <div className="perf-footer-label">F1 Score</div>
          <div className="perf-footer-val">
            {hasData ? (metrics.f1 || 0).toFixed(3) : "—"}
          </div>
        </div>
      </div>
    </div>
  );
}
