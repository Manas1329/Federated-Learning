import { useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer,
} from "recharts";

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div style={{
        background: "var(--bg-card)",
        border: "1px solid var(--border-light)",
        borderRadius: 8,
        padding: "8px 12px",
        fontSize: 12,
      }}>
        <div style={{ color: "var(--text-muted)", marginBottom: 4 }}>{label}</div>
        <div style={{ color: "#60a5fa", fontWeight: 700 }}>
          {payload[0]?.name === "accuracy"
            ? `Accuracy: ${payload[0].value}%`
            : `Loss: ${payload[0].value}`}
        </div>
      </div>
    );
  }
  return null;
};

const CustomDot = (props) => {
  const { cx, cy } = props;
  return <circle cx={cx} cy={cy} r={5} fill="#3b82f6" stroke="#1e40af" strokeWidth={2} />;
};

// Placeholder shown before any training data exists
function EmptyChart() {
  return (
    <div style={{
      height: 210,
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      color: "var(--text-muted)",
      fontSize: 13,
      border: "1px dashed var(--border)",
      borderRadius: 8,
      flexDirection: "column",
      gap: 8,
    }}>
      <span style={{ fontSize: 24 }}>📊</span>
      <span>Accuracy graph will update after each FL round</span>
    </div>
  );
}

export default function PerformanceChart({ history = [], summary = {} }) {
  const [metric, setMetric] = useState("Accuracy");

  const hasData = history.length > 0;

  return (
    <div className="card">
      <div className="card-header">
        <span className="card-title">Global Model Performance</span>
        <select
          className="chart-dropdown"
          value={metric}
          onChange={(e) => setMetric(e.target.value)}
        >
          <option>Accuracy</option>
          <option>Loss</option>
        </select>
      </div>

      {!hasData ? (
        <EmptyChart />
      ) : (
        <ResponsiveContainer width="100%" height={210}>
          <LineChart data={history} margin={{ top: 14, right: 20, left: -10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
            <XAxis
              dataKey="round"
              tick={{ fill: "var(--text-muted)", fontSize: 11 }}
              axisLine={{ stroke: "var(--border)" }}
              tickLine={false}
              label={{ value: "Rounds", position: "insideBottom", offset: -2, fill: "var(--text-muted)", fontSize: 11 }}
            />
            <YAxis
              domain={metric === "Accuracy" ? [50, 100] : [0, 1]}
              tick={{ fill: "var(--text-muted)", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(v) => metric === "Accuracy" ? `${v}%` : v.toFixed(2)}
              label={{
                value: metric === "Accuracy" ? "Accuracy (%)" : "Loss",
                angle: -90,
                position: "insideLeft",
                offset: 14,
                fill: "var(--text-muted)",
                fontSize: 11,
              }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Line
              type="monotone"
              dataKey={metric === "Accuracy" ? "accuracy" : "loss"}
              name={metric === "Accuracy" ? "accuracy" : "loss"}
              stroke="#3b82f6"
              strokeWidth={2.5}
              dot={<CustomDot />}
              activeDot={{ r: 7, fill: "#60a5fa", stroke: "#1d4ed8", strokeWidth: 2 }}
              label={{
                position: "top",
                fill: "#93c5fd",
                fontSize: 11,
                fontWeight: 600,
                formatter: (v) =>
                  metric === "Accuracy" ? `${v}%` : v.toFixed(3),
              }}
            />
          </LineChart>
        </ResponsiveContainer>
      )}

      {/* Metric summary cards */}
      <div className="chart-metrics-row">
        <div className="metric-mini">
          <div className="metric-mini-label" style={{ color: "#60a5fa" }}>Best Accuracy</div>
          <div className="metric-mini-val">
            {summary.bestAccuracy > 0 ? `${summary.bestAccuracy}%` : "—"}
          </div>
          <div className="metric-mini-sub">{summary.bestAccuracyRound || "—"}</div>
        </div>
        <div className="metric-mini">
          <div className="metric-mini-label" style={{ color: "#a78bfa" }}>Average Accuracy</div>
          <div className="metric-mini-val">
            {summary.avgAccuracy > 0 ? `${summary.avgAccuracy}%` : "—"}
          </div>
        </div>
        <div className="metric-mini">
          <div className="metric-mini-label" style={{ color: "#34d399" }}>Best Loss</div>
          <div className="metric-mini-val">
            {summary.bestLoss > 0 ? summary.bestLoss : "—"}
          </div>
          <div className="metric-mini-sub">{summary.bestLossRound || "—"}</div>
        </div>
        <div className="metric-mini">
          <div className="metric-mini-label" style={{ color: "#fb923c" }}>Current Loss</div>
          <div className="metric-mini-val">
            {summary.currentLoss > 0 ? summary.currentLoss : "—"}
          </div>
          {summary.lossChange !== 0 && summary.lossChange !== undefined && (
            <div
              className="metric-mini-change"
              style={{ color: summary.lossChange < 0 ? "var(--green)" : "var(--red)" }}
            >
              {summary.lossChange < 0 ? "↓" : "↑"} {Math.abs(summary.lossChange).toFixed(4)}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
