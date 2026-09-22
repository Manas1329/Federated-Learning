// Event type → dot color
const EVENT_COLORS = {
  update:      "#3b82f6", // blue
  aggregation: "#a855f7", // purple
  round:       "#3b82f6", // blue
  success:     "#22c55e", // green
  connect:     "#22c55e", // green
  warn:        "#f59e0b", // orange
  error:       "#ef4444", // red
  training:    "#06b6d4", // teal
};

function EventDot({ type }) {
  const color = EVENT_COLORS[type] || "#64748b";
  return (
    <span
      className="event-dot"
      style={{ background: color, boxShadow: `0 0 5px ${color}66` }}
    />
  );
}

export default function RecentEvents({ events, onViewAll }) {
  return (
    <div className="card" style={{ flex: 1 }}>
      <div className="card-header">
        <span className="card-title">Recent System Events</span>
        <span className="card-link" onClick={onViewAll}>View All</span>
      </div>

      <div className="events-list">
        {events.map((ev) => (
          <div key={ev.id} className="event-item">
            <EventDot type={ev.type} />
            <span className="event-time">{ev.time}</span>
            <span className="event-msg">{ev.message}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
