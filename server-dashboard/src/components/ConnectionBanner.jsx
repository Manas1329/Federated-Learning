// ConnectionBanner.jsx
// Shows a warning bar when the WebSocket is disconnected / API is unreachable

import { WifiOff } from "lucide-react";

export default function ConnectionBanner({ message }) {
  return (
    <div style={{
      background:   "#7c2d12",
      borderBottom: "1px solid #b91c1c",
      padding:      "8px 20px",
      display:      "flex",
      alignItems:   "center",
      gap:          10,
      fontSize:     12.5,
      color:        "#fecaca",
      flexShrink:   0,
    }}>
      <WifiOff size={14} color="#f87171" />
      <strong style={{ color: "#f87171" }}>API Disconnected:</strong>
      {message}
      <span style={{ marginLeft: "auto", opacity: 0.7 }}>
        Dashboard will auto-reconnect. Make sure <code>dashboard_api.py</code> is running on port 8000.
      </span>
    </div>
  );
}
