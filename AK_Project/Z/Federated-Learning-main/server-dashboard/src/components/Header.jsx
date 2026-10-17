import { useState, useEffect } from "react";
import { ShieldCheck, Bell, Calendar, Clock, ChevronDown, Sun, Moon, Wifi, WifiOff } from "lucide-react";

function formatDate(date) {
  return date.toLocaleDateString("en-US", {
    month: "short", day: "numeric", year: "numeric",
  });
}

function formatTime(date) {
  return date.toLocaleTimeString("en-US", {
    hour: "2-digit", minute: "2-digit", second: "2-digit",
  });
}

export default function Header({ theme, onToggleTheme, isConnected }) {
  const [now, setNow] = useState(new Date());

  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <header className="header">
      {/* Left */}
      <div className="header-left">
        <div className="header-icon">
          <ShieldCheck size={20} />
        </div>
        <div className="header-title">
          <span>Federated Healthcare AI - Central Server</span>
          <span>Admin Dashboard</span>
        </div>
      </div>

      {/* Right */}
      <div className="header-right">
        {/* Connection indicator */}
        <div
          title={isConnected ? "API Connected" : "API Disconnected"}
          style={{
            display:    "flex",
            alignItems: "center",
            gap:        5,
            fontSize:   11.5,
            color:      isConnected ? "var(--green)" : "var(--red)",
            padding:    "3px 8px",
            borderRadius: 6,
            background: isConnected ? "var(--green-glow)" : "var(--red-glow)",
            border:     `1px solid ${isConnected ? "rgba(34,197,94,0.25)" : "rgba(239,68,68,0.25)"}`,
          }}
        >
          {isConnected
            ? <><Wifi size={12}/> Live</>
            : <><WifiOff size={12}/> Offline</>
          }
        </div>

        {/* Theme toggle */}
        <button
          id="theme-toggle-btn"
          onClick={onToggleTheme}
          title={theme === "dark" ? "Switch to Light Mode" : "Switch to Dark Mode"}
          style={{
            background:   "var(--bg-input)",
            border:       "1px solid var(--border-light)",
            borderRadius: 8,
            padding:      "6px 8px",
            cursor:       "pointer",
            color:        "var(--text-secondary)",
            display:      "flex",
            alignItems:   "center",
            transition:   "all 0.2s",
          }}
          onMouseOver={(e) => e.currentTarget.style.color = "var(--text-primary)"}
          onMouseOut={(e)  => e.currentTarget.style.color = "var(--text-secondary)"}
        >
          {theme === "dark"
            ? <Sun  size={16} color="#f59e0b" />
            : <Moon size={16} color="#818cf8" />
          }
        </button>

        {/* Notification */}
        <div className="header-notif">
          <Bell size={19} />
          <span className="notif-badge">3</span>
        </div>

        {/* Date */}
        <div className="header-time-block">
          <Calendar size={14} />
          <span>{formatDate(now)}</span>
        </div>

        {/* Time */}
        <div className="header-time-block">
          <Clock size={14} />
          <span>{formatTime(now)}</span>
        </div>

        {/* User */}
        <div className="header-user">
          <div className="user-avatar">A</div>
          <div className="user-info">
            <span>Admin</span>
            <span>Super Administrator</span>
          </div>
          <ChevronDown size={14} color="var(--text-muted)" />
        </div>
      </div>
    </header>
  );
}
