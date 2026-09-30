import {
  LayoutDashboard, Users, Globe, Activity, BarChart2,
  ShieldCheck, Trash2, RefreshCw, ClipboardCheck,
  FileText, Settings, ChevronDown
} from "lucide-react";
import { mockSystemStatus } from "../data/mockServerData";

const NAV_ITEMS = [
  { label: "System Overview",      icon: LayoutDashboard },
  { label: "Hospital Clients",     icon: Users },
  { label: "Global Model",         icon: Globe },
  { label: "Federated Training",   icon: Activity },
  { label: "Model Comparison",     icon: BarChart2 },
  { label: "Trust & Robustness",   icon: ShieldCheck },
  { label: "Federated Unlearning", icon: Trash2 },
  { label: "Model Recovery",       icon: RefreshCw },
  { label: "Verification & Audit", icon: ClipboardCheck },
  { label: "Reports & Logs",       icon: FileText },
  { label: "System Settings",      icon: Settings },
];

export default function Sidebar({ activeItem, onNavClick }) {
  const { services, version, year } = mockSystemStatus;

  return (
    <aside className="sidebar">
      {/* Logo */}
      <div className="sidebar-logo">
        <div className="sidebar-logo-icon">
          <ShieldCheck size={18} />
        </div>
        <div className="sidebar-logo-text">
          <span>Healthcare AI</span>
          <span>Federated Network</span>
        </div>
      </div>

      {/* Navigation */}
      <nav className="sidebar-nav">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.label}
              className={`nav-item ${activeItem === item.label ? "active" : ""}`}
              onClick={() => onNavClick?.(item.label)}
            >
              <Icon />
              {item.label}
            </button>
          );
        })}
      </nav>

      {/* Bottom: System Status + Footer */}
      <div className="sidebar-bottom">
        <div className="system-status-card">
          <div className="status-title">System Status</div>
          <div className="status-all-running">
            <span className="status-dot" />
            All Services Running
          </div>
          {services.map((svc) => (
            <div key={svc.name} className="status-item">
              <span className="status-dot-sm" />
              {svc.name}
            </div>
          ))}
        </div>

        <div className="sidebar-footer">
          <div>Version {version}</div>
          <div>© {year} Healthcare AI</div>
        </div>
      </div>
    </aside>
  );
}
