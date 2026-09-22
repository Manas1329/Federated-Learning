import { Lock, Shield, Wifi } from "lucide-react";

function SecurityItem({ icon: Icon, label, enabled }) {
  return (
    <div className="security-item" style={{ color: enabled ? "var(--text-secondary)" : "var(--text-muted)" }}>
      <Icon size={13} style={{ color: enabled ? "var(--green)" : "var(--text-muted)" }} />
      {label}
      {!enabled && (
        <span style={{
          fontSize: 10, color: "var(--text-muted)", marginLeft: 4,
          background: "var(--bg-hover)", padding: "1px 5px",
          borderRadius: 4, border: "1px solid var(--border)",
        }}>
          Disabled
        </span>
      )}
    </div>
  );
}

export default function SecurityFooter({
  useQuantization = true,
  useDp           = false,
  isOperational   = false,
}) {
  return (
    <footer className="security-footer">
      <div className="security-left">
        <span style={{ fontSize: 12, fontWeight: 600, color: "var(--text-secondary)", whiteSpace: "nowrap" }}>
          System Security:
        </span>

        <SecurityItem icon={Lock}   label="Secure Aggregation Enabled" enabled={true} />
        <span className="security-separator">•</span>
        <SecurityItem icon={Shield} label="Differential Privacy"         enabled={useDp} />
        <span className="security-separator">•</span>
        <SecurityItem icon={Wifi}   label="End-to-End Encryption Active" enabled={true} />
        {useQuantization && (
          <>
            <span className="security-separator">•</span>
            <div className="security-item" style={{ color: "var(--green)" }}>
              <Wifi size={13} color="var(--green)" />
              INT8 Quantization Active
            </div>
          </>
        )}
      </div>

      <div className="security-right">
        {isOperational ? "All systems operational" : "Dashboard API offline"}
        <span
          className="op-dot"
          style={{ background: isOperational ? "var(--green)" : "var(--red)" }}
        />
      </div>
    </footer>
  );
}
