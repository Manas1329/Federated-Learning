/**
 * TrustStatus.jsx
 * ===============
 * Client Trust / Tagging dashboard panel.
 * Shows per-hospital trust scores, tags, component breakdown, and round history.
 *
 * Data comes from liveState.trust_data which is populated by trust_manager.py
 * via the trust_{suffix}.csv polling in dashboard_api.py.
 *
 * No fake/static data — all values come from the backend.
 */

import React, { useState } from "react";

// ============================================================
// Constants (must match TRUST_CONFIG in trust_manager.py)
// ============================================================

const TAG_CONFIG = {
  TRUSTED: {
    label:    "TRUSTED",
    emoji:    "🟢",
    color:    "var(--trust-trusted)",
    bg:       "var(--trust-trusted-bg)",
    border:   "var(--trust-trusted-border)",
    badgeCls: "trust-badge-trusted",
  },
  SUSPICIOUS: {
    label:    "SUSPICIOUS",
    emoji:    "🟡",
    color:    "var(--trust-suspicious)",
    bg:       "var(--trust-suspicious-bg)",
    border:   "var(--trust-suspicious-border)",
    badgeCls: "trust-badge-suspicious",
  },
  UNTRUSTED: {
    label:    "UNTRUSTED",
    emoji:    "🔴",
    color:    "var(--trust-untrusted)",
    bg:       "var(--trust-untrusted-bg)",
    border:   "var(--trust-untrusted-border)",
    badgeCls: "trust-badge-untrusted",
  },
};

function getTagConfig(tag) {
  return TAG_CONFIG[tag] || TAG_CONFIG.TRUSTED;
}

// Score bar — thin progress bar that fills based on score (0-100)
function ScoreBar({ value, tag }) {
  const tc = getTagConfig(tag);
  const pct = Math.min(100, Math.max(0, value));
  return (
    <div className="trust-score-bar-track" title={`${value} / 100`}>
      <div
        className="trust-score-bar-fill"
        style={{ width: `${pct}%`, backgroundColor: tc.color }}
      />
    </div>
  );
}

// Component score row (label + score + mini bar)
function ComponentRow({ label, value, tag }) {
  const pct  = Math.min(100, Math.max(0, value));
  const col  = pct >= 80 ? "var(--trust-trusted)" : pct >= 50 ? "var(--trust-suspicious)" : "var(--trust-untrusted)";
  return (
    <div className="trust-component-row">
      <span className="trust-component-label">{label}</span>
      <div className="trust-component-bar-track">
        <div className="trust-component-bar-fill" style={{ width: `${pct}%`, backgroundColor: col }} />
      </div>
      <span className="trust-component-value">{value.toFixed(1)}</span>
    </div>
  );
}

// History badge for a single round entry
function HistoryBadge({ round, score, tag }) {
  const tc = getTagConfig(tag);
  return (
    <div className="trust-history-badge" style={{ borderColor: tc.border, backgroundColor: tc.bg }}>
      <span className="trust-history-round">R{round}</span>
      <span className="trust-history-score" style={{ color: tc.color }}>{score}</span>
      <span className="trust-history-emoji">{tc.emoji}</span>
    </div>
  );
}

// Individual hospital trust card
function TrustCard({ clientName, data }) {
  const [expanded, setExpanded] = useState(false);
  const tc = getTagConfig(data.tag);

  return (
    <div
      className="trust-card"
      style={{ borderColor: tc.border }}
      id={`trust-card-${clientName.replace(/[^a-zA-Z0-9]/g, "_")}`}
    >
      {/* Header */}
      <div className="trust-card-header" style={{ backgroundColor: tc.bg }}>
        <div className="trust-card-title-row">
          <span className="trust-card-name">{clientName}</span>
          <span className={`trust-badge ${tc.badgeCls}`}>
            {tc.emoji} {tc.label}
          </span>
        </div>

        {/* Main score */}
        <div className="trust-card-score-row">
          <span className="trust-card-score" style={{ color: tc.color }}>
            {data.trust_score.toFixed(1)}
          </span>
          <span className="trust-card-score-denom">/ 100</span>
          <span className="trust-card-round-label">Round {data.round}</span>
        </div>

        {/* Main score bar */}
        <ScoreBar value={data.trust_score} tag={data.tag} />
      </div>

      {/* Component breakdown */}
      <div className="trust-card-body">
        <p className="trust-components-title">Component Scores</p>
        <ComponentRow label="Update Behaviour"      value={data.update_score}      tag={data.tag} />
        <ComponentRow label="Training Behaviour"    value={data.training_score}    tag={data.tag} />
        <ComponentRow label="Historical Reputation" value={data.historical_score}  tag={data.tag} />
        <ComponentRow label="Participation"         value={data.reliability_score} tag={data.tag} />
      </div>

      {/* History section (collapsible) */}
      {data.history && data.history.length > 0 && (
        <div className="trust-card-history">
          <button
            className="trust-history-toggle"
            onClick={() => setExpanded(e => !e)}
            id={`trust-history-toggle-${clientName.replace(/[^a-zA-Z0-9]/g, "_")}`}
          >
            {expanded ? "▲ Hide History" : `▼ Round History (${data.history.length})`}
          </button>
          {expanded && (
            <div className="trust-history-grid">
              {data.history.map(([round, score, tag], i) => (
                <HistoryBadge key={i} round={round} score={score} tag={tag} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ============================================================
// Main exported component
// ============================================================

export function TrustStatus({ trustData }) {
  const clients = Object.keys(trustData || {}).sort();
  const hasData = clients.length > 0;

  return (
    <section className="trust-section card" id="trust-status-section">
      {/* Section header */}
      <div className="card-header">
        <div className="trust-header-row">
          <h2 className="card-title" style={{ fontSize: "1rem", margin: 0 }}>
            🛡️ Client Trust Status
          </h2>
          <span className="trust-header-note">
            {hasData
              ? `${clients.length} client${clients.length !== 1 ? "s" : ""} monitored`
              : "Awaiting first FL round…"}
          </span>
        </div>
        <p className="trust-header-sub">
          Multi-factor trust scoring — Non-IID safe. Each client scored against its own history.
        </p>
      </div>

      {/* Cards grid */}
      {hasData ? (
        <div className="trust-cards-grid">
          {clients.map(name => (
            <TrustCard key={name} clientName={name} data={trustData[name]} />
          ))}
        </div>
      ) : (
        <div className="trust-empty-state">
          <div className="trust-empty-icon">🛡️</div>
          <p className="trust-empty-title">No trust data yet</p>
          <p className="trust-empty-sub">
            Trust scores will appear here after the first FL round completes.
          </p>
        </div>
      )}

      {/* Legend */}
      {hasData && (
        <div className="trust-legend">
          {Object.values(TAG_CONFIG).map(tc => (
            <div key={tc.label} className="trust-legend-item">
              <span className={`trust-badge ${tc.badgeCls}`} style={{ fontSize: "0.72rem" }}>
                {tc.emoji} {tc.label}
              </span>
            </div>
          ))}
          <span className="trust-legend-note">80–100 / 50–79 / 0–49</span>
        </div>
      )}
    </section>
  );
}

export default TrustStatus;
