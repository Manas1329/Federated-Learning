/**
 * useLiveState.js
 * ===============
 * Custom React hook that:
 *  1. Connects to the dashboard WebSocket at ws://localhost:8000/ws/dashboard
 *  2. Parses incoming LIVE_STATE JSON and updates React state
 *  3. Falls back to REST polling (/api/status every 3 s) if WebSocket disconnects
 *
 * Returns: { liveState, isConnected, wsError, sendCommand }
 *
 * To connect to a different host (e.g., server on a LAN IP):
 *   Set VITE_API_URL=http://192.168.x.x:8000 in a .env file in server-dashboard/
 */

import { useState, useEffect, useRef, useCallback } from "react";

// ── API base URLs ────────────────────────────────────────────────
const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";
const WS_BASE  = API_BASE.replace(/^http/, "ws");

// ── Default state (mirrors dashboard_state.LIVE_STATE structure) ─
const DEFAULT_STATE = {
  training_status:     "Ready",
  current_round:       0,
  total_rounds:        5,
  connected_clients:   [],
  client_details:      {},
  performance_history: [],
  latest_metrics: {
    accuracy: 0, loss: 0, f1: 0, precision: 0, recall: 0,
  },
  update_stats: {
    total_updates_received:   0,
    updates_this_round:       0,
    avg_update_size_raw_mb:   0,
    avg_update_size_quant_mb: 0,
    compression_ratio:        0,
    dp_epsilon_avg:           0,
  },
  dropout_info: {
    expected_clients:  3,
    active_clients:    0,
    dropped_clients:   0,
    minimum_required:  2,
    overall_status:    "all_active",
  },
  events: [],
  config: {
    use_quantization: true,
    use_dp:           false,
    target_clients:   3,
    min_clients:      2,
    fl_port:          8080,
  },
  // Trust / Tagging module data (updated after each FL round)
  trust_data:   {},
  server_pid:   null,
  version:      0,
  last_updated: "",
};


// ── Reconnect settings ────────────────────────────────────────────
const RECONNECT_DELAY_MS  = 3000;
const POLL_FALLBACK_MS    = 3000;
const MAX_RECONNECT_TRIES = 10;

export function useLiveState() {
  const [liveState, setLiveState]   = useState(DEFAULT_STATE);
  const [isConnected, setConnected] = useState(false);
  const [wsError, setWsError]       = useState(null);

  const wsRef            = useRef(null);
  const reconnectTimer   = useRef(null);
  const pollTimer        = useRef(null);
  const reconnectCount   = useRef(0);
  const usingFallback    = useRef(false);

  // ── Merge incoming state ────────────────────────────────────────
  const applyState = useCallback((incoming) => {
    if (!incoming || incoming.ping) return;          // ignore ping-only frames
    setLiveState((prev) => {
      // Only update if the server state is newer
      if ((incoming.version ?? 0) >= (prev.version ?? 0)) {
        return { ...DEFAULT_STATE, ...incoming };
      }
      return prev;
    });
  }, []);

  // ── REST fallback poll ──────────────────────────────────────────
  const startPolling = useCallback(() => {
    if (usingFallback.current) return;
    usingFallback.current = true;
    const poll = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/status`);
        if (res.ok) {
          const data = await res.json();
          applyState(data);
        }
      } catch (_) {/* server not yet up */}
      pollTimer.current = setTimeout(poll, POLL_FALLBACK_MS);
    };
    poll();
  }, [applyState]);

  const stopPolling = useCallback(() => {
    usingFallback.current = false;
    if (pollTimer.current) {
      clearTimeout(pollTimer.current);
      pollTimer.current = null;
    }
  }, []);

  // ── WebSocket connect ───────────────────────────────────────────
  const connect = useCallback(() => {
    if (wsRef.current &&
        (wsRef.current.readyState === WebSocket.CONNECTING ||
         wsRef.current.readyState === WebSocket.OPEN)) {
      return;
    }

    const ws = new WebSocket(`${WS_BASE}/ws/dashboard`);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      setWsError(null);
      reconnectCount.current = 0;
      stopPolling();
    };

    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        applyState(data);
      } catch (_) {}
    };

    ws.onerror = () => {
      setWsError("WebSocket error — falling back to REST polling");
      startPolling();
    };

    ws.onclose = () => {
      setConnected(false);
      startPolling();

      if (reconnectCount.current < MAX_RECONNECT_TRIES) {
        reconnectCount.current += 1;
        reconnectTimer.current = setTimeout(connect, RECONNECT_DELAY_MS);
      } else {
        setWsError(
          "Could not connect to dashboard API. " +
          "Make sure dashboard_api.py is running on port 8000."
        );
      }
    };
  }, [applyState, startPolling, stopPolling]);

  // ── Mount / unmount ─────────────────────────────────────────────
  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      clearTimeout(pollTimer.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [connect]);

  // ── sendCommand: POST to REST API ────────────────────────────────
  const sendCommand = useCallback(async (endpoint, body = {}) => {
    try {
      const url    = `${API_BASE}${endpoint}`;
      const params = new URLSearchParams(body).toString();
      const res    = await fetch(params ? `${url}?${params}` : url, {
        method: "POST",
      });
      return await res.json();
    } catch (e) {
      console.error("API command failed:", e);
      return { error: e.message };
    }
  }, []);

  return { liveState, isConnected, wsError, sendCommand };
}
