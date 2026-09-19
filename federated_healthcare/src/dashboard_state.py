"""
dashboard_state.py
==================
Thread-safe shared live state between the FL server process and the
dashboard API. This module is imported by both server_instrumented.py
(which writes state) and dashboard_api.py (which reads/broadcasts state).

NO existing FL files are modified. This module only ADDS new hooks.
"""

import json
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

# ──────────────────────────────────────────────────────────────────
# Path for the shared state JSON file
# ──────────────────────────────────────────────────────────────────

_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(_SRC_DIR)
STATE_FILE = os.path.join(_BASE_DIR, "dashboard", "state.json")

os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)

# ──────────────────────────────────────────────────────────────────
# Thread safety
# ──────────────────────────────────────────────────────────────────

_lock = threading.Lock()

# ──────────────────────────────────────────────────────────────────
# Timestamp helper — defined early so LIVE_STATE dict can use it
# ──────────────────────────────────────────────────────────────────

def _now_str() -> str:
    return datetime.now().strftime("%I:%M:%S %p")


# ──────────────────────────────────────────────────────────────────
# Read .env config once
# ──────────────────────────────────────────────────────────────────

def _load_env() -> Dict[str, str]:
    env: Dict[str, str] = {}
    env_file = os.path.join(_BASE_DIR, ".env")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip()
    return env

_ENV = _load_env()

USE_QUANTIZATION = os.environ.get("USE_QUANTIZATION", _ENV.get("USE_QUANTIZATION", "1")) == "1"
USE_DP           = os.environ.get("USE_DP",           _ENV.get("USE_DP",           "0")) == "1"
TARGET_CLIENTS   = int(os.environ.get("TARGET_CLIENTS", _ENV.get("TARGET_CLIENTS", "3")))
import math
MIN_CLIENTS      = max(2, math.ceil(0.6 * TARGET_CLIENTS))

# ──────────────────────────────────────────────────────────────────
# Live State (single source of truth)
# ──────────────────────────────────────────────────────────────────

LIVE_STATE: Dict[str, Any] = {
    # Training control
    "training_status": "Ready",    # Ready | Waiting | Training | Completed | Error | Stopped
    "current_round":   0,
    "total_rounds":    5,

    # Client tracking
    "connected_clients": [],       # list of client name strings
    "client_details": {},          # {name: {status, accuracy, loss, update_size, last_update}}

    # Per-round metrics (list, appended after each round)
    "performance_history": [],     # [{round, accuracy, loss, f1, precision, recall}]

    # Summary metrics (latest round)
    "latest_metrics": {
        "accuracy":  0.0,
        "loss":      0.0,
        "f1":        0.0,
        "precision": 0.0,
        "recall":    0.0,
    },

    # Model update statistics
    "update_stats": {
        "total_updates_received":    0,
        "updates_this_round":        0,
        "avg_update_size_raw_mb":    0.0,
        "avg_update_size_quant_mb":  0.0,
        "compression_ratio":         0.0,
        "dp_epsilon_avg":            0.0,
    },

    # Dropout monitor
    "dropout_info": {
        "expected_clients":  TARGET_CLIENTS,
        "active_clients":    0,
        "dropped_clients":   0,
        "minimum_required":  MIN_CLIENTS,
        "overall_status":    "all_active",   # all_active | dropped_out | insufficient_clients
    },

    # Event log (newest first)
    "events": [],

    # Configuration (from .env)
    "config": {
        "use_quantization": USE_QUANTIZATION,
        "use_dp":           USE_DP,
        "target_clients":   TARGET_CLIENTS,
        "min_clients":      MIN_CLIENTS,
        "fl_port":          8080,
    },

    # Server process PID (set by dashboard_api when it spawns the subprocess)
    "server_pid": None,

    # Trust / Tagging data (populated by trust_manager after each round)
    # Format: {client_name: {trust_score, tag, update_score, training_score,
    #                        historical_score, reliability_score, round, history}}
    "trust_data": {},

    # State version — incremented on every change so WebSocket can detect updates
    "version": 0,
    "last_updated": _now_str(),
}


# ──────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────
# Version + write helpers
# ──────────────────────────────────────────────────────────────────

def _bump_version():
    """Increment version counter and set last_updated timestamp."""
    LIVE_STATE["version"]      += 1
    LIVE_STATE["last_updated"]  = _now_str()


def write_state_file():
    """Atomically write LIVE_STATE to STATE_FILE as JSON."""
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(LIVE_STATE, f, indent=2)
    os.replace(tmp, STATE_FILE)


def _update(fn):
    """
    Convenience wrapper: acquire lock → call fn() → bump version →
    write state file.  fn() receives no arguments and modifies
    LIVE_STATE in place.
    """
    with _lock:
        fn()
        _bump_version()
        write_state_file()


# ──────────────────────────────────────────────────────────────────
# Event helpers
# ──────────────────────────────────────────────────────────────────

EVENT_TYPES = {
    "connect":     "#22c55e",   # green
    "disconnect":  "#ef4444",   # red
    "training":    "#06b6d4",   # teal
    "round":       "#3b82f6",   # blue
    "update":      "#3b82f6",   # blue
    "aggregation": "#a855f7",   # purple
    "success":     "#22c55e",   # green
    "warn":        "#f59e0b",   # orange
    "error":       "#ef4444",   # red
    "dropout":     "#ef4444",   # red
    "info":        "#64748b",   # muted
}


def add_event(message: str, event_type: str = "info"):
    """Thread-safe: prepend a new event to the live event log."""
    import itertools
    event = {
        "id":      int(time.time() * 1000),
        "time":    _now_str(),
        "message": message,
        "type":    event_type,
    }
    def _do():
        LIVE_STATE["events"] = [event] + LIVE_STATE["events"][:49]
    _update(_do)


# ──────────────────────────────────────────────────────────────────
# Callback hooks — called by server_instrumented.py
# ──────────────────────────────────────────────────────────────────

def on_server_start(total_rounds: int, target_clients: int, min_clients: int):
    """Called when the FL server starts up."""
    def _do():
        LIVE_STATE["training_status"] = "Waiting"
        LIVE_STATE["total_rounds"]    = total_rounds
        LIVE_STATE["current_round"]   = 0
        LIVE_STATE["connected_clients"] = []
        LIVE_STATE["performance_history"] = []
        LIVE_STATE["events"] = []
        LIVE_STATE["config"]["target_clients"] = target_clients
        LIVE_STATE["config"]["min_clients"]    = min_clients
        LIVE_STATE["dropout_info"]["expected_clients"] = target_clients
        LIVE_STATE["dropout_info"]["minimum_required"] = min_clients
        LIVE_STATE["dropout_info"]["active_clients"]   = 0
        LIVE_STATE["dropout_info"]["dropped_clients"]  = 0
    _update(_do)
    add_event("Federated Learning server started", "training")
    add_event(f"Waiting for {target_clients} clients to connect...", "info")


def on_client_connect(client_name: str, total_connected: int, target: int):
    """Called when a hospital client successfully registers."""
    def _do():
        name = client_name or f"Client_{total_connected}"
        if name not in LIVE_STATE["connected_clients"]:
            LIVE_STATE["connected_clients"].append(name)
        LIVE_STATE["dropout_info"]["active_clients"] = total_connected
        # Initialise client detail entry
        if name not in LIVE_STATE["client_details"]:
            LIVE_STATE["client_details"][name] = {
                "status":      "Connected",
                "accuracy":    None,
                "loss":        None,
                "update_size": None,
                "last_update": _now_str(),
            }
        # If all clients connected → move to Training status
        if total_connected >= target:
            LIVE_STATE["training_status"] = "Training"
    _update(_do)
    add_event(f"{client_name} connected ({total_connected}/{target})", "connect")
    if total_connected >= target:
        add_event("All clients connected — starting federated training", "success")


def on_round_start(server_round: int, total_rounds: int):
    """Called at the start of each federated round (configure_fit)."""
    def _do():
        LIVE_STATE["training_status"] = "Training"
        LIVE_STATE["current_round"]   = server_round
        LIVE_STATE["total_rounds"]    = total_rounds
        LIVE_STATE["update_stats"]["updates_this_round"] = 0
        # Mark all known clients as Training
        for name in LIVE_STATE["client_details"]:
            LIVE_STATE["client_details"][name]["status"] = "Training"
    _update(_do)
    add_event(f"Round {server_round} / {total_rounds} started", "round")


def on_client_update_received(
    client_name: str,
    server_round: int,
    accuracy: Optional[float] = None,
    loss: Optional[float] = None,
    payload_raw_mb: Optional[float] = None,
    payload_quant_mb: Optional[float] = None,
    compression_ratio: Optional[float] = None,
    epsilon: Optional[float] = None,
):
    """Called when a client FitRes is received during fit_round."""
    def _do():
        LIVE_STATE["update_stats"]["total_updates_received"] += 1
        LIVE_STATE["update_stats"]["updates_this_round"]     += 1

        # Update running averages
        n = LIVE_STATE["update_stats"]["updates_this_round"]
        if payload_raw_mb is not None:
            old = LIVE_STATE["update_stats"]["avg_update_size_raw_mb"]
            LIVE_STATE["update_stats"]["avg_update_size_raw_mb"] = (
                old + (payload_raw_mb - old) / n
            )
        if payload_quant_mb is not None:
            old = LIVE_STATE["update_stats"]["avg_update_size_quant_mb"]
            LIVE_STATE["update_stats"]["avg_update_size_quant_mb"] = (
                old + (payload_quant_mb - old) / n
            )
        if compression_ratio is not None:
            old = LIVE_STATE["update_stats"]["compression_ratio"]
            LIVE_STATE["update_stats"]["compression_ratio"] = (
                old + (compression_ratio - old) / n
            )
        if epsilon is not None and epsilon > 0:
            old = LIVE_STATE["update_stats"]["dp_epsilon_avg"]
            LIVE_STATE["update_stats"]["dp_epsilon_avg"] = (
                old + (epsilon - old) / n
            )

        # Update per-client detail
        detail = LIVE_STATE["client_details"].get(client_name, {})
        detail["status"]      = "Update Submitted"
        detail["last_update"] = _now_str()
        if accuracy is not None:
            detail["accuracy"] = round(accuracy * 100, 2)
        if loss is not None:
            detail["loss"] = round(loss, 4)
        if payload_quant_mb is not None and USE_QUANTIZATION:
            detail["update_size"] = f"{payload_quant_mb:.2f} MB"
        elif payload_raw_mb is not None:
            detail["update_size"] = f"{payload_raw_mb:.2f} MB"
        LIVE_STATE["client_details"][client_name] = detail
    _update(_do)
    add_event(f"Model update received from {client_name}", "update")


def on_round_complete(
    server_round: int,
    accuracy: float,
    loss: float,
    f1: float = 0.0,
    precision: float = 0.0,
    recall: float = 0.0,
    successful_clients: int = 0,
    failed_clients: int = 0,
):
    """Called after aggregate_fit + evaluate_metrics_aggregation_fn."""
    def _do():
        entry = {
            "round":     server_round,
            "accuracy":  round(accuracy * 100, 2),
            "loss":      round(loss, 4),
            "f1":        round(f1, 4),
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
        }
        LIVE_STATE["performance_history"].append(entry)
        LIVE_STATE["latest_metrics"] = {
            "accuracy":  round(accuracy * 100, 2),
            "loss":      round(loss, 4),
            "f1":        round(f1, 4),
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
        }
        # Update dropout info
        LIVE_STATE["dropout_info"]["active_clients"]  = successful_clients
        LIVE_STATE["dropout_info"]["dropped_clients"] = failed_clients
        if failed_clients > 0:
            LIVE_STATE["dropout_info"]["overall_status"] = "dropped_out"
        else:
            LIVE_STATE["dropout_info"]["overall_status"] = "all_active"

        # Reset client statuses to Connected after round
        for name in LIVE_STATE["client_details"]:
            if LIVE_STATE["client_details"][name]["status"] == "Training":
                LIVE_STATE["client_details"][name]["status"] = "Connected"

    _update(_do)
    add_event(
        f"Round {server_round} aggregation completed — "
        f"Accuracy: {accuracy*100:.2f}%  Loss: {loss:.4f}",
        "aggregation",
    )
    if failed_clients > 0:
        add_event(
            f"{failed_clients} client(s) dropped out — "
            f"training continues with {successful_clients} clients",
            "dropout",
        )


def on_training_complete(total_rounds: int):
    """Called after all FL rounds finish."""
    def _do():
        LIVE_STATE["training_status"] = "Completed"
        for name in LIVE_STATE["client_details"]:
            LIVE_STATE["client_details"][name]["status"] = "Connected"
    _update(_do)
    add_event(f"Federated training completed — {total_rounds} rounds done", "success")


def on_training_error(error_msg: str):
    """Called if the FL process crashes."""
    def _do():
        LIVE_STATE["training_status"] = "Error"
    _update(_do)
    add_event(f"Training error: {error_msg}", "error")


def reset_state():
    """Reset to initial state (used by /api/training/reset)."""
    global LIVE_STATE
    with _lock:
        LIVE_STATE.update({
            "training_status":    "Ready",
            "current_round":      0,
            "connected_clients":  [],
            "client_details":     {},
            "performance_history": [],
            "latest_metrics": {
                "accuracy": 0.0, "loss": 0.0,
                "f1": 0.0, "precision": 0.0, "recall": 0.0,
            },
            "update_stats": {
                "total_updates_received": 0,
                "updates_this_round": 0,
                "avg_update_size_raw_mb": 0.0,
                "avg_update_size_quant_mb": 0.0,
                "compression_ratio": 0.0,
                "dp_epsilon_avg": 0.0,
            },
            "dropout_info": {
                "expected_clients": TARGET_CLIENTS,
                "active_clients": 0,
                "dropped_clients": 0,
                "minimum_required": MIN_CLIENTS,
                "overall_status": "all_active",
            },
            "events":     [],
            "server_pid": None,
            "trust_data": {},   # Clear trust scores on reset
        })
        _bump_version()
        write_state_file()



# Write initial state on module load
write_state_file()
