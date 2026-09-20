"""
server_instrumented.py
======================
Thin instrumented wrapper around the existing server.py logic.

Subclasses SaveModelStrategy and AdaptiveServer to inject dashboard_state
callbacks WITHOUT modifying any existing file.

Usage (replaces demo_server.py when the dashboard is running):
    python server_instrumented.py --num_rounds 5 --target_clients 3

The existing demo_server.py / server.py still work standalone — unchanged.
"""

import argparse
import concurrent.futures
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

# ──────────────────────────────────────────────────────────────────
# Path setup — identical to demo_server.py
# ──────────────────────────────────────────────────────────────────

SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Load .env (same logic as every other file in the project)
_ENV_FILE = os.path.join(BASE_DIR, ".env")
if os.path.exists(_ENV_FILE):
    with open(_ENV_FILE) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                _k = _k.strip()
                if _k not in os.environ:
                    os.environ[_k] = _v.strip()

# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Instrumented FL Server (adds dashboard hooks)"
)
parser.add_argument("--port",           type=int, default=8080)
parser.add_argument("--num_rounds",     type=int, default=5)
parser.add_argument("--target_clients", type=int,
                    default=int(os.environ.get("TARGET_CLIENTS", "3")))
parser.add_argument("--min_clients",    type=int, default=None)
args = parser.parse_args()

PORT           = args.port
NUM_ROUNDS     = args.num_rounds
TARGET_CLIENTS = args.target_clients
MIN_CLIENTS    = (args.min_clients
                  if args.min_clients is not None
                  else max(2, math.ceil(0.6 * TARGET_CLIENTS)))

os.environ["TARGET_CLIENTS"] = str(TARGET_CLIENTS)

# ──────────────────────────────────────────────────────────────────
# Import existing server logic (unchanged)
# ──────────────────────────────────────────────────────────────────

import flwr as fl
from flwr.common import FitRes, EvaluateRes, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

from server import (
    SaveModelStrategy,
    evaluate_metrics_aggregation_fn as _original_eval_fn,
    RESULTS_DIR,
    MODEL_DIR,
    MODEL_PATH,
    SUFFIX,
    USE_DP,
    USE_QUANTIZATION,
)
from dropout_handler import AdaptiveServer

# ──────────────────────────────────────────────────────────────────
# Import dashboard state (new — no existing file is modified)
# ──────────────────────────────────────────────────────────────────

import dashboard_state as ds

# ──────────────────────────────────────────────────────────────────
# Instrumented Strategy
# ──────────────────────────────────────────────────────────────────

class InstrumentedStrategy(SaveModelStrategy):
    """
    Extends SaveModelStrategy only to fire dashboard_state hooks.
    All existing behaviour (dequantization, FedAvg, CSV writing,
    model saving) is preserved via super() calls.
    """

    def configure_fit(self, server_round, parameters, client_manager):
        # ── Dashboard hook ──
        ds.on_round_start(server_round, NUM_ROUNDS)
        # ── Existing logic ──
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        # ── Existing logic ──
        aggregated = super().aggregate_fit(server_round, results, failures)

        # ── Dashboard hook: extract per-client metrics from FitRes ──
        for client_proxy, fit_res in results:
            m = fit_res.metrics or {}
            ds.on_client_update_received(
                client_name       = m.get("client_name", f"Client_{client_proxy.cid[:6]}"),
                server_round      = server_round,
                accuracy          = None,                           # accuracy comes from evaluate
                loss              = None,
                payload_raw_mb    = m.get("payload_size_mb"),
                payload_quant_mb  = m.get("quantized_payload_mb"),
                compression_ratio = m.get("compression_ratio"),
                epsilon           = m.get("epsilon") if m.get("epsilon", -1) > 0 else None,
            )

        return aggregated


# ──────────────────────────────────────────────────────────────────
# Instrumented evaluate_metrics_aggregation_fn
# ──────────────────────────────────────────────────────────────────

def instrumented_eval_fn(metrics):
    """
    Calls the original evaluate_metrics_aggregation_fn (which writes
    CSVs and prints the banner), then fires the dashboard round-complete
    hook with the resulting aggregated metrics.
    """
    result = _original_eval_fn(metrics)

    # Determine which round this is (same logic as original fn)
    import pandas as pd
    from server import METRICS_FILE
    if os.path.exists(METRICS_FILE):
        df = pd.read_csv(METRICS_FILE)
        current_round = len(df)
    else:
        current_round = 1

    # Extract accuracy from each client's metrics for the client table
    for num_examples, m in metrics:
        client_name = m.get("client_name", "")
        if client_name and "accuracy" in m and "loss" in m:
            # Update client detail with evaluation results
            detail = ds.LIVE_STATE["client_details"].get(client_name, {})
            detail["accuracy"]    = round(m["accuracy"] * 100, 2)
            detail["loss"]        = round(m["loss"], 4)
            detail["status"]      = "Connected"
            detail["last_update"] = ds._now_str()
            ds.LIVE_STATE["client_details"][client_name] = detail

    ds.on_round_complete(
        server_round       = current_round,
        accuracy           = result.get("accuracy",  0.0),
        loss               = result.get("loss",      0.0),
        f1                 = result.get("f1",        0.0),
        precision          = result.get("precision", 0.0),
        recall             = result.get("recall",    0.0),
        successful_clients = len(metrics),
        failed_clients     = 0,   # failures counted separately in fit_round
    )
    return result


# ──────────────────────────────────────────────────────────────────
# Instrumented AdaptiveServer
# ──────────────────────────────────────────────────────────────────

class InstrumentedAdaptiveServer(AdaptiveServer):
    """
    Extends AdaptiveServer to track per-round dropout info in the
    dashboard state.  All existing grace-period / EMA logic is unchanged.
    """

    def fit_round(self, server_round, timeout):
        result = super().fit_round(server_round, timeout)

        if result is not None:
            _, _, (results, failures) = result
            # Update dropout counts after each round
            def _update_dropout():
                active  = len(results)
                dropped = len(failures)
                ds.LIVE_STATE["dropout_info"]["active_clients"]  = active
                ds.LIVE_STATE["dropout_info"]["dropped_clients"] = dropped
                if dropped > 0 and active >= self.min_clients:
                    ds.LIVE_STATE["dropout_info"]["overall_status"] = "dropped_out"
                elif active < self.min_clients:
                    ds.LIVE_STATE["dropout_info"]["overall_status"] = "insufficient_clients"
                else:
                    ds.LIVE_STATE["dropout_info"]["overall_status"] = "all_active"
                ds._bump_version()
                ds.write_state_file()

            with ds._lock:
                _update_dropout()

        return result


# ──────────────────────────────────────────────────────────────────
# Build strategy and server
# ──────────────────────────────────────────────────────────────────

strategy = InstrumentedStrategy(
    fraction_fit               = 1.0,
    fraction_evaluate          = 1.0,
    min_fit_clients            = TARGET_CLIENTS,
    min_evaluate_clients       = MIN_CLIENTS,
    min_available_clients      = MIN_CLIENTS,
    evaluate_metrics_aggregation_fn = instrumented_eval_fn,
)

client_manager = fl.server.SimpleClientManager()

# Timeouts (identical to demo_server.py)
if USE_DP:
    init_grace = 120.0
    max_grace  = 180.0
    round_to   = 1800.0
else:
    init_grace = 45.0
    max_grace  = 60.0
    round_to   = 300.0

server = InstrumentedAdaptiveServer(
    client_manager     = client_manager,
    strategy           = strategy,
    target_clients     = TARGET_CLIENTS,
    min_clients        = MIN_CLIENTS,
    initial_grace_period = init_grace,
    max_grace_period   = max_grace,
    round_timeout      = round_to,
    suffix             = SUFFIX,
    models_dir         = MODEL_DIR,
)

# ──────────────────────────────────────────────────────────────────
# Patch client_manager.register to capture connections
# ──────────────────────────────────────────────────────────────────

_original_register = client_manager.register

def _patched_register(client):
    result = _original_register(client)
    if result:
        num_connected = len(client_manager.all())
        # Try to get a human-readable name — we'll use cid as fallback
        name = f"Client_{client.cid[:8]}"
        ds.on_client_connect(name, num_connected, TARGET_CLIENTS)
        print(
            f"  [InstrumentedServer] Client joined (cid={client.cid}) "
            f"-- {num_connected}/{TARGET_CLIENTS} connected"
        )
    return result

client_manager.register = _patched_register

# ──────────────────────────────────────────────────────────────────
# Notify dashboard of startup
# ──────────────────────────────────────────────────────────────────

ds.on_server_start(
    total_rounds   = NUM_ROUNDS,
    target_clients = TARGET_CLIENTS,
    min_clients    = MIN_CLIENTS,
)

import socket
def _get_lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

SERVER_IP = _get_lan_ip()

print("\n")
print("=" * 60)
print("   INSTRUMENTED FL SERVER  -  STARTING")
print("=" * 60)
print(f"   Server IP      : {SERVER_IP}")
print(f"   FL Port        : {PORT}")
print(f"   Dashboard API  : http://localhost:8000")
print(f"   FL Mode        : {'DP-SGD' if USE_DP else 'INT8 Quantized' if USE_QUANTIZATION else 'Pure FedAvg'}")
print(f"   Target Clients : {TARGET_CLIENTS}")
print(f"   Min Clients    : {MIN_CLIENTS}")
print(f"   FL Rounds      : {NUM_ROUNDS}")
print("=" * 60)
print()

# ──────────────────────────────────────────────────────────────────
# Start Flower server
# ──────────────────────────────────────────────────────────────────

try:
    fl.server.start_server(
        server_address       = f"0.0.0.0:{PORT}",
        server               = server,
        config               = fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        grpc_max_message_length = 1024 * 1024 * 1024,
    )
    ds.on_training_complete(NUM_ROUNDS)

except Exception as e:
    print(f"\n[InstrumentedServer] ERROR: {e}")
    ds.on_training_error(str(e))
    raise

print()
print("=" * 60)
print("   FEDERATED LEARNING COMPLETED")
print(f"   Results: {RESULTS_DIR}")
print(f"   Model  : {MODEL_PATH}")
print("=" * 60)
print()
