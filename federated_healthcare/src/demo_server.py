"""
demo_server.py
==============
Multi-laptop live demonstration wrapper for the Federated Learning Server.

Detects the LAN IP automatically, prints a clear startup banner, then
starts the existing Flower-based FL server (server.py logic) with the
AdaptiveServer dropout handler -- unchanged.

Usage (run from the 'src' directory):
    python demo_server.py
    python demo_server.py --port 8080 --num_rounds 5
    python demo_server.py --target_clients 3 --min_clients 2 --num_rounds 5

The existing .env file settings (USE_QUANTIZATION, USE_DP, etc.) are
respected exactly as before.
"""

import argparse
import math
import os
import socket
import sys
import time

# ------------------------------------------------------------------
# Ensure 'src' is importable when run from any working directory
# ------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ------------------------------------------------------------------
# Load .env (same logic as existing server.py / client.py)
# ------------------------------------------------------------------
_BASE_DIR = os.path.dirname(SRC_DIR)
_ENV_FILE = os.path.join(_BASE_DIR, ".env")
if os.path.exists(_ENV_FILE):
    with open(_ENV_FILE) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                _k = _k.strip()
                if _k not in os.environ:
                    os.environ[_k] = _v.strip()

# ------------------------------------------------------------------
# CLI Arguments
# ------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="FL Server -- Live Multi-Laptop Demonstration"
)
parser.add_argument(
    "--port",
    type=int,
    default=8080,
    help="Port to bind the FL server on (default: 8080)"
)
parser.add_argument(
    "--num_rounds",
    type=int,
    default=5,
    help="Number of federated learning rounds (default: 5)"
)
parser.add_argument(
    "--target_clients",
    type=int,
    default=int(os.environ.get("TARGET_CLIENTS", "3")),
    help="Target number of clients per round (default: 3)"
)
parser.add_argument(
    "--min_clients",
    type=int,
    default=None,
    help="Minimum clients required for aggregation (default: ceil(0.6 x target_clients))"
)
parser.add_argument(
    "--dropout_hard_deadline",
    type=float,
    default=float(os.environ.get("DROPOUT_HARD_DEADLINE", "180.0" if os.environ.get("USE_DP", "0") == "1" else "60.0")),
    help="Hard deadline for adaptive dropout engine predictions (seconds)."
)
parser.add_argument(
    "--round_timeout",
    type=float,
    default=float(os.environ.get("ROUND_TIMEOUT", "1800.0" if os.environ.get("USE_DP", "0") == "1" else "300.0")),
    help="Absolute timeout for the Flower server round (seconds)."
)
args = parser.parse_args()

PORT           = args.port
NUM_ROUNDS     = args.num_rounds
TARGET_CLIENTS = args.target_clients
MIN_CLIENTS    = args.min_clients if args.min_clients is not None \
                 else max(2, math.ceil(0.6 * TARGET_CLIENTS))
DROPOUT_HARD_DEADLINE = args.dropout_hard_deadline
ROUND_TIMEOUT  = args.round_timeout

# Propagate to env so existing server.py / dropout_handler.py picks them up
os.environ["TARGET_CLIENTS"] = str(TARGET_CLIENTS)
os.environ["NUM_ROUNDS"] = str(NUM_ROUNDS)

# ------------------------------------------------------------------
# Detect LAN IP
# ------------------------------------------------------------------

def get_lan_ip():
    """Return the LAN IPv4 address of this machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

SERVER_IP = get_lan_ip()

# ------------------------------------------------------------------
# Startup Banner
# ------------------------------------------------------------------

USE_DP           = os.environ.get("USE_DP", "0") == "1"
USE_QUANTIZATION = os.environ.get("USE_QUANTIZATION", "1") == "1"

if USE_DP:
    MODE = "Differential Privacy (DP-SGD)"
elif USE_QUANTIZATION:
    MODE = "INT8 Quantization"
else:
    MODE = "Pure FedAvg"

print("\n")
print("=" * 60)
print("   FEDERATED LEARNING SERVER  -  STARTING")
print("=" * 60)
print(f"   Server IP          : {SERVER_IP}")
print(f"   Port               : {PORT}")
print(f"   FL Mode            : {MODE}")
print(f"   Target Clients     : {TARGET_CLIENTS}")
print(f"   Minimum Clients    : {MIN_CLIENTS}")
print(f"   FL Rounds          : {NUM_ROUNDS}")
print(f"   Dropout Hard Deadline : {DROPOUT_HARD_DEADLINE} sec")
print(f"   Round Timeout         : {ROUND_TIMEOUT} sec")
print("=" * 60)
print(f"\n   Clients should connect with:")
print(f"   python demo_client.py --server_ip {SERVER_IP} --client_id Hospital_X")
print("\n")
print(f"   Waiting for {TARGET_CLIENTS} client(s) to join...")
print("=" * 60)
print()

# ------------------------------------------------------------------
# Import existing FL server logic
# ------------------------------------------------------------------

import flwr as fl
from server import (
    SaveModelStrategy,
    evaluate_metrics_aggregation_fn,
    RESULTS_DIR,
    MODEL_DIR,
    MODEL_PATH,
    SUFFIX,
)
from dropout_handler import AdaptiveServer

# ------------------------------------------------------------------
# Build strategy (mirrors existing server.py __main__)
# ------------------------------------------------------------------

strategy = SaveModelStrategy(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=TARGET_CLIENTS,
    min_evaluate_clients=MIN_CLIENTS,
    min_available_clients=MIN_CLIENTS,
    evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
)

client_manager = fl.server.SimpleClientManager()

server = AdaptiveServer(
    client_manager=client_manager,
    strategy=strategy,
    target_clients=TARGET_CLIENTS,
    min_clients=MIN_CLIENTS,
    total_rounds=NUM_ROUNDS,
    hard_deadline=DROPOUT_HARD_DEADLINE,
    alpha=0.3,
    beta=0.3,
    k=1.0,
    suffix=SUFFIX,
    models_dir=MODEL_DIR,
    adaptive_dropout_enabled=(os.environ.get("ADAPTIVE_DROPOUT_ENABLED", "1") == "1")
)

# ------------------------------------------------------------------
# Connection hook -- print when each client joins
# ------------------------------------------------------------------

_original_register = client_manager.register

def _patched_register(client):
    result = _original_register(client)
    if result:
        num_connected = len(client_manager.all())
        print(
            f"  [SERVER] Client joined "
            f"(cid={client.cid}) "
            f"-- {num_connected}/{TARGET_CLIENTS} connected"
        )
        if num_connected >= TARGET_CLIENTS:
            print()
            print("=" * 60)
            print("  ALL REQUIRED CLIENTS CONNECTED")
            print("  Starting Federated Learning...")
            print("=" * 60)
            print()
    return result

client_manager.register = _patched_register

# ------------------------------------------------------------------
# Start Server
# ------------------------------------------------------------------

print(f"[SERVER] Binding on 0.0.0.0:{PORT}  (LAN IP: {SERVER_IP})")
print()

fl.server.start_server(
    server_address=f"0.0.0.0:{PORT}",
    server=server,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS, round_timeout=ROUND_TIMEOUT),
    grpc_max_message_length=1024 * 1024 * 1024,
)

print()
print("=" * 60)
print("   FEDERATED LEARNING COMPLETED")
print(f"   Results saved to : {RESULTS_DIR}")
print(f"   Global model     : {MODEL_PATH}")
print("=" * 60)
print()
