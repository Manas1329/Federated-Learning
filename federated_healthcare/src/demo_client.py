"""
demo_client.py
==============
Multi-laptop live demonstration wrapper for the Federated Learning Client.

Accepts CLI arguments for server IP, client identity, and data path, then
starts the existing HospitalClient (client.py logic) -- unchanged.

Usage (run from the 'src' directory on each hospital laptop):

    python demo_client.py --server_ip 192.168.x.x --client_id Hospital_A
    python demo_client.py --server_ip 192.168.x.x --client_id Hospital_B --data_path ../data/hospital_B
    python demo_client.py --server_ip 192.168.x.x --client_id Hospital_C --port 8080

The existing .env file settings (USE_QUANTIZATION, USE_DP, etc.) are
respected exactly as before.
"""

import argparse
import os
import sys
import socket

# ------------------------------------------------------------------
# Ensure 'src' is importable
# ------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ------------------------------------------------------------------
# Load .env first (before argparse, so defaults can use env values)
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
    description="FL Client -- Live Multi-Laptop Demonstration"
)
parser.add_argument(
    "--server_ip",
    type=str,
    required=True,
    help="LAN IP address of the FL server laptop (e.g. 192.168.1.10)"
)
parser.add_argument(
    "--port",
    type=int,
    default=8080,
    help="FL server port (default: 8080)"
)
parser.add_argument(
    "--client_id",
    type=str,
    default="Hospital_A",
    help="Client identity shown in logs (e.g. Hospital_A, Hospital_B, Hospital_C)"
)
parser.add_argument(
    "--data_path",
    type=str,
    default=None,
    help="Path to this client's hospital data folder. "
         "Defaults to ../data/<client_id_lowercase> relative to src/"
)
args = parser.parse_args()

# ------------------------------------------------------------------
# Resolve data path
# ------------------------------------------------------------------

if args.data_path:
    DATA_PATH = os.path.abspath(args.data_path)
else:
    # Auto-resolve: Hospital_A -> data/hospital_A
    # Matches existing folder names: hospital_A, hospital_B, hospital_c
    folder_name = args.client_id.lower().replace("-", "_")
    DATA_PATH = os.path.abspath(
        os.path.join(_BASE_DIR, "data", folder_name)
    )
    # Fallback: if the lowercase folder does not exist, try original casing
    if not os.path.exists(DATA_PATH):
        # Try exact match from known folder list
        data_dir = os.path.join(_BASE_DIR, "data")
        if os.path.isdir(data_dir):
            for entry in os.listdir(data_dir):
                if entry.lower() == folder_name:
                    DATA_PATH = os.path.join(data_dir, entry)
                    break

SERVER_ADDRESS = f"{args.server_ip}:{args.port}"
CLIENT_NAME    = args.client_id

# Inject into environment so client.py picks them up
os.environ["SERVER_ADDRESS"] = SERVER_ADDRESS
os.environ["CLIENT_NAME"]    = CLIENT_NAME
os.environ["DATA_PATH"]      = DATA_PATH

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

def get_own_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown"

OWN_IP = get_own_ip()

print("\n")
print("=" * 60)
print(f"   FL CLIENT  :  {CLIENT_NAME}")
print("=" * 60)
print(f"   This laptop IP     : {OWN_IP}")
print(f"   Server address     : {SERVER_ADDRESS}")
print(f"   FL Mode            : {MODE}")
print(f"   Data path          : {DATA_PATH}")
print("=" * 60)

# Verify data path exists before attempting connection
if not os.path.exists(DATA_PATH):
    print()
    print(f"  [ERROR] Data path not found: {DATA_PATH}")
    print(f"  Please specify the correct path with --data_path")
    print()
    sys.exit(1)

print(f"\n   Connecting to FL Server at {SERVER_ADDRESS} ...")
print("=" * 60)
print()

# ------------------------------------------------------------------
# Import existing FL client logic and start
# ------------------------------------------------------------------

import flwr as fl

# Import HospitalClient from existing client.py
# client.py reads CLIENT_NAME, SERVER_ADDRESS, DATA_PATH from env
# which we have already set above
from client import HospitalClient

print(f"[{CLIENT_NAME}] Data loaded. Connecting to server...")
print()

fl.client.start_client(
    server_address=SERVER_ADDRESS,
    client=HospitalClient().to_client(),
    grpc_max_message_length=1024 * 1024 * 1024,
)

print()
print("=" * 60)
print(f"   {CLIENT_NAME}  -  ALL ROUNDS COMPLETED")
print("=" * 60)
print()
