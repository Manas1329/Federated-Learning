"""
run_local.py  ─  LOCAL DEVELOPMENT SIMULATION ONLY
====================================================
THIS IS NOT THE PRIMARY DEPLOYMENT ARCHITECTURE.

Primary architecture (see DEMO_INSTRUCTIONS.md):
  - Device 1: Hospital A  →  python federated_healthcare/src/client.py
  - Device 2: Hospital B  →  python federated_healthcare/src/client.py
  - Device 3: Hospital C  →  python federated_healthcare/src/client.py
  - Device 4: FL Server   →  python federated_healthcare/src/server.py

This script is a convenience tool for LOCAL DEVELOPMENT AND TESTING ONLY.
It starts the server and all 3 hospital processes on the same machine
by launching independent subprocesses — it does NOT use Python threads
for training, and does NOT let the server train hospital models.

Each hospital process is fully isolated:
  - its own Python interpreter
  - its own ChestCNN model + optimizer
  - its own LOCAL data path (data/hospital_A, data/hospital_B, data/hospital_C)
  - its own environment variables

The Flower gRPC server then collects their updates concurrently, which is
the same mechanism used in the real 4-device deployment.

GPU Safety Note:
    On a single machine, all 3 client processes share the GPU.
    Set FORCE_CPU=1 in .env or pass --force_cpu 1 to avoid GPU OOM errors.
    ChestCNN is small enough to train on CPU for development purposes.

Usage (local dev/simulation only):
    python run_local.py
    python run_local.py --num_rounds 5
    python run_local.py --use_dp 1
    python run_local.py --force_cpu 1

For real 4-device deployment, see DEMO_INSTRUCTIONS.md
"""

import argparse
import os
import sys
import subprocess
import time
import threading
import signal
from datetime import datetime
from pathlib import Path

# ------------------------------------------------------------------
# Resolve project paths (works regardless of cwd)
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "federated_healthcare" / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------------------
# Load .env (same logic as server.py / client.py)
# ------------------------------------------------------------------
_ENV_FILE = PROJECT_ROOT / ".env"
_env_defaults = {}
if _ENV_FILE.exists():
    with open(_ENV_FILE) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                _k = _k.strip()
                _env_defaults[_k] = _v.strip()
                if _k not in os.environ:
                    os.environ[_k] = _v.strip()

# ------------------------------------------------------------------
# CLI Arguments
# ------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Federated Learning — Local Concurrent Simulation Launcher",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__
)
parser.add_argument("--num_rounds",       type=int,   default=int(os.environ.get("NUM_ROUNDS", "5")),
                    help="Number of FL rounds (default: 5)")
parser.add_argument("--target_clients",   type=int,   default=int(os.environ.get("TARGET_CLIENTS", "3")),
                    help="Number of hospital clients (default: 3)")
parser.add_argument("--server_port",      type=int,   default=8080,
                    help="FL server port (default: 8080)")
parser.add_argument("--use_quantization", type=int,   default=int(os.environ.get("USE_QUANTIZATION", "1")),
                    choices=[0, 1], help="Enable INT8 quantization (default: 1)")
parser.add_argument("--use_dp",           type=int,   default=int(os.environ.get("USE_DP", "0")),
                    choices=[0, 1], help="Enable Differential Privacy / DP-SGD (default: 0)")
parser.add_argument("--force_cpu",        type=int,   default=int(os.environ.get("FORCE_CPU", "0")),
                    choices=[0, 1],
                    help="Force CPU training for all clients — recommended when "
                         "running all clients on a single GPU machine (default: 0)")
parser.add_argument("--server_wait_sec",  type=float, default=4.0,
                    help="Seconds to wait for the server to bind before launching clients (default: 4.0)")
parser.add_argument("--timeout_sec",      type=float, default=0,
                    help="Kill everything after N seconds (0 = no timeout, default: 0)")

args = parser.parse_args()

HOSPITAL_CODES = ["Hospital_A", "Hospital_B", "Hospital_C"][:args.target_clients]
SERVER_ADDRESS = f"127.0.0.1:{args.server_port}"

# ------------------------------------------------------------------
# Decide mode label for banner
# ------------------------------------------------------------------
if args.use_dp:
    MODE = "Differential Privacy (DP-SGD)"
elif args.use_quantization:
    MODE = "INT8 Quantization + FedAvg"
else:
    MODE = "Pure FedAvg"

# ------------------------------------------------------------------
# Banner
# ------------------------------------------------------------------
print("\n" + "=" * 64)
print("   FEDERATED LEARNING — LOCAL CONCURRENT SIMULATION")
print("=" * 64)
print(f"   Mode               : {MODE}")
print(f"   Rounds             : {args.num_rounds}")
print(f"   Hospitals          : {len(HOSPITAL_CODES)}  ({', '.join(HOSPITAL_CODES)})")
print(f"   Server             : {SERVER_ADDRESS}")
print(f"   Quantization       : {'ON' if args.use_quantization else 'OFF'}")
print(f"   Differential Privacy: {'ON' if args.use_dp else 'OFF'}")
print(f"   Force CPU          : {'YES' if args.force_cpu else 'NO (auto-detect GPU)'}")
print(f"   Concurrency        : All hospital processes start simultaneously")
print("=" * 64)
print()

# ------------------------------------------------------------------
# Build shared environment
# ------------------------------------------------------------------
def build_env(extra: dict = None) -> dict:
    """Build a clean subprocess environment based on the current env + overrides."""
    env = os.environ.copy()

    # Add project root to PYTHONPATH
    pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + str(SRC_DIR) + (os.pathsep + pp if pp else "")

    # FL configuration
    env["USE_QUANTIZATION"]   = str(args.use_quantization)
    env["USE_DP"]             = str(args.use_dp)
    env["FORCE_CPU"]          = str(args.force_cpu)
    env["NUM_ROUNDS"]         = str(args.num_rounds)
    env["TARGET_CLIENTS"]     = str(args.target_clients)

    # Prevent CPU oversubscription when multiple training processes share hardware
    # Each process already uses PyTorch's internal threading; extra threads
    # from OpenMP/MKL compound contention without improving throughput.
    env["OMP_NUM_THREADS"]     = "2"
    env["MKL_NUM_THREADS"]     = "2"
    env["OPENBLAS_NUM_THREADS"] = "2"
    env["NUMEXPR_NUM_THREADS"]  = "2"

    if extra:
        for k, v in extra.items():
            env[k] = str(v)
    return env

# ------------------------------------------------------------------
# Log streaming (non-blocking, prefix each line with hospital name)
# ------------------------------------------------------------------
def _stream(pipe, prefix: str, stop_event: threading.Event):
    try:
        for raw in iter(pipe.readline, ""):
            if stop_event.is_set():
                break
            line = raw.rstrip("\r\n")
            if line:
                print(f"[{prefix}] {line}", flush=True)
    except Exception:
        pass
    finally:
        try:
            pipe.close()
        except Exception:
            pass

# ------------------------------------------------------------------
# Process registry
# ------------------------------------------------------------------
_procs: dict = {}          # label -> subprocess.Popen
_stop_event = threading.Event()

def _launch(label: str, cmd: list, env: dict, cwd: str):
    """Launch a subprocess and stream its output asynchronously."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=str(cwd),
        text=True,
        bufsize=1,
    )
    _procs[label] = proc
    threading.Thread(target=_stream, args=(proc.stdout, label, _stop_event), daemon=True).start()
    threading.Thread(target=_stream, args=(proc.stderr, label, _stop_event), daemon=True).start()
    return proc

def _kill_all():
    """Terminate all subprocesses gracefully, then kill if needed."""
    _stop_event.set()
    for label, proc in list(_procs.items()):
        if proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass
    time.sleep(2)
    for label, proc in list(_procs.items()):
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass

# ------------------------------------------------------------------
# SIGINT / SIGTERM handler for clean Ctrl+C shutdown
# ------------------------------------------------------------------
def _handle_sigint(signum, frame):
    print("\n\n[run_local] Received interrupt — shutting down all processes...")
    _kill_all()
    sys.exit(1)

signal.signal(signal.SIGINT,  _handle_sigint)
if hasattr(signal, "SIGTERM"):
    signal.signal(signal.SIGTERM, _handle_sigint)

# ==================================================================
# STEP 1 — Start Flower Server
# ==================================================================
print("=" * 64)
print("[run_local] STEP 1 — Starting Flower Server")
print("=" * 64)

wall_start = time.perf_counter()
session_start_iso = datetime.utcnow().isoformat() + "Z"

server_env = build_env({
    "SERVER_ADDRESS": f"0.0.0.0:{args.server_port}",
})

server_proc = _launch(
    label="SERVER",
    cmd=[sys.executable, str(SRC_DIR / "demo_server.py"),
         "--port",          str(args.server_port),
         "--num_rounds",    str(args.num_rounds),
         "--target_clients", str(args.target_clients)],
    env=server_env,
    cwd=PROJECT_ROOT,
)

print(f"[run_local] Server started (PID {server_proc.pid}). "
      f"Waiting {args.server_wait_sec:.1f}s for it to bind...")
time.sleep(args.server_wait_sec)

if server_proc.poll() is not None:
    print(f"[run_local] ERROR: Server exited prematurely (code {server_proc.returncode}). Aborting.")
    _kill_all()
    sys.exit(1)

# ==================================================================
# STEP 2 — Start all hospital clients CONCURRENTLY
#
# Each hospital gets its own subprocess with its own environment.
# None of them wait for the others before starting.
# Flower itself provides the rendezvous point: all clients connect,
# and the server begins a round only when min_clients are registered.
# ==================================================================
print()
print("=" * 64)
print("[run_local] STEP 2 — Launching Hospital Clients (CONCURRENT)")
print("=" * 64)

client_start_times: dict = {}
client_procs: dict = {}

for code in HOSPITAL_CODES:
    data_path = PROJECT_ROOT / "data" / code.lower()
    if not data_path.exists():
        print(f"[run_local] WARNING: Data directory not found: {data_path}")
        print(f"[run_local]          Client {code} may fail to start.")

    c_env = build_env({
        "CLIENT_NAME":    code,
        "DATA_PATH":      str(data_path),
        "SERVER_ADDRESS": SERVER_ADDRESS,
    })

    t0 = time.perf_counter()
    proc = _launch(
        label=code,
        cmd=[sys.executable, str(SRC_DIR / "client.py")],
        env=c_env,
        cwd=PROJECT_ROOT,
    )
    client_start_times[code] = t0
    client_procs[code] = proc
    print(f"[run_local] {code} started (PID {proc.pid})")

print()
print("[run_local] All clients launched. Federated rounds in progress...")
print("[run_local] Press Ctrl+C to stop all processes.")
print()

# ==================================================================
# STEP 3 — Monitor until all clients and server finish
# ==================================================================
optional_timeout = args.timeout_sec if args.timeout_sec > 0 else None

try:
    while True:
        time.sleep(1.0)
        elapsed = time.perf_counter() - wall_start

        # Optional hard timeout
        if optional_timeout and elapsed > optional_timeout:
            print(f"\n[run_local] Timeout ({optional_timeout:.0f}s) reached — killing all processes.")
            break

        # Check if server finished
        if server_proc.poll() is not None:
            print(f"\n[run_local] Server exited (code {server_proc.returncode}).")
            break

        # Check if all clients finished
        alive = [c for c, p in client_procs.items() if p.poll() is None]
        if not alive:
            # Wait briefly for server to wrap up
            print("\n[run_local] All clients have completed. Waiting for server to finalize...")
            try:
                server_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                print("[run_local] Server did not exit cleanly — terminating.")
            break

except KeyboardInterrupt:
    print("\n[run_local] Interrupted.")

finally:
    wall_end = time.perf_counter()
    total_wall = wall_end - wall_start
    _kill_all()

# ==================================================================
# STEP 4 — Report
# ==================================================================
print()
print("=" * 64)
print("   FEDERATED LEARNING — SESSION COMPLETE")
print("=" * 64)
print(f"   Session started     : {session_start_iso}")
print(f"   Total wall-clock    : {total_wall:.2f} sec  ({total_wall/60:.1f} min)")
print()

print("   Client exit codes:")
for code, proc in client_procs.items():
    rc = proc.returncode if proc.returncode is not None else "?"
    status = "OK" if rc == 0 else f"FAILED (code {rc})"
    print(f"     {code:<15}: {status}")

server_rc = server_proc.returncode if server_proc.returncode is not None else "?"
server_status = "OK" if server_rc == 0 else f"FAILED (code {server_rc})"
print(f"     {'Server':<15}: {server_status}")
print()
print("   Results and metrics:")
print(f"     federated_healthcare/dashboard/results/")
print()

# Performance benchmark note
print("   To benchmark sequential vs concurrent execution:")
print("   Compare `client_wall_clock_time_sec` column across Hospital A/B/C")
print("   in the per-client CSV files inside the results directory.")
print("   Concurrent: all three overlap in wall-clock time.")
print("   Sequential: Hospital B would only start after Hospital A finishes.")
print("=" * 64)
print()
