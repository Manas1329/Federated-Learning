"""
dashboard_api.py
================
FastAPI REST + WebSocket server that bridges the React dashboard frontend
with the Flower-based federated learning backend.

Works in TWO modes:
  1. server_instrumented.py mode: receives real-time hooks via dashboard_state
  2. server.py direct mode: polls CSV files that server.py writes every round

Run from federated_healthcare/src/:
    python dashboard_api.py

Requires: pip install fastapi uvicorn websockets pandas
"""

import asyncio
import glob
import json
import math
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# Path setup
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Load .env
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

import dashboard_state as ds

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import uvicorn
except ImportError:
    print("\n[dashboard_api] ERROR: FastAPI not installed.")
    print("Run:  pip install fastapi uvicorn websockets\n")
    sys.exit(1)

try:
    import pandas as pd
    _PANDAS_OK = True
except ImportError:
    _PANDAS_OK = False


# ============================================================
# WebSocket connection manager
# ============================================================

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(json.dumps(data))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()

# ============================================================
# CSV polling state
# ============================================================

_csv_last_round_seen  = 0
_csv_last_client_rows = {}   # {client_name: row_count}
_fl_process: Optional[subprocess.Popen] = None
_last_broadcast_version = -1


def _get_results_info():
    """Return (suffix, results_dir) based on current .env settings."""
    use_dp    = os.environ.get("USE_DP",           "0") == "1"
    use_quant = os.environ.get("USE_QUANTIZATION", "1") == "1"
    if use_dp:
        suffix = "c_dp"
    elif use_quant:
        suffix = "b_quantized"
    else:
        suffix = "a_pure"
    results_dir = Path(BASE_DIR) / "dashboard" / "results" / suffix
    return suffix, results_dir


def _poll_csv_metrics():
    """
    Poll the CSV files server.py writes every round.
    Feeds live data into LIVE_STATE so the dashboard updates
    even when server.py is run directly (not via the dashboard button).
    """
    global _csv_last_round_seen, _csv_last_client_rows
    if not _PANDAS_OK:
        return

    suffix, results_dir = _get_results_info()
    metrics_csv = results_dir / f"metrics_{suffix}.csv"
    round_csv   = results_dir / f"round_metrics_{suffix}.csv"

    # 1. Global metrics: accuracy / loss per round
    try:
        if metrics_csv.exists():
            df = pd.read_csv(metrics_csv)
            if len(df) > _csv_last_round_seen:
                history = []
                for _, row in df.iterrows():
                    history.append({
                        "round":     int(row["Round"]),
                        "accuracy":  round(float(row["Accuracy"]) * 100, 2),
                        "loss":      round(float(row["Loss"]), 4),
                        "f1":        0.0,
                        "precision": 0.0,
                        "recall":    0.0,
                    })
                latest = history[-1]
                new_entries = history[_csv_last_round_seen:]

                with ds._lock:
                    ds.LIVE_STATE["performance_history"] = history
                    ds.LIVE_STATE["latest_metrics"]["accuracy"] = latest["accuracy"]
                    ds.LIVE_STATE["latest_metrics"]["loss"]     = latest["loss"]
                    ds.LIVE_STATE["training_status"] = "Training"
                    ds.LIVE_STATE["current_round"]   = latest["round"]
                    ds._bump_version()
                    ds.write_state_file()

                for r in new_entries:
                    ds.add_event(
                        f"Round {r['round']} completed — "
                        f"Accuracy: {r['accuracy']}%  Loss: {r['loss']}",
                        "aggregation",
                    )
                _csv_last_round_seen = len(df)
    except Exception:
        pass

    # 2. Round metrics: client counts
    try:
        if round_csv.exists():
            df_r = pd.read_csv(round_csv)
            if len(df_r) > 0:
                row     = df_r.iloc[-1]
                success = int(row.get("Successful_Clients", 0))
                failed  = int(row.get("Failed_Clients",     0))
                with ds._lock:
                    ds.LIVE_STATE["dropout_info"]["active_clients"]  = success
                    ds.LIVE_STATE["dropout_info"]["dropped_clients"] = failed
                    ds.LIVE_STATE["dropout_info"]["overall_status"] = (
                        "dropped_out" if failed > 0 else "all_active"
                    )
    except Exception:
        pass

    # 3. Per-client CSVs: hospital table + update stats
    try:
        pattern = str(results_dir / f"*_{suffix}.csv")
        client_csvs = [
            f for f in glob.glob(pattern)
            if not Path(f).name.startswith("metrics_")
            and not Path(f).name.startswith("round_metrics_")
        ]

        for csv_path in client_csvs:
            client_name = Path(csv_path).stem.replace(f"_{suffix}", "")
            prev_count  = _csv_last_client_rows.get(client_name, 0)

            try:
                df_c = pd.read_csv(csv_path)
            except Exception:
                continue

            if len(df_c) == prev_count:
                continue
            _csv_last_client_rows[client_name] = len(df_c)

            # Latest accuracy
            acc_col  = "accuracy" if "accuracy" in df_c.columns else None
            loss_col = "loss"     if "loss"     in df_c.columns else None

            acc, loss = None, None
            if acc_col:
                valid = df_c[df_c[acc_col].notna() & (df_c[acc_col] != "")]
                if len(valid) > 0:
                    acc  = float(valid[acc_col].iloc[-1])
                    if loss_col:
                        loss = float(valid[loss_col].iloc[-1])

            # Payload size
            raw_mb, quant_mb = None, None
            if "payload_size_mb" in df_c.columns:
                t = df_c[df_c["payload_size_mb"].notna() & (df_c["payload_size_mb"] != "")]
                if len(t) > 0:
                    raw_mb = float(t["payload_size_mb"].iloc[-1])
            # Check multiple possible column names for quantized payload
            for qcol in ["quantized_payload_mb", "quant_payload_mb"]:
                if qcol in df_c.columns:
                    t = df_c[df_c[qcol].notna() & (df_c[qcol] != "")]
                    if len(t) > 0:
                        quant_mb = float(t[qcol].iloc[-1])
                    break

            with ds._lock:
                is_new = client_name not in ds.LIVE_STATE["connected_clients"]
                if is_new:
                    ds.LIVE_STATE["connected_clients"].append(client_name)
                    ds.LIVE_STATE["dropout_info"]["active_clients"] = \
                        len(ds.LIVE_STATE["connected_clients"])
                    ds.LIVE_STATE["training_status"] = "Training"

                detail = ds.LIVE_STATE["client_details"].get(client_name, {})
                detail["status"]      = "Update Submitted"
                detail["last_update"] = ds._now_str()
                if acc  is not None: detail["accuracy"] = round(acc  * 100, 2)
                if loss is not None: detail["loss"]     = round(loss, 4)
                if quant_mb is not None:
                    detail["update_size"] = f"{quant_mb:.2f} MB (INT8)"
                elif raw_mb is not None:
                    detail["update_size"] = f"{raw_mb:.2f} MB"
                ds.LIVE_STATE["client_details"][client_name] = detail

                # Update stats
                if raw_mb   is not None:
                    ds.LIVE_STATE["update_stats"]["avg_update_size_raw_mb"]   = round(raw_mb,   3)
                if quant_mb is not None:
                    ds.LIVE_STATE["update_stats"]["avg_update_size_quant_mb"] = round(quant_mb, 3)
                if raw_mb and quant_mb and quant_mb > 0:
                    ds.LIVE_STATE["update_stats"]["compression_ratio"] = \
                        round(raw_mb / quant_mb, 2)
                ds.LIVE_STATE["update_stats"]["total_updates_received"] += 1
                ds._bump_version()
                ds.write_state_file()

            if is_new:
                ds.add_event(f"{client_name} connected (via CSV)", "connect")

    except Exception:
        pass


def _try_read_state_json():
    """Read state.json if server_instrumented.py wrote a newer version."""
    try:
        state_path = Path(ds.STATE_FILE)
        if not state_path.exists():
            return
        with open(state_path) as f:
            disk_state = json.load(f)
        if disk_state.get("version", 0) > ds.LIVE_STATE.get("version", 0):
            with ds._lock:
                ds.LIVE_STATE.update(disk_state)
    except Exception:
        pass


# ============================================================
# Background broadcaster
# ============================================================

async def _state_broadcaster():
    global _last_broadcast_version, _fl_process
    while True:
        await asyncio.sleep(1.5)

        # Check if spawned subprocess finished
        if _fl_process is not None:
            retcode = _fl_process.poll()
            if retcode is not None:
                _fl_process = None
                if ds.LIVE_STATE["training_status"] == "Training":
                    ds.on_training_complete(ds.LIVE_STATE["total_rounds"])
                ds.LIVE_STATE["server_pid"] = None
                ds.write_state_file()

        # Read state.json (from server_instrumented.py if used)
        _try_read_state_json()

        # Always poll CSVs as fallback (works with plain server.py too)
        _poll_csv_metrics()

        # Broadcast if state changed
        if manager.active and ds.LIVE_STATE["version"] != _last_broadcast_version:
            _last_broadcast_version = ds.LIVE_STATE["version"]
            await manager.broadcast(ds.LIVE_STATE)


# ============================================================
# App lifespan
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_state_broadcaster())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title       = "Federated Healthcare AI — Dashboard API",
    description = "REST + WebSocket bridge between the React dashboard and the Flower FL server.",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ============================================================
# REST Endpoints
# ============================================================

@app.get("/api/status")
async def get_status():
    return JSONResponse(ds.LIVE_STATE)


@app.get("/api/config")
async def get_config():
    target = int(os.environ.get("TARGET_CLIENTS", "3"))
    return {
        "use_quantization": os.environ.get("USE_QUANTIZATION", "1") == "1",
        "use_dp":           os.environ.get("USE_DP",           "0") == "1",
        "target_clients":   target,
        "min_clients":      max(2, math.ceil(0.6 * target)),
        "fl_port":          8080,
        "dashboard_port":   8000,
    }


@app.get("/api/metrics/history")
async def get_metrics_history():
    return {"history": ds.LIVE_STATE["performance_history"]}


@app.get("/api/clients")
async def get_clients():
    return {
        "connected_clients": ds.LIVE_STATE["connected_clients"],
        "client_details":    ds.LIVE_STATE["client_details"],
    }


@app.get("/api/events")
async def get_events():
    return {"events": ds.LIVE_STATE["events"]}


@app.post("/api/training/start")
async def start_training(
    num_rounds: int = 5,
    target_clients: int = 3,
    port: int = 8080,
):
    global _fl_process

    if ds.LIVE_STATE["training_status"] in ("Training", "Waiting"):
        raise HTTPException(
            status_code=409,
            detail="Training is already running. Stop or reset first.",
        )

    instrumented_script = os.path.join(SRC_DIR, "server_instrumented.py")
    if not os.path.exists(instrumented_script):
        raise HTTPException(status_code=500,
                            detail="server_instrumented.py not found.")

    cmd = [
        sys.executable, instrumented_script,
        "--num_rounds",     str(num_rounds),
        "--target_clients", str(target_clients),
        "--port",           str(port),
    ]

    ds.reset_state()

    try:
        _fl_process = subprocess.Popen(
            cmd,
            cwd    = SRC_DIR,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text   = True,
            bufsize= 1,
        )
        ds.LIVE_STATE["server_pid"] = _fl_process.pid
        ds.LIVE_STATE["training_status"] = "Waiting"
        ds.LIVE_STATE["total_rounds"] = num_rounds
        ds.write_state_file()

        asyncio.create_task(_read_subprocess_logs(_fl_process))

        return {
            "status": "started",
            "pid":    _fl_process.pid,
            "cmd":    " ".join(cmd),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _read_subprocess_logs(proc: subprocess.Popen):
    loop = asyncio.get_event_loop()

    def _read_line():
        return proc.stdout.readline() if proc.stdout else ""

    while True:
        line = await loop.run_in_executor(None, _read_line)
        if not line:
            if proc.poll() is not None:
                break
            await asyncio.sleep(0.1)
            continue
        line = line.rstrip()
        if not line:
            continue
        print(f"[FL] {line}")
        _parse_log_line(line)


def _parse_log_line(line: str):
    lower = line.lower()
    if "grace period expired" in lower:
        ds.add_event("Grace period expired — some clients dropped", "dropout")
    elif "minimum clients" in lower and "not reached" in lower:
        ds.add_event("Minimum clients not reached — round aborted", "error")


@app.post("/api/training/stop")
async def stop_training():
    global _fl_process
    if _fl_process is None or _fl_process.poll() is not None:
        return {"status": "not_running"}

    try:
        _fl_process.terminate()
        await asyncio.sleep(2)
        if _fl_process.poll() is None:
            _fl_process.kill()
        _fl_process = None

        with ds._lock:
            ds.LIVE_STATE["training_status"] = "Stopped"
            ds.LIVE_STATE["server_pid"]      = None
            ds._bump_version()
            ds.write_state_file()

        ds.add_event("Training stopped by admin", "warn")
        return {"status": "stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/training/reset")
async def reset_training():
    global _fl_process, _csv_last_round_seen, _csv_last_client_rows
    if _fl_process is not None and _fl_process.poll() is None:
        raise HTTPException(
            status_code=409,
            detail="Cannot reset while training is running. Stop first.",
        )
    _csv_last_round_seen  = 0
    _csv_last_client_rows = {}
    ds.reset_state()
    return {"status": "reset"}


# ============================================================
# WebSocket endpoint
# ============================================================

@app.websocket("/ws/dashboard")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        await ws.send_text(json.dumps(ds.LIVE_STATE))
        while True:
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                await ws.send_text(json.dumps({"ping": True}))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        manager.disconnect(ws)


# ============================================================
# Health
# ============================================================

@app.get("/")
async def root():
    return {
        "service": "Federated Healthcare AI — Dashboard API",
        "status":  "running",
        "docs":    "http://localhost:8000/docs",
    }


@app.get("/health")
async def health():
    return {"status": "ok", "training": ds.LIVE_STATE["training_status"]}


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
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

    LAN_IP = _get_lan_ip()

    print("\n")
    print("=" * 60)
    print("  FEDERATED HEALTHCARE AI  --  DASHBOARD API")
    print("=" * 60)
    print(f"  REST API  : http://localhost:8000")
    print(f"  WebSocket : ws://localhost:8000/ws/dashboard")
    print(f"  Docs      : http://localhost:8000/docs")
    print(f"  LAN IP    : {LAN_IP}")
    print(f"  CSV poll  : ENABLED (reads server.py CSVs directly)")
    print("=" * 60)
    print()

    uvicorn.run(
        "dashboard_api:app",
        host      = "0.0.0.0",
        port      = 8000,
        reload    = False,
        log_level = "info",
    )
