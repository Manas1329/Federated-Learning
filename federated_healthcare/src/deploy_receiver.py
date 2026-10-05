"""
deploy_receiver.py
==================
NETWORK-MODE hospital receiver.  Run on each HOSPITAL LAPTOP.

What it does
------------
1. Listens for the server's HTTP POST to /receive_model.
2. Verifies the SHA256 checksum of the received model bytes.
3. Saves the model as <save_dir>/deployed_model.pth.
4. Optionally runs local inference immediately after receiving the model
   (inference uses THIS laptop's local data — patient data is NEVER sent back).

ADDITIVE ONLY: does not modify any FL, trust, dropout, or unlearning code.

Usage (HOSPITAL LAPTOP)
-----------------------
    cd federated_healthcare/src
    python deploy_receiver.py [options]

Options
-------
    --client-name NAME     Hospital name, e.g. Hospital_A  (default: CLIENT_NAME env)
    --save-dir DIR         Where to save received deployed_model.pth
                           (default: ../models/deployed/<suffix>/<client_name>/)
    --port PORT            Port to listen on (default: 9765)
    --suffix SUFFIX        Used for default save path (auto-detected from .env)
    --no-inference         Skip local inference after receiving model
    --once                 Exit after receiving and saving one model
"""

from __future__ import annotations

import os
import sys
import json
import time
import hashlib
import argparse
import threading
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Load .env
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                if _k.strip() not in os.environ:
                    os.environ[_k.strip()] = _v.strip()

from paths import MODELS_DIR, DATA_DIR  # noqa: E402


# ===========================================================================
# Helpers
# ===========================================================================

def _banner(title: str, width: int = 60):
    print("\n" + "=" * width)
    print("  " + title)
    print("=" * width)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _resolve_suffix(cli_suffix):
    if cli_suffix:
        return cli_suffix
    if "EXPERIMENT_NAME" in os.environ:
        return os.environ["EXPERIMENT_NAME"]
    use_dp    = os.environ.get("USE_DP",           "0") == "1"
    use_quant = os.environ.get("USE_QUANTIZATION", "1") == "1"
    if use_dp:
        return "c_dp"
    elif use_quant:
        return "b_quantized"
    else:
        return "a_pure"


# ===========================================================================
# Local inference  (patient data stays 100% local)
# ===========================================================================

def _run_local_inference(client_name, deployed_model_path):
    """
    Load the received model and run inference on THIS hospital's local data.
    Patient data never leaves this machine.
    Returns result dict or None on failure.
    """
    try:
        import torch
        import numpy as np
        from model import ChestCNN, test        # existing model
        from utils import load_hospital_data    # existing data loader
    except ImportError as exc:
        print("[Receiver] Could not import model/utils: {}".format(exc))
        return None

    local_data_path = DATA_DIR / client_name.lower()
    if not local_data_path.exists():
        print("[Receiver] Local data path not found: {}".format(local_data_path))
        print("[Receiver] Skipping inference — no local data available.")
        return None

    try:
        _trainloader, testloader = load_hospital_data(str(local_data_path))
    except Exception as exc:
        print("[Receiver] Could not load local data: {}".format(exc))
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ChestCNN().to(device)
    try:
        state_dict = torch.load(str(deployed_model_path), map_location=device)
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        print("[Receiver] ERROR loading model: {}".format(exc))
        return None

    model.eval()

    try:
        loss, accuracy = test(model, testloader)
    except Exception as exc:
        print("[Receiver] ERROR during inference: {}".format(exc))
        return None

    all_preds, all_labels = [], []
    try:
        import torch
        with torch.no_grad():
            for images, labels in testloader:
                images = images.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy().tolist())
                all_labels.extend(labels.numpy().tolist())
    except Exception as exc:
        print("[Receiver] WARNING: detailed prediction collection failed: {}".format(exc))

    import numpy as np
    preds_arr           = np.array(all_preds)
    normal_predicted    = int(np.sum(preds_arr == 0))
    pneumonia_predicted = int(np.sum(preds_arr == 1))
    total               = len(preds_arr)

    return {
        "client_name":         client_name,
        "local_data_path":     str(local_data_path),
        "total_samples":       total,
        "normal_predicted":    normal_predicted,
        "pneumonia_predicted": pneumonia_predicted,
        "loss":                loss,
        "accuracy":            accuracy,
    }


# ===========================================================================
# HTTP receiver server
# ===========================================================================

class _ReceiverState:
    """Mutable singleton state shared between HTTP handler and main thread."""
    received_event = threading.Event()
    received_path  = None     # Path where last model was saved
    received_ok    = False    # True if SHA256 verified
    client_name    = "Unknown"
    save_dir       = None     # Path
    no_inference   = False
    run_once       = False
    _lock          = threading.Lock()


class _ReceiverHandler(BaseHTTPRequestHandler):
    """Receives POST /receive_model from the deploy_server."""

    def do_GET(self):
        if self.path == "/health":
            self._text(200, "OK — deploy_receiver running as {}".format(
                _ReceiverState.client_name
            ))
        else:
            self._text(404, "Not found. POST to /receive_model")

    def do_POST(self):
        if self.path != "/receive_model":
            self._text(404, "Unknown endpoint")
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._text(400, "Empty body — no model data received")
            return

        data         = self.rfile.read(content_length)
        remote_sha   = self.headers.get("X-Model-SHA256", "")
        client_name  = self.headers.get("X-Client-Name",  _ReceiverState.client_name)
        model_name   = self.headers.get("X-Model-Name",   "deployed_model.pth")

        # Verify integrity
        local_sha = _sha256_bytes(data)
        if remote_sha and local_sha != remote_sha:
            msg = (
                "SHA256 mismatch! expected={} got={}".format(remote_sha, local_sha)
            )
            print("[Receiver] ERROR: {}".format(msg))
            self._text(400, "CHECKSUM_MISMATCH: " + msg)
            return

        # Save model
        save_dir = _ReceiverState.save_dir
        if save_dir is None:
            self._text(500, "save_dir not configured")
            return

        save_dir.mkdir(parents=True, exist_ok=True)
        dest = save_dir / "deployed_model.pth"

        try:
            dest.write_bytes(data)
        except Exception as exc:
            self._text(500, "SAVE ERROR: {}".format(exc))
            return

        print("\n[Receiver] Model received from {} ({} bytes)".format(
            self.client_address[0], len(data)
        ))
        print("[Receiver] SHA256 verified: {}".format(local_sha))
        print("[Receiver] Saved to       : {}".format(dest))

        # Signal main thread
        with _ReceiverState._lock:
            _ReceiverState.received_path = dest
            _ReceiverState.received_ok   = True
        _ReceiverState.received_event.set()

        self._text(200, "RECEIVED_OK sha256={}".format(local_sha))

        # If run-once, trigger shutdown from background thread
        if _ReceiverState.run_once:
            threading.Thread(target=lambda: (time.sleep(0.3), _shutdown_flag.set()),
                             daemon=True).start()

    def _text(self, code, text):
        body = text.encode()
        self.send_response(code)
        self.send_header("Content-Type",   "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # Suppress Apache-style logs


_shutdown_flag = threading.Event()


# ===========================================================================
# Main orchestrator
# ===========================================================================

def run_receiver(
    client_name=None,
    save_dir=None,
    port=9765,
    suffix=None,
    no_inference=False,
    run_once=False,
):
    """
    Start the HTTP receiver and wait for the model to be pushed by the server.
    """
    effective_suffix = _resolve_suffix(suffix)
    effective_name   = (
        client_name
        or os.environ.get("CLIENT_NAME", "Hospital_A")
    )

    effective_save_dir = (
        Path(save_dir)
        if save_dir
        else MODELS_DIR / "deployed" / effective_suffix / effective_name
    )
    effective_save_dir.mkdir(parents=True, exist_ok=True)

    # Configure shared state
    _ReceiverState.client_name  = effective_name
    _ReceiverState.save_dir     = effective_save_dir
    _ReceiverState.no_inference = no_inference
    _ReceiverState.run_once     = run_once

    _banner("HOSPITAL DEPLOYMENT RECEIVER")
    print("  Hospital     : {}".format(effective_name))
    print("  Listening on : 0.0.0.0:{}".format(port))
    print("  Save dir     : {}".format(effective_save_dir))
    print("  Suffix       : {}".format(effective_suffix))
    print("  Local data   : {}".format(DATA_DIR / effective_name.lower()))
    print("  Mode         : {}".format("once" if run_once else "continuous"))
    print("  Inference    : {}".format("disabled" if no_inference else "enabled"))

    import socket
    try:
        my_ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        my_ip = "127.0.0.1"
    print("\n[Receiver] Waiting for model from server...")
    print("[Receiver] Health check: http://{}:{}/health".format(my_ip, port))
    print("[Receiver] Endpoint    : POST http://{}:{}/receive_model".format(my_ip, port))

    httpd = HTTPServer(("0.0.0.0", port), _ReceiverHandler)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    try:
        while not _shutdown_flag.is_set():
            received = _ReceiverState.received_event.wait(timeout=1.0)
            if received:
                _ReceiverState.received_event.clear()

                saved_path = _ReceiverState.received_path
                _banner("MODEL RECEIVED")
                print("  Hospital : {}".format(effective_name))
                print("  Saved at : {}".format(saved_path))

                # Run local inference if enabled
                if not no_inference and saved_path is not None:
                    _banner("LOCAL HOSPITAL INFERENCE")
                    print("  PATIENT DATA STAYS LOCAL — no data sent back to server\n")
                    result = _run_local_inference(effective_name, saved_path)
                    if result:
                        print("  {:25s}: {}".format("Hospital", result["client_name"]))
                        print("  {:25s}: {}".format("Local data", result["local_data_path"]))
                        print("  {:25s}: {}".format("Test samples", result["total_samples"]))
                        print("  {:25s}: {:.4f}".format("Loss", result["loss"]))
                        print("  {:25s}: {:.2f}%".format("Accuracy", result["accuracy"] * 100))
                        print("  {:25s}: {}".format("NORMAL predicted",    result["normal_predicted"]))
                        print("  {:25s}: {}".format("PNEUMONIA predicted", result["pneumonia_predicted"]))
                    else:
                        print("  [WARNING] Inference skipped (no local data or model load error).")

                if run_once:
                    break

    except KeyboardInterrupt:
        print("\n[Receiver] Keyboard interrupt — stopping.")
    finally:
        httpd.shutdown()
        print("[Receiver] Server stopped.")


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        prog="deploy_receiver.py",
        description=(
            "Hospital-side model receiver for network-mode deployment. "
            "Run on each hospital laptop. Waits for the server to push the "
            "final global model, saves it, and optionally runs local inference. "
            "ADDITIVE — no FL or training logic is touched."
        ),
    )
    p.add_argument("--client-name",  default=None, dest="client_name",
                   help="This hospital's name, e.g. Hospital_A.")
    p.add_argument("--save-dir",     default=None, dest="save_dir",
                   help="Directory to save deployed_model.pth.")
    p.add_argument("--port",         type=int, default=9765,
                   help="Port to listen on (default 9765).")
    p.add_argument("--suffix",       default=None,
                   help="Model suffix (auto-detected from .env).")
    p.add_argument("--no-inference", action="store_true", dest="no_inference",
                   help="Skip local inference after receiving model.")
    p.add_argument("--once",         action="store_true",
                   help="Exit after receiving and processing one model.")
    args = p.parse_args()

    run_receiver(
        client_name  = args.client_name,
        save_dir     = args.save_dir,
        port         = args.port,
        suffix       = args.suffix,
        no_inference = args.no_inference,
        run_once     = args.once,
    )
