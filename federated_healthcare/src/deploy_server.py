"""
deploy_server.py
================
NETWORK-MODE deployment server for Final Global Model Distribution.

Run this on the SERVER LAPTOP after FL training (and optional unlearning /
DACM recovery) is fully complete.

Capabilities
------------
1. Exposes a lightweight HTTP pull endpoint (GET /model) so hospital laptops
   can pull the final model on demand.
2. Proactively HTTP-POSTs the final model to each eligible hospital laptop
   that is running deploy_receiver.py.

Eligibility rules (read-only — nothing is written)
---------------------------------------------------
* Scans MODELS_DIR for existing dacm_recovered_*_{suffix}.pth artefacts
  produced by the unlearning pipeline → those clients are permanently
  excluded and will NOT receive the model.
* Temporarily dropped / straggling clients are NOT excluded.
* --excluded flag adds extra manual exclusions.

ADDITIVE ONLY: does not modify server.py, trust_manager, dropout_handler,
FedAvg, quantization, dacu, or unlearn_workflow.

Usage (SERVER LAPTOP)
---------------------
    cd federated_healthcare/src
    python deploy_server.py [options]

Options
-------
    --suffix SUFFIX           Model suffix (auto-detected from .env)
    --model-path PATH         Explicit .pth path (skips auto-detect)
    --known-clients LIST      Comma-separated hospital names
    --excluded LIST           Comma-separated extra exclusions
    --hospital-ips MAPPING    Hospital_A=192.168.1.101,Hospital_B=192.168.1.102
    --receiver-port PORT      Port where deploy_receiver.py listens (default 9765)
    --serve-port PORT         Port for GET /model endpoint (default 9764)
    --no-push                 Serve only; do not push to hospitals
    --no-serve                Push only; do not start HTTP server
"""

from __future__ import annotations

import os
import sys
import json
import time
import socket
import hashlib
import argparse
import threading
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import urlopen, Request
from urllib.error import URLError

# ---------------------------------------------------------------------------
# Ensure src package is importable regardless of cwd
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Load .env (mirrors server.py / deploy_model.py behaviour exactly)
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                if _k.strip() not in os.environ:
                    os.environ[_k.strip()] = _v.strip()

from paths import MODELS_DIR  # noqa: E402


# ===========================================================================
# Helpers
# ===========================================================================

def _banner(title: str, width: int = 64):
    print("\n" + "=" * width)
    print("  " + title)
    print("=" * width)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ===========================================================================
# Suffix resolution  (mirrors deploy_model.py / server.py exactly)
# ===========================================================================

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
# Final model identification  (same priority as deploy_model.py)
# ===========================================================================

def _find_final_model(suffix, cli_model_path):
    """
    Priority:
      1. Explicit --model-path
      2. Most-recently modified dacm_recovered_*_{suffix}.pth
      3. global_model_{suffix}.pth
    """
    if cli_model_path:
        p = Path(cli_model_path)
        if not p.exists():
            raise FileNotFoundError("[ERROR] --model-path not found: {}".format(p))
        return p, "Explicitly specified model"

    recovered = sorted(
        MODELS_DIR.glob("dacm_recovered_*_{}.pth".format(suffix)),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if recovered:
        chosen = recovered[0]
        return chosen, "DACM-recovered model after unlearning ({})".format(chosen.name)

    normal = MODELS_DIR / "global_model_{}.pth".format(suffix)
    if normal.exists():
        return normal, "Normal final global model (no unlearning)"

    raise FileNotFoundError(
        "[ERROR] No model found for suffix '{}'. Run FL training first.".format(suffix)
    )


# ===========================================================================
# Eligible / excluded client resolution  (READ-ONLY, mirrors deploy_model.py)
# ===========================================================================

def _resolve_clients(cli_known, cli_excluded, suffix):
    if cli_known:
        known = [c.strip() for c in cli_known.split(",") if c.strip()]
    else:
        env_k = os.environ.get("KNOWN_CLIENTS", "Hospital_A,Hospital_B,Hospital_C")
        known = [c.strip() for c in env_k.split(",") if c.strip()]
    if not known:
        raise ValueError("[ERROR] No known clients found.")

    # Auto-detect permanently excluded from DACM artefacts
    auto_ex     = set()
    prefix      = "dacm_recovered_"
    suffix_part = "_{}".format(suffix)
    for rp in MODELS_DIR.glob("dacm_recovered_*_{}.pth".format(suffix)):
        stem = rp.stem
        if stem.startswith(prefix) and stem.endswith(suffix_part):
            cname = stem[len(prefix): len(stem) - len(suffix_part)]
            if cname in known:
                auto_ex.add(cname)

    explicit_ex = set()
    if cli_excluded:
        explicit_ex = {c.strip() for c in cli_excluded.split(",") if c.strip()}

    all_ex   = auto_ex | explicit_ex
    eligible = sorted([c for c in known if c not in all_ex])
    excluded = sorted(all_ex)
    return eligible, excluded


# ===========================================================================
# Hospital IP map parsing
# ===========================================================================

def _parse_hospital_ips(raw):
    """
    Accept Hospital_A=192.168.1.101,Hospital_B=192.168.1.102
    or JSON {"Hospital_A": "192.168.1.101"}.
    """
    if not raw:
        return {}
    raw = raw.strip()
    if raw.startswith("{"):
        return json.loads(raw)
    result = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if "=" in pair:
            k, v = pair.split("=", 1)
            result[k.strip()] = v.strip()
    return result


# ===========================================================================
# Background HTTP model server  (GET /model  GET /sha256  GET /health)
# ===========================================================================

_SERVED_MODEL_PATH = None
_MODEL_SHA256      = None
_SERVER_READY      = threading.Event()


class _ModelHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler that serves the model binary on GET /model."""

    def do_GET(self):
        if self.path == "/health":
            self._text(200, "OK")
        elif self.path == "/sha256":
            self._text(200, _MODEL_SHA256 or "")
        elif self.path == "/model":
            if _SERVED_MODEL_PATH is None or not _SERVED_MODEL_PATH.exists():
                self._text(503, "Model not ready")
                return
            data = _SERVED_MODEL_PATH.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type",   "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Model-SHA256", _MODEL_SHA256 or "")
            self.send_header("X-Model-Name",   _SERVED_MODEL_PATH.name)
            self.end_headers()
            self.wfile.write(data)
            print("[DeployServer] GET /model -> {} ({} bytes)".format(
                self.client_address[0], len(data)
            ))
        else:
            self._text(404, "Not found")

    def _text(self, code, text):
        body = text.encode()
        self.send_response(code)
        self.send_header("Content-Type",   "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # Suppress default Apache-style access logs


def _start_http_server(model_path, serve_port):
    global _SERVED_MODEL_PATH, _MODEL_SHA256
    _SERVED_MODEL_PATH = model_path
    _MODEL_SHA256      = _sha256(model_path)
    httpd = HTTPServer(("0.0.0.0", serve_port), _ModelHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    _SERVER_READY.set()
    return httpd


# ===========================================================================
# Push model to one hospital receiver
# ===========================================================================

def _push_to_hospital(client_name, ip, model_path, sha256, receiver_port, timeout=60):
    """HTTP POST the model binary to the hospital's deploy_receiver.py."""
    url  = "http://{}:{}/receive_model".format(ip, receiver_port)
    data = model_path.read_bytes()
    req  = Request(url, data=data, method="POST")
    req.add_header("Content-Type",   "application/octet-stream")
    req.add_header("Content-Length", str(len(data)))
    req.add_header("X-Client-Name",  client_name)
    req.add_header("X-Model-SHA256", sha256)
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode()
            if resp.status == 200:
                print("[DeployServer] {:20s} -> {} | PUSH OK | {} bytes | {}".format(
                    client_name, ip, len(data), body.strip()
                ))
                return True
            print("[DeployServer] {:20s} -> {} | PUSH FAIL HTTP {} | {}".format(
                client_name, ip, resp.status, body.strip()
            ))
            return False
    except (URLError, OSError) as exc:
        print("[DeployServer] {:20s} -> {} | CONNECTION ERROR: {}".format(
            client_name, ip, exc
        ))
        return False


# ===========================================================================
# Main orchestrator
# ===========================================================================

def run_deploy_server(
    suffix=None, model_path=None, known_clients=None, excluded=None,
    hospital_ips=None, receiver_port=9765, serve_port=9764,
    no_push=False, no_serve=False,
):
    """
    Full network-mode deployment pipeline.
    Returns True if all pushes succeeded (or if no pushes were attempted).
    """
    try:
        # 1 — Identify final model
        effective_suffix = _resolve_suffix(suffix)
        final_model_path, model_type = _find_final_model(effective_suffix, model_path)
        model_sha256 = _sha256(final_model_path)

        _banner("NETWORK DEPLOYMENT SERVER")
        print("  Mode         : NETWORK (HTTP)")
        print("  Suffix       : {}".format(effective_suffix))
        print("  Final Model  : {}".format(final_model_path))
        print("  Model Type   : {}".format(model_type))
        print("  SHA256       : {}".format(model_sha256))

        # 2 — Resolve eligible / excluded
        eligible_clients, excluded_clients = _resolve_clients(
            known_clients, excluded, effective_suffix
        )
        print("  Eligible     : {}".format(", ".join(eligible_clients) or "None"))
        print("  Excluded     : {}".format(", ".join(excluded_clients) or "None"))

        if not eligible_clients:
            print("[DeployServer] WARNING: No eligible hospitals. Nothing to deploy.")
            return False

        # 3 — Parse IP map
        ip_map = _parse_hospital_ips(hospital_ips)
        my_ip  = socket.gethostbyname(socket.gethostname())
        print("  Server IP    : {}".format(my_ip))
        print("  Serve port   : {}".format(serve_port))
        print("  Receiver port: {}".format(receiver_port))
        print("  Hospital IPs : {}".format(ip_map or "(none provided — push skipped)"))

        # 4 — Start background HTTP model server
        httpd = None
        if not no_serve:
            httpd = _start_http_server(final_model_path, serve_port)
            _SERVER_READY.wait()
            print("\n[DeployServer] HTTP model server started:")
            print("  Pull URL  : http://{}:{}/model".format(my_ip, serve_port))
            print("  Health    : http://{}:{}/health".format(my_ip, serve_port))
            print("  SHA256 URL: http://{}:{}/sha256".format(my_ip, serve_port))

        # 5 — Proactive push to each eligible hospital
        results = {}
        if not no_push:
            if not ip_map:
                print(
                    "\n[DeployServer] --hospital-ips not provided. Skipping proactive push.\n"
                    "  Hospitals can pull from: http://{}:{}/model".format(my_ip, serve_port)
                )
            else:
                _banner("PUSHING MODEL TO HOSPITAL LAPTOPS")
                for client in eligible_clients:
                    ip = ip_map.get(client)
                    if not ip:
                        print("[DeployServer] {:20s} -> NO IP CONFIGURED — skipping".format(client))
                        results[client] = "NO_IP"
                        continue
                    print("[DeployServer] {:20s} -> {}:{} pushing ...".format(
                        client, ip, receiver_port
                    ))
                    ok = _push_to_hospital(
                        client, ip, final_model_path, model_sha256, receiver_port
                    )
                    results[client] = "SUCCESS" if ok else "FAILED"

                for client in excluded_clients:
                    print("[DeployServer] {:20s} -> EXCLUDED (not pushed)".format(client))
                    results[client] = "EXCLUDED"

                _banner("PUSH SUMMARY")
                icons = {"SUCCESS": "PASS", "FAILED": "FAIL", "EXCLUDED": "SKIP", "NO_IP": "SKIP"}
                for client, status in results.items():
                    print("  [{:4s}] {:20s} : {}".format(icons.get(status, "?"), client, status))

        # 6 — Keep serving if background server was started
        if httpd and not no_serve:
            print("\n[DeployServer] Serving model. Press Ctrl+C to stop.\n")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n[DeployServer] Keyboard interrupt — shutting down HTTP server.")
                httpd.shutdown()

        return all(v == "SUCCESS" for v in results.values()) if results else True

    except FileNotFoundError as exc:
        print(str(exc))
        return False
    except Exception as exc:
        print("[DeployServer ERROR] {}".format(exc))
        import traceback
        traceback.print_exc()
        return False


# ===========================================================================
# CLI entry point
# ===========================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        prog="deploy_server.py",
        description=(
            "Network-mode Final Global Model Deployment Server. "
            "Run AFTER FL training. Serves / pushes model to eligible hospital "
            "laptops. ADDITIVE — no FL, trust, or unlearning logic is touched."
        ),
    )
    p.add_argument("--suffix",         default=None,
                   help="Model suffix (auto-detected from .env).")
    p.add_argument("--model-path",     default=None, dest="model_path",
                   help="Explicit path to .pth file.")
    p.add_argument("--known-clients",  default=None, dest="known_clients",
                   help="Comma-separated hospital names.")
    p.add_argument("--excluded",       default=None,
                   help="Comma-separated extra hospitals to exclude.")
    p.add_argument("--hospital-ips",   default=None, dest="hospital_ips",
                   help="Hospital_A=192.168.1.101,Hospital_B=192.168.1.102")
    p.add_argument("--receiver-port",  type=int, default=9765, dest="receiver_port",
                   help="Port where deploy_receiver.py listens (default 9765).")
    p.add_argument("--serve-port",     type=int, default=9764, dest="serve_port",
                   help="Port to expose GET /model (default 9764).")
    p.add_argument("--no-push",        action="store_true", dest="no_push",
                   help="Only serve; skip proactive push.")
    p.add_argument("--no-serve",       action="store_true", dest="no_serve",
                   help="Only push; skip starting HTTP server.")
    args = p.parse_args()

    ok = run_deploy_server(
        suffix        = args.suffix,
        model_path    = args.model_path,
        known_clients = args.known_clients,
        excluded      = args.excluded,
        hospital_ips  = args.hospital_ips,
        receiver_port = args.receiver_port,
        serve_port    = args.serve_port,
        no_push       = args.no_push,
        no_serve      = args.no_serve,
    )
    sys.exit(0 if ok else 1)
