"""
unlearn_server.py
=================
Lightweight HTTP server that receives post-training unlearning requests
and routes them through an admin approval gate before executing the workflow.

Usage (run AFTER FL training has finished):
    cd src
    python unlearn_server.py

Environment overrides:
    UNLEARN_PORT=8765          # Port for this HTTP server (default: 8765)
    KNOWN_CLIENTS=Hospital_A,Hospital_B,Hospital_C
    DACM_TAU_SAFE=0.50
    DACM_ALPHA=3.0
    DACM_RECOVERY_EPOCHS=2
    UNLEARN_LR=5e-3
    UNLEARN_STEPS=5
"""

import os
import sys
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# Ensure src is importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Load .env if present (mirrors server.py behaviour)
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                if _k.strip() not in os.environ:
                    os.environ[_k.strip()] = _v.strip()

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
UNLEARN_PORT: int = int(os.environ.get("UNLEARN_PORT", "8765"))

def _get_known_clients() -> set:
    if "KNOWN_CLIENTS" in os.environ:
        return {c.strip() for c in os.environ["KNOWN_CLIENTS"].split(",") if c.strip()}
    from dacm import discover_clients
    return set(discover_clients())

KNOWN_CLIENTS: set = _get_known_clients()

# Active client registry — starts as all known clients; updated after each
# successful unlearning so repeated requests are correctly rejected.
_active_clients: set = set(KNOWN_CLIENTS)

# State flags
_pending_request: dict = {}       # {"client_id": str} when a request is pending
_pending_lock = threading.Lock()  # serialises concurrent requests
_unlearning_running = False       # True while a workflow is executing


# -----------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------

def _validate_request(client_id: str) -> tuple[bool, str]:
    """
    Validate an incoming unlearning request.

    Returns (ok: bool, reason: str)
    """
    global _unlearning_running, _pending_request

    if not client_id:
        return False, "Missing client_id in request body."

    if client_id not in KNOWN_CLIENTS:
        return False, (
            f"Unknown client '{client_id}'. "
            f"Known clients: {sorted(KNOWN_CLIENTS)}"
        )

    if client_id not in _active_clients:
        return False, (
            f"Client '{client_id}' has already been unlearned "
            f"and is no longer active."
        )

    if _unlearning_running:
        return False, (
            "Another unlearning operation is already running. "
            "Please wait for it to complete."
        )

    if _pending_request:
        pending_id = _pending_request.get("client_id", "?")
        return False, (
            f"A request for '{pending_id}' is already pending admin approval. "
            "Please resolve it before submitting a new request."
        )

    return True, "OK"


# -----------------------------------------------------------------------
# Admin approval (terminal prompt — modular, replaceable by dashboard)
# -----------------------------------------------------------------------

def _admin_approval_prompt(client_id: str) -> bool:
    """
    Block until the administrator types 'y' or 'n'.
    Returns True if accepted, False if rejected.

    This function is intentionally isolated so it can later be replaced
    with a dashboard button / REST callback without touching any other logic.
    """
    print("\n")
    print("=" * 42)
    print("      UNLEARNING REQUEST RECEIVED      ")
    print("=" * 42)
    print(f"\n  Client: {client_id}")
    print(f"  Status: PENDING\n")

    while True:
        try:
            answer = input("  Accept request? [y/n]: ").strip().lower()
        except EOFError:
            # Non-interactive environment (e.g. piped input)
            answer = "n"

        if answer in ("y", "yes"):
            print(f"\n[ADMIN] Request ACCEPTED")
            print(f"[ADMIN] Target client: {client_id}")
            return True
        elif answer in ("n", "no"):
            print(f"\n[ADMIN] Request REJECTED")
            return False
        else:
            print("  Please enter 'y' or 'n'.")


# -----------------------------------------------------------------------
# Approval + workflow runner (runs in background thread)
# -----------------------------------------------------------------------

def _handle_approved_or_rejected(client_id: str):
    """
    Run the full approval → workflow sequence in a background thread
    so the HTTP response can be sent immediately while this runs.
    """
    global _pending_request, _unlearning_running, _active_clients

    accepted = _admin_approval_prompt(client_id)

    if not accepted:
        print(f"[REQUEST] Unlearning request from {client_id} was REJECTED by admin.")
        print(f"[UNLEARNING] No unlearning performed.")
        print(f"[DACM] No recovery triggered.")
        with _pending_lock:
            _pending_request = {}
        return

    # Mark as running
    with _pending_lock:
        _unlearning_running = True
        _pending_request = {}

    try:
        from unlearn_workflow import run_unlearn_workflow, SUFFIX

        print(f"[UNLEARNING] Target client: {client_id}")
        print(f"[UNLEARNING] Existing unlearning procedure started")

        # Pass a *copy* of the current active set so the workflow can mutate it
        active_copy = set(_active_clients)

        success = run_unlearn_workflow(
            target_client=client_id,
            suffix=SUFFIX,
            active_clients=active_copy,
        )

        if success:
            # Persist removal in the server's registry
            _active_clients.discard(client_id)
            print(f"\n[SERVER] Active clients updated: {sorted(_active_clients)}")
        else:
            print(f"\n[SERVER] Workflow reported failure for {client_id}.")

    except Exception as exc:
        print(f"\n[ERROR] Unlearning workflow raised an exception: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        with _pending_lock:
            _unlearning_running = False


# -----------------------------------------------------------------------
# HTTP Request Handler
# -----------------------------------------------------------------------

class UnlearningRequestHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):  # noqa: A002
        # Suppress default HTTP access log; our own prints are clearer
        pass

    # ----------------------------------------------------------------
    # POST /request-unlearning
    # ----------------------------------------------------------------
    def do_POST(self):  # noqa: N802
        if self.path != "/request-unlearning":
            self._send_json(404, {"error": f"Unknown endpoint: {self.path}"})
            return

        # Parse body
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length) if content_length else b""

        try:
            body = json.loads(raw_body) if raw_body else {}
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON body."})
            return

        client_id = body.get("client_id", "").strip()
        request_type = body.get("request", "").strip()

        if request_type != "unlearn":
            self._send_json(400, {
                "error": f"Unknown request type '{request_type}'. Expected 'unlearn'."
            })
            return

        print(f"\n[REQUEST] Unlearning request received from {client_id}")

        # Validate
        with _pending_lock:
            ok, reason = _validate_request(client_id)
            if not ok:
                print(f"[VALIDATION] REJECTED — {reason}")
                self._send_json(409, {"status": "REJECTED", "reason": reason})
                return

            # Mark as pending
            _pending_request["client_id"] = client_id

        print(f"[VALIDATION] Client {client_id} is active")
        print(f"[REQUEST] Status: PENDING")

        # Respond immediately — admin interaction runs in background thread
        self._send_json(202, {
            "status": "PENDING",
            "client_id": client_id,
            "message": (
                "Request received and is pending admin approval. "
                "Check the server terminal."
            ),
        })

        # Fire-and-forget background thread for approval + workflow
        t = threading.Thread(
            target=_handle_approved_or_rejected,
            args=(client_id,),
            daemon=True,
        )
        t.start()

    # ----------------------------------------------------------------
    # GET /status — handy health-check
    # ----------------------------------------------------------------
    def do_GET(self):  # noqa: N802
        if self.path == "/status":
            self._send_json(200, {
                "server": "unlearn_server",
                "known_clients": sorted(KNOWN_CLIENTS),
                "active_clients": sorted(_active_clients),
                "pending_request": _pending_request or None,
                "unlearning_running": _unlearning_running,
            })
        else:
            self._send_json(404, {"error": f"Unknown endpoint: {self.path}"})

    # ----------------------------------------------------------------
    # Helper
    # ----------------------------------------------------------------
    def _send_json(self, status_code: int, payload: dict):
        body = json.dumps(payload, indent=2).encode()
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  UNLEARNING REQUEST SERVER")
    print("=" * 60)
    print(f"  Port            : {UNLEARN_PORT}")
    print(f"  Known clients   : {sorted(KNOWN_CLIENTS)}")
    print(f"  Active clients  : {sorted(_active_clients)}")
    print(f"  Admin approval  : terminal prompt [y/n]")
    print("=" * 60)
    print(f"\n[UNLEARN SERVER] Listening on http://0.0.0.0:{UNLEARN_PORT}")
    print(f"[UNLEARN SERVER] Waiting for unlearning requests...")
    print(f"[UNLEARN SERVER] Press Ctrl+C to stop.\n")

    httpd = HTTPServer(("0.0.0.0", UNLEARN_PORT), UnlearningRequestHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[UNLEARN SERVER] Shutting down.")
        httpd.server_close()
