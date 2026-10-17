"""
request_unlearning.py
=====================
Lightweight CLI tool for submitting an unlearning request to the
running unlearn_server.py.

This file performs NO unlearning itself.
Its only responsibility is to send the request and display the server's response.

Usage:
    python request_unlearning.py --client-id Hospital_B
    python request_unlearning.py --client-id Hospital_A
    python request_unlearning.py --client-id Hospital_C

Optional flags:
    --server  http://localhost:8765   (default)
    --timeout 10                      (seconds, default: 10)
"""

import argparse
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path

# Load .env if present so UNLEARN_PORT can be read from it
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    import os
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                if _k.strip() not in os.environ:
                    os.environ[_k.strip()] = _v.strip()

import os
_default_port = int(os.environ.get("UNLEARN_PORT", "8765"))
_default_server = f"http://localhost:{_default_port}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Submit a post-training unlearning request to the unlearn_server. "
            "The server admin must then Accept [y] or Reject [n] in their terminal."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python request_unlearning.py --client-id Hospital_B
  python request_unlearning.py --client-id Hospital_A --server http://192.168.1.10:8765
        """,
    )
    parser.add_argument(
        "--client-id",
        required=True,
        metavar="CLIENT_ID",
        help="ID of the hospital requesting to be unlearned (e.g. Hospital_B).",
    )
    parser.add_argument(
        "--server",
        default=_default_server,
        metavar="URL",
        help=f"Base URL of the unlearn_server (default: {_default_server}).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        metavar="SECONDS",
        help="Request timeout in seconds (default: 10).",
    )
    return parser


def submit_request(client_id: str, server_url: str, timeout: int) -> int:
    """
    POST the unlearning request to the server.
    Returns the HTTP status code, or -1 on connection error.
    """
    endpoint = f"{server_url.rstrip('/')}/request-unlearning"

    payload = json.dumps({
        "client_id": client_id,
        "request": "unlearn",
    }).encode("utf-8")

    req = urllib.request.Request(
        url=endpoint,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    print(f"\n[REQUEST] Submitting unlearning request for '{client_id}' ...")
    print(f"[REQUEST] Server endpoint : {endpoint}")
    print(f"[REQUEST] Payload         : {{\"client_id\": \"{client_id}\", \"request\": \"unlearn\"}}")

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status = response.status
            raw = response.read()
            body = json.loads(raw) if raw else {}

            print(f"\n[RESPONSE] HTTP {status}")
            print(f"[RESPONSE] Status  : {body.get('status', 'N/A')}")

            if body.get("status") == "PENDING":
                print(f"[RESPONSE] Message : {body.get('message', '')}")
                print(
                    "\n[INFO] Your request is pending admin approval.\n"
                    "[INFO] Check the server terminal for the Accept/Reject prompt.\n"
                    "[INFO] The unlearning will run automatically if accepted."
                )
            else:
                print(f"[RESPONSE] Body    : {json.dumps(body, indent=2)}")

            return status

    except urllib.error.HTTPError as exc:
        # Server responded with an error status (4xx / 5xx)
        status = exc.code
        try:
            raw = exc.read()
            body = json.loads(raw) if raw else {}
        except Exception:
            body = {}

        print(f"\n[ERROR] HTTP {status}")
        reason = body.get("reason") or body.get("error") or exc.reason
        print(f"[ERROR] {reason}")

        # Helpful hints for common errors
        if status == 409:
            print(
                "[HINT] The server rejected the request due to a conflict.\n"
                "[HINT] Possible causes:\n"
                "[HINT]   - Client has already been unlearned\n"
                "[HINT]   - Another request is already pending or running\n"
                "[HINT]   - Unknown client ID"
            )
        return status

    except urllib.error.URLError as exc:
        print(f"\n[ERROR] Could not connect to unlearn_server at {endpoint}")
        print(f"[ERROR] Reason: {exc.reason}")
        print(
            "[HINT] Make sure unlearn_server.py is running:\n"
            "[HINT]   cd src && python unlearn_server.py"
        )
        return -1

    except Exception as exc:
        print(f"\n[ERROR] Unexpected error: {exc}")
        return -1


def main():
    parser = build_parser()
    args = parser.parse_args()

    status = submit_request(
        client_id=args.client_id,
        server_url=args.server,
        timeout=args.timeout,
    )

    # Exit code: 0 = accepted/pending, 1 = error
    sys.exit(0 if status in (200, 201, 202) else 1)


if __name__ == "__main__":
    main()
