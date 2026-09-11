"""Remote Execution Utility for Azerbaijani Banknote Experiments.

Executes Python code directly against the remote Jupyter kernel over WebSockets,
streaming standard output and error in real-time.
"""

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Optional

import requests
import websocket

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.config import RemoteClusterConfig

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("RemoteExec")


def run_remote(
    code_str: str,
    timeout: int = 120,
    config: Optional[RemoteClusterConfig] = None,
) -> bool:
    """Sends a code snippet to the remote Jupyter server and streams the execution output.

    Args:
        code_str: Python code string to be executed on the remote kernel.
        timeout: Network timeout in seconds for kernel creation and connection.
        config: Optional RemoteClusterConfig instance. Defaults to standard configuration.

    Returns:
        True if the remote code ran to completion, False otherwise.
    """
    cfg = config or RemoteClusterConfig()
    base_url = cfg.base_url.rstrip("/")
    headers = cfg.get_auth_headers()

    try:
        resp = requests.post(
            f"{base_url}/api/kernels",
            headers=headers,
            json={"name": "python3"},
            timeout=timeout,
        )
        resp.raise_for_status()
        kernel_id = resp.json()["id"]
        ws_url = cfg.get_websocket_url(kernel_id)
        ws = websocket.create_connection(ws_url, timeout=timeout)
    except Exception as exc:
        logger.error("Failed to establish remote connection: %s", exc)
        return False

    msg_id = uuid.uuid4().hex
    payload = {
        "header": {
            "msg_id": msg_id,
            "username": "client",
            "session": msg_id,
            "msg_type": "execute_request",
            "version": "5.3",
        },
        "metadata": {},
        "content": {
            "code": code_str,
            "silent": False,
            "store_history": True,
            "user_expressions": {},
            "allow_stdin": False,
        },
        "parent_header": {},
        "channel": "shell",
    }
    ws.send(json.dumps(payload))

    try:
        while True:
            raw = ws.recv()
            msg = json.loads(raw)
            if msg.get("parent_header", {}).get("msg_id") == msg_id:
                mtype = msg.get("msg_type")
                if mtype == "stream":
                    sys.stdout.write(msg["content"]["text"])
                    sys.stdout.flush()
                elif mtype == "error":
                    sys.stderr.write("\n".join(msg["content"]["traceback"]) + "\n")
                    sys.stderr.flush()
                elif mtype == "status" and msg["content"]["execution_state"] == "idle":
                    break
    except Exception as exc:
        logger.error("Error during streaming execution: %s", exc)
        return False
    finally:
        ws.close()
        try:
            requests.delete(f"{base_url}/api/kernels/{kernel_id}", headers=headers, timeout=10)
        except Exception:
            pass

    return True


def main() -> None:
    """Command-line entrypoint for remote execution."""
    parser = argparse.ArgumentParser(
        description="Execute Python code on remote GPU cluster kernel"
    )
    parser.add_argument(
        "code",
        nargs="?",
        help="Python code string to execute",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=Path,
        help="Path to Python file containing code to execute",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Execution timeout in seconds (default: 120)",
    )
    args = parser.parse_args()

    if args.file:
        code_to_run = args.file.read_text(encoding="utf-8")
    elif args.code:
        code_to_run = args.code
    else:
        code_to_run = sys.stdin.read()

    success = run_remote(code_to_run, timeout=args.timeout)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
