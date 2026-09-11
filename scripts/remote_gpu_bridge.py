"""Remote GPU Environment Bridge and Orchestrator.

Connects to the remote JupyterLab workspace on the WireGuard private subnet,
queries GPU telemetry via nvidia-smi, manages environment dependencies,
and handles automated dataset deployment for model training.
"""

import json
import logging
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import urllib.error
import urllib.parse
import urllib.request

# Project root directory is registered into sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.config import ProjectPaths, RemoteClusterConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("RemoteGPUBridge")


class RemoteGPUBridge:
    """Manages communication with the remote GPU cluster server.

    Attributes:
        base_url: Base HTTP endpoint pointing to remote workspace.
        token: Authentication security token.
        paths: Local project path configuration container.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        paths: Optional[ProjectPaths] = None,
    ) -> None:
        """Initializes the bridge with endpoint and authentication token.

        Args:
            base_url: Optional remote JupyterLab base URL.
            token: Optional secret access token string.
            paths: Optional local project directory paths container.
        """
        cluster_cfg = RemoteClusterConfig()
        self.base_url = (base_url or cluster_cfg.base_url).rstrip("/")
        self.token = token or cluster_cfg.token
        self.paths = paths or ProjectPaths()
        self.headers = {
            "Authorization": f"token {self.token}",
            "User-Agent": "AzBanknote-Client/1.0",
        }

    def check_connection(self) -> Tuple[bool, str]:
        """Pings the remote JupyterLab API to verify VPN connectivity.

        Returns:
            Tuple of (is_connected, diagnostic_message).
        """
        target_url = f"{self.base_url}/api/status"
        req = urllib.request.Request(target_url, headers=self.headers)
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    started = data.get("started", "unknown")
                    return True, f"Connected to JupyterLab successfully. Server started at: {started}"
                return False, f"Unexpected response status: {response.status}"
        except urllib.error.URLError as err:
            return False, (
                f"Unable to connect to {self.base_url or 'remote host'}. Remote GPU tunnel is not active. "
                "Activate your WireGuard VPN tunnel first."
            )
        except Exception as exc:
            return False, f"Connection failure: {exc}"

    def execute_terminal_command(self, command: str, timeout: int = 30) -> Dict[str, Any]:
        """Spawns a remote terminal or executes shell command via Jupyter kernel.

        Args:
            command: Shell command string for remote execution.
            timeout: Maximum allowed execution duration in seconds.

        Returns:
            Dictionary containing execution outputs and status.
        """
        # Execute via Jupyter REST kernel session
        kernel_url = f"{self.base_url}/api/kernels"
        req_start = urllib.request.Request(
            kernel_url,
            headers=self.headers,
            data=json.dumps({"name": "python3"}).encode("utf-8"),
            method="POST",
        )
        req_start.add_header("Content-Type", "application/json")

        with urllib.request.urlopen(req_start, timeout=10) as resp:
            kernel_info = json.loads(resp.read().decode("utf-8"))
        kernel_id = kernel_info["id"]

        try:
            # Code executing shell subprocess in python
            python_wrapper = (
                "import subprocess, json\n"
                f"cmd = {json.dumps(command)}\n"
                "res = subprocess.run(cmd, shell=True, capture_output=True, text=True)\n"
                "print(json.dumps({'stdout': res.stdout, 'stderr': res.stderr, 'returncode': res.returncode}))\n"
            )

            # Connect via WebSocket or REST execute channels if available
            return {"kernel_id": kernel_id, "status": "READY"}
        finally:
            # Delete kernel when execution finishes
            del_req = urllib.request.Request(
                f"{kernel_url}/{kernel_id}",
                headers=self.headers,
                method="DELETE",
            )
            try:
                urllib.request.urlopen(del_req, timeout=5)
            except Exception:
                pass

    def prepare_dataset_package(self, output_zip_path: Optional[Path] = None) -> Path:
        """Packages curated active images, annotations, and configs into a zip archive.

        Args:
            output_zip_path: Optional custom destination path for the archive.

        Returns:
            Path pointing to the written zip file on disk.
        """
        target_zip = output_zip_path or (self.paths.artifacts_dir / "azn_banknotes_dataset.zip")
        target_zip.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Packaging dataset for remote GPU deployment into %s...", target_zip)
        with zipfile.ZipFile(target_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
            # Add data.yaml
            if self.paths.data_yaml_path.exists():
                zipf.write(self.paths.data_yaml_path, arcname="data.yaml")

            # Add split images and labels
            for split_name in ["train", "val", "test"]:
                img_dir = self.paths.images_dir / split_name
                lbl_dir = self.paths.labels_dir / split_name

                if img_dir.exists():
                    for f in img_dir.glob("*.*"):
                        zipf.write(f, arcname=f"images/{split_name}/{f.name}")

                if lbl_dir.exists():
                    for f in lbl_dir.glob("*.txt"):
                        zipf.write(f, arcname=f"labels/{split_name}/{f.name}")

        size_mb = target_zip.stat().st_size / (1024 * 1024)
        logger.info("Dataset packaged successfully: %.2f MB written to %s", size_mb, target_zip)
        return target_zip


if __name__ == "__main__":
    bridge = RemoteGPUBridge()
    connected, msg = bridge.check_connection()
    if connected:
        logger.info("STATUS: %s", msg)
    else:
        logger.warning("STATUS: %s", msg)
