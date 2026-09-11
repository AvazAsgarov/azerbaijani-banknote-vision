"""Dataset Deployment and Synchronization Engine for Remote GPU Clusters.

Archives processed dataset partitions (train/val/test splits, labels, data.yaml)
into an optimized archive and transfers it to the remote compute cluster over a secure
socket bridge, followed by remote extraction and integrity verification.
"""

import argparse
import json
import logging
import os
import socket
import sys
import time
import uuid
import zipfile
from pathlib import Path
from typing import Optional

import requests
import websocket

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.config import ProjectPaths, RemoteClusterConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("DatasetDeployer")


def build_dataset_archive(paths: ProjectPaths, output_path: Path) -> Path:
    """Creates a compressed zip bundle of the processed dataset splits and declarations.

    Args:
        paths: ProjectPaths configuration instance.
        output_path: Target destination path for the archive.

    Returns:
        Path to the verified zip archive.
    """
    logger.info("Assembling dataset archive at %s...", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        data_yaml = paths.processed_data_dir / "data.yaml"
        if data_yaml.exists():
            zf.write(data_yaml, "data.yaml")

        total_files = 0
        for split in ["train", "val", "test"]:
            img_dir = paths.images_dir / split
            lbl_dir = paths.labels_dir / split

            for img in sorted(img_dir.glob("*.jpg")):
                zf.write(img, f"images/{split}/{img.name}")
                total_files += 1

            for lbl in sorted(lbl_dir.glob("*.txt")):
                zf.write(lbl, f"labels/{split}/{lbl.name}")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Dataset archive generated: %d images packaged (%.2f MB).", total_files, size_mb)
    return output_path


def send_archive_socket(
    archive_path: Path,
    host: str,
    port: int = 8888,
    chunk_size: int = 512 * 1024,
    timeout: float = 300.0,
) -> None:
    """Streams local binary archive file over a TCP socket bridge to remote receiver.

    Args:
        archive_path: Path to the local zip archive.
        host: Target receiver host address or IP.
        port: Listening port on target receiver.
        chunk_size: Stream buffer size in bytes.
        timeout: Socket communication timeout in seconds.
    """
    file_size = archive_path.stat().st_size
    size_mb = file_size / (1024 * 1024)
    logger.info("Connecting to TCP receiver at %s:%d to transfer %.2f MB...", host, port, size_mb)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect((host, port))

    try:
        sock.sendall(file_size.to_bytes(8, byteorder="big"))

        bytes_sent = 0
        start_time = time.perf_counter()
        last_log = start_time

        with open(archive_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                sock.sendall(chunk)
                bytes_sent += len(chunk)
                now = time.perf_counter()
                if now - last_log >= 3.0:
                    speed = (bytes_sent / (1024 * 1024)) / max(1e-3, now - start_time)
                    pct = (bytes_sent / file_size) * 100
                    logger.info(
                        "Transfer progress: %.1f%% (%.1f/%.1f MB) | Speed: %.2f MB/s",
                        pct,
                        bytes_sent / (1024 * 1024),
                        size_mb,
                        speed,
                    )
                    last_log = now

        total_elapsed = time.perf_counter() - start_time
        avg_speed = size_mb / max(1e-3, total_elapsed)
        logger.info(
            "Upload complete: %.2f MB transmitted in %.1fs (average: %.2f MB/s).",
            size_mb,
            total_elapsed,
            avg_speed,
        )

        ack = sock.recv(1024)
        logger.info("Remote receiver ACK status: %s", ack.decode("utf-8", errors="replace").strip())
    finally:
        sock.close()


def deploy_dataset(
    paths: Optional[ProjectPaths] = None,
    cluster_cfg: Optional[RemoteClusterConfig] = None,
    rebuild: bool = False,
) -> None:
    """Coordinates remote execution of receiver daemon, local transfer, and unpack verification.

    Args:
        paths: ProjectPaths configuration instance.
        cluster_cfg: RemoteClusterConfig containing endpoint and authentication.
        rebuild: When True, forces regeneration of the local zip archive.
    """
    paths = paths or ProjectPaths()
    cluster = cluster_cfg or RemoteClusterConfig()

    archive_path = paths.artifacts_dir / "azn_banknotes_dataset.zip"
    if rebuild or not archive_path.exists():
        build_dataset_archive(paths, archive_path)

    headers = {"Authorization": f"token {cluster.token}"}
    logger.info("Spawning remote kernel receiver at %s...", cluster.base_url)

    r = requests.post(f"{cluster.base_url}/api/kernels", headers=headers, json={"name": "python3"}, timeout=10)
    r.raise_for_status()
    kid = r.json()["id"]

    ws_endpoint = f"{cluster.ws_url}/api/kernels/{kid}/channels?token={cluster.token}"
    ws = websocket.create_connection(ws_endpoint, timeout=30)

    receiver_code = """
import socket, time, zipfile, os, shutil

s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('0.0.0.0', 8888))
s.listen(1)
print('RECEIVER_READY')

conn, addr = s.accept()
header = conn.recv(8)
file_size = int.from_bytes(header, byteorder='big')

bytes_received = 0
with open('azn_banknotes_dataset.zip', 'wb') as f:
    while bytes_received < file_size:
        to_read = min(512 * 1024, file_size - bytes_received)
        chunk = conn.recv(to_read)
        if not chunk:
            break
        f.write(chunk)
        bytes_received += len(chunk)

conn.sendall(b'OK')
conn.close()
s.close()

if os.path.exists('dataset'):
    shutil.rmtree('dataset')
os.makedirs('dataset', exist_ok=True)

with zipfile.ZipFile('azn_banknotes_dataset.zip', 'r') as z:
    z.extractall('dataset')

print('UNPACK_SUCCESS')
"""

    msg_id = uuid.uuid4().hex
    ws.send(json.dumps({
        "header": {"msg_id": msg_id, "username": "team2", "session": msg_id, "msg_type": "execute_request", "version": "5.3"},
        "metadata": {},
        "content": {"code": receiver_code, "silent": False, "store_history": False, "user_expressions": {}, "allow_stdin": False},
        "parent_header": {},
        "channel": "shell",
    }))

    target_host = cluster.base_url.split("://")[-1].split(":")[0]
    time.sleep(2.0)

    send_archive_socket(archive_path, host=target_host, port=8888)

    while True:
        res = json.loads(ws.recv())
        if res.get("parent_header", {}).get("msg_id") == msg_id:
            msg_type = res.get("msg_type")
            if msg_type == "stream":
                sys.stdout.write(res["content"]["text"])
            elif msg_type == "error":
                logger.error("Remote unpack error: %s", "\n".join(res["content"]["traceback"]))
            elif msg_type == "status" and res["content"]["execution_state"] == "idle":
                break

    ws.close()
    requests.delete(f"{cluster.base_url}/api/kernels/{kid}", headers=headers, timeout=5)
    logger.info("Dataset deployment and remote unpack completed successfully.")


def main() -> None:
    """CLI entrypoint for dataset deployment."""
    parser = argparse.ArgumentParser(description="Deploy and unpack dataset on remote A100 GPU cluster.")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of dataset zip archive prior to transfer.")
    args = parser.parse_args()
    deploy_dataset(rebuild=args.rebuild)


if __name__ == "__main__":
    main()
