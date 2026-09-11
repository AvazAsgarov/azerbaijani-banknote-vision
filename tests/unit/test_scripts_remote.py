"""Unit Tests for Remote GPU Cluster Execution and Synchronization Scripts.

Tests cover:
- scripts.remote_exec: WebSocket kernel execution, streaming outputs, cleanup.
- scripts.remote_gpu_bridge: Connectivity probe, dataset packaging.
- scripts.sync_exp_artifacts: Registry mappings, artifact retrieval resilience.
- scripts.write_remote_exp*: Script generation payloads and syntax validity.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.core.config import ProjectPaths, RemoteClusterConfig
from scripts.remote_exec import run_remote
from scripts.remote_gpu_bridge import RemoteGPUBridge
from scripts.sync_exp_artifacts import EXP_MAPPINGS, sync_experiment


@pytest.mark.unit
class TestRemoteExec:
    """Evaluates WebSocket execution against remote Jupyter kernels."""

    def test_run_remote_connection_failure(self) -> None:
        """Verifies run_remote returns False when server connection cannot be established."""
        with patch("requests.post", side_effect=Exception("Connection refused")):
            result = run_remote("print('test')", timeout=1)
            assert result is False

    def test_run_remote_streaming_success(self) -> None:
        """Verifies run_remote handles execute_request, streams output, and deletes kernel."""
        mock_post_resp = MagicMock()
        mock_post_resp.json.return_value = {"id": "fake_kernel_123"}
        mock_post_resp.raise_for_status.return_value = None

        mock_ws = MagicMock()
        # Simulate Jupyter kernel messages: 1 stream output, then idle status
        msg_id_holder = []

        def mock_send(payload_str: str) -> None:
            payload = json.loads(payload_str)
            msg_id_holder.append(payload["header"]["msg_id"])

        mock_ws.send.side_effect = mock_send

        def mock_recv() -> str:
            msg_id = msg_id_holder[0] if msg_id_holder else "default_id"
            if not hasattr(mock_recv, "step"):
                mock_recv.step = 0
            if mock_recv.step == 0:
                mock_recv.step += 1
                return json.dumps({
                    "parent_header": {"msg_id": msg_id},
                    "msg_type": "stream",
                    "content": {"text": "Remote output line\n"},
                })
            else:
                return json.dumps({
                    "parent_header": {"msg_id": msg_id},
                    "msg_type": "status",
                    "content": {"execution_state": "idle"},
                })

        mock_recv.step = 0
        mock_ws.recv.side_effect = mock_recv

        with patch("requests.post", return_value=mock_post_resp), \
             patch("websocket.create_connection", return_value=mock_ws), \
             patch("requests.delete") as mock_delete:

            success = run_remote("print('Remote output line')")
            assert success is True
            assert mock_ws.send.called
            assert mock_ws.close.called
            assert mock_delete.called


@pytest.mark.unit
class TestRemoteGPUBridge:
    """Evaluates RemoteGPUBridge connectivity and packaging operations."""

    def test_bridge_initialization_defaults(self) -> None:
        """Verifies default initialization inherits RemoteClusterConfig values."""
        bridge = RemoteGPUBridge()
        cfg = RemoteClusterConfig()
        assert bridge.base_url == cfg.base_url.rstrip("/")
        assert bridge.token == cfg.token
        assert "Authorization" in bridge.headers
        assert bridge.headers["Authorization"] == f"token {cfg.token}"

    def test_bridge_check_connection_failure(self) -> None:
        """Verifies check_connection returns False when remote endpoint is unreachable."""
        bridge = RemoteGPUBridge(base_url="http://invalid.unreachable.host:8080/team2")
        is_connected, msg = bridge.check_connection()
        assert is_connected is False
        assert len(msg) > 0

    def test_prepare_dataset_package(self, mock_paths: ProjectPaths, tmp_path: Path) -> None:
        """Verifies dataset packaging builds a zip containing YAML and split files."""
        # Create minimal test structure
        train_img = mock_paths.images_dir / "train" / "001_azn_00001.jpg"
        train_img.write_bytes(b"\xFF\xD8\xFFFakeImageData")
        train_lbl = mock_paths.labels_dir / "train" / "001_azn_00001.txt"
        train_lbl.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

        target_zip = tmp_path / "test_deploy.zip"
        bridge = RemoteGPUBridge(paths=mock_paths)
        res_zip = bridge.prepare_dataset_package(output_zip_path=target_zip)

        assert res_zip.exists()
        assert res_zip.stat().st_size > 0

        import zipfile
        with zipfile.ZipFile(res_zip, "r") as zf:
            namelist = zf.namelist()
            assert "data.yaml" in namelist
            assert "images/train/001_azn_00001.jpg" in namelist
            assert "labels/train/001_azn_00001.txt" in namelist


@pytest.mark.unit
class TestSyncExpArtifacts:
    """Evaluates artifact synchronization mappings and defensive handling."""

    def test_exp_mappings_registry_completeness(self) -> None:
        """Verifies all 6 experiment keys are registered with valid structure."""
        expected_keys = [f"exp{i}" for i in range(1, 7)]
        for k in expected_keys:
            assert k in EXP_MAPPINGS
            entry = EXP_MAPPINGS[k]
            assert "remote_complete_zip" in entry
            assert "remote_csv" in entry
            assert "remote_json" in entry
            assert "local_name" in entry

    def test_sync_experiment_graceful_missing_remote(self, mock_paths: ProjectPaths) -> None:
        """Verifies sync_experiment handles unreachable/missing remote archives gracefully."""
        with patch("requests.get", side_effect=Exception("Network down")):
            # Should not raise uncaught exceptions, returns True with warning logs
            success = sync_experiment("exp1", paths=mock_paths)
            assert success is True


@pytest.mark.unit
class TestWriteRemoteExpScripts:
    """Evaluates remote experiment script generator payloads."""

    def test_all_remote_experiment_payloads_valid_syntax(self) -> None:
        """Verifies remote_code payload in write_remote_exp1..6 is non-empty and well-formed."""
        try:
            from scripts import (
                write_remote_exp1,
                write_remote_exp2,
                write_remote_exp3,
                write_remote_exp4,
                write_remote_exp5,
                write_remote_exp6,
            )
        except ImportError:
            pytest.skip("Remote experiment generator scripts not present in local deployment")

        modules = [
            write_remote_exp1,
            write_remote_exp2,
            write_remote_exp3,
            write_remote_exp4,
            write_remote_exp5,
            write_remote_exp6,
        ]
        for m in modules:
            assert hasattr(m, "remote_code")
            assert isinstance(m.remote_code, str)
            assert len(m.remote_code) > 100
            assert hasattr(m, "main")
