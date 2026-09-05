"""Unit tests for MJPEG frame streamer."""
import time
import pytest
from src.hardware.mjpeg_streamer import MJPEGStreamer

class TestMJPEGStreamer:
    def test_initial_no_frame(self):
        s = MJPEGStreamer("http://192.168.1.99/capture")
        assert s.latest_frame is None
    def test_not_running_before_start(self):
        assert not MJPEGStreamer("http://192.168.1.99/capture").is_running()
    def test_start_creates_daemon(self):
        s = MJPEGStreamer("http://192.168.1.99/capture", poll_interval_s=100)
        s.start()
        try:
            assert s.is_running() and s._thread.daemon
        finally:
            s.stop()
    def test_stop_terminates(self):
        s = MJPEGStreamer("http://192.168.1.99/capture", poll_interval_s=100)
        s.start(); s.stop(); time.sleep(0.1)
        assert not s.is_running()
    def test_frame_injection(self):
        s = MJPEGStreamer("http://192.168.1.99/capture", poll_interval_s=100)
        with s._lock: s._latest = b"FAKE_JPEG"
        assert s.latest_frame == b"FAKE_JPEG"
