"""MJPEG frame streamer thread for ESP32 Wi-Fi camera hardware bridge."""
import threading, time, urllib.request
from typing import Optional

class MJPEGStreamer:
    def __init__(self, url, poll_interval_s=0.05, timeout_s=1.5):
        self.url = url; self.poll_interval = poll_interval_s; self.timeout = timeout_s
        self._thread = None; self._stop_event = threading.Event()
        self._lock = threading.Lock(); self._latest = None; self._frame_count = 0

    @property
    def latest_frame(self):
        with self._lock: return self._latest

    @property
    def frame_count(self): return self._frame_count

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread: self._thread.join(timeout=3.0)

    def _poll_loop(self):
        while not self._stop_event.is_set():
            try:
                with urllib.request.urlopen(self.url, timeout=self.timeout) as resp:
                    with self._lock: self._latest = resp.read()
                self._frame_count += 1
            except Exception: pass
            time.sleep(self.poll_interval)

    def is_running(self): return self._thread is not None and self._thread.is_alive()
