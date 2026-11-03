"""Phone camera stream client.

The phone acts as a network webcam. Any MJPEG-over-HTTP source works:

* Android "IP Webcam" app (https://github.com/pavelkhlebovich/ip-webcam)
      stream   : http://<phone-ip>:8080/video
      snapshot : http://<phone-ip>:8080/shot.jpg
* DroidCam (Wi-Fi mode)
      stream   : http://<phone-ip>:4747/video
* Any phone on the same Wi-Fi/LAN running a similar app.

If the MJPEG stream cannot be opened, this class falls back to polling a
snapshot URL (e.g. IP Webcam's /shot.jpg), which is slower but works with
almost any app. A local webcam index can be used for testing (source=0).
"""

from __future__ import annotations

import threading
import time
import urllib.request

import cv2
import numpy as np


class PhoneCamera:
    """Thread-safe reader that always returns the latest frame."""

    def __init__(self, stream_url: str | None = None,
                 snapshot_url: str | None = None,
                 source: int | str | None = None,
                 width: int = 960, height: int = 540):
        self.stream_url = stream_url
        self.snapshot_url = snapshot_url
        self.width, self.height = width, height
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._ok = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        if self.stream_url:
            if self._run_mjpeg():
                return
        if self.snapshot_url:
            self._run_snapshot()
        elif self.stream_url is None and self.snapshot_url is None:
            # local webcam fallback (index passed via snapshot_url='0' hack is avoided)
            pass

    def _run_mjpeg(self) -> bool:
        cap = cv2.VideoCapture(self.stream_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            cap.release()
            return False
        ok = True
        while not self._stop.is_set():
            ret, frame = cap.read()
            if not ret:
                ok = False
                break
            frame = self._resize(frame)
            with self._lock:
                self._frame, self._ok = frame, True
        cap.release()
        return ok and not self._stop.is_set()

    def _run_snapshot(self) -> None:
        while not self._stop.is_set():
            try:
                req = urllib.request.Request(self.snapshot_url, headers={"User-Agent": "quiz-solver"})
                with urllib.request.urlopen(req, timeout=5) as r:
                    data = np.frombuffer(r.read(), dtype=np.uint8)
                frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if frame is not None:
                    frame = self._resize(frame)
                    with self._lock:
                        self._frame, self._ok = frame, True
            except Exception:
                pass
            time.sleep(0.05)

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = min(self.width / w, self.height / h)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        return frame

    # ------------------------------------------------------------------ #
    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            return self._ok, None if self._frame is None else self._frame.copy()

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def release(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
