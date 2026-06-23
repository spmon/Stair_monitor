from __future__ import annotations

import os
import time
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

Frame: TypeAlias = NDArray[np.uint8]


@dataclass(frozen=True)
class LatestFrame:
    frame: Frame
    sequence_id: int
    captured_at_monotonic: float


class LatestFrameCapture:
    def __init__(self, source_url: str) -> None:
        self.source_url = source_url
        self._cap = self._open_capture(source_url)
        self._lock = Lock()
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._latest_frame: Frame | None = None
        self._latest_sequence_id = -1
        self._latest_captured_at_monotonic = 0.0

    def _open_capture(self, source_url: str) -> cv2.VideoCapture:
        if source_url.lower().startswith("rtsp://"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            cap = cv2.VideoCapture(source_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = cv2.VideoCapture(source_url)

        if not cap.isOpened():
            if source_url.lower().startswith("rtsp://"):
                raise RuntimeError(
                    "Khong the mo RTSP tunnel. Kiem tra SSH tunnel localhost:9999."
                )
            raise RuntimeError(f"Khong the mo video input: {source_url}")

        return cap

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            copied_frame = frame.copy()
            captured_at_monotonic = time.monotonic()

            with self._lock:
                self._latest_frame = copied_frame
                self._latest_sequence_id += 1
                self._latest_captured_at_monotonic = captured_at_monotonic

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = Thread(
            target=self._capture_loop,
            name="LatestFrameCapture",
            daemon=True,
        )
        self._thread.start()

    def read_latest(self) -> LatestFrame | None:
        with self._lock:
            if self._latest_frame is None:
                return None

            return LatestFrame(
                frame=self._latest_frame.copy(),
                sequence_id=self._latest_sequence_id,
                captured_at_monotonic=self._latest_captured_at_monotonic,
            )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def release(self) -> None:
        self._cap.release()

    def get_source_fps(self) -> float:
        return float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
