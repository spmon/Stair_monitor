from __future__ import annotations

import time
from abc import ABC, abstractmethod
from threading import Condition, Thread
from typing import Protocol

import cv2
from loguru import logger

from stair_monitor.common.types import GStreamerCodec
from stair_monitor.input.video_capture import (
    open_ffmpeg_rtsp_capture,
    open_gstreamer_rtsp_capture,
    open_video_file_capture,
)
from stair_monitor.runtime.runtime_types import FramePacket

RTSP_TUNNEL_URL = (
    "rtsp://admin:vna%40123456@localhost:9999/"
    "cam/realmonitor?channel=1&subtype=0"
)


class FrameSource(Protocol):
    def open(self) -> None: ...

    def read(self) -> FramePacket | None: ...

    def source_fps(self) -> float: ...

    def release(self) -> None: ...


def resolve_input_path(input_path: str) -> str:
    normalized_input_path = input_path.strip()
    if normalized_input_path == "RTSP_TUNNEL":
        return RTSP_TUNNEL_URL
    return normalized_input_path


def resolve_output_fps(source_fps: float) -> float:
    return source_fps if source_fps > 0.0 else 15.0


class OpenCvFrameSource(ABC):
    """Base class for sequential cv2-backed frame sources."""

    def __init__(
        self,
        *,
        source_name: str,
        max_failed_reads: int,
    ) -> None:
        self._source_name = source_name
        self._max_failed_reads = max(1, max_failed_reads)
        self._capture: cv2.VideoCapture | None = None
        self._is_open = False
        self._frame_index = -1
        self._source_fps = 0.0
        self._failed_reads = 0

    @abstractmethod
    def _open_capture(self) -> cv2.VideoCapture:
        raise NotImplementedError

    def open(self) -> None:
        if self._is_open:
            return

        self._capture = self._open_capture()
        self._source_fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self._is_open = True

    def read(self) -> FramePacket | None:
        if not self._is_open:
            self.open()

        if self._capture is None:
            return None

        while self._failed_reads < self._max_failed_reads:
            ret, frame = self._capture.read()
            if ret:
                self._failed_reads = 0
                self._frame_index += 1
                return FramePacket(
                    frame=frame,
                    frame_index=self._frame_index,
                    source_fps=self._source_fps,
                    captured_at_monotonic=time.monotonic(),
                    source_name=self._source_name,
                )

            self._failed_reads += 1
            logger.warning(
                "Khong doc duoc frame tu source {} ({}/{})",
                self._source_name,
                self._failed_reads,
                self._max_failed_reads,
            )
            self._handle_failed_read_delay()

        return None

    def _handle_failed_read_delay(self) -> None:
        return None

    def source_fps(self) -> float:
        if not self._is_open:
            self.open()
        return self._source_fps

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._is_open = False


class VideoFileFrameSource(OpenCvFrameSource):
    """Sequential offline video file source without frame dropping."""

    def __init__(self, input_path: str) -> None:
        self.input_path = input_path
        super().__init__(source_name="video_file", max_failed_reads=1)

    def _open_capture(self) -> cv2.VideoCapture:
        return open_video_file_capture(self.input_path)


class RtspFfmpegFrameSource(OpenCvFrameSource):
    """RTSP source opened through OpenCV FFmpeg TCP transport."""

    def __init__(self, rtsp_url: str) -> None:
        self.rtsp_url = rtsp_url
        super().__init__(source_name="rtsp_ffmpeg", max_failed_reads=5)

    def _open_capture(self) -> cv2.VideoCapture:
        return open_ffmpeg_rtsp_capture(self.rtsp_url)

    def _handle_failed_read_delay(self) -> None:
        time.sleep(0.01)


class RtspGStreamerFrameSource(OpenCvFrameSource):
    """RTSP source opened through OpenCV GStreamer, with optional FFmpeg fallback."""

    def __init__(
        self,
        rtsp_url: str,
        *,
        codec: GStreamerCodec,
        latency_ms: int,
        fallback_to_ffmpeg_when_gstreamer_fails: bool,
    ) -> None:
        self.rtsp_url = rtsp_url
        self.codec = codec
        self.latency_ms = latency_ms
        self.fallback_to_ffmpeg_when_gstreamer_fails = (
            fallback_to_ffmpeg_when_gstreamer_fails
        )
        super().__init__(source_name="rtsp_gstreamer", max_failed_reads=5)

    def _open_capture(self) -> cv2.VideoCapture:
        return open_gstreamer_rtsp_capture(
            self.rtsp_url,
            codec=self.codec,
            latency_ms=self.latency_ms,
            fallback_to_ffmpeg_when_gstreamer_fails=(
                self.fallback_to_ffmpeg_when_gstreamer_fails
            ),
        )

    def _handle_failed_read_delay(self) -> None:
        time.sleep(0.01)


class LatestFrameSource:
    """Wrap a live source and expose only the newest frame to the runtime."""

    def __init__(self, inner_source: FrameSource) -> None:
        self._inner_source = inner_source
        self._condition = Condition()
        self._thread: Thread | None = None
        self._latest_packet: FramePacket | None = None
        self._last_delivered_frame_index = -1
        self._closed = False
        self._exhausted = False

    def open(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self._inner_source.open()
        self._closed = False
        self._thread = Thread(
            target=self._capture_loop,
            name="LatestFrameSource",
            daemon=True,
        )
        self._thread.start()

    def _capture_loop(self) -> None:
        while not self._closed:
            packet = self._inner_source.read()
            if packet is None:
                with self._condition:
                    self._exhausted = True
                    self._condition.notify_all()
                return

            with self._condition:
                self._latest_packet = packet
                self._condition.notify_all()

    def read(self) -> FramePacket | None:
        if self._thread is None:
            self.open()

        with self._condition:
            while True:
                if self._latest_packet is not None:
                    if (
                        self._latest_packet.frame_index
                        != self._last_delivered_frame_index
                    ):
                        self._last_delivered_frame_index = (
                            self._latest_packet.frame_index
                        )
                        return self._latest_packet
                if self._exhausted:
                    return None
                self._condition.wait(timeout=0.05)

    def source_fps(self) -> float:
        return self._inner_source.source_fps()

    def release(self) -> None:
        self._closed = True
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._inner_source.release()
