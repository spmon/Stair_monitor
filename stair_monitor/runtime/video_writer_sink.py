from __future__ import annotations

import cv2

from stair_monitor.runtime.runtime_types import Frame


class VideoWriterSink:
    """Lazy output writer for the Windows/demo runtime."""

    def __init__(
        self,
        output_path: str,
        output_fps: float,
        enabled: bool,
    ) -> None:
        self.output_path = output_path
        self.output_fps = output_fps
        self.enabled = enabled
        self._writer: cv2.VideoWriter | None = None

    def write(self, frame: Frame) -> None:
        if not self.enabled:
            return

        if self._writer is None:
            frame_h, frame_w = frame.shape[:2]
            self._writer = cv2.VideoWriter(
                self.output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.output_fps,
                (frame_w, frame_h),
            )

        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
