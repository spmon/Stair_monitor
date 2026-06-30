from __future__ import annotations

import os

import cv2
from loguru import logger

from stair_monitor.common.types import GStreamerCodec


def is_opencv_gstreamer_available() -> bool:
    build_info = cv2.getBuildInformation()
    for build_line in build_info.splitlines():
        normalized_line = " ".join(build_line.strip().lower().split())
        if normalized_line.startswith("gstreamer:"):
            return "yes" in normalized_line
    return False


def _get_gstreamer_codec_elements(
    codec: GStreamerCodec,
) -> tuple[str, str, str]:
    if codec == "h265":
        return "rtph265depay", "h265parse", "avdec_h265"
    return "rtph264depay", "h264parse", "avdec_h264"


def _escape_gstreamer_value(raw_value: str) -> str:
    return raw_value.replace("\\", "\\\\").replace('"', '\\"')


def build_gstreamer_rtsp_pipeline(
    rtsp_url: str,
    codec: GStreamerCodec,
    latency_ms: int,
) -> str:
    depay, parser, decoder = _get_gstreamer_codec_elements(codec)
    escaped_url = _escape_gstreamer_value(rtsp_url)
    normalized_latency_ms = max(0, latency_ms)
    return (
        f'rtspsrc location="{escaped_url}" protocols=tcp latency={normalized_latency_ms} '
        "drop-on-latency=true "
        f"! {depay} "
        f"! {parser} "
        f"! {decoder} "
        "! videoconvert "
        "! video/x-raw,format=BGR "
        "! appsink max-buffers=1 drop=true sync=false"
    )


def _open_ffmpeg_rtsp_capture(rtsp_url: str) -> cv2.VideoCapture:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    capture = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def open_video_file_capture(input_path: str) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(input_path)
    if not capture.isOpened():
        raise RuntimeError(f"Khong the mo video input: {input_path}")
    return capture


def open_ffmpeg_rtsp_capture(rtsp_url: str) -> cv2.VideoCapture:
    capture = _open_ffmpeg_rtsp_capture(rtsp_url)
    if capture.isOpened():
        return capture

    capture.release()
    raise RuntimeError(
        "Khong the mo RTSP tunnel bang FFmpeg. Kiem tra SSH tunnel localhost:9999."
    )


def try_open_gstreamer_rtsp_capture(
    rtsp_url: str,
    codec: GStreamerCodec,
    latency_ms: int,
) -> cv2.VideoCapture | None:
    if not is_opencv_gstreamer_available():
        logger.warning("OpenCV build does not have GStreamer support.")
        return None

    pipeline = build_gstreamer_rtsp_pipeline(rtsp_url, codec, latency_ms)
    logger.info("Using GStreamer RTSP pipeline with appsink drop=true")
    capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if capture.isOpened():
        return capture

    capture.release()
    logger.warning("GStreamer RTSP open failed, falling back to FFmpeg backend.")
    return None


def open_gstreamer_rtsp_capture(
    rtsp_url: str,
    *,
    codec: GStreamerCodec,
    latency_ms: int,
    fallback_to_ffmpeg_when_gstreamer_fails: bool,
) -> cv2.VideoCapture:
    capture = try_open_gstreamer_rtsp_capture(rtsp_url, codec, latency_ms)
    if capture is not None:
        return capture

    if fallback_to_ffmpeg_when_gstreamer_fails:
        logger.warning("GStreamer unavailable or failed, falling back to FFmpeg")
        return open_ffmpeg_rtsp_capture(rtsp_url)

    raise RuntimeError(
        "Khong the mo RTSP tunnel bang GStreamer. "
        "Kiem tra OpenCV GStreamer support va SSH tunnel localhost:9999."
    )

