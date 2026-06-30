from __future__ import annotations

from stair_monitor.config.settings import VideoConfig
from stair_monitor.runtime.frame_source import (
    FrameSource,
    LatestFrameSource,
    RtspFfmpegFrameSource,
    RtspGStreamerFrameSource,
    VideoFileFrameSource,
    resolve_input_path,
)


def create_frame_source(video_settings: VideoConfig) -> FrameSource:
    if video_settings.input_mode == "file":
        return VideoFileFrameSource(resolve_input_path(video_settings.input_path))

    rtsp_url = resolve_input_path(video_settings.rtsp_url)
    if video_settings.video_capture_backend == "gstreamer":
        frame_source: FrameSource = RtspGStreamerFrameSource(
            rtsp_url,
            codec=video_settings.gstreamer_codec,
            latency_ms=video_settings.gstreamer_rtsp_latency_ms,
            fallback_to_ffmpeg_when_gstreamer_fails=(
                video_settings.fallback_to_ffmpeg_when_gstreamer_fails
            ),
        )
    else:
        frame_source = RtspFfmpegFrameSource(rtsp_url)

    if video_settings.use_latest_frame_for_live:
        return LatestFrameSource(frame_source)
    return frame_source
