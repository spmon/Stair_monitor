from __future__ import annotations

from stair_monitor.output.alert_event_logger import AlertEventLogger
from stair_monitor.runtime.display_sink import AlertOnlyDisplaySink
from stair_monitor.runtime.runtime_types import OutputFrameContext, ViolationEventSink
from stair_monitor.runtime.violation_event_factory import build_violation_events
from stair_monitor.runtime.video_writer_sink import VideoWriterSink


class RuntimeOutputCoordinator:
    """Coordinate display, video writing, and violation logging for one frame."""

    def __init__(
        self,
        display_sink: AlertOnlyDisplaySink,
        video_writer_sink: VideoWriterSink,
        event_sinks: tuple[ViolationEventSink, ...],
        alert_logger: AlertEventLogger | None,
    ) -> None:
        self.display_sink = display_sink
        self.video_writer_sink = video_writer_sink
        self.event_sinks = event_sinks
        self.alert_logger = alert_logger

    def handle_frame(self, frame_context: OutputFrameContext) -> bool:
        events = build_violation_events(frame_context)
        for event_sink in self.event_sinks:
            event_sink.log_events(events, frame_context.frame_index)

        self.video_writer_sink.write(frame_context.overlay_frame)
        return self.display_sink.render(
            frame_context.overlay_frame,
            frame_context.active_alert_until_frame,
            frame_context.person_results,
        )

    def close(self) -> None:
        self.video_writer_sink.close()
        for event_sink in self.event_sinks:
            event_sink.close()
        self.display_sink.close()
