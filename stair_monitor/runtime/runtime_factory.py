from __future__ import annotations

from stair_monitor.common.types import CameraConfigDict
from stair_monitor.config.settings import AppSettings
from stair_monitor.core.analyzer import BehaviorAnalyzer
from stair_monitor.output.alert_event_logger import AlertEventLogger
from stair_monitor.runtime.analysis_pipeline import StairAnalysisPipeline
from stair_monitor.runtime.display_sink import AlertOnlyDisplaySink
from stair_monitor.runtime.frame_source import resolve_output_fps
from stair_monitor.runtime.output_coordinator import RuntimeOutputCoordinator
from stair_monitor.runtime.pose_estimator import YoloPoseEstimator
from stair_monitor.runtime.runtime_app import RuntimeAppSettings, StairMonitorRuntimeApp
from stair_monitor.runtime.runtime_types import ViolationEventSink
from stair_monitor.runtime.source_factory import create_frame_source
from stair_monitor.runtime.video_writer_sink import VideoWriterSink
from stair_monitor.state.person_identity import PersonIdentityManager


def _build_alert_logger(
    settings: AppSettings,
    output_fps: float,
) -> AlertEventLogger | None:
    if not settings.logging.enable_violation_csv:
        return None

    violation_cooldown_frames = max(
        1,
        int(round(output_fps * settings.logging.violation_cooldown_seconds)),
    )
    return AlertEventLogger(
        log_dir=settings.logging.log_dir,
        cooldown_frames=violation_cooldown_frames,
        enabled=settings.logging.enable_violation_csv,
    )


def create_runtime_app(
    settings: AppSettings,
    behavior_config: CameraConfigDict,
    extra_event_sinks: tuple[ViolationEventSink, ...] = (),
) -> StairMonitorRuntimeApp:
    frame_source = create_frame_source(settings.video)
    frame_source.open()

    output_fps = resolve_output_fps(frame_source.source_fps())
    demo_alert_hold_frames = max(
        1,
        int(round(output_fps * settings.demo.demo_alert_hold_seconds)),
    )

    analyzer = BehaviorAnalyzer(behavior_config)
    identity_manager = PersonIdentityManager(behavior_config)
    analysis_pipeline = StairAnalysisPipeline(
        analyzer=analyzer,
        identity_manager=identity_manager,
        alert_hold_frames=demo_alert_hold_frames,
    )
    display_sink = AlertOnlyDisplaySink(
        enabled=settings.video.show_live_window,
        window_name="Stair Monitor Demo Live",
        scale=settings.video.live_window_scale,
        analyzer=analyzer,
        config=behavior_config,
        demo_settings=settings.demo,
    )
    alert_logger = _build_alert_logger(settings, output_fps)
    event_sinks = (
        ((alert_logger,) if alert_logger is not None else ())
        + extra_event_sinks
    )
    output_coordinator = RuntimeOutputCoordinator(
        display_sink=display_sink,
        event_sinks=event_sinks,
        alert_logger=alert_logger,
        video_writer_sink=VideoWriterSink(
            enabled=settings.video.save_output_video,
            output_path=settings.video.output_path,
            output_fps=output_fps,
        ),
    )
    pose_estimator = YoloPoseEstimator(
        model_path=settings.model.model_path,
        confidence=settings.model.confidence,
        iou=settings.model.iou,
        tracker_config=settings.model.tracker_config,
    )
    return StairMonitorRuntimeApp(
        frame_source=frame_source,
        pose_estimator=pose_estimator,
        analysis_pipeline=analysis_pipeline,
        output_coordinator=output_coordinator,
        settings=RuntimeAppSettings(
            video=settings.video,
            demo=settings.demo,
            performance=settings.performance,
        ),
    )
