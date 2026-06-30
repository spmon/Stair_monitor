from __future__ import annotations

import time
from dataclasses import dataclass

from loguru import logger

from stair_monitor.config.settings import (
    DemoSettings,
    PerformanceConfig,
    VideoSettings,
)
from stair_monitor.debug.performance import (
    accumulate_analysis_perf,
    log_perf,
    new_perf_totals,
)
from stair_monitor.debug.snapshots import (
    ensure_debug_snapshot_dirs,
    save_debug_snapshots,
)
from stair_monitor.runtime.analysis_pipeline import StairAnalysisPipeline
from stair_monitor.runtime.frame_source import FrameSource
from stair_monitor.runtime.output_coordinator import RuntimeOutputCoordinator
from stair_monitor.runtime.pose_estimator import YoloPoseEstimator
from stair_monitor.runtime.runtime_types import OutputFrameContext


@dataclass(frozen=True, slots=True)
class RuntimeAppSettings:
    video: VideoSettings
    demo: DemoSettings
    performance: PerformanceConfig


class StairMonitorRuntimeApp:
    """Own the Windows/demo main loop without changing behavior logic."""

    def __init__(
        self,
        frame_source: FrameSource,
        pose_estimator: YoloPoseEstimator,
        analysis_pipeline: StairAnalysisPipeline,
        output_coordinator: RuntimeOutputCoordinator,
        settings: RuntimeAppSettings,
    ) -> None:
        self.frame_source = frame_source
        self.pose_estimator = pose_estimator
        self.analysis_pipeline = analysis_pipeline
        self.output_coordinator = output_coordinator
        self.settings = settings

    def run(self) -> None:
        self.frame_source.open()

        logger.info(
            "Frame source implementation={}",
            type(self.frame_source).__name__,
        )
        logger.info("Live window enabled={}", self.output_coordinator.display_sink.enabled)
        if (
            self.output_coordinator.alert_logger is not None
            and self.output_coordinator.alert_logger.file_path is not None
        ):
            logger.info("Alert CSV: {}", self.output_coordinator.alert_logger.file_path)

        total_start_time = time.perf_counter()
        perf_totals = (
            new_perf_totals()
            if self.settings.performance.enable_perf_log
            else None
        )
        perf_window_start = (
            time.perf_counter()
            if self.settings.performance.enable_perf_log
            else None
        )
        perf_window_frames = 0

        debug_frame_index = 0
        model_input_dir, overlay_dir = ensure_debug_snapshot_dirs()

        try:
            while True:
                frame_start = (
                    time.perf_counter()
                    if self.settings.performance.enable_perf_log
                    else None
                )

                packet = self.frame_source.read()
                if packet is None:
                    break

                debug_frame_index = packet.frame_index + 1

                frame_raw = packet.frame.copy()
                model_frame = frame_raw.copy()
                overlay_frame = frame_raw.copy()

                model_input_snapshot = (
                    model_frame.copy()
                    if self.settings.video.save_model_input_debug
                    else None
                )

                model_start = (
                    time.perf_counter()
                    if self.settings.performance.enable_perf_log
                    else None
                )
                detections = self.pose_estimator.estimate(model_frame)
                if perf_totals is not None and model_start is not None:
                    perf_totals["model"] += (
                        time.perf_counter() - model_start
                    ) * 1000.0

                frame_result = self.analysis_pipeline.process_frame(detections)

                if perf_totals is not None:
                    accumulate_analysis_perf(
                        perf_totals,
                        frame_result.analysis_perf,
                    )

                overlay_start = (
                    time.perf_counter()
                    if self.settings.performance.enable_perf_log
                    else None
                )
                should_continue = self.output_coordinator.handle_frame(
                    OutputFrameContext(
                        frame_index=frame_result.frame_index,
                        overlay_frame=overlay_frame,
                        active_alert_until_frame=frame_result.active_alert_until_frame,
                        person_results=frame_result.person_results,
                    )
                )
                if perf_totals is not None and overlay_start is not None:
                    perf_totals["overlay"] += (
                        time.perf_counter() - overlay_start
                    ) * 1000.0

                save_debug_snapshots(
                    debug_frame_index,
                    (
                        model_input_snapshot
                        if model_input_snapshot is not None
                        else model_frame
                    ),
                    overlay_frame,
                    model_input_dir,
                    overlay_dir,
                )

                if not should_continue:
                    break

                if perf_totals is not None and frame_start is not None:
                    perf_totals["total"] += (
                        time.perf_counter() - frame_start
                    ) * 1000.0

                if self.settings.performance.enable_perf_log:
                    perf_window_frames += 1
                    if (
                        perf_window_start is not None
                        and perf_window_frames
                        >= max(1, self.settings.performance.perf_log_interval)
                    ):
                        elapsed_s = time.perf_counter() - perf_window_start
                        log_perf(perf_totals, perf_window_frames, elapsed_s)
                        perf_totals = new_perf_totals()
                        perf_window_start = time.perf_counter()
                        perf_window_frames = 0
        finally:
            self.frame_source.release()
            self.analysis_pipeline.cleanup()
            self.output_coordinator.close()

        if not self.settings.demo.demo_mode:
            logger.info(
                "Tong thoi gian xu ly: {:.3f}s",
                time.perf_counter() - total_start_time,
            )
