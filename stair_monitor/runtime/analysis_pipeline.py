from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from stair_monitor.config.settings import SETTINGS
from stair_monitor.common.types import AnalysisResult, PerfStats, Point, PoseFeatures
from stair_monitor.debug.performance import accumulate_analysis_perf
from stair_monitor.core.analyzer import BehaviorAnalyzer
from stair_monitor.runtime.runtime_types import (
    FrameAnalysisResult,
    PersonAnalysisResult,
    PoseDetection,
)
from stair_monitor.state.person_identity import PersonIdentityManager
from stair_monitor.vision.geometry import extract_pose_features


def _read_text_field(raw_value: object, default: str) -> str:
    if raw_value is None:
        return default
    value_text = str(raw_value).strip()
    return value_text if value_text else default


def _read_warning_tuple(raw_value: object) -> tuple[str, ...]:
    if not isinstance(raw_value, Sequence) or isinstance(raw_value, (str, bytes)):
        return ()

    warnings: list[str] = []
    for warning in raw_value:
        warning_text = str(warning).strip()
        if warning_text:
            warnings.append(warning_text)
    return tuple(warnings)


class StairAnalysisPipeline:
    """Connect detections to extract/features, identity, analyzer, and alerts."""

    def __init__(
        self,
        analyzer: BehaviorAnalyzer,
        identity_manager: PersonIdentityManager,
        alert_hold_frames: int,
    ) -> None:
        self._analyzer = analyzer
        self._identity_manager = identity_manager
        self._alert_hold_frames = max(1, alert_hold_frames)
        self._active_alert_until_frame: dict[str, int] = {}

    def begin_frame(self) -> None:
        self._analyzer.begin_frame()
        self._identity_manager.begin_frame(self._analyzer.frame_index)

    def _update_active_alerts(self, warnings: tuple[str, ...]) -> None:
        for warning in warnings:
            if warning not in SETTINGS.violation.count_labels:
                continue
            self._active_alert_until_frame[warning] = max(
                self._active_alert_until_frame.get(warning, -1),
                self._analyzer.frame_index + self._alert_hold_frames - 1,
            )

    def _build_person_result(
        self,
        detection: PoseDetection,
        features: PoseFeatures,
        p_lane: Point | None,
        p_motion: Point | None,
        analysis: AnalysisResult,
        analysis_subject_id: str,
        person_uid_label: str,
    ) -> PersonAnalysisResult:
        return PersonAnalysisResult(
            frame_index=self._analyzer.frame_index,
            yolo_track_id=detection.yolo_track_id,
            analysis_subject_id=analysis_subject_id,
            person_uid=_read_text_field(analysis.get("person_uid"), ""),
            warnings=_read_warning_tuple(analysis.get("warnings")),
            direction=_read_text_field(
                analysis.get("direction"),
                "UNKNOWN",
            ),
            status=_read_text_field(
                analysis.get("display_status"),
                _read_text_field(analysis.get("status"), "ANALYZING"),
            ),
            bbox=detection.bbox,
            keypoints=detection.keypoints,
            features=features,
            p_lane=p_lane,
            p_motion=p_motion,
            analysis=analysis,
            person_uid_label=person_uid_label,
        )

    def _prune_inactive_alerts(self) -> None:
        self._active_alert_until_frame = {
            warning: until_frame
            for warning, until_frame in self._active_alert_until_frame.items()
            if until_frame >= self._analyzer.frame_index
        }

    def _analyze_detection(
        self,
        detection: PoseDetection,
        analysis_perf: PerfStats,
    ) -> PersonAnalysisResult:
        bbox_array = np.asarray(detection.bbox, dtype=np.float32)
        features = extract_pose_features(detection.keypoints, bbox_array)
        identity_result = self._identity_manager.update_detection(
            yolo_track_id=detection.yolo_track_id,
            bbox=bbox_array,
            features=features,
            frame_index=self._analyzer.frame_index,
        )

        p_lane = features.get("feet_point")
        p_motion = features.get("motion_point")

        merge_from_subject_id = identity_result.merged_from_analysis_subject_id
        if merge_from_subject_id:
            self._analyzer.merge_behavior_history(
                merge_from_subject_id,
                identity_result.analysis_subject_id,
            )

        if identity_result.analysis_subject_id:
            analysis = self._analyzer.analyze(
                identity_result.analysis_subject_id,
                p_lane,
                p_motion,
                detection.keypoints,
                box=bbox_array,
                features=features,
            )
            if identity_result.person_uid is not None:
                self._identity_manager.update_from_analysis(
                    identity_result.person_uid,
                    analysis,
                )
            self._identity_manager.augment_analysis_result(
                analysis,
                identity_result,
            )
            warnings = _read_warning_tuple(analysis.get("warnings"))
            self._update_active_alerts(warnings)
            accumulate_analysis_perf(
                analysis_perf,
                analysis.get("perf"),
            )
        else:
            analysis = self._identity_manager.build_overlay_result(identity_result)

        return self._build_person_result(
            detection=detection,
            features=features,
            p_lane=p_lane,
            p_motion=p_motion,
            analysis=analysis,
            analysis_subject_id=identity_result.analysis_subject_id,
            person_uid_label=identity_result.person_uid_label,
        )

    def _analyze_detections(
        self,
        detections: tuple[PoseDetection, ...],
    ) -> FrameAnalysisResult:
        person_results: list[PersonAnalysisResult] = []
        analysis_perf: PerfStats = {}

        for detection in detections:
            person_results.append(self._analyze_detection(detection, analysis_perf))

        self._prune_inactive_alerts()
        return FrameAnalysisResult(
            frame_index=self._analyzer.frame_index,
            person_results=tuple(person_results),
            active_alert_until_frame=dict(self._active_alert_until_frame),
            analysis_perf=analysis_perf,
        )

    def end_frame(self) -> None:
        self._identity_manager.end_frame()
        self._analyzer.cleanup_inactive_tracks(
            self._identity_manager.get_retained_analysis_subject_ids()
        )

    def process_frame(
        self,
        detections: tuple[PoseDetection, ...],
    ) -> FrameAnalysisResult:
        self.begin_frame()
        frame_result = self._analyze_detections(detections)
        self.end_frame()
        return frame_result

    def cleanup(self) -> None:
        """Keep a typed cleanup hook for future runtime-owned resources."""
