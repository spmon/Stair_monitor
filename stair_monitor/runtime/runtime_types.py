from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

from stair_monitor.common.types import AnalysisResult, PerfStats, Point, PoseFeatures

Frame: TypeAlias = NDArray[np.uint8]
BBox: TypeAlias = tuple[float, float, float, float]
DetectionBBox: TypeAlias = BBox
Keypoints: TypeAlias = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class PoseDetection:
    yolo_track_id: int
    bbox: BBox
    keypoints: Keypoints
    confidence: float


@dataclass(frozen=True, slots=True)
class FramePacket:
    frame: Frame
    frame_index: int
    source_fps: float
    captured_at_monotonic: float
    source_name: str


@dataclass(frozen=True, slots=True)
class PersonAnalysisResult:
    frame_index: int
    yolo_track_id: int | None
    analysis_subject_id: str
    person_uid: str
    warnings: tuple[str, ...]
    direction: str
    status: str
    bbox: BBox
    keypoints: Keypoints
    features: PoseFeatures
    p_lane: Point | None
    p_motion: Point | None
    analysis: AnalysisResult
    person_uid_label: str


PersonRuntimeResult = PersonAnalysisResult


@dataclass(frozen=True, slots=True)
class FrameAnalysisResult:
    frame_index: int
    person_results: tuple[PersonAnalysisResult, ...]
    active_alert_until_frame: dict[str, int]
    analysis_perf: PerfStats


@dataclass(frozen=True, slots=True)
class ViolationEvent:
    timestamp: str
    frame_index: int
    yolo_track_id: int | None
    analysis_subject_id: str
    person_uid: str
    direction: str
    warnings: tuple[str, ...]
    status: str


class ViolationEventSink(Protocol):
    def log_events(
        self,
        events: tuple[ViolationEvent, ...],
        current_frame_index: int,
    ) -> None:
        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True, slots=True)
class OutputFrameContext:
    frame_index: int
    overlay_frame: Frame
    active_alert_until_frame: dict[str, int]
    person_results: tuple[PersonAnalysisResult, ...]
