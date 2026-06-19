from __future__ import annotations

from dataclasses import dataclass

from stair_monitor.common.types import (
    AnalysisSubjectID,
    BBoxArray,
    KeypointsArray,
    Point,
    PoseFeatures,
)


@dataclass(slots=True)
class PersonContext:
    track_id: AnalysisSubjectID
    keypoints: KeypointsArray
    box: BBoxArray | None
    features: PoseFeatures
    p_lane: Point | None
    p_motion: Point | None
