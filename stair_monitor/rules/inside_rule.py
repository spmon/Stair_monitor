from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

import cv2

from stair_monitor.common.types import AnalysisSubjectID, LinePoints, Point, PoseFeatures

if TYPE_CHECKING:
    from stair_monitor.core.analyzer import BehaviorAnalyzer


class InsideStairsAnalysisUpdate(TypedDict):
    inside_stairs: bool
    inside_feet_point: Point | None
    inside_feet_point_source: str
    ankle_valid_count: int
    feet_reliable: bool
    inside_raw_by_feet: bool | None
    inside_reason: str
    inside_grace_left: int
    left_ankle_valid: bool
    right_ankle_valid: bool
    left_foot_in: bool
    right_foot_in: bool
    valid_foot_count: int
    track_zone_state: str


@dataclass(frozen=True, slots=True)
class InsideStairsInput:
    stairs_polygon: LinePoints
    last_inside_state: bool
    left_ankle: Point | None
    right_ankle: Point | None
    inside_feet_point: Point | None
    inside_feet_point_source: str
    feet_unavailable_reason: str
    ankle_valid_count: int
    feet_reliable: bool


@dataclass(frozen=True, slots=True)
class InsideStairsResult:
    inside_stairs: bool
    inside_feet_point: Point | None
    inside_feet_point_source: str
    ankle_valid_count: int
    feet_reliable: bool
    inside_raw_by_feet: bool | None
    inside_reason: str
    inside_grace_left: int
    left_ankle_valid: bool
    right_ankle_valid: bool
    left_foot_in: bool
    right_foot_in: bool
    valid_foot_count: int
    track_zone_state: str
    last_inside_state_update: bool | None

    def to_analysis_update(self) -> InsideStairsAnalysisUpdate:
        return {
            "inside_stairs": self.inside_stairs,
            "inside_feet_point": self.inside_feet_point,
            "inside_feet_point_source": self.inside_feet_point_source,
            "ankle_valid_count": self.ankle_valid_count,
            "feet_reliable": self.feet_reliable,
            "inside_raw_by_feet": self.inside_raw_by_feet,
            "inside_reason": self.inside_reason,
            "inside_grace_left": self.inside_grace_left,
            "left_ankle_valid": self.left_ankle_valid,
            "right_ankle_valid": self.right_ankle_valid,
            "left_foot_in": self.left_foot_in,
            "right_foot_in": self.right_foot_in,
            "valid_foot_count": self.valid_foot_count,
            "track_zone_state": self.track_zone_state,
        }


def is_inside_stairs(stairs_poly: LinePoints, p_lane: Point | None) -> bool:
    if p_lane is None or len(stairs_poly) < 3:
        return False

    return cv2.pointPolygonTest(
        stairs_poly,
        (float(p_lane[0]), float(p_lane[1])),
        False,
    ) >= 0


def build_inside_stairs_input(
    features: PoseFeatures,
    stairs_polygon: LinePoints,
    last_inside_state: bool,
) -> InsideStairsInput:
    left_ankle = features.get("left_ankle")
    right_ankle = features.get("right_ankle")
    valid_foot_count = int(left_ankle is not None) + int(right_ankle is not None)
    ankle_valid_count = int(
        features.get("ankle_valid_count", valid_foot_count) or 0
    )
    return InsideStairsInput(
        stairs_polygon=stairs_polygon,
        last_inside_state=last_inside_state,
        left_ankle=left_ankle,
        right_ankle=right_ankle,
        inside_feet_point=features.get("inside_feet_point"),
        inside_feet_point_source=features.get(
            "inside_feet_point_source",
            "FEET_UNAVAILABLE",
        ),
        feet_unavailable_reason=str(
            features.get("feet_unavailable_reason", "NO_SHOULDER_NO_HIP")
        ),
        ankle_valid_count=ankle_valid_count,
        feet_reliable=bool(features.get("feet_reliable", False)),
    )


def _build_result(
    rule_input: InsideStairsInput,
    *,
    inside_stairs: bool,
    inside_raw_by_feet: bool | None,
    inside_reason: str,
    track_zone_state: str,
    last_inside_state_update: bool | None,
    left_foot_in: bool,
    right_foot_in: bool,
) -> InsideStairsResult:
    left_ankle_valid = rule_input.left_ankle is not None
    right_ankle_valid = rule_input.right_ankle is not None
    valid_foot_count = int(left_ankle_valid) + int(right_ankle_valid)
    return InsideStairsResult(
        inside_stairs=inside_stairs,
        inside_feet_point=rule_input.inside_feet_point,
        inside_feet_point_source=rule_input.inside_feet_point_source,
        ankle_valid_count=rule_input.ankle_valid_count,
        feet_reliable=rule_input.feet_reliable,
        inside_raw_by_feet=inside_raw_by_feet,
        inside_reason=inside_reason,
        inside_grace_left=0,
        left_ankle_valid=left_ankle_valid,
        right_ankle_valid=right_ankle_valid,
        left_foot_in=left_foot_in,
        right_foot_in=right_foot_in,
        valid_foot_count=valid_foot_count,
        track_zone_state=track_zone_state,
        last_inside_state_update=last_inside_state_update,
    )


def evaluate_inside_stairs_typed(
    rule_input: InsideStairsInput,
) -> InsideStairsResult:
    left_foot_in = False
    right_foot_in = False
    feet_is_real = rule_input.inside_feet_point_source.startswith("REAL_")

    if rule_input.left_ankle is not None:
        left_foot_in = is_inside_stairs(rule_input.stairs_polygon, rule_input.left_ankle)
    if rule_input.right_ankle is not None:
        right_foot_in = is_inside_stairs(
            rule_input.stairs_polygon,
            rule_input.right_ankle,
        )

    if rule_input.inside_feet_point is None:
        return _build_result(
            rule_input,
            inside_stairs=rule_input.last_inside_state,
            inside_raw_by_feet=None,
            inside_reason=rule_input.feet_unavailable_reason,
            track_zone_state=(
                "INSIDE_KEEP_LAST_FEET_UNAVAILABLE"
                if rule_input.last_inside_state
                else "OUTSIDE_FEET_UNAVAILABLE"
            ),
            last_inside_state_update=None,
            left_foot_in=left_foot_in,
            right_foot_in=right_foot_in,
        )

    inside_raw_by_feet = is_inside_stairs(
        rule_input.stairs_polygon,
        rule_input.inside_feet_point,
    )
    if inside_raw_by_feet:
        return _build_result(
            rule_input,
            inside_stairs=True,
            inside_raw_by_feet=True,
            inside_reason=(
                "ENTERED_BY_FOOT" if feet_is_real else "ENTERED_BY_VIRTUAL_FOOT"
            ),
            track_zone_state=(
                "INSIDE_CONFIRMED_BY_FOOT"
                if feet_is_real
                else "INSIDE_CONFIRMED_BY_VIRTUAL_FOOT"
            ),
            last_inside_state_update=True,
            left_foot_in=left_foot_in,
            right_foot_in=right_foot_in,
        )

    return _build_result(
        rule_input,
        inside_stairs=False,
        inside_raw_by_feet=False,
        inside_reason=(
            "ALL_VISIBLE_FEET_OUT" if feet_is_real else "SELECTED_FEET_OUTSIDE"
        ),
        track_zone_state=(
            "OUTSIDE_CONFIRMED_BY_FOOT"
            if feet_is_real
            else "OUTSIDE_CONFIRMED_BY_SELECTED_FEET"
        ),
        last_inside_state_update=False,
        left_foot_in=left_foot_in,
        right_foot_in=right_foot_in,
    )


def evaluate_inside_stairs(
    analyzer: BehaviorAnalyzer,
    track_id: AnalysisSubjectID,
    features: PoseFeatures,
    fallback_lane_point: Point | None,
) -> InsideStairsAnalysisUpdate:
    del fallback_lane_point
    rule_input = build_inside_stairs_input(
        features=features,
        stairs_polygon=analyzer.stairs_poly,
        last_inside_state=bool(analyzer.inside_last_state.get(track_id, False)),
    )
    result = evaluate_inside_stairs_typed(rule_input)
    if result.last_inside_state_update is not None:
        analyzer.inside_last_state[track_id] = result.last_inside_state_update
    return result.to_analysis_update()
