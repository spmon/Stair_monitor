from __future__ import annotations

from typing import TYPE_CHECKING

import cv2

from stair_monitor.common.types import LinePoints, Point, PoseFeatures

if TYPE_CHECKING:
    from stair_monitor.core.analyzer import BehaviorAnalyzer


def is_inside_stairs(stairs_poly: LinePoints, p_lane: Point | None) -> bool:
    if p_lane is None or len(stairs_poly) < 3:
        return False

    return cv2.pointPolygonTest(
        stairs_poly,
        (float(p_lane[0]), float(p_lane[1])),
        False,
    ) >= 0


def evaluate_inside_stairs(
    analyzer: BehaviorAnalyzer,
    track_id: int,
    features: PoseFeatures,
    fallback_lane_point: Point | None,
) -> dict[str, object]:
    _ = fallback_lane_point
    left_ankle = features.get("left_ankle")
    right_ankle = features.get("right_ankle")
    inside_feet_point = features.get("inside_feet_point")
    inside_feet_point_source = features.get(
        "inside_feet_point_source",
        "FEET_UNAVAILABLE",
    )
    feet_unavailable_reason = str(
        features.get("feet_unavailable_reason", "NO_SHOULDER_NO_HIP")
    )
    left_ankle_valid = left_ankle is not None
    right_ankle_valid = right_ankle is not None
    valid_foot_count = int(left_ankle_valid) + int(right_ankle_valid)
    ankle_valid_count = int(features.get("ankle_valid_count", valid_foot_count) or 0)
    feet_reliable = bool(features.get("feet_reliable", False))
    feet_is_real = inside_feet_point_source.startswith("REAL_")
    inside_raw_by_feet = None
    inside_grace_left = 0
    left_foot_in = False
    right_foot_in = False
    track_zone_state = "OUTSIDE_FEET_UNAVAILABLE"
    selected_feet_available = inside_feet_point is not None

    if left_ankle_valid:
        left_foot_in = is_inside_stairs(analyzer.stairs_poly, left_ankle)
    if right_ankle_valid:
        right_foot_in = is_inside_stairs(analyzer.stairs_poly, right_ankle)

    if not selected_feet_available:
        last_inside_state = bool(analyzer.inside_last_state.get(track_id, False))
        if last_inside_state:
            track_zone_state = "INSIDE_KEEP_LAST_FEET_UNAVAILABLE"
        return {
            "inside_stairs": last_inside_state,
            "inside_feet_point": inside_feet_point,
            "inside_feet_point_source": inside_feet_point_source,
            "ankle_valid_count": ankle_valid_count,
            "feet_reliable": feet_reliable,
            "inside_raw_by_feet": inside_raw_by_feet,
            "inside_reason": feet_unavailable_reason,
            "inside_grace_left": inside_grace_left,
            "left_ankle_valid": left_ankle_valid,
            "right_ankle_valid": right_ankle_valid,
            "left_foot_in": left_foot_in,
            "right_foot_in": right_foot_in,
            "valid_foot_count": valid_foot_count,
            "track_zone_state": track_zone_state,
        }

    inside_raw_by_feet = is_inside_stairs(analyzer.stairs_poly, inside_feet_point)
    analyzer.inside_last_state[track_id] = inside_raw_by_feet
    if inside_raw_by_feet:
        return {
            "inside_stairs": True,
            "inside_feet_point": inside_feet_point,
            "inside_feet_point_source": inside_feet_point_source,
            "ankle_valid_count": ankle_valid_count,
            "feet_reliable": feet_reliable,
            "inside_raw_by_feet": inside_raw_by_feet,
            "inside_reason": (
                "ENTERED_BY_FOOT" if feet_is_real else "ENTERED_BY_VIRTUAL_FOOT"
            ),
            "inside_grace_left": inside_grace_left,
            "left_ankle_valid": left_ankle_valid,
            "right_ankle_valid": right_ankle_valid,
            "left_foot_in": left_foot_in,
            "right_foot_in": right_foot_in,
            "valid_foot_count": valid_foot_count,
            "track_zone_state": (
                "INSIDE_CONFIRMED_BY_FOOT"
                if feet_is_real
                else "INSIDE_CONFIRMED_BY_VIRTUAL_FOOT"
            ),
        }

    return {
        "inside_stairs": False,
        "inside_feet_point": inside_feet_point,
        "inside_feet_point_source": inside_feet_point_source,
        "ankle_valid_count": ankle_valid_count,
        "feet_reliable": feet_reliable,
        "inside_raw_by_feet": inside_raw_by_feet,
        "inside_reason": (
            "ALL_VISIBLE_FEET_OUT" if feet_is_real else "SELECTED_FEET_OUTSIDE"
        ),
        "inside_grace_left": inside_grace_left,
        "left_ankle_valid": left_ankle_valid,
        "right_ankle_valid": right_ankle_valid,
        "left_foot_in": left_foot_in,
        "right_foot_in": right_foot_in,
        "valid_foot_count": valid_foot_count,
        "track_zone_state": (
            "OUTSIDE_CONFIRMED_BY_FOOT"
            if feet_is_real
            else "OUTSIDE_CONFIRMED_BY_SELECTED_FEET"
        ),
    }
