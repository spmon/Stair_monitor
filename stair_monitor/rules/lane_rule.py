from __future__ import annotations

from typing import TYPE_CHECKING

from stair_monitor.common.types import LinePoints, Numeric, Point, PoseFeatures
from stair_monitor.config.settings import SETTINGS

if TYPE_CHECKING:
    from stair_monitor.core.analyzer import BehaviorAnalyzer


def compute_lane_side(
    point: Point | None,
    line_points: LinePoints | None,
) -> Numeric | None:
    if point is None or line_points is None or len(line_points) < 2:
        return None

    a, b = line_points[0], line_points[1]
    return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (
        point[0] - a[0]
    )


def is_left_lane_side(side_value: Numeric | None) -> bool:
    if side_value is None or side_value == 0:
        return False
    return side_value * SETTINGS.camera.lane_left_side_sign > 0


def is_right_lane_side(side_value: Numeric | None) -> bool:
    if side_value is None or side_value == 0:
        return False
    return side_value * SETTINGS.camera.lane_left_side_sign < 0


def is_wrong_lane_side(
    direction: str,
    side_value: Numeric | None,
    sign_normal: bool,
) -> bool | None:
    if side_value is None or direction not in ("UP", "DOWN"):
        return None

    if not SETTINGS.camera.use_current_camera_angle:
        if direction == "DOWN":
            return not is_left_lane_side(side_value)
        return not is_right_lane_side(side_value)

    if sign_normal:
        return (direction == "UP" and side_value < 0) or (
            direction == "DOWN" and side_value > 0
        )
    return (direction == "UP" and side_value > 0) or (
        direction == "DOWN" and side_value < 0
    )


def get_lane_side_label(side_value: Numeric | None) -> str:
    if side_value is None:
        return "UNKNOWN"
    if is_left_lane_side(side_value):
        return "LEFT"
    if is_right_lane_side(side_value):
        return "RIGHT"
    return "ON_LINE"


def get_correct_lane_side(direction: str, sign_normal: bool) -> str:
    if direction not in ("UP", "DOWN"):
        return "UNKNOWN"

    if not SETTINGS.camera.use_current_camera_angle:
        return "LEFT" if direction == "DOWN" else "RIGHT"

    if sign_normal:
        return "RIGHT" if direction == "UP" else "LEFT"
    return "LEFT" if direction == "UP" else "RIGHT"


def get_camera_angle_profile() -> str:
    return (
        "CURRENT_CAMERA"
        if SETTINGS.camera.use_current_camera_angle
        else "BOTTOM_STAIR_CAMERA"
    )


def select_p_lane_for_lane(features: PoseFeatures) -> tuple[Point | None, str, str]:
    feet_point = features.get("feet_point")
    feet_point_source = features.get("feet_point_source", "FEET_UNAVAILABLE")

    if feet_point is None:
        return None, "NO_FOOT", "NO_VALID_FOOT_FOR_LANE"

    if feet_point_source == "REAL_BOTH_ANKLES":
        return feet_point, "FEET_MIDPOINT", "FEET_VISIBLE"
    if feet_point_source == "REAL_LEFT_ANKLE":
        return feet_point, "LEFT_FOOT_ONLY", "FEET_VISIBLE"
    if feet_point_source == "REAL_RIGHT_ANKLE":
        return feet_point, "RIGHT_FOOT_ONLY", "FEET_VISIBLE"
    return feet_point, feet_point_source, "VIRTUAL_FEET"

def update_lane_history(
    analyzer: BehaviorAnalyzer,
    track_id: int,
    wrong_lane_raw: bool | None,
) -> tuple[int, bool]:
    return analyzer._update_lane_history(track_id, wrong_lane_raw)


def evaluate_lane_violation(
    analyzer: BehaviorAnalyzer,
    track_id: int,
    features: PoseFeatures,
    direction: str,
) -> dict[str, object]:
    camera_angle_profile = get_camera_angle_profile()
    lane_direction = direction
    lane_mapping_source = camera_angle_profile
    lane_side_value = None
    lane_side_label = "UNKNOWN"
    correct_lane_side = "UNKNOWN"
    wrong_lane_raw = False
    wrong_lane = False
    lane_wrong_hits = 0
    lane_status = "UNKNOWN"
    lane_reason = "NA"
    lane_source = "NO_LANE"
    p_lane_source = "NONE"
    lane_missing_feet_grace_left = 0
    foot_lane_side = None
    v = None

    selected_p_lane, p_lane_source, _selection_reason = select_p_lane_for_lane(features)
    p_lane = selected_p_lane
    if p_lane is not None:
        lane_source = "FOOT_LANE"
        if len(analyzer.center_line) >= 2:
            foot_lane_side = compute_lane_side(p_lane, analyzer.center_line)
            v = foot_lane_side
            lane_side_value = foot_lane_side
            lane_side_label = get_lane_side_label(foot_lane_side)
            correct_lane_side = get_correct_lane_side(
                direction,
                SETTINGS.lane.sign_normal,
            )
        if direction not in ("UP", "DOWN"):
            lane_wrong_hits, wrong_lane = analyzer._get_lane_history_state(track_id)
            lane_status = "UNKNOWN"
            lane_reason = "LANE_BY_FEET"
        elif len(analyzer.center_line) >= 2:
            wrong_lane_raw = is_wrong_lane_side(
                direction,
                foot_lane_side,
                SETTINGS.lane.sign_normal,
            )
            lane_wrong_hits, wrong_lane = update_lane_history(
                analyzer,
                track_id,
                wrong_lane_raw,
            )
            lane_status = "EVALUATED"
            lane_reason = "LANE_BY_FEET"
        else:
            lane_status = "UNKNOWN"
            lane_reason = "CENTER_LINE_MISSING"
    else:
        last_seen = analyzer.lane_last_seen.get(track_id)
        if (
            last_seen is not None
            and analyzer.frame_index - last_seen
            <= SETTINGS.lane.missing_feet_grace_frames
        ):
            lane_missing_feet_grace_left = max(
                0,
                SETTINGS.lane.missing_feet_grace_frames
                - (analyzer.frame_index - last_seen)
                + 1,
            )
            last_lane_state = analyzer.lane_last_state.get(track_id, {})
            wrong_lane_raw = last_lane_state.get("wrong_lane_raw")
            wrong_lane = last_lane_state.get("wrong_lane", False)
            lane_wrong_hits = last_lane_state.get("lane_wrong_hits", 0)
            lane_status = last_lane_state.get("lane_status", "UNKNOWN")
            lane_direction = last_lane_state.get("lane_direction", lane_direction)
            v = last_lane_state.get("lane_v")
            lane_source = "LANE_KEEP_LAST"
            foot_lane_side = last_lane_state.get("foot_lane_side")
            lane_side_value = last_lane_state.get("lane_side_value")
            lane_side_label = last_lane_state.get("lane_side_label", "UNKNOWN")
            correct_lane_side = last_lane_state.get("correct_lane_side", "UNKNOWN")
            lane_mapping_source = last_lane_state.get(
                "lane_mapping_source",
                lane_mapping_source,
            )
            lane_reason = "MISSING_FEET_KEEP_LAST"
            p_lane_source = "NO_FOOT"
        else:
            wrong_lane_raw = None
            wrong_lane = False
            lane_wrong_hits = 0
            lane_status = "UNKNOWN"
            lane_source = "NO_LANE"
            lane_reason = "MISSING_FEET_NO_DISPLAY"
            p_lane_source = "NO_FOOT"
            lane_side_value = None
            lane_side_label = "UNKNOWN"
            correct_lane_side = "UNKNOWN"
    if lane_source == "FOOT_LANE":
        analyzer.lane_last_seen[track_id] = analyzer.frame_index
        analyzer.lane_last_state[track_id] = {
            "wrong_lane_raw": wrong_lane_raw,
            "wrong_lane": wrong_lane,
            "lane_wrong_hits": lane_wrong_hits,
            "lane_status": lane_status,
            "lane_direction": lane_direction,
            "lane_v": v,
            "foot_lane_side": foot_lane_side,
            "lane_side_value": lane_side_value,
            "lane_side_label": lane_side_label,
            "correct_lane_side": correct_lane_side,
            "lane_mapping_source": lane_mapping_source,
        }

    return {
        "v": v,
        "p_lane": p_lane,
        "p_lane_source": p_lane_source,
        "lane_source": lane_source,
        "lane_mapping_source": lane_mapping_source,
        "lane_direction": lane_direction,
        "foot_lane_side": foot_lane_side,
        "lane_side_value": lane_side_value,
        "lane_side_label": lane_side_label,
        "correct_lane_side": correct_lane_side,
        "wrong_lane_raw": wrong_lane_raw,
        "wrong_lane": wrong_lane,
        "lane_wrong_hits": lane_wrong_hits,
        "lane_status": lane_status,
        "lane_reason": lane_reason,
        "lane_missing_feet_grace_left": lane_missing_feet_grace_left,
    }
