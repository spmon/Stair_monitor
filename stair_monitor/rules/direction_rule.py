from __future__ import annotations

from typing import TYPE_CHECKING

from stair_monitor.common.types import Point
from stair_monitor.config.settings import SETTINGS

if TYPE_CHECKING:
    from stair_monitor.core.analyzer import BehaviorAnalyzer


def apply_direction_history(
    analyzer: BehaviorAnalyzer,
    track_id: int,
    p_motion: Point | None,
) -> None:
    if track_id not in analyzer.track_history:
        analyzer.track_history[track_id] = []
    if p_motion is not None:
        analyzer.track_history[track_id].append(p_motion[1])


def compute_direction_from_motion(
    analyzer: BehaviorAnalyzer,
    track_id: int,
) -> dict[str, object]:
    dy = analyzer.track_history[track_id][-1] - analyzer.track_history[track_id][0]
    analyzer.track_history[track_id] = analyzer.track_history[track_id][
        -SETTINGS.direction.history_len:
    ]

    if SETTINGS.camera.use_current_camera_angle:
        if SETTINGS.direction.sign_normal:
            if dy < -SETTINGS.direction.pixel_threshold:
                direction = "UP"
                direction_reason = "LEGACY_SIGN_NORMAL_Y_DECREASE_UP"
            elif dy > SETTINGS.direction.pixel_threshold:
                direction = "DOWN"
                direction_reason = "LEGACY_SIGN_NORMAL_Y_INCREASE_DOWN"
            else:
                direction = "IDLE"
                direction_reason = "LEGACY_DIRECTION_IDLE_THRESHOLD"
        else:
            if dy < -SETTINGS.direction.pixel_threshold:
                direction = "DOWN"
                direction_reason = "LEGACY_SIGN_INVERTED_Y_DECREASE_DOWN"
            elif dy > SETTINGS.direction.pixel_threshold:
                direction = "UP"
                direction_reason = "LEGACY_SIGN_INVERTED_Y_INCREASE_UP"
            else:
                direction = "IDLE"
                direction_reason = "LEGACY_DIRECTION_IDLE_THRESHOLD"
    else:
        if dy > SETTINGS.direction.pixel_threshold:
            direction = "DOWN"
            direction_reason = "NEW_CAMERA_Y_INCREASE_DOWN"
        elif dy < -SETTINGS.direction.pixel_threshold:
            direction = "UP"
            direction_reason = "NEW_CAMERA_Y_DECREASE_UP"
        else:
            direction = "IDLE"
            direction_reason = "NEW_CAMERA_DIRECTION_IDLE_THRESHOLD"

    return {
        "direction": direction,
        "dy": dy,
        "direction_dy": dy,
        "direction_reason": direction_reason,
    }


def update_direction(
    analyzer: BehaviorAnalyzer,
    track_id: int,
) -> dict[str, object]:
    return compute_direction_from_motion(analyzer, track_id)
