from __future__ import annotations

from typing import TYPE_CHECKING

from stair_monitor.common.types import Point, PoseFeatures
from stair_monitor.config.settings import SETTINGS

if TYPE_CHECKING:
    from stair_monitor.core.analyzer import BehaviorAnalyzer

# File nay suy ra huong di tu lich su monitor point.
# FLOW: monitor_point_hip / monitor_point_shoulder -> history theo nhieu frame -> final_direction.
# WHY: Direction khong nen ket luan tren 1 frame vi bbox/keypoint co the rung.

def _ensure_motion_history(
    history_map: dict[int, list[int]],
    track_id: int,
) -> None:
    if track_id not in history_map:
        history_map[track_id] = []


def _append_monitor_point_history(
    history_map: dict[int, list[int]],
    track_id: int,
    monitor_point: Point | None,
) -> None:
    _ensure_motion_history(history_map, track_id)
    if monitor_point is None:
        return

    history_map[track_id].append(int(monitor_point[1]))
    history_map[track_id] = history_map[track_id][-SETTINGS.direction.history_len :]


def apply_direction_history(
    analyzer: BehaviorAnalyzer,
    track_id: int,
    features: PoseFeatures,
) -> None:
    """Cap nhat lich su motion cho 2 luong hip va shoulder.

    WHY:
        - Giu 2 history rieng giup direction on dinh hon so voi chi nhin 1 diem.
    """
    _append_monitor_point_history(
        analyzer.hip_motion_history,
        track_id,
        features.get("monitor_point_hip"),
    )
    _append_monitor_point_history(
        analyzer.shoulder_motion_history,
        track_id,
        features.get("monitor_point_shoulder"),
    )


def _map_dy_to_direction(dy: int, point_prefix: str) -> tuple[str, str]:
    if SETTINGS.camera.use_current_camera_angle:
        if SETTINGS.direction.sign_normal:
            if dy < -SETTINGS.direction.pixel_threshold:
                return "UP", f"{point_prefix}_LEGACY_SIGN_NORMAL_Y_DECREASE_UP"
            if dy > SETTINGS.direction.pixel_threshold:
                return "DOWN", f"{point_prefix}_LEGACY_SIGN_NORMAL_Y_INCREASE_DOWN"
            return "IDLE", f"{point_prefix}_LEGACY_DIRECTION_IDLE_THRESHOLD"
        if dy < -SETTINGS.direction.pixel_threshold:
            return "DOWN", f"{point_prefix}_LEGACY_SIGN_INVERTED_Y_DECREASE_DOWN"
        if dy > SETTINGS.direction.pixel_threshold:
            return "UP", f"{point_prefix}_LEGACY_SIGN_INVERTED_Y_INCREASE_UP"
        return "IDLE", f"{point_prefix}_LEGACY_DIRECTION_IDLE_THRESHOLD"

    if dy > SETTINGS.direction.pixel_threshold:
        return "DOWN", f"{point_prefix}_NEW_CAMERA_Y_INCREASE_DOWN"
    if dy < -SETTINGS.direction.pixel_threshold:
        return "UP", f"{point_prefix}_NEW_CAMERA_Y_DECREASE_UP"
    return "IDLE", f"{point_prefix}_NEW_CAMERA_DIRECTION_IDLE_THRESHOLD"


def compute_direction_from_point_history(
    history: list[int],
    point_prefix: str,
    monitor_point: Point | None,
) -> dict[str, object]:
    if monitor_point is None:
        return {
            "direction": "UNKNOWN",
            "dy": None,
            "direction_dy": None,
            "direction_reason": f"{point_prefix}_MONITOR_POINT_MISSING",
        }

    if len(history) < SETTINGS.direction.min_frames:
        return {
            "direction": "ANALYZING",
            "dy": None,
            "direction_dy": None,
            "direction_reason": f"{point_prefix}_HISTORY_NOT_ENOUGH_FRAMES",
        }

    dy = int(history[-1] - history[0])
    direction, direction_reason = _map_dy_to_direction(dy, point_prefix)
    return {
        "direction": direction,
        "dy": dy,
        "direction_dy": dy,
        "direction_reason": direction_reason,
    }


def _is_valid_movement_direction(direction: str) -> bool:
    return direction in ("UP", "DOWN")


def _is_usable_monitor_direction(direction: str) -> bool:
    return direction in ("UP", "DOWN", "IDLE")


def update_direction(
    analyzer: BehaviorAnalyzer,
    track_id: int,
    features: PoseFeatures,
) -> dict[str, object]:
    """Tong hop direction tu 2 history hip/shoulder.

    OUTPUT:
        - `direction`, `final_direction`, `direction_source`, `direction_reason`.

    WHY:
        - Neu 2 luong mau thuan, runtime uu tien giu `last_valid_direction`
          thay vi flip lien tuc tung frame.
    """
    hip_direction_info = compute_direction_from_point_history(
        analyzer.hip_motion_history.get(track_id, []),
        "HIP",
        features.get("monitor_point_hip"),
    )
    shoulder_direction_info = compute_direction_from_point_history(
        analyzer.shoulder_motion_history.get(track_id, []),
        "SHOULDER",
        features.get("monitor_point_shoulder"),
    )

    hip_direction = str(hip_direction_info["direction"])
    shoulder_direction = str(shoulder_direction_info["direction"])
    hip_dy = hip_direction_info.get("direction_dy")
    shoulder_dy = shoulder_direction_info.get("direction_dy")

    if (
        _is_valid_movement_direction(hip_direction)
        and _is_valid_movement_direction(shoulder_direction)
        and hip_direction == shoulder_direction
    ):
        return {
            "direction": hip_direction,
            "final_direction": hip_direction,
            "hip_direction": hip_direction,
            "shoulder_direction": shoulder_direction,
            "direction_source": "HIP_SHOULDER_AGREE",
            "direction_reason": "HIP_SHOULDER_AGREE",
            "dy": hip_dy if hip_dy is not None else shoulder_dy,
            "direction_dy": hip_dy if hip_dy is not None else shoulder_dy,
        }

    if (
        _is_valid_movement_direction(hip_direction)
        and _is_valid_movement_direction(shoulder_direction)
        and hip_direction != shoulder_direction
    ):
        last_valid_direction = analyzer.last_valid_direction.get(track_id)
        final_direction = (
            last_valid_direction
            if last_valid_direction in ("UP", "DOWN")
            else "ANALYZING"
        )
        return {
            "direction": final_direction,
            "final_direction": final_direction,
            "hip_direction": hip_direction,
            "shoulder_direction": shoulder_direction,
            "direction_source": "HIP_SHOULDER_CONFLICT",
            "direction_reason": "HIP_SHOULDER_CONFLICT",
            "dy": None,
            "direction_dy": None,
        }

    if hip_direction == shoulder_direction and _is_usable_monitor_direction(hip_direction):
        return {
            "direction": hip_direction,
            "final_direction": hip_direction,
            "hip_direction": hip_direction,
            "shoulder_direction": shoulder_direction,
            "direction_source": "HIP_SHOULDER_AGREE",
            "direction_reason": "HIP_SHOULDER_AGREE",
            "dy": hip_dy if hip_dy is not None else shoulder_dy,
            "direction_dy": hip_dy if hip_dy is not None else shoulder_dy,
        }

    if _is_usable_monitor_direction(hip_direction) and not _is_usable_monitor_direction(
        shoulder_direction
    ):
        return {
            "direction": hip_direction,
            "final_direction": hip_direction,
            "hip_direction": hip_direction,
            "shoulder_direction": shoulder_direction,
            "direction_source": "HIP_ONLY",
            "direction_reason": str(hip_direction_info["direction_reason"]),
            "dy": hip_dy,
            "direction_dy": hip_dy,
        }

    if _is_usable_monitor_direction(shoulder_direction) and not _is_usable_monitor_direction(
        hip_direction
    ):
        return {
            "direction": shoulder_direction,
            "final_direction": shoulder_direction,
            "hip_direction": hip_direction,
            "shoulder_direction": shoulder_direction,
            "direction_source": "SHOULDER_ONLY",
            "direction_reason": str(shoulder_direction_info["direction_reason"]),
            "dy": shoulder_dy,
            "direction_dy": shoulder_dy,
        }

    if _is_valid_movement_direction(hip_direction) and shoulder_direction == "IDLE":
        return {
            "direction": hip_direction,
            "final_direction": hip_direction,
            "hip_direction": hip_direction,
            "shoulder_direction": shoulder_direction,
            "direction_source": "HIP_ONLY",
            "direction_reason": str(hip_direction_info["direction_reason"]),
            "dy": hip_dy,
            "direction_dy": hip_dy,
        }

    if _is_valid_movement_direction(shoulder_direction) and hip_direction == "IDLE":
        return {
            "direction": shoulder_direction,
            "final_direction": shoulder_direction,
            "hip_direction": hip_direction,
            "shoulder_direction": shoulder_direction,
            "direction_source": "SHOULDER_ONLY",
            "direction_reason": str(shoulder_direction_info["direction_reason"]),
            "dy": shoulder_dy,
            "direction_dy": shoulder_dy,
        }

    return {
        "direction": "ANALYZING",
        "final_direction": "ANALYZING",
        "hip_direction": hip_direction,
        "shoulder_direction": shoulder_direction,
        "direction_source": "NO_VALID_MONITOR_DIRECTION",
        "direction_reason": "NO_VALID_MONITOR_DIRECTION",
        "dy": None,
        "direction_dy": None,
    }
