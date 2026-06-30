from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypedDict

from stair_monitor.common.types import AnalysisSubjectID, Point, PoseFeatures
from stair_monitor.config.settings import SETTINGS

if TYPE_CHECKING:
    from stair_monitor.core.analyzer import BehaviorAnalyzer

# File nay suy ra huong di tu lich su monitor point.
# FLOW: monitor_point_hip / monitor_point_shoulder -> history theo nhieu frame -> final_direction.
# WHY: Direction khong nen ket luan tren 1 frame vi bbox/keypoint co the rung.


class DirectionAnalysisUpdate(TypedDict):
    direction: str
    final_direction: str
    hip_direction: str
    shoulder_direction: str
    direction_source: str
    direction_reason: str
    dy: int | None
    direction_dy: int | None


@dataclass(frozen=True, slots=True)
class DirectionInput:
    subject_id: AnalysisSubjectID
    hip_monitor_point: Point | None
    shoulder_monitor_point: Point | None
    hip_history: tuple[int, ...]
    shoulder_history: tuple[int, ...]
    last_valid_direction: str | None


@dataclass(frozen=True, slots=True)
class PointDirectionResult:
    direction: str
    dy: int | None
    direction_dy: int | None
    direction_reason: str


@dataclass(frozen=True, slots=True)
class DirectionResult:
    direction: str
    final_direction: str
    hip_direction: str
    shoulder_direction: str
    direction_source: str
    direction_reason: str
    dy: int | None
    direction_dy: int | None

    def to_analysis_update(self) -> DirectionAnalysisUpdate:
        return {
            "direction": self.direction,
            "final_direction": self.final_direction,
            "hip_direction": self.hip_direction,
            "shoulder_direction": self.shoulder_direction,
            "direction_source": self.direction_source,
            "direction_reason": self.direction_reason,
            "dy": self.dy,
            "direction_dy": self.direction_dy,
        }


class DirectionHistoryAdapter(Protocol):
    hip_motion_history: dict[AnalysisSubjectID, list[int]]
    shoulder_motion_history: dict[AnalysisSubjectID, list[int]]
    last_valid_direction: dict[AnalysisSubjectID, str]


def _ensure_motion_history(
    history_map: dict[AnalysisSubjectID, list[int]],
    track_id: AnalysisSubjectID,
) -> None:
    if track_id not in history_map:
        history_map[track_id] = []


def _append_monitor_point_history(
    history_map: dict[AnalysisSubjectID, list[int]],
    track_id: AnalysisSubjectID,
    monitor_point: Point | None,
) -> None:
    _ensure_motion_history(history_map, track_id)
    if monitor_point is None:
        return

    history_map[track_id].append(int(monitor_point[1]))
    history_map[track_id] = history_map[track_id][-SETTINGS.direction.history_len :]


def build_direction_input(
    track_id: AnalysisSubjectID,
    features: PoseFeatures,
    history_adapter: DirectionHistoryAdapter,
) -> DirectionInput:
    return DirectionInput(
        subject_id=track_id,
        hip_monitor_point=features.get("monitor_point_hip"),
        shoulder_monitor_point=features.get("monitor_point_shoulder"),
        hip_history=tuple(history_adapter.hip_motion_history.get(track_id, [])),
        shoulder_history=tuple(history_adapter.shoulder_motion_history.get(track_id, [])),
        last_valid_direction=history_adapter.last_valid_direction.get(track_id),
    )


def apply_direction_history_typed(
    rule_input: DirectionInput,
    history_adapter: DirectionHistoryAdapter,
) -> None:
    _append_monitor_point_history(
        history_adapter.hip_motion_history,
        rule_input.subject_id,
        rule_input.hip_monitor_point,
    )
    _append_monitor_point_history(
        history_adapter.shoulder_motion_history,
        rule_input.subject_id,
        rule_input.shoulder_monitor_point,
    )


def apply_direction_history(
    analyzer: BehaviorAnalyzer,
    track_id: AnalysisSubjectID,
    features: PoseFeatures,
) -> None:
    """Cap nhat lich su motion cho 2 luong hip va shoulder.

    WHY:
        - Giu 2 history rieng giup direction on dinh hon so voi chi nhin 1 diem.
    """
    rule_input = build_direction_input(track_id, features, analyzer)
    apply_direction_history_typed(rule_input, analyzer)


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
    history: tuple[int, ...],
    point_prefix: str,
    monitor_point: Point | None,
) -> PointDirectionResult:
    if monitor_point is None:
        return PointDirectionResult(
            direction="UNKNOWN",
            dy=None,
            direction_dy=None,
            direction_reason=f"{point_prefix}_MONITOR_POINT_MISSING",
        )

    if len(history) < SETTINGS.direction.min_frames:
        return PointDirectionResult(
            direction="ANALYZING",
            dy=None,
            direction_dy=None,
            direction_reason=f"{point_prefix}_HISTORY_NOT_ENOUGH_FRAMES",
        )

    dy = int(history[-1] - history[0])
    direction, direction_reason = _map_dy_to_direction(dy, point_prefix)
    return PointDirectionResult(
        direction=direction,
        dy=dy,
        direction_dy=dy,
        direction_reason=direction_reason,
    )


def _is_valid_movement_direction(direction: str) -> bool:
    return direction in ("UP", "DOWN")


def _is_usable_monitor_direction(direction: str) -> bool:
    return direction in ("UP", "DOWN", "IDLE")


def _build_direction_result(
    *,
    direction: str,
    final_direction: str,
    hip_direction: str,
    shoulder_direction: str,
    direction_source: str,
    direction_reason: str,
    dy: int | None,
    direction_dy: int | None,
) -> DirectionResult:
    return DirectionResult(
        direction=direction,
        final_direction=final_direction,
        hip_direction=hip_direction,
        shoulder_direction=shoulder_direction,
        direction_source=direction_source,
        direction_reason=direction_reason,
        dy=dy,
        direction_dy=direction_dy,
    )


def update_direction_typed(rule_input: DirectionInput) -> DirectionResult:
    hip_direction_info = compute_direction_from_point_history(
        rule_input.hip_history,
        "HIP",
        rule_input.hip_monitor_point,
    )
    shoulder_direction_info = compute_direction_from_point_history(
        rule_input.shoulder_history,
        "SHOULDER",
        rule_input.shoulder_monitor_point,
    )

    hip_direction = hip_direction_info.direction
    shoulder_direction = shoulder_direction_info.direction
    hip_dy = hip_direction_info.direction_dy
    shoulder_dy = shoulder_direction_info.direction_dy

    if (
        _is_valid_movement_direction(hip_direction)
        and _is_valid_movement_direction(shoulder_direction)
        and hip_direction == shoulder_direction
    ):
        agreed_dy = hip_dy if hip_dy is not None else shoulder_dy
        return _build_direction_result(
            direction=hip_direction,
            final_direction=hip_direction,
            hip_direction=hip_direction,
            shoulder_direction=shoulder_direction,
            direction_source="HIP_SHOULDER_AGREE",
            direction_reason="HIP_SHOULDER_AGREE",
            dy=agreed_dy,
            direction_dy=agreed_dy,
        )

    if (
        _is_valid_movement_direction(hip_direction)
        and _is_valid_movement_direction(shoulder_direction)
        and hip_direction != shoulder_direction
    ):
        final_direction = (
            rule_input.last_valid_direction
            if rule_input.last_valid_direction in ("UP", "DOWN")
            else "ANALYZING"
        )
        return _build_direction_result(
            direction=final_direction,
            final_direction=final_direction,
            hip_direction=hip_direction,
            shoulder_direction=shoulder_direction,
            direction_source="HIP_SHOULDER_CONFLICT",
            direction_reason="HIP_SHOULDER_CONFLICT",
            dy=None,
            direction_dy=None,
        )

    if hip_direction == shoulder_direction and _is_usable_monitor_direction(hip_direction):
        agreed_dy = hip_dy if hip_dy is not None else shoulder_dy
        return _build_direction_result(
            direction=hip_direction,
            final_direction=hip_direction,
            hip_direction=hip_direction,
            shoulder_direction=shoulder_direction,
            direction_source="HIP_SHOULDER_AGREE",
            direction_reason="HIP_SHOULDER_AGREE",
            dy=agreed_dy,
            direction_dy=agreed_dy,
        )

    if _is_usable_monitor_direction(hip_direction) and not _is_usable_monitor_direction(
        shoulder_direction
    ):
        return _build_direction_result(
            direction=hip_direction,
            final_direction=hip_direction,
            hip_direction=hip_direction,
            shoulder_direction=shoulder_direction,
            direction_source="HIP_ONLY",
            direction_reason=hip_direction_info.direction_reason,
            dy=hip_dy,
            direction_dy=hip_dy,
        )

    if _is_usable_monitor_direction(shoulder_direction) and not _is_usable_monitor_direction(
        hip_direction
    ):
        return _build_direction_result(
            direction=shoulder_direction,
            final_direction=shoulder_direction,
            hip_direction=hip_direction,
            shoulder_direction=shoulder_direction,
            direction_source="SHOULDER_ONLY",
            direction_reason=shoulder_direction_info.direction_reason,
            dy=shoulder_dy,
            direction_dy=shoulder_dy,
        )

    if _is_valid_movement_direction(hip_direction) and shoulder_direction == "IDLE":
        return _build_direction_result(
            direction=hip_direction,
            final_direction=hip_direction,
            hip_direction=hip_direction,
            shoulder_direction=shoulder_direction,
            direction_source="HIP_ONLY",
            direction_reason=hip_direction_info.direction_reason,
            dy=hip_dy,
            direction_dy=hip_dy,
        )

    if _is_valid_movement_direction(shoulder_direction) and hip_direction == "IDLE":
        return _build_direction_result(
            direction=shoulder_direction,
            final_direction=shoulder_direction,
            hip_direction=hip_direction,
            shoulder_direction=shoulder_direction,
            direction_source="SHOULDER_ONLY",
            direction_reason=shoulder_direction_info.direction_reason,
            dy=shoulder_dy,
            direction_dy=shoulder_dy,
        )

    return _build_direction_result(
        direction="ANALYZING",
        final_direction="ANALYZING",
        hip_direction=hip_direction,
        shoulder_direction=shoulder_direction,
        direction_source="NO_VALID_MONITOR_DIRECTION",
        direction_reason="NO_VALID_MONITOR_DIRECTION",
        dy=None,
        direction_dy=None,
    )


def update_direction(
    analyzer: BehaviorAnalyzer,
    track_id: AnalysisSubjectID,
    features: PoseFeatures,
) -> DirectionAnalysisUpdate:
    """Tong hop direction tu 2 history hip/shoulder.

    OUTPUT:
        - `direction`, `final_direction`, `direction_source`, `direction_reason`.

    WHY:
        - Neu 2 luong mau thuan, runtime uu tien giu `last_valid_direction`
          thay vi flip lien tuc tung frame.
    """
    rule_input = build_direction_input(track_id, features, analyzer)
    result = update_direction_typed(rule_input)
    return result.to_analysis_update()
