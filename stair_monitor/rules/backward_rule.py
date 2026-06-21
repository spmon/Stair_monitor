from __future__ import annotations

from typing import TYPE_CHECKING

from stair_monitor.config.settings import SETTINGS
from stair_monitor.rules.handrail_rule import is_back_to_camera, is_front_to_camera

if TYPE_CHECKING:
    from stair_monitor.core.analyzer import BehaviorAnalyzer

# File nay xac nhan `Di Lui` tu direction va body facing.
# WHY: Chi nhin motion la khong du; cung chieu di do nhung huong than nguoi khac nhau se cho nghiep vu khac nhau.


def is_backward_by_direction_and_facing(direction: str, body_facing: str) -> bool:
    if SETTINGS.camera.use_current_camera_angle:
        return (
            (direction == "DOWN" and is_front_to_camera(body_facing))
            or (direction == "UP" and is_back_to_camera(body_facing))
        )
    return (
        (direction == "DOWN" and is_back_to_camera(body_facing))
        or (direction == "UP" and is_front_to_camera(body_facing))
    )


def evaluate_backward(
    analyzer: BehaviorAnalyzer,
    track_id: int,
    direction: str,
    body_facing: str,
    body_facing_confidence: float,
    body_facing_evidence_count: int,
    hip_pair_valid: bool,
    shoulder_pair_valid: bool,
    ear_pair_valid: bool,
    head_valid: bool,
) -> dict[str, object]:
    """Danh gia backward raw va backward confirmed cho 1 frame."""
    backward_history_value = False
    body_facing_reliable = (
        body_facing in ("FRONT_TO_CAMERA", "BACK_TO_CAMERA")
        and body_facing_confidence > 0.5
        and body_facing_evidence_count >= SETTINGS.backward.min_valid_evidence
    )
    upper_body_occluded = not (
        hip_pair_valid
        and shoulder_pair_valid
        and (ear_pair_valid or head_valid)
    )

    # WHY: Khi upper body bi che hoac body facing chua du evidence, history backward khong nen hoc tu du lieu mo ho.
    if not body_facing_reliable:
        backward_raw = False
        backward_history_value = None
        if (
            upper_body_occluded
            or body_facing_evidence_count < SETTINGS.backward.min_valid_evidence
        ):
            backward_reason = "UNKNOWN_OCCLUDED"
        else:
            backward_reason = "UNKNOWN_NOT_ENOUGH_EVIDENCE"
    else:
        if SETTINGS.camera.use_current_camera_angle:
            if direction == "DOWN" and is_front_to_camera(body_facing):
                backward_raw = True
                backward_history_value = True
                backward_reason = "OK_DOWN_FRONT"
            elif direction == "UP" and is_back_to_camera(body_facing):
                backward_raw = True
                backward_history_value = True
                backward_reason = "OK_UP_BACK"
            else:
                backward_raw = False
                backward_history_value = False
                backward_reason = "NORMAL_DIRECTION_FACING"
        else:
            if direction == "DOWN" and is_back_to_camera(body_facing):
                backward_raw = True
                backward_history_value = True
                backward_reason = "NEW_CAMERA_DOWN_BACK_IS_BACKWARD"
            elif direction == "UP" and is_front_to_camera(body_facing):
                backward_raw = True
                backward_history_value = True
                backward_reason = "NEW_CAMERA_UP_FRONT_IS_BACKWARD"
            else:
                backward_raw = False
                backward_history_value = False
                backward_reason = "NORMAL_DIRECTION_FACING"

    backward_hits, backward_confirmed = analyzer._update_backward_history(
        track_id,
        backward_history_value,
    )
    return {
        "backward_raw": backward_raw,
        "backward_hits": backward_hits,
        "backward_confirmed": backward_confirmed,
        "backward_mapping_source": (
            "CURRENT_CAMERA"
            if SETTINGS.camera.use_current_camera_angle
            else "BOTTOM_STAIR_CAMERA"
        ),
        "backward_reason": backward_reason,
    }
