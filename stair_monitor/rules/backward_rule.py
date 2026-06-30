from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypedDict

from stair_monitor.config.settings import SETTINGS
from stair_monitor.rules.handrail_rule import is_back_to_camera, is_front_to_camera
from stair_monitor.common.types import AnalysisSubjectID

if TYPE_CHECKING:
    from stair_monitor.core.analyzer import BehaviorAnalyzer

# File nay xac nhan `Di Lui` tu direction va body facing.
# WHY: Chi nhin motion la khong du; cung chieu di do nhung huong than nguoi khac nhau se cho nghiep vu khac nhau.


class BackwardAnalysisUpdate(TypedDict):
    backward_raw: bool
    backward_hits: int
    backward_confirmed: bool
    backward_mapping_source: str
    backward_reason: str


@dataclass(frozen=True, slots=True)
class BackwardInput:
    subject_id: AnalysisSubjectID
    direction: str
    body_facing: str
    body_facing_confidence: float
    body_facing_evidence_count: int
    hip_pair_valid: bool
    shoulder_pair_valid: bool
    ear_pair_valid: bool
    head_valid: bool


@dataclass(frozen=True, slots=True)
class BackwardResult:
    backward_raw: bool
    backward_hits: int
    backward_confirmed: bool
    backward_mapping_source: str
    backward_reason: str

    def to_analysis_update(self) -> BackwardAnalysisUpdate:
        return {
            "backward_raw": self.backward_raw,
            "backward_hits": self.backward_hits,
            "backward_confirmed": self.backward_confirmed,
            "backward_mapping_source": self.backward_mapping_source,
            "backward_reason": self.backward_reason,
        }


class BackwardHistoryUpdater(Protocol):
    def _update_backward_history(
        self,
        track_id: AnalysisSubjectID,
        backward_raw: bool | None,
    ) -> tuple[int, bool]:
        ...


@dataclass(frozen=True, slots=True)
class BackwardFrameDecision:
    backward_raw: bool
    backward_history_value: bool | None
    backward_reason: str


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


def build_backward_input(
    subject_id: AnalysisSubjectID,
    direction: str,
    body_facing: str,
    body_facing_confidence: float,
    body_facing_evidence_count: int,
    hip_pair_valid: bool,
    shoulder_pair_valid: bool,
    ear_pair_valid: bool,
    head_valid: bool,
) -> BackwardInput:
    return BackwardInput(
        subject_id=subject_id,
        direction=direction,
        body_facing=body_facing,
        body_facing_confidence=body_facing_confidence,
        body_facing_evidence_count=body_facing_evidence_count,
        hip_pair_valid=hip_pair_valid,
        shoulder_pair_valid=shoulder_pair_valid,
        ear_pair_valid=ear_pair_valid,
        head_valid=head_valid,
    )


def _is_body_facing_reliable(rule_input: BackwardInput) -> bool:
    return (
        rule_input.body_facing in ("FRONT_TO_CAMERA", "BACK_TO_CAMERA")
        and rule_input.body_facing_confidence > 0.5
        and rule_input.body_facing_evidence_count >= SETTINGS.backward.min_valid_evidence
    )


def _is_upper_body_occluded(rule_input: BackwardInput) -> bool:
    return not (
        rule_input.hip_pair_valid
        and rule_input.shoulder_pair_valid
        and (rule_input.ear_pair_valid or rule_input.head_valid)
    )


def _resolve_unreliable_backward_decision(
    rule_input: BackwardInput,
) -> BackwardFrameDecision:
    if (
        _is_upper_body_occluded(rule_input)
        or rule_input.body_facing_evidence_count < SETTINGS.backward.min_valid_evidence
    ):
        backward_reason = "UNKNOWN_OCCLUDED"
    else:
        backward_reason = "UNKNOWN_NOT_ENOUGH_EVIDENCE"

    return BackwardFrameDecision(
        backward_raw=False,
        backward_history_value=None,
        backward_reason=backward_reason,
    )


def _resolve_reliable_backward_decision(
    rule_input: BackwardInput,
) -> BackwardFrameDecision:
    if SETTINGS.camera.use_current_camera_angle:
        if rule_input.direction == "DOWN" and is_front_to_camera(rule_input.body_facing):
            return BackwardFrameDecision(
                backward_raw=True,
                backward_history_value=True,
                backward_reason="OK_DOWN_FRONT",
            )
        if rule_input.direction == "UP" and is_back_to_camera(rule_input.body_facing):
            return BackwardFrameDecision(
                backward_raw=True,
                backward_history_value=True,
                backward_reason="OK_UP_BACK",
            )
        return BackwardFrameDecision(
            backward_raw=False,
            backward_history_value=False,
            backward_reason="NORMAL_DIRECTION_FACING",
        )

    if rule_input.direction == "DOWN" and is_back_to_camera(rule_input.body_facing):
        return BackwardFrameDecision(
            backward_raw=True,
            backward_history_value=True,
            backward_reason="NEW_CAMERA_DOWN_BACK_IS_BACKWARD",
        )
    if rule_input.direction == "UP" and is_front_to_camera(rule_input.body_facing):
        return BackwardFrameDecision(
            backward_raw=True,
            backward_history_value=True,
            backward_reason="NEW_CAMERA_UP_FRONT_IS_BACKWARD",
        )
    return BackwardFrameDecision(
        backward_raw=False,
        backward_history_value=False,
        backward_reason="NORMAL_DIRECTION_FACING",
    )


def evaluate_backward_typed(
    rule_input: BackwardInput,
    history_updater: BackwardHistoryUpdater,
) -> BackwardResult:
    if not _is_body_facing_reliable(rule_input):
        frame_decision = _resolve_unreliable_backward_decision(rule_input)
    else:
        frame_decision = _resolve_reliable_backward_decision(rule_input)

    backward_hits, backward_confirmed = history_updater._update_backward_history(
        rule_input.subject_id,
        frame_decision.backward_history_value,
    )
    return BackwardResult(
        backward_raw=frame_decision.backward_raw,
        backward_hits=backward_hits,
        backward_confirmed=backward_confirmed,
        backward_mapping_source=(
            "CURRENT_CAMERA"
            if SETTINGS.camera.use_current_camera_angle
            else "BOTTOM_STAIR_CAMERA"
        ),
        backward_reason=frame_decision.backward_reason,
    )


def evaluate_backward(
    analyzer: BehaviorAnalyzer,
    track_id: AnalysisSubjectID,
    direction: str,
    body_facing: str,
    body_facing_confidence: float,
    body_facing_evidence_count: int,
    hip_pair_valid: bool,
    shoulder_pair_valid: bool,
    ear_pair_valid: bool,
    head_valid: bool,
) -> BackwardAnalysisUpdate:
    """Danh gia backward raw va backward confirmed cho 1 frame."""
    rule_input = build_backward_input(
        subject_id=track_id,
        direction=direction,
        body_facing=body_facing,
        body_facing_confidence=body_facing_confidence,
        body_facing_evidence_count=body_facing_evidence_count,
        hip_pair_valid=hip_pair_valid,
        shoulder_pair_valid=shoulder_pair_valid,
        ear_pair_valid=ear_pair_valid,
        head_valid=head_valid,
    )
    result = evaluate_backward_typed(rule_input, analyzer)
    return result.to_analysis_update()
