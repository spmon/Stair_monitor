from __future__ import annotations

from dataclasses import dataclass, field

from stair_monitor.common.types import AnalysisSubjectID, Point


@dataclass
class FootStepState:
    candidate_step_index: int | None = None
    candidate_count: int = 0
    last_planted_step_index: int | None = None
    last_valid_step_index: int | None = None
    last_valid_step_frame: int | None = None
    filtered_step_index: int | None = None
    raw_step_index: int | None = None
    filter_reason: str = "NOT_EVALUATED"
    last_ankle_point: Point | None = None
    last_seen_frame: int | None = None
    is_planted: bool = False
    ankle_speed_px: float | None = None
    reason: str = "NOT_EVALUATED"
    step_history: list[int] = field(default_factory=list)


@dataclass
class AnalyzerState:
    hip_motion_history: dict[AnalysisSubjectID, list[int]] = field(default_factory=dict)
    shoulder_motion_history: dict[AnalysisSubjectID, list[int]] = field(default_factory=dict)
    inside_last_state: dict[AnalysisSubjectID, bool] = field(default_factory=dict)
    lane_history: dict[AnalysisSubjectID, list[bool | None]] = field(default_factory=dict)
    lane_last_state: dict[AnalysisSubjectID, dict[str, object]] = field(default_factory=dict)
    lane_last_seen: dict[AnalysisSubjectID, int] = field(default_factory=dict)
    hold_status_history: dict[AnalysisSubjectID, list[str]] = field(default_factory=dict)
    left_handrail_hit_history: dict[AnalysisSubjectID, list[bool]] = field(
        default_factory=dict
    )
    right_handrail_hit_history: dict[AnalysisSubjectID, list[bool]] = field(
        default_factory=dict
    )
    last_valid_direction: dict[AnalysisSubjectID, str] = field(default_factory=dict)
    last_valid_direction_frame: dict[AnalysisSubjectID, int] = field(
        default_factory=dict
    )
    hand_claim_state: dict[AnalysisSubjectID, dict[str, dict[str, int | str | None]]] = field(
        default_factory=dict
    )
    front_carry_history: dict[AnalysisSubjectID, list[bool]] = field(default_factory=dict)
    front_carry_one_arm_history: dict[AnalysisSubjectID, list[bool]] = field(default_factory=dict)
    backward_history: dict[AnalysisSubjectID, list[bool]] = field(default_factory=dict)
    standing_history: dict[AnalysisSubjectID, list[bool]] = field(default_factory=dict)
    two_step_skip_history: dict[AnalysisSubjectID, list[bool]] = field(default_factory=dict)
    left_foot_step_states: dict[AnalysisSubjectID, FootStepState] = field(default_factory=dict)
    right_foot_step_states: dict[AnalysisSubjectID, FootStepState] = field(default_factory=dict)
    standing_motion_history: dict[AnalysisSubjectID, list[tuple[int, int]]] = field(
        default_factory=dict
    )
    frame_index: int = -1
