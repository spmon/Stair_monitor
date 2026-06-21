from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from typing import Literal, TypeAlias, TypedDict

Point: TypeAlias = tuple[int, int]
BBox: TypeAlias = tuple[int, int, int, int]
FloatPoint: TypeAlias = tuple[float, float]
ColorBGR: TypeAlias = tuple[int, int, int]
Numeric: TypeAlias = int | float
FrameArray: TypeAlias = NDArray[np.uint8]
KeypointsArray: TypeAlias = NDArray[np.float32] | NDArray[np.float64]
BBoxArray: TypeAlias = NDArray[np.float32] | NDArray[np.float64]
PointList: TypeAlias = Sequence[Point]
LinePoints: TypeAlias = PointList | NDArray[np.int32] | NDArray[np.float32] | NDArray[np.float64]
PersonUID: TypeAlias = int
AnalysisSubjectID: TypeAlias = int | str
Direction: TypeAlias = Literal["UP", "DOWN", "IDLE", "ANALYZING", "UNKNOWN"]
LaneSide: TypeAlias = Literal["LEFT", "RIGHT", "CENTER", "UNKNOWN"]
BodyFacingLabel: TypeAlias = Literal["FRONT_TO_CAMERA", "BACK_TO_CAMERA", "UNKNOWN"]
ArmSideOrderLabel: TypeAlias = Literal[
    "LEFT_ARM_ON_IMAGE_LEFT",
    "LEFT_ARM_ON_IMAGE_RIGHT",
    "UNKNOWN",
]
DirectionLabel: TypeAlias = Direction | Literal["NA"]
LaneSideLabel: TypeAlias = Literal["LEFT", "RIGHT", "ON_LINE", "UNKNOWN"]
HoldStatusLabel: TypeAlias = Literal[
    "CORRECT",
    "WRONG_SIDE",
    "NONE",
    "UNKNOWN",
    "ANALYZING",
    "OUTSIDE",
]
HandrailStatusLabel: TypeAlias = Literal[
    "OK",
    "KHONG_VIN",
    "VIN_SAI_BEN",
    "SKIP_BACKWARD",
    "WAIT_HOLD_CONFIRM",
    "OUTSIDE",
    "NOT_EVALUATED",
]
IdentityStatusLabel: TypeAlias = Literal[
    "CANDIDATE",
    "NEW",
    "ACTIVE",
    "RELINKED",
    "TEMP_REID_CANDIDATE",
    "AMBIGUOUS_REID",
    "LOST",
    "EXITED",
]
PersonSessionStatusLabel: TypeAlias = Literal[
    "CANDIDATE",
    "ACTIVE",
    "LOST",
    "COMPLETED",
]
PersonSessionLifecycleLabel: TypeAlias = Literal[
    "CANDIDATE_NO_FEET",
    "OCCLUDED_ENTRY_CANDIDATE",
    "CANDIDATE_WAIT_ENTER",
    "CANDIDATE_OUTSIDE",
    "UNASSIGNED_INSIDE_CANDIDATE",
    "ACTIVE_INSIDE",
    "LOST_INSIDE",
    "TEMP_REID_CANDIDATE",
    "AMBIGUOUS_REID",
    "EXITED",
]
IdentityEntryReasonLabel: TypeAlias = Literal[
    "NONE",
    "CONFIRMED_ENTER",
    "OCCLUDED_ENTRY",
]
ReLinkStateLabel: TypeAlias = Literal[
    "NONE",
    "RELINK_NO_MATCH",
    "RELINK_WAIT_MORE_FRAMES",
    "RELINK_CONFIRMED",
    "RELINK_REJECTED_AMBIGUOUS",
    "RELINK_REJECTED_ORDER_CONFLICT",
]
CountEventReasonLabel: TypeAlias = Literal[
    "NO_COUNT_EVENT",
    "ENTER_COUNTED_BY_FEET",
    "COUNT_ENTER_OCCLUDED",
    "EXIT_COUNTED_BY_FEET",
    "ENTER_ALREADY_COUNTED",
    "EXIT_ALREADY_COUNTED",
    "TRACK_LOST_NO_EXIT_COUNT",
    "BBOX_OUTSIDE_IGNORED_NO_EXIT_COUNT",
    "RELINK_NO_RECOUNT",
]
DebugFocusMode: TypeAlias = Literal[
    "all",
    "none",
    "handrail",
    "carry",
    "lane",
    "backward",
    "standing",
    "two_step",
    "feet",
    "identity",
]
TextAnchor: TypeAlias = Literal["left", "right"]
DebugInfoDict: TypeAlias = dict[str, object]
KeypointValidDict: TypeAlias = dict[str, bool]
PerfStats: TypeAlias = dict[str, float]
WarningList: TypeAlias = list[str]

JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
JsonDict: TypeAlias = dict[str, JsonValue]

PointListJson: TypeAlias = list[list[int]]


class StepLineJson(TypedDict):
    id: int
    p1: list[int]
    p2: list[int]


StepLineListJson: TypeAlias = list[StepLineJson]


class CameraConfigDict(TypedDict, total=False):
    ROI: PointListJson
    CENTER_LINE: PointListJson
    HANDRAIL_LEFT_LINE: PointListJson
    HANDRAIL_RIGHT_LINE: PointListJson
    HANDRAIL_LEFT_POLY: PointListJson
    HANDRAIL_RIGHT_POLY: PointListJson
    STEP_BOTTOM: PointListJson
    STEP_TOP: PointListJson
    STEP_LINES: StepLineListJson


class PoseFeatures(TypedDict, total=False):
    bbox: BBox | None
    bbox_center: Point | None
    bbox_bottom_center: Point | None
    bbox_height: int | None
    nose: Point | None
    left_eye: Point | None
    right_eye: Point | None
    left_ear: Point | None
    right_ear: Point | None
    left_ankle: Point | None
    right_ankle: Point | None
    left_ankle_conf: float
    right_ankle_conf: float
    left_wrist: Point | None
    right_wrist: Point | None
    left_elbow: Point | None
    right_elbow: Point | None
    left_shoulder: Point | None
    right_shoulder: Point | None
    left_hip: Point | None
    right_hip: Point | None
    hip_center: Point | None
    shoulder_center: Point | None
    torso_center: Point | None
    head_center: Point | None
    monitor_point_hip: Point | None
    monitor_point_hip_source: str
    monitor_point_shoulder: Point | None
    monitor_point_shoulder_source: str
    motion_point: Point | None
    feet_point: Point | None
    feet_point_source: str
    feet_available: bool
    feet_unavailable_reason: str
    real_feet_point: Point | None
    real_feet_source: str
    sh_hip_visible_shoulder_count: int
    sh_hip_visible_hip_count: int
    sh_hip_selected_pair: str
    sh_hip_virtual_feet_point: Point | None
    sh_hip_virtual_feet_source: str
    sh_hip_anchor_shoulder_point: Point | None
    sh_hip_anchor_hip_point: Point | None
    virtual_feet_from_shoulder_hip: Point | None
    virtual_feet_from_shoulder_hip_source: str
    selected_feet_point: Point | None
    selected_feet_source: str
    shoulder_hip_feet_dx: int | None
    shoulder_hip_feet_dy: int | None
    shoulder_hip_feet_distance: float | None
    shoulder_hip_feet_compare_available: bool
    shoulder_hip_scale_used: float
    inside_feet_point: Point | None
    inside_feet_point_source: str
    ankle_valid_count: int
    feet_reliable: bool
    left_arm_angle: float | None
    right_arm_angle: float | None
    body_facing: BodyFacingLabel
    body_facing_confidence: float
    body_facing_evidence_count: int
    body_facing_front_votes: int
    body_facing_back_votes: int
    body_facing_reason: str
    hip_pair_valid: bool
    shoulder_pair_valid: bool
    ear_pair_valid: bool
    head_valid: bool
    arm_side_order: ArmSideOrderLabel
    keypoint_valid: KeypointValidDict
    torso_box: BBox | None


class AnalysisResult(TypedDict, total=False):
    status: str
    display_status: str
    color: ColorBGR
    track_id: AnalysisSubjectID
    person_uid: PersonUID
    person_uid_label: str
    analysis_subject_id: str
    analysis_subject_label: str
    merged_from_analysis_subject_id: str
    yolo_track_id: int | None
    previous_yolo_track_id: int | None
    identity_status: IdentityStatusLabel
    session_status: PersonSessionStatusLabel
    session_lifecycle: PersonSessionLifecycleLabel
    identity_debug: str
    identity_feet_source: str
    identity_feet_reason: str
    identity_gate_reason: str
    identity_inside_test: str
    identity_entry_reason: IdentityEntryReasonLabel
    identity_outside_proof: bool
    identity_enter_confirm_hits: int
    identity_enter_confirm_target: int
    has_active_person_id: bool
    relink_score: float | None
    relink_frame_gap: int
    relink_score_gap: float | None
    relink_best_candidate: str
    relink_second_candidate: str
    relink_state: ReLinkStateLabel
    total_entered_count: int
    total_confirmed_entered_count: int
    total_occluded_entered_count: int
    total_exited_count: int
    entered_count: int
    exited_count: int
    active_inside_count: int
    lost_inside_count: int
    active_or_lost_inside_count: int
    current_person_id: str
    person_lifecycle_state: PersonSessionLifecycleLabel
    has_counted_enter: bool
    has_counted_exit: bool
    count_event_reason: CountEventReasonLabel
    occluded_entry_candidate_age_frames: int
    occluded_entry_age_target: int
    occluded_entry_inside_frames: int
    occluded_entry_inside_target: int
    occluded_entry_motion_frames: int
    occluded_entry_motion_target: int
    occluded_entry_block_reason: str
    occluded_entry_nearest_ghost_score: float | None
    occluded_entry_nearest_active_iou: float | None
    occluded_entry_nearest_active_distance: float | None
    dy: Numeric | None
    direction_dy: Numeric | None
    lane_v: Numeric | None
    direction: Direction
    final_direction: Direction
    hip_direction: Direction
    shoulder_direction: Direction
    direction_source: str
    direction_reason: str
    monitor_point_hip: Point | None
    monitor_point_hip_source: str
    monitor_point_shoulder: Point | None
    monitor_point_shoulder_source: str
    use_current_camera_angle: bool
    camera_angle_profile: str
    lane_raw: bool | None
    lane_hits: int
    lane_conf: bool
    wrong_lane: bool
    lane_status: str
    lane_reason: str
    lane_source: str
    lane_direction: Direction
    lane_side_value: Numeric | None
    lane_side_label: LaneSideLabel
    correct_lane_side: LaneSideLabel
    lane_mapping_source: str
    p_lane_source: str
    lane_missing_feet_grace_left: int
    foot_lane_side: Numeric | None
    inside_stairs: bool
    inside_feet_point: Point | None
    feet_point_source: str
    inside_feet_point_source: str
    feet_available: bool
    feet_unavailable_reason: str
    sh_hip_visible_shoulder_count: int
    sh_hip_visible_hip_count: int
    sh_hip_selected_pair: str
    sh_hip_virtual_feet_point: Point | None
    sh_hip_virtual_feet_source: str
    ankle_valid_count: int
    feet_reliable: bool
    bbox_height: int | None
    left_ankle_valid: bool
    right_ankle_valid: bool
    left_ankle_point: Point | None
    right_ankle_point: Point | None
    left_ankle_raw_point: Point | None
    right_ankle_raw_point: Point | None
    left_ankle_step_point: Point | None
    right_ankle_step_point: Point | None
    left_ankle_conf: float
    right_ankle_conf: float
    left_current_ankle_step: int | None
    right_current_ankle_step: int | None
    left_raw_ankle_step: int | None
    right_raw_ankle_step: int | None
    left_raw_step: int | None
    right_raw_step: int | None
    left_filtered_step: int | None
    right_filtered_step: int | None
    left_last_valid_step: int | None
    right_last_valid_step: int | None
    left_step_filter_reason: str
    right_step_filter_reason: str
    left_adjusted_step: int | None
    right_adjusted_step: int | None
    ankle_step_offset_x: int
    ankle_step_offset_y: int
    ankle_step_offset_direction: str
    left_step_reason: str
    right_step_reason: str
    left_step_nearest_band_id: int | None
    right_step_nearest_band_id: int | None
    left_step_nearest_distance: float | None
    right_step_nearest_distance: float | None
    left_step_is_inside: bool
    right_step_is_inside: bool
    left_step_is_near_boundary: bool
    right_step_is_near_boundary: bool
    left_step_index: int | None
    right_step_index: int | None
    foot_gap: int | None
    step_gap: int | None
    left_planted_step: int | None
    right_planted_step: int | None
    planted_step_gap: int | None
    left_is_planted: bool
    right_is_planted: bool
    left_ankle_speed: float | None
    right_ankle_speed: float | None
    left_foot_step_reason: str
    right_foot_step_reason: str
    left_foot_state_label: str
    right_foot_state_label: str
    left_foot_landed: bool
    right_foot_landed: bool
    two_step_skip_check_available: bool
    two_step_skip_reason: str
    two_step_skip_confirmed: bool
    left_foot_in: bool
    right_foot_in: bool
    valid_foot_count: int
    track_zone_state: str
    inside_raw_by_feet: bool | None
    inside_reason: str
    inside_grace_left: int
    holding: bool
    holding_raw: bool
    hold_status: HoldStatusLabel
    hold_raw_status: HoldStatusLabel
    hold_confirmed_status: HoldStatusLabel
    hold_correct_hits: int
    hold_wrong_side_hits: int
    hold_none_hits: int
    hold_unknown_hits: int
    hold_not_hold_evidence_hits: int
    not_hold_by_evidence: bool
    holding_correct_raw: bool
    holding_wrong_raw: bool
    left_hand_claim: str
    right_hand_claim: str
    left_hold_claim_hits: int
    right_hold_claim_hits: int
    left_carry_claim_hits: int
    right_carry_claim_hits: int
    left_hold_raw_before_claim: bool
    right_hold_raw_before_claim: bool
    left_hold_raw_after_claim: bool
    right_hold_raw_after_claim: bool
    left_carry_raw_before_claim: bool
    right_carry_raw_before_claim: bool
    left_carry_raw_after_claim: bool
    right_carry_raw_after_claim: bool
    left_wrist_valid: bool
    right_wrist_valid: bool
    left_wrist_hit: bool
    right_wrist_hit: bool
    left_wrist_hit_count: int
    right_wrist_hit_count: int
    left_wrist_miss_count: int
    right_wrist_miss_count: int
    left_wrist_confirm_required: int
    right_wrist_confirm_required: int
    left_wrist_distance_to_left_rail: float | None
    left_wrist_distance_to_right_rail: float | None
    right_wrist_distance_to_left_rail: float | None
    right_wrist_distance_to_right_rail: float | None
    left_wrist_nearest_distance: float | None
    right_wrist_nearest_distance: float | None
    left_wrist_nearest_rail: str
    right_wrist_nearest_rail: str
    left_wrist_nearest_point: Point | None
    right_wrist_nearest_point: Point | None
    left_holding: bool
    right_holding: bool
    handrail_status: HandrailStatusLabel
    handrail_reason: str
    handrail_debug_reason: str
    is_carrying: bool
    carry_type: str
    left_arm_angle: Numeric | None
    right_arm_angle: Numeric | None
    left_wrist_in_torso: bool
    right_wrist_in_torso: bool
    body_scale: Numeric | None
    shoulder_width: Numeric | None
    torso_height: Numeric | None
    wrist_dx: Numeric | None
    wrist_dx_threshold: Numeric | None
    wrist_dy: Numeric | None
    wrist_dy_threshold: Numeric | None
    wrist_distance: Numeric | None
    left_bent: bool
    right_bent: bool
    wrists_close: bool
    any_wrist_in_torso: bool
    both_wrist_in_torso: bool
    front_carry: bool
    front_carry_raw: bool
    front_carry_hits: int
    front_carry_two_hand_raw: bool
    front_carry_two_hand_hits: int
    front_carry_one_arm_raw: bool
    front_carry_one_arm_hits: int
    front_carry_confirmed: bool
    left_carry: bool
    right_carry: bool
    left_carry_allowed: bool
    right_carry_allowed: bool
    left_carry_evidence: bool
    right_carry_evidence: bool
    left_carry_score: Numeric | None
    right_carry_score: Numeric | None
    one_hand_carry_side: str
    carry_reason: str
    carrying_arm: str
    p_lane: Point | None
    p_motion: Point | None
    body_facing: BodyFacingLabel
    body_facing_confidence: float
    body_facing_evidence_count: int
    body_facing_front_votes: int
    body_facing_back_votes: int
    body_facing_reason: str
    hip_pair_valid: bool
    shoulder_pair_valid: bool
    ear_pair_valid: bool
    head_valid: bool
    arm_side_order: ArmSideOrderLabel
    backward_raw: bool
    backward_hits: int
    backward_confirmed: bool
    backward_mapping_source: str
    handrail_mapping_source: str
    backward_reason: str
    standing_raw: bool
    standing_hits: int
    standing_still_confirmed: bool
    standing_motion_range: float | None
    standing_len: int
    warnings: list[str]
    perf: PerfStats | None
    debug_info: DebugInfoDict | None


AnalysisLike: TypeAlias = AnalysisResult | DebugInfoDict
