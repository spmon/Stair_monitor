from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Literal, TypeAlias, TypedDict

Point: TypeAlias = tuple[int, int]
BBox: TypeAlias = tuple[int, int, int, int]
FloatPoint: TypeAlias = tuple[float, float]
ColorBGR: TypeAlias = tuple[int, int, int]
Numeric: TypeAlias = int | float
KeypointsArray: TypeAlias = NDArray[np.float32] | NDArray[np.float64]
BBoxArray: TypeAlias = NDArray[np.float32] | NDArray[np.float64]
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
DebugInfoDict: TypeAlias = dict[str, object]
KeypointValidDict: TypeAlias = dict[str, bool]
PerfStats: TypeAlias = dict[str, float]

JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]
JsonDict: TypeAlias = dict[str, JsonValue]

PointListJson: TypeAlias = list[list[int]]


class CameraConfigDict(TypedDict, total=False):
    ROI: PointListJson
    CENTER_LINE: PointListJson
    HANDRAIL_LEFT_LINE: PointListJson
    HANDRAIL_RIGHT_LINE: PointListJson
    HANDRAIL_LEFT_POLY: PointListJson
    HANDRAIL_RIGHT_POLY: PointListJson
    STEP_BOTTOM: PointListJson
    STEP_TOP: PointListJson


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
    motion_point: Point | None
    feet_point: Point | None
    feet_point_source: str
    real_feet_point: Point | None
    real_feet_source: str
    virtual_feet_from_shoulder_hip: Point | None
    virtual_feet_from_shoulder_hip_source: str
    virtual_feet_from_two_shoulders: Point | None
    virtual_feet_from_two_shoulders_source: str
    selected_feet_point: Point | None
    selected_feet_source: str
    shoulder_hip_feet_dx: int | None
    shoulder_hip_feet_dy: int | None
    shoulder_hip_feet_distance: float | None
    shoulder_hip_feet_compare_available: bool
    two_shoulders_feet_dx: int | None
    two_shoulders_feet_dy: int | None
    two_shoulders_feet_distance: float | None
    two_shoulders_feet_compare_available: bool
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
    track_id: int
    dy: Numeric | None
    direction_dy: Numeric | None
    lane_v: Numeric | None
    direction: Direction
    direction_reason: str
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
    ankle_valid_count: int
    feet_reliable: bool
    bbox_height: int | None
    left_ankle_valid: bool
    right_ankle_valid: bool
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
    hold_status_correct: HoldStatusLabel
    hold_status_wrong: HoldStatusLabel
    dist_wrist: Numeric
    wrist_side: str
    best_wrist: str
    best_wrist_point: Point | None
    dist_correct: Numeric
    dist_wrong: Numeric
    seg_dist_correct: Numeric | None
    seg_dist_wrong: Numeric | None
    t_correct: Numeric | None
    t_wrong: Numeric | None
    wrist_side_correct: str
    wrist_side_wrong: str
    best_wrist_correct: str
    best_wrist_wrong: str
    best_wrist_correct_point: Point | None
    best_wrist_wrong_point: Point | None
    correct_line_name: str
    wrong_line_name: str
    correct_rule: str
    wrong_rule: str
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
