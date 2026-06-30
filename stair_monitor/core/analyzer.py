from __future__ import annotations

import time
from collections.abc import Iterable
from typing import cast

import numpy as np

from stair_monitor.common.types import (
    AnalysisSubjectID,
    AnalysisResult,
    BBoxArray,
    CameraConfigDict,
    ColorBGR,
    PerfStats,
    KeypointsArray,
    Point,
    PoseFeatures,
)
from stair_monitor.config.settings import SETTINGS
from stair_monitor.core.person_context import PersonContext
from stair_monitor.output.result_builder import ResultBuilderMixin
from stair_monitor.rules.backward_rule import evaluate_backward
from stair_monitor.rules.carry_analysis import CarryAnalysisMixin, evaluate_carry
from stair_monitor.rules.direction_rule import apply_direction_history, update_direction
from stair_monitor.rules.handrail_rule import (
    HandrailAnalysisMixin,
    evaluate_handrail,
)
from stair_monitor.rules.inside_rule import (
    build_inside_stairs_input,
    evaluate_inside_stairs_typed,
)
from stair_monitor.rules.lane_rule import evaluate_lane_violation, get_camera_angle_profile
from stair_monitor.rules.standing_rule import evaluate_standing_still
from stair_monitor.state.behavior_history import BehaviorHistoryMixin
from stair_monitor.state.track_state import AnalyzerState
from stair_monitor.rules.two_step_rule import TwoStepAnalyzer, build_two_step_input
from stair_monitor.vision.geometry import extract_pose_features
from stair_monitor.vision.step_lines import (
    build_step_bands,
    normalize_step_lines,
)

AnalysisContext = dict[str, object]

# File nay la trung tam phan tich hanh vi cua ban Windows/demo.
# FLOW: features cua 1 nguoi -> inside/direction/lane/handrail/carry/backward/standing -> warnings/result dict.
# WHY: Tach analyzer ra khoi `test-cauthang.py` de file entry point chi con viec dieu phoi frame/model/render.

class BehaviorAnalyzer(
    BehaviorHistoryMixin,
    CarryAnalysisMixin,
    HandrailAnalysisMixin,
    ResultBuilderMixin,
):
    """Trung tam dieu phoi cac rule cho stair_monitor Windows/demo.

    INPUT:
        - Camera config da duoc load tu JSON.
        - Du lieu tung nguoi da qua `extract_pose_features()`.

    OUTPUT:
        - `AnalysisResult` chua status, warnings, color va cac field debug.

    WHY:
        - Moi rule can history rieng theo subject, nhung renderer va main loop
          lai can mot result dict cuoi cung de de ve/log.
    """

    def __init__(self, config: CameraConfigDict | dict[str, object]) -> None:
        # INPUT: `config` chua ROI cau thang, center line, handrail line va step lines.
        # WHY: Analyzer cache san cac hinh hoc nay de moi frame khong phai parse lai JSON.
        self.center_line = config.get("CENTER_LINE", [[0, 0], [0, 0]])

        self.left_line = np.array(
            config.get("HANDRAIL_LEFT_LINE", config.get("HANDRAIL_LEFT_POLY", [])),
            np.int32,
        )
        self.right_line = np.array(
            config.get("HANDRAIL_RIGHT_LINE", config.get("HANDRAIL_RIGHT_POLY", [])),
            np.int32,
        )

        step_bottom = config.get("STEP_BOTTOM", [[0, 0], [0, 0]])
        step_top = config.get("STEP_TOP", [[0, 0], [0, 0]])
        bottom_left = [
            step_bottom[0][0],
            step_bottom[0][1],
        ]
        bottom_right = [
            step_bottom[1][0],
            step_bottom[1][1],
        ]
        top_right = [
            step_top[1][0],
            step_top[1][1],
        ]
        top_left = [
            step_top[0][0],
            step_top[0][1],
        ]
        self.stairs_poly = np.array(
            [bottom_left, bottom_right, top_right, top_left],
            np.int32,
        )
        self.step_lines = normalize_step_lines(config.get("STEP_LINES"))
        self.step_bands = build_step_bands(self.step_lines)

        self.state = AnalyzerState()
        self.hip_motion_history = self.state.hip_motion_history
        self.shoulder_motion_history = self.state.shoulder_motion_history
        self.inside_last_state = self.state.inside_last_state
        self.lane_history = self.state.lane_history
        self.lane_last_state = self.state.lane_last_state
        self.lane_last_seen = self.state.lane_last_seen
        self.hold_status_history = self.state.hold_status_history
        self.left_handrail_hit_history = self.state.left_handrail_hit_history
        self.right_handrail_hit_history = self.state.right_handrail_hit_history
        self.last_valid_direction = self.state.last_valid_direction
        self.last_valid_direction_frame = self.state.last_valid_direction_frame
        self.hand_claim_state = self.state.hand_claim_state
        self.front_carry_history = self.state.front_carry_history
        self.front_carry_one_arm_history = self.state.front_carry_one_arm_history
        self.backward_history = self.state.backward_history
        self.standing_history = self.state.standing_history
        self.two_step_skip_history = self.state.two_step_skip_history
        self.left_foot_step_states = self.state.left_foot_step_states
        self.right_foot_step_states = self.state.right_foot_step_states
        self.standing_motion_history = self.state.standing_motion_history
        self.two_step_analyzer = TwoStepAnalyzer(self.step_bands, self.state)

    @property
    def frame_index(self) -> int:
        return self.state.frame_index

    @frame_index.setter
    def frame_index(self, value: int) -> None:
        self.state.frame_index = value

    def begin_frame(self) -> None:
        self.frame_index += 1

    @staticmethod
    def _record_perf(
        perf: PerfStats | None,
        key: str,
        start_time: float | None,
    ) -> None:
        if perf is None or start_time is None:
            return
        perf[key] = perf.get(key, 0.0) + (time.perf_counter() - start_time) * 1000.0

    def _resolve_pose_features(
        self,
        keypoints: KeypointsArray,
        box: BBoxArray | None,
        features: PoseFeatures | None,
    ) -> PoseFeatures:
        # FLOW: Neu caller chua truyen `features` thi analyzer tu extract lai.
        # WHY: Giu API mem deo, nhung van uu tien tai su dung features da co san de tranh tinh lap.
        return cast(
            PoseFeatures,
            features or extract_pose_features(keypoints, box),
        )

    def _build_person_context(
        self,
        track_id: AnalysisSubjectID,
        keypoints: KeypointsArray,
        box: BBoxArray | None,
        features: PoseFeatures,
        p_lane: Point | None,
        p_motion: Point | None,
    ) -> PersonContext:
        resolved_p_lane = p_lane
        if resolved_p_lane is None:
            resolved_p_lane = features.get("feet_point")

        resolved_p_motion = p_motion
        if resolved_p_motion is None:
            resolved_p_motion = features.get("motion_point")

        return PersonContext(
            track_id=track_id,
            keypoints=keypoints,
            box=box,
            features=features,
            p_lane=resolved_p_lane,
            p_motion=resolved_p_motion,
        )

    def _create_analysis_context(
        self,
        track_id: AnalysisSubjectID,
        person: PersonContext,
    ) -> AnalysisContext:
        features = person.features
        use_current_camera_angle = SETTINGS.camera.use_current_camera_angle
        camera_angle_profile = get_camera_angle_profile()
        body_facing = features.get("body_facing", "UNKNOWN")
        body_facing_confidence = float(
            features.get("body_facing_confidence", 0.0) or 0.0
        )
        body_facing_evidence_count = int(
            features.get("body_facing_evidence_count", 0) or 0
        )
        body_facing_front_votes = int(
            features.get("body_facing_front_votes", 0) or 0
        )
        body_facing_back_votes = int(
            features.get("body_facing_back_votes", 0) or 0
        )
        body_facing_reason = features.get("body_facing_reason", "UNKNOWN")
        hip_pair_valid = bool(features.get("hip_pair_valid", False))
        shoulder_pair_valid = bool(features.get("shoulder_pair_valid", False))
        ear_pair_valid = bool(features.get("ear_pair_valid", False))
        head_valid = bool(features.get("head_valid", False))
        arm_side_order = features.get("arm_side_order", "UNKNOWN")

        # DEBUG: `analysis_context` la bo nho tam giu moi field ma result_builder va renderer can doc.
        # WHY: Gom field vao 1 dict chung giup build result cuoi cung on dinh, khong phai truyen tay tung bien.
        return {
            "track_id": track_id,
            "p_lane": person.p_lane,
            "p_motion": person.p_motion,
            "use_current_camera_angle": use_current_camera_angle,
            "camera_angle_profile": camera_angle_profile,
            "lane_mapping_source": camera_angle_profile,
            "handrail_mapping_source": camera_angle_profile,
            "backward_mapping_source": camera_angle_profile,
            "direction": "ANALYZING",
            "final_direction": "ANALYZING",
            "hip_direction": "UNKNOWN",
            "shoulder_direction": "UNKNOWN",
            "direction_source": "NO_VALID_MONITOR_DIRECTION",
            "direction_reason": "ANALYZING",
            "dy": None,
            "direction_dy": None,
            "v": None,
            "lane_direction": "ANALYZING",
            "monitor_point_hip": features.get("monitor_point_hip"),
            "monitor_point_hip_source": features.get(
                "monitor_point_hip_source",
                "NO_HIP_CENTER",
            ),
            "monitor_point_shoulder": features.get("monitor_point_shoulder"),
            "monitor_point_shoulder_source": features.get(
                "monitor_point_shoulder_source",
                "NO_SHOULDER_CENTER",
            ),
            "feet_point_source": features.get("feet_point_source", "FEET_UNAVAILABLE"),
            "inside_feet_point": features.get("inside_feet_point"),
            "inside_feet_point_source": features.get(
                "inside_feet_point_source",
                "FEET_UNAVAILABLE",
            ),
            "feet_available": bool(features.get("feet_available", False)),
            "feet_unavailable_reason": str(
                features.get("feet_unavailable_reason", "NO_SHOULDER_NO_HIP")
            ),
            "sh_hip_visible_shoulder_count": int(
                features.get("sh_hip_visible_shoulder_count", 0) or 0
            ),
            "sh_hip_visible_hip_count": int(
                features.get("sh_hip_visible_hip_count", 0) or 0
            ),
            "sh_hip_selected_pair": str(features.get("sh_hip_selected_pair", "NONE")),
            "sh_hip_virtual_feet_point": features.get("sh_hip_virtual_feet_point"),
            "sh_hip_virtual_feet_source": str(
                features.get("sh_hip_virtual_feet_source", "FEET_UNAVAILABLE")
            ),
            "ankle_valid_count": int(features.get("ankle_valid_count", 0) or 0),
            "feet_reliable": bool(features.get("feet_reliable", False)),
            "bbox_height": features.get("bbox_height"),
            "left_ankle_point": features.get("left_ankle"),
            "right_ankle_point": features.get("right_ankle"),
            "left_ankle_raw_point": features.get("left_ankle"),
            "right_ankle_raw_point": features.get("right_ankle"),
            "left_ankle_step_point": features.get("left_ankle"),
            "right_ankle_step_point": features.get("right_ankle"),
            "left_ankle_conf": float(features.get("left_ankle_conf", 0.0) or 0.0),
            "right_ankle_conf": float(
                features.get("right_ankle_conf", 0.0) or 0.0
            ),
            "left_current_ankle_step": None,
            "right_current_ankle_step": None,
            "left_raw_ankle_step": None,
            "right_raw_ankle_step": None,
            "left_adjusted_step": None,
            "right_adjusted_step": None,
            "ankle_step_offset_x": 0,
            "ankle_step_offset_y": 0,
            "ankle_step_offset_direction": "UNKNOWN",
            "left_step_reason": "NOT_EVALUATED",
            "right_step_reason": "NOT_EVALUATED",
            "left_step_nearest_band_id": None,
            "right_step_nearest_band_id": None,
            "left_step_nearest_distance": None,
            "right_step_nearest_distance": None,
            "left_step_is_inside": False,
            "right_step_is_inside": False,
            "left_step_is_near_boundary": False,
            "right_step_is_near_boundary": False,
            "left_step_index": None,
            "right_step_index": None,
            "foot_gap": None,
            "step_gap": None,
            "left_planted_step": None,
            "right_planted_step": None,
            "planted_step_gap": None,
            "left_is_planted": False,
            "right_is_planted": False,
            "left_ankle_speed": None,
            "right_ankle_speed": None,
            "left_foot_step_reason": "NOT_EVALUATED",
            "right_foot_step_reason": "NOT_EVALUATED",
            "left_foot_state_label": "WAIT",
            "right_foot_state_label": "WAIT",
            "left_foot_landed": False,
            "right_foot_landed": False,
            "two_step_skip_check_available": False,
            "two_step_skip_status": "NOT_EVALUATED",
            "two_step_skip_reason": "NOT_EVALUATED",
            "two_step_skip_confirmed": False,
            "body_facing": body_facing,
            "body_facing_confidence": body_facing_confidence,
            "body_facing_evidence_count": body_facing_evidence_count,
            "body_facing_front_votes": body_facing_front_votes,
            "body_facing_back_votes": body_facing_back_votes,
            "body_facing_reason": body_facing_reason,
            "hip_pair_valid": hip_pair_valid,
            "shoulder_pair_valid": shoulder_pair_valid,
            "ear_pair_valid": ear_pair_valid,
            "head_valid": head_valid,
            "arm_side_order": arm_side_order,
            "left_wrist_valid": False,
            "right_wrist_valid": False,
            "left_wrist_hit": False,
            "right_wrist_hit": False,
            "left_wrist_hit_count": 0,
            "right_wrist_hit_count": 0,
            "left_wrist_miss_count": 0,
            "right_wrist_miss_count": 0,
            "left_wrist_confirm_required": self._get_left_handrail_confirm_required(),
            "right_wrist_confirm_required": self._get_right_handrail_confirm_required(),
            "left_wrist_distance_to_left_rail": None,
            "left_wrist_distance_to_right_rail": None,
            "right_wrist_distance_to_left_rail": None,
            "right_wrist_distance_to_right_rail": None,
            "left_wrist_nearest_distance": None,
            "right_wrist_nearest_distance": None,
            "left_wrist_nearest_rail": "NONE",
            "right_wrist_nearest_rail": "NONE",
            "left_wrist_nearest_point": None,
            "right_wrist_nearest_point": None,
            "left_holding": False,
            "right_holding": False,
            "handrail_status": "NOT_EVALUATED",
            "handrail_reason": "NOT_EVALUATED",
            "handrail_debug_reason": "NOT_EVALUATED",
            "warnings": [],
        }

    def _build_analysis_result(
        self,
        analysis_context: AnalysisContext,
        status: str,
        color: ColorBGR,
        perf: PerfStats | None,
        analyze_start: float | None,
        display_status: str = "",
        carry_info_override: dict[str, object] | None = None,
        **overrides: object,
    ) -> AnalysisResult:
        # OUTPUT: Moi duong return cuoi cung deu di qua helper nay de format result dict thong nhat.
        self._record_perf(perf, "analyze", analyze_start)
        return self._build_result_from_context(
            analysis_context,
            status=status,
            display_status=display_status,
            color=color,
            perf=perf,
            carry_info=carry_info_override,
            **overrides,
        )

    def _update_hold_context(
        self,
        track_id: AnalysisSubjectID,
        person: PersonContext,
        analysis_context: AnalysisContext,
        perf: PerfStats | None,
        hold_direction: str | None,
        skip_handrail: bool = False,
    ) -> dict[str, object]:
        # FLOW: Handrail duoc xu ly qua 2 tang:
        # 1. lay bang chung wrist/rail raw
        # 2. qua history + hand-claim de ra hold final on dinh hon.
        hold_start = time.perf_counter() if perf is not None else None
        hold_state, carry_pose, hand_claim_state = evaluate_handrail(
            self,
            track_id,
            hold_direction,
            person.keypoints,
            person.features,
        )
        self._record_perf(perf, "hold", hold_start)

        analysis_context.update(
            {
                "left_hand_claim": hand_claim_state["left"]["claim"] or "NONE",
                "right_hand_claim": hand_claim_state["right"]["claim"] or "NONE",
                "left_hold_claim_hits": hand_claim_state["left"]["hold_hits"],
                "right_hold_claim_hits": hand_claim_state["right"]["hold_hits"],
                "left_carry_claim_hits": hand_claim_state["left"]["carry_hits"],
                "right_carry_claim_hits": hand_claim_state["right"]["carry_hits"],
                "left_hold_raw_before_claim": hold_state[
                    "left_hold_raw_before_claim"
                ],
                "right_hold_raw_before_claim": hold_state[
                    "right_hold_raw_before_claim"
                ],
                "left_hold_raw_after_claim": hold_state["left_hold_raw_after_claim"],
                "right_hold_raw_after_claim": hold_state[
                    "right_hold_raw_after_claim"
                ],
                "left_carry_raw_before_claim": carry_pose[
                    "left_carry_raw_before_claim"
                ],
                "right_carry_raw_before_claim": carry_pose[
                    "right_carry_raw_before_claim"
                ],
                "left_carry_raw_after_claim": carry_pose[
                    "left_carry_raw_after_claim"
                ],
                "right_carry_raw_after_claim": carry_pose[
                    "right_carry_raw_after_claim"
                ],
            }
        )

        (
            holding_raw,
            hold_raw_status,
            holding_correct_raw,
            holding_wrong_raw,
            left_wrist_valid,
            right_wrist_valid,
            _left_holding_raw_effective,
            _right_holding_raw_effective,
            _handrail_status_raw,
            _handrail_reason_raw,
        ) = self._copy_hold_state_fields(hold_state)
        handrail_hit_debug = self._update_handrail_hit_history(
            track_id,
            bool(hold_state["left_wrist_hit"]),
            bool(hold_state["right_wrist_hit"]),
        )

        analysis_context.update(
            {
                "holding_raw": holding_raw,
                "hold_raw_status": hold_raw_status,
                "holding_correct_raw": holding_correct_raw,
                "holding_wrong_raw": holding_wrong_raw,
                "left_wrist_valid": left_wrist_valid,
                "right_wrist_valid": right_wrist_valid,
                "left_wrist_hit": hold_state["left_wrist_hit"],
                "right_wrist_hit": hold_state["right_wrist_hit"],
                "left_wrist_hit_count": handrail_hit_debug[
                    "left_wrist_hit_count"
                ],
                "right_wrist_hit_count": handrail_hit_debug[
                    "right_wrist_hit_count"
                ],
                "left_wrist_miss_count": handrail_hit_debug[
                    "left_wrist_miss_count"
                ],
                "right_wrist_miss_count": handrail_hit_debug[
                    "right_wrist_miss_count"
                ],
                "left_wrist_confirm_required": handrail_hit_debug[
                    "left_wrist_confirm_required"
                ],
                "right_wrist_confirm_required": handrail_hit_debug[
                    "right_wrist_confirm_required"
                ],
                "left_wrist_distance_to_left_rail": hold_state[
                    "left_wrist_distance_to_left_rail"
                ],
                "left_wrist_distance_to_right_rail": hold_state[
                    "left_wrist_distance_to_right_rail"
                ],
                "right_wrist_distance_to_left_rail": hold_state[
                    "right_wrist_distance_to_left_rail"
                ],
                "right_wrist_distance_to_right_rail": hold_state[
                    "right_wrist_distance_to_right_rail"
                ],
                "left_wrist_nearest_distance": hold_state[
                    "left_wrist_nearest_distance"
                ],
                "right_wrist_nearest_distance": hold_state[
                    "right_wrist_nearest_distance"
                ],
                "left_wrist_nearest_rail": hold_state["left_wrist_nearest_rail"],
                "right_wrist_nearest_rail": hold_state["right_wrist_nearest_rail"],
                "left_wrist_nearest_point": hold_state["left_wrist_nearest_point"],
                "right_wrist_nearest_point": hold_state["right_wrist_nearest_point"],
            }
        )

        if skip_handrail:
            (
                hold_correct_hits,
                hold_wrong_side_hits,
                hold_none_hits,
                hold_unknown_hits,
                hold_not_hold_evidence_hits,
                hold_final_status,
                holding,
            ) = self._get_hold_status_history_state(track_id)
        else:
            (
                hold_correct_hits,
                hold_wrong_side_hits,
                hold_none_hits,
                hold_unknown_hits,
                hold_not_hold_evidence_hits,
                hold_final_status,
                holding,
            ) = self._update_hold_status_history(track_id, hold_raw_status)

        left_holding_confirmed = False
        right_holding_confirmed = False
        if not skip_handrail:
            left_holding_confirmed = hold_final_status == "WRONG_SIDE"
            right_holding_confirmed = hold_final_status == "CORRECT"

        if skip_handrail:
            handrail_decision = {
                "handrail_status": "SKIP_BACKWARD",
                "handrail_reason": "BACKWARD_SKIP_HANDRAIL",
                "handrail_debug_reason": "BACKWARD_SKIP_HANDRAIL",
            }
        else:
            handrail_decision = self._build_confirmed_handrail_decision(
                left_wrist_valid=left_wrist_valid,
                right_wrist_valid=right_wrist_valid,
                left_hit=bool(hold_state["left_wrist_hit"]),
                right_hit=bool(hold_state["right_wrist_hit"]),
                left_hit_count=int(handrail_hit_debug["left_wrist_hit_count"]),
                right_hit_count=int(handrail_hit_debug["right_wrist_hit_count"]),
                left_confirm_required=int(
                    handrail_hit_debug["left_wrist_confirm_required"]
                ),
                right_confirm_required=int(
                    handrail_hit_debug["right_wrist_confirm_required"]
                ),
                left_holding=left_holding_confirmed,
                right_holding=right_holding_confirmed,
                left_hand_claim=str(analysis_context["left_hand_claim"]),
                right_hand_claim=str(analysis_context["right_hand_claim"]),
            )

        analysis_context.update(
            {
                "hold_correct_hits": hold_correct_hits,
                "hold_wrong_side_hits": hold_wrong_side_hits,
                "hold_none_hits": hold_none_hits,
                "hold_unknown_hits": hold_unknown_hits,
                "hold_not_hold_evidence_hits": hold_not_hold_evidence_hits,
                "hold_final_status": hold_final_status,
                "holding": holding,
                "left_holding": left_holding_confirmed,
                "right_holding": right_holding_confirmed,
                "handrail_status": handrail_decision["handrail_status"],
                "handrail_reason": handrail_decision["handrail_reason"],
                "handrail_debug_reason": handrail_decision[
                    "handrail_debug_reason"
                ],
            }
        )
        return carry_pose

    def _evaluate_motion_and_position(
        self,
        person: PersonContext,
        analysis_context: AnalysisContext,
        perf: PerfStats | None,
    ) -> None:
        # FLOW: Bat dau pipeline phan tich cho frame hien tai.
        # WHY: Direction can lich su nhieu frame, nen phai append history som truoc khi rule direction chay.
        apply_direction_history(self, person.track_id, person.features)

        # FLOW: Kiem tra vi tri trong ROI va dung yen truoc.
        # WHY: Standing chi co y nghia khi nguoi dang o trong vung cau thang.
        standing_start = time.perf_counter() if perf is not None else None
        inside_result = evaluate_inside_stairs_typed(
            build_inside_stairs_input(
                features=person.features,
                stairs_polygon=self.stairs_poly,
                last_inside_state=bool(
                    self.inside_last_state.get(person.track_id, False)
                ),
            )
        )
        if inside_result.last_inside_state_update is not None:
            self.inside_last_state[person.track_id] = (
                inside_result.last_inside_state_update
            )
        analysis_context.update(inside_result.to_analysis_update())
        analysis_context.update(
            evaluate_standing_still(
                self,
                person.track_id,
                person.p_motion,
                bool(analysis_context["inside_stairs"]),
            )
        )

        # FLOW: Tinh direction sau khi da co lich su motion moi nhat.
        # WHY: Lane, backward, handrail direction-aware va two-step deu phu thuoc direction.
        direction_start = time.perf_counter() if perf is not None else None
        analysis_context.update(update_direction(self, person.track_id, person.features))
        self._record_perf(perf, "direction", direction_start)

        # FLOW: BehaviorAnalyzer chi dieu phoi va lay ket qua tu rule two-step rieng.
        # WHY: Day phan tinh step gap/planted/history sang module rieng giup `analyze()` ngan hon.
        two_step_input = build_two_step_input(
            track_id=person.track_id,
            direction=str(analysis_context["direction"]),
            features=person.features,
            inside_stairs=bool(analysis_context["inside_stairs"]),
            frame_index=self.frame_index,
            direction_source=str(analysis_context["direction_source"]),
        )
        two_step_result = self.two_step_analyzer.analyze_input(two_step_input)
        analysis_context.update(two_step_result.to_analysis_context())
        self._record_perf(perf, "standing", standing_start)

    def _build_analyzing_result_if_needed(
        self,
        track_id: AnalysisSubjectID,
        person: PersonContext,
        analysis_context: AnalysisContext,
        perf: PerfStats | None,
        analyze_start: float | None,
    ) -> AnalysisResult | None:
        # WHY: Direction can du frame history moi xac nhan.
        # Khi chua du history, analyzer chi duoc tra ket qua tam va tranh sinh warning manh mot cach som.
        if analysis_context["direction"] != "ANALYZING":
            return None

        if not analysis_context["inside_stairs"]:
            # WHY: Nguoi da ngoai ROI ma direction van chua ro thi reset history som de tranh giu rac frame cu.
            self._reset_behavior_histories(track_id)
            analysis_context["hold_raw_status"] = "OUTSIDE"
            analysis_context["hold_final_status"] = "OUTSIDE"
            return self._build_analysis_result(
                analysis_context,
                status="ANALYZING...",
                display_status="" if SETTINGS.demo.demo_mode else "ANALYZING...",
                color=SETTINGS.violation.outside_color,
                perf=perf,
                analyze_start=analyze_start,
            )

        hold_direction = self.last_valid_direction.get(track_id)
        # WHY: Du direction chua confirm, handrail/standing/two-step van co the can hien debug tam thoi.
        self._update_hold_context(
            track_id,
            person,
            analysis_context,
            perf,
            hold_direction,
        )
        warnings = self._collect_warnings(
            str(analysis_context["direction"]),
            False,
            analysis_context["hold_final_status"],
            False,
            analysis_context["standing_still_confirmed"],
            two_step_skip_confirmed=bool(
                analysis_context["two_step_skip_confirmed"]
            ),
        )
        analysis_context["warnings"] = warnings
        if warnings:
            status, display_status, color = self._summarize_result(
                analysis_context["direction"],
                warnings,
                analysis_context["hold_final_status"],
                SETTINGS.violation.unknown_color
                if SETTINGS.demo.demo_mode
                else SETTINGS.violation.analyzing_color,
                "ANALYZING...",
            )
            return self._build_analysis_result(
                analysis_context,
                status=status,
                display_status=display_status,
                color=color,
                perf=perf,
                analyze_start=analyze_start,
            )

        return self._build_analysis_result(
            analysis_context,
            status="ANALYZING...",
            display_status="" if SETTINGS.demo.demo_mode else "ANALYZING...",
            color=(
                SETTINGS.violation.unknown_color
                if SETTINGS.demo.demo_mode
                else SETTINGS.violation.analyzing_color
            ),
            perf=perf,
            analyze_start=analyze_start,
        )

    def _build_outside_result_if_needed(
        self,
        track_id: AnalysisSubjectID,
        analysis_context: AnalysisContext,
        perf: PerfStats | None,
        analyze_start: float | None,
    ) -> AnalysisResult | None:
        # WHY: Ra khoi ROI phai reset history hanh vi de warning cu khong "di theo" subject khi quay lai.
        if analysis_context["inside_stairs"]:
            return None

        self._reset_behavior_histories(track_id)
        analysis_context["warnings"] = []
        return self._build_analysis_result(
            analysis_context,
            status=f"{analysis_context['direction']} | Ngoai Vung",
            display_status=(
                ""
                if SETTINGS.demo.demo_mode
                else f"{analysis_context['direction']} | Ngoai Vung"
            ),
            color=SETTINGS.violation.outside_color,
            perf=perf,
            analyze_start=analyze_start,
            lane_raw=False,
            lane_hits=0,
            lane_conf=False,
            inside_stairs=False,
            hold_status="OUTSIDE",
            hold_raw_status="OUTSIDE",
            hold_confirmed_status="OUTSIDE",
            hold_correct_hits=0,
            hold_wrong_side_hits=0,
            hold_none_hits=0,
            hold_unknown_hits=0,
            holding_correct_raw=False,
            holding_wrong_raw=False,
            left_wrist_valid=False,
            right_wrist_valid=False,
            left_wrist_hit=False,
            right_wrist_hit=False,
            left_wrist_distance_to_left_rail=None,
            left_wrist_distance_to_right_rail=None,
            right_wrist_distance_to_left_rail=None,
            right_wrist_distance_to_right_rail=None,
            left_wrist_nearest_distance=None,
            right_wrist_nearest_distance=None,
            left_wrist_nearest_rail="NONE",
            right_wrist_nearest_rail="NONE",
            left_wrist_nearest_point=None,
            right_wrist_nearest_point=None,
            left_holding=False,
            right_holding=False,
            handrail_status="OUTSIDE",
            handrail_reason="OUTSIDE_STAIRS",
            handrail_debug_reason="OUTSIDE_STAIRS",
        )

    def _update_last_valid_direction(
        self,
        track_id: AnalysisSubjectID,
        analysis_context: AnalysisContext,
    ) -> None:
        # FLOW: Khi da co direction hop le thi ghi nho last valid direction cho cac frame mo ho sau do.
        if analysis_context["direction"] in ("UP", "DOWN"):
            self.last_valid_direction[track_id] = str(analysis_context["direction"])
            self.last_valid_direction_frame[track_id] = self.frame_index

    def _evaluate_behavior_rules(
        self,
        track_id: AnalysisSubjectID,
        person: PersonContext,
        analysis_context: AnalysisContext,
        perf: PerfStats | None,
    ) -> dict[str, object]:
        body_facing = str(analysis_context["body_facing"])
        body_facing_confidence = float(analysis_context["body_facing_confidence"] or 0.0)
        body_facing_evidence_count = int(
            analysis_context["body_facing_evidence_count"] or 0
        )
        hip_pair_valid = bool(analysis_context["hip_pair_valid"])
        shoulder_pair_valid = bool(analysis_context["shoulder_pair_valid"])
        ear_pair_valid = bool(analysis_context["ear_pair_valid"])
        head_valid = bool(analysis_context["head_valid"])
        direction = str(analysis_context["direction"])

        # FLOW: Backward dung direction + body_facing, khong dung lane/handrail.
        backward_start = time.perf_counter() if perf is not None else None
        analysis_context.update(
            evaluate_backward(
                self,
                track_id,
                direction,
                body_facing,
                body_facing_confidence,
                body_facing_evidence_count,
                hip_pair_valid,
                shoulder_pair_valid,
                ear_pair_valid,
                head_valid,
            )
        )
        self._record_perf(perf, "backward", backward_start)

        # FLOW: Lane dung center line + direction + feet point.
        lane_start = time.perf_counter() if perf is not None else None
        analysis_context.update(
            evaluate_lane_violation(
                self,
                track_id,
                person.features,
                direction,
            )
        )
        self._record_perf(perf, "lane", lane_start)

        hold_direction: str | None = direction
        if hold_direction not in ("UP", "DOWN"):
            hold_direction = self.last_valid_direction.get(track_id)

        # WHY: Khi backward da confirm thi skip handrail final.
        # Rule nghiep vu hien tai uu tien `Di Lui`, tranh vua bao backward vua ep logic Khong Vin/Vin Sai Ben.
        return self._update_hold_context(
            track_id,
            person,
            analysis_context,
            perf,
            hold_direction,
            skip_handrail=bool(analysis_context["backward_confirmed"]),
        )

    def _build_idle_result_if_needed(
        self,
        analysis_context: AnalysisContext,
        perf: PerfStats | None,
        analyze_start: float | None,
    ) -> AnalysisResult | None:
        # WHY: `IDLE` nhung chua du bang chung `Dung Yen` thi khong nen goi la vi pham.
        if (
            analysis_context["direction"] == "IDLE"
            and not analysis_context["standing_still_confirmed"]
        ):
            return self._build_analysis_result(
                analysis_context,
                status="IDLE",
                display_status="",
                color=(
                    SETTINGS.violation.safe_color
                    if SETTINGS.demo.demo_mode
                    else (200, 200, 200)
                ),
                perf=perf,
                analyze_start=analyze_start,
            )

        # FLOW: `IDLE` + da confirm standing thi tong hop warning ngay, khong can qua carry.
        if analysis_context["direction"] != "IDLE":
            return None

        warnings = self._collect_warnings(
            str(analysis_context["direction"]),
            analysis_context["wrong_lane"],
            analysis_context["hold_final_status"],
            analysis_context["backward_confirmed"],
            analysis_context["standing_still_confirmed"],
            two_step_skip_confirmed=bool(
                analysis_context["two_step_skip_confirmed"]
            ),
        )
        analysis_context["warnings"] = warnings
        status, display_status, color = self._summarize_result(
            analysis_context["direction"],
            warnings,
            analysis_context["hold_final_status"],
            (
                SETTINGS.violation.safe_color
                if SETTINGS.demo.demo_mode
                else (200, 200, 200)
            ),
            analysis_context["direction"],
        )
        return self._build_analysis_result(
            analysis_context,
            status=status,
            display_status=display_status,
            color=color,
            perf=perf,
            analyze_start=analyze_start,
        )

    def _finalize_carry_result(
        self,
        track_id: AnalysisSubjectID,
        person: PersonContext,
        analysis_context: AnalysisContext,
        perf: PerfStats | None,
        analyze_start: float | None,
        carry_pose: dict[str, object],
    ) -> AnalysisResult:
        # FLOW: Carry duoc danh gia sau handrail vi no can hand claim/hold state de tranh dam logic.
        # WARNING: Carry chi sinh warning `Mang Vac`, khong duoc ghi de len handrail final.
        carry_start = time.perf_counter() if perf is not None else None
        carry_info = evaluate_carry(
            self,
            track_id,
            person.keypoints,
            analysis_context["holding_raw"],
            features=person.features,
            carry_pose=carry_pose,
            handrail_state={
                "left_wrist_hit": analysis_context["left_wrist_hit"],
                "right_wrist_hit": analysis_context["right_wrist_hit"],
                "left_holding": analysis_context["left_holding"],
                "right_holding": analysis_context["right_holding"],
            },
        )
        self._record_perf(perf, "carry", carry_start)
        warnings = self._collect_warnings(
            str(analysis_context["direction"]),
            analysis_context["wrong_lane"],
            analysis_context["hold_final_status"],
            analysis_context["backward_confirmed"],
            analysis_context["standing_still_confirmed"],
            is_carrying=carry_info["is_carrying"],
            two_step_skip_confirmed=bool(
                analysis_context["two_step_skip_confirmed"]
            ),
        )
        analysis_context["warnings"] = warnings
        status, display_status, color = self._summarize_result(
            analysis_context["direction"],
            warnings,
            analysis_context["hold_final_status"],
            SETTINGS.violation.safe_color,
            f"{analysis_context['direction']} | An Toan",
        )
        return self._build_analysis_result(
            analysis_context,
            status=status,
            display_status=display_status,
            color=color,
            perf=perf,
            analyze_start=analyze_start,
            carry_info_override=carry_info,
        )

    def analyze(
        self,
        track_id: AnalysisSubjectID,
        p_lane: Point | None,
        p_motion: Point | None,
        keypoints: KeypointsArray,
        box: BBoxArray | None = None,
        features: PoseFeatures | None = None,
    ) -> AnalysisResult:
        """Phan tich hanh vi cua 1 subject trong 1 frame.

        INPUT:
            - `track_id`: ID ma analyzer dung de giu history cua subject hien tai.
            - `p_lane`: Diem uu tien cho inside/lane.
            - `p_motion`: Diem uu tien cho direction/backward/standing.
            - `keypoints`, `box`, `features`: Dau vao pose cua subject.

        OUTPUT:
            - `AnalysisResult` co status, warnings, color va cac field debug.

        WHY:
            - Ham nay la noi gom ket qua tu nhieu rule doc lap, nhung van phai
              giu behavior on dinh qua nhieu frame.
        """
        perf = {} if SETTINGS.performance.enable_perf_log else None
        analyze_start = time.perf_counter() if perf is not None else None
        if self.frame_index < 0:
            self.frame_index = 0

        resolved_features = self._resolve_pose_features(
            keypoints,
            box,
            features,
        )
        person = self._build_person_context(
            track_id,
            keypoints,
            box,
            resolved_features,
            p_lane,
            p_motion,
        )
        analysis_context = self._create_analysis_context(track_id, person)

        self._evaluate_motion_and_position(
            person,
            analysis_context,
            perf,
        )

        analyzing_result = self._build_analyzing_result_if_needed(
            track_id,
            person,
            analysis_context,
            perf,
            analyze_start,
        )
        if analyzing_result is not None:
            return analyzing_result

        outside_result = self._build_outside_result_if_needed(
            track_id,
            analysis_context,
            perf,
            analyze_start,
        )
        if outside_result is not None:
            return outside_result

        self._update_last_valid_direction(track_id, analysis_context)
        carry_pose = self._evaluate_behavior_rules(
            track_id,
            person,
            analysis_context,
            perf,
        )

        idle_result = self._build_idle_result_if_needed(
            analysis_context,
            perf,
            analyze_start,
        )
        if idle_result is not None:
            return idle_result

        return self._finalize_carry_result(
            track_id,
            person,
            analysis_context,
            perf,
            analyze_start,
            carry_pose,
        )

    def merge_behavior_history(
        self,
        from_subject_id: AnalysisSubjectID,
        to_subject_id: AnalysisSubjectID,
    ) -> None:
        # WHY: Khi candidate duoc relink/promote sang subject moi, warning history phai di theo de khong reset canh bao.
        if from_subject_id == to_subject_id:
            return

        list_history_maps = [
            self.hip_motion_history,
            self.shoulder_motion_history,
            self.lane_history,
            self.hold_status_history,
            self.left_handrail_hit_history,
            self.right_handrail_hit_history,
            self.front_carry_history,
            self.front_carry_one_arm_history,
            self.backward_history,
            self.standing_history,
            self.two_step_skip_history,
            self.standing_motion_history,
        ]
        for history_map in list_history_maps:
            if from_subject_id not in history_map:
                continue
            from_value = history_map.pop(from_subject_id)
            to_value = history_map.get(to_subject_id)
            if isinstance(to_value, list) and isinstance(from_value, list):
                history_map[to_subject_id] = to_value + from_value
            else:
                history_map[to_subject_id] = from_value

        latest_value_maps = [
            self.inside_last_state,
            self.lane_last_state,
            self.lane_last_seen,
            self.last_valid_direction,
            self.last_valid_direction_frame,
            self.hand_claim_state,
            self.left_foot_step_states,
            self.right_foot_step_states,
        ]
        for history_map in latest_value_maps:
            if from_subject_id in history_map:
                history_map[to_subject_id] = history_map.pop(from_subject_id)

    def cleanup_inactive_tracks(
        self,
        active_track_ids: Iterable[AnalysisSubjectID],
    ) -> None:
        active_track_ids = set(active_track_ids)
        history_maps = [
            self.hip_motion_history,
            self.shoulder_motion_history,
            self.lane_history,
            self.lane_last_state,
            self.lane_last_seen,
            self.hold_status_history,
            self.left_handrail_hit_history,
            self.right_handrail_hit_history,
            self.last_valid_direction,
            self.last_valid_direction_frame,
            self.hand_claim_state,
            self.front_carry_history,
            self.front_carry_one_arm_history,
            self.backward_history,
            self.standing_history,
            self.two_step_skip_history,
            self.left_foot_step_states,
            self.right_foot_step_states,
            self.standing_motion_history,
        ]

        for history_map in history_maps:
            inactive_ids = [
                current_track_id
                for current_track_id in history_map.keys()
                if current_track_id not in active_track_ids
            ]
            for current_track_id in inactive_ids:
                del history_map[current_track_id]
