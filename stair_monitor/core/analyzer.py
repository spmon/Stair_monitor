from __future__ import annotations

import time
from typing import cast

import numpy as np

from stair_monitor.common.types import (
    AnalysisResult,
    BBoxArray,
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
from stair_monitor.rules.inside_rule import evaluate_inside_stairs
from stair_monitor.rules.lane_rule import evaluate_lane_violation, get_camera_angle_profile
from stair_monitor.rules.standing_rule import evaluate_standing_still
from stair_monitor.state.behavior_history import BehaviorHistoryMixin
from stair_monitor.state.track_state import AnalyzerState
from stair_monitor.vision.geometry import extract_pose_features


class BehaviorAnalyzer(
    BehaviorHistoryMixin,
    CarryAnalysisMixin,
    HandrailAnalysisMixin,
    ResultBuilderMixin,
):
    """Trung tam dieu phoi cac rule cho stair_monitor Windows/demo."""

    def __init__(self, config):
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
        bottom_right = step_bottom[1]
        top_right = step_top[1]
        top_left = [
            step_top[0][0],
            step_top[0][1],
        ]
        self.stairs_poly = np.array(
            [bottom_left, bottom_right, top_right, top_left],
            np.int32,
        )

        self.state = AnalyzerState()
        self.hip_motion_history = self.state.hip_motion_history
        self.shoulder_motion_history = self.state.shoulder_motion_history
        self.lane_history = self.state.lane_history
        self.lane_last_state = self.state.lane_last_state
        self.lane_last_seen = self.state.lane_last_seen
        self.hold_status_history = self.state.hold_status_history
        self.last_valid_direction = self.state.last_valid_direction
        self.hand_claim_state = self.state.hand_claim_state
        self.front_carry_history = self.state.front_carry_history
        self.front_carry_one_arm_history = self.state.front_carry_one_arm_history
        self.backward_history = self.state.backward_history
        self.standing_history = self.state.standing_history
        self.standing_motion_history = self.state.standing_motion_history

    @property
    def frame_index(self) -> int:
        return self.state.frame_index

    @frame_index.setter
    def frame_index(self, value: int) -> None:
        self.state.frame_index = value

    def begin_frame(self):
        self.frame_index += 1

    @staticmethod
    def _record_perf(perf, key, start_time):
        if perf is None or start_time is None:
            return
        perf[key] = perf.get(key, 0.0) + (time.perf_counter() - start_time) * 1000.0

    def analyze(
        self,
        track_id: int,
        p_lane: Point | None,
        p_motion: Point | None,
        keypoints: KeypointsArray,
        box: BBoxArray | None = None,
        features: PoseFeatures | None = None,
    ) -> AnalysisResult:
        perf = {} if SETTINGS.performance.enable_perf_log else None
        analyze_start = time.perf_counter() if perf is not None else None
        if self.frame_index < 0:
            self.frame_index = 0

        features = cast(
            PoseFeatures,
            features or extract_pose_features(keypoints, box),
        )
        if p_lane is None:
            p_lane = features.get("feet_point")
        if p_motion is None:
            p_motion = features.get("motion_point")

        person = PersonContext(
            track_id=track_id,
            keypoints=keypoints,
            box=box,
            features=features,
            p_lane=p_lane,
            p_motion=p_motion,
        )

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

        analysis_context: dict[str, object] = {
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
            "feet_point_source": features.get("feet_point_source", "NO_FEET_POINT"),
            "inside_feet_point": features.get("inside_feet_point"),
            "inside_feet_point_source": features.get(
                "inside_feet_point_source",
                "NO_FEET_POINT",
            ),
            "ankle_valid_count": int(features.get("ankle_valid_count", 0) or 0),
            "feet_reliable": bool(features.get("feet_reliable", False)),
            "bbox_height": features.get("bbox_height"),
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
            "warnings": [],
        }

        def build_result(
            status: str,
            color,
            display_status: str = "",
            carry_info_override=None,
            **overrides,
        ) -> AnalysisResult:
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

        def update_hold_context(hold_direction):
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
                hold_status_correct,
                hold_status_wrong,
                dist_correct,
                dist_wrong,
                seg_dist_correct,
                seg_dist_wrong,
                t_correct,
                t_wrong,
                wrist_side_correct,
                wrist_side_wrong,
                best_wrist_correct,
                best_wrist_wrong,
                best_wrist_correct_point,
                best_wrist_wrong_point,
                correct_line_name,
                wrong_line_name,
                correct_rule,
                wrong_rule,
                best_wrist,
                best_wrist_point,
                dist_wrist,
                wrist_side,
            ) = self._copy_hold_state_fields(hold_state)

            analysis_context.update(
                {
                    "holding_raw": holding_raw,
                    "hold_raw_status": hold_raw_status,
                    "holding_correct_raw": holding_correct_raw,
                    "holding_wrong_raw": holding_wrong_raw,
                    "hold_status_correct": hold_status_correct,
                    "hold_status_wrong": hold_status_wrong,
                    "dist_correct": dist_correct,
                    "dist_wrong": dist_wrong,
                    "seg_dist_correct": seg_dist_correct,
                    "seg_dist_wrong": seg_dist_wrong,
                    "t_correct": t_correct,
                    "t_wrong": t_wrong,
                    "wrist_side_correct": wrist_side_correct,
                    "wrist_side_wrong": wrist_side_wrong,
                    "best_wrist_correct": best_wrist_correct,
                    "best_wrist_wrong": best_wrist_wrong,
                    "best_wrist_correct_point": best_wrist_correct_point,
                    "best_wrist_wrong_point": best_wrist_wrong_point,
                    "correct_line_name": correct_line_name,
                    "wrong_line_name": wrong_line_name,
                    "correct_rule": correct_rule,
                    "wrong_rule": wrong_rule,
                    "best_wrist": best_wrist,
                    "best_wrist_point": best_wrist_point,
                    "dist_wrist": dist_wrist,
                    "wrist_side": wrist_side,
                }
            )

            (
                hold_correct_hits,
                hold_wrong_side_hits,
                hold_none_hits,
                hold_unknown_hits,
                hold_not_hold_evidence_hits,
                hold_final_status,
                holding,
            ) = self._update_hold_status_history(track_id, hold_raw_status)

            analysis_context.update(
                {
                    "hold_correct_hits": hold_correct_hits,
                    "hold_wrong_side_hits": hold_wrong_side_hits,
                    "hold_none_hits": hold_none_hits,
                    "hold_unknown_hits": hold_unknown_hits,
                    "hold_not_hold_evidence_hits": hold_not_hold_evidence_hits,
                    "hold_final_status": hold_final_status,
                    "holding": holding,
                }
            )
            return carry_pose

        apply_direction_history(self, track_id, person.features)

        standing_start = time.perf_counter() if perf is not None else None
        analysis_context.update(
            evaluate_inside_stairs(
                self,
                track_id,
                person.features,
                person.p_lane,
            )
        )
        analysis_context.update(
            evaluate_standing_still(
                self,
                track_id,
                person.p_motion,
                bool(analysis_context["inside_stairs"]),
            )
        )
        self._record_perf(perf, "standing", standing_start)

        direction_start = time.perf_counter() if perf is not None else None
        analysis_context.update(update_direction(self, track_id, person.features))
        self._record_perf(perf, "direction", direction_start)

        if analysis_context["direction"] == "ANALYZING":
            if not analysis_context["inside_stairs"]:
                self._reset_behavior_histories(track_id)
                analysis_context["hold_raw_status"] = "OUTSIDE"
                analysis_context["hold_final_status"] = "OUTSIDE"
                return build_result(
                    status="ANALYZING...",
                    display_status="" if SETTINGS.demo.demo_mode else "ANALYZING...",
                    color=SETTINGS.violation.outside_color,
                )

            hold_direction = self.last_valid_direction.get(track_id)
            update_hold_context(hold_direction)
            warnings = self._collect_warnings(
                False,
                analysis_context["hold_final_status"],
                False,
                analysis_context["standing_still_confirmed"],
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
                return build_result(
                    status=status,
                    display_status=display_status,
                    color=color,
                )

            return build_result(
                status="ANALYZING...",
                display_status="" if SETTINGS.demo.demo_mode else "ANALYZING...",
                color=(
                    SETTINGS.violation.unknown_color
                    if SETTINGS.demo.demo_mode
                    else SETTINGS.violation.analyzing_color
                ),
            )

        if not analysis_context["inside_stairs"]:
            self._reset_behavior_histories(track_id)
            analysis_context["warnings"] = []
            return build_result(
                status=f"{analysis_context['direction']} | Ngoai Vung",
                display_status=(
                    ""
                    if SETTINGS.demo.demo_mode
                    else f"{analysis_context['direction']} | Ngoai Vung"
                ),
                color=SETTINGS.violation.outside_color,
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
                hold_status_correct="OUTSIDE",
                hold_status_wrong="OUTSIDE",
                dist_wrist=-999,
                wrist_side="UNKNOWN",
                best_wrist="NONE",
                best_wrist_point=None,
                dist_correct=-999,
                dist_wrong=-999,
                wrist_side_correct="UNKNOWN",
                wrist_side_wrong="UNKNOWN",
                best_wrist_correct="NONE",
                best_wrist_wrong="NONE",
                best_wrist_correct_point=None,
                best_wrist_wrong_point=None,
                correct_line_name="NONE",
                wrong_line_name="NONE",
                correct_rule="NA",
                wrong_rule="NA",
            )

        if analysis_context["direction"] in ("UP", "DOWN"):
            self.last_valid_direction[track_id] = analysis_context["direction"]

        backward_start = time.perf_counter() if perf is not None else None
        analysis_context.update(
            evaluate_backward(
                self,
                track_id,
                analysis_context["direction"],
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

        lane_start = time.perf_counter() if perf is not None else None
        analysis_context.update(
            evaluate_lane_violation(
                self,
                track_id,
                person.features,
                analysis_context["direction"],
            )
        )
        self._record_perf(perf, "lane", lane_start)

        hold_direction = analysis_context["direction"]
        if hold_direction not in ("UP", "DOWN"):
            hold_direction = self.last_valid_direction.get(track_id)
        carry_pose = update_hold_context(hold_direction)

        if (
            analysis_context["direction"] == "IDLE"
            and not analysis_context["standing_still_confirmed"]
        ):
            return build_result(
                status="IDLE",
                display_status="",
                color=(
                    SETTINGS.violation.safe_color
                    if SETTINGS.demo.demo_mode
                    else (200, 200, 200)
                ),
            )

        if analysis_context["direction"] == "IDLE":
            warnings = self._collect_warnings(
                analysis_context["wrong_lane"],
                analysis_context["hold_final_status"],
                analysis_context["backward_confirmed"],
                analysis_context["standing_still_confirmed"],
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
            return build_result(
                status=status,
                display_status=display_status,
                color=color,
            )

        carry_start = time.perf_counter() if perf is not None else None
        carry_info = evaluate_carry(
            self,
            track_id,
            person.keypoints,
            analysis_context["holding_raw"],
            analysis_context["best_wrist"],
            features=person.features,
            carry_pose=carry_pose,
        )
        self._record_perf(perf, "carry", carry_start)
        warnings = self._collect_warnings(
            analysis_context["wrong_lane"],
            analysis_context["hold_final_status"],
            analysis_context["backward_confirmed"],
            analysis_context["standing_still_confirmed"],
            is_carrying=carry_info["is_carrying"],
        )
        analysis_context["warnings"] = warnings
        status, display_status, color = self._summarize_result(
            analysis_context["direction"],
            warnings,
            analysis_context["hold_final_status"],
            SETTINGS.violation.safe_color,
            f"{analysis_context['direction']} | An Toan",
        )
        return build_result(
            status=status,
            display_status=display_status,
            color=color,
            carry_info_override=carry_info,
        )

    def cleanup_inactive_tracks(self, active_track_ids):
        active_track_ids = set(active_track_ids)
        history_maps = [
            self.hip_motion_history,
            self.shoulder_motion_history,
            self.lane_history,
            self.lane_last_state,
            self.lane_last_seen,
            self.hold_status_history,
            self.last_valid_direction,
            self.hand_claim_state,
            self.front_carry_history,
            self.front_carry_one_arm_history,
            self.backward_history,
            self.standing_history,
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
