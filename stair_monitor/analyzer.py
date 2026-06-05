import cv2
import time
import numpy as np

from stair_monitor.behavior_history import BehaviorHistoryMixin
from stair_monitor.carry_analysis import CarryAnalysisMixin
from stair_monitor.geometry import extract_pose_features
from stair_monitor.handrail_analysis import (
    HandrailAnalysisMixin,
    compute_handrail_evidence,
    is_back_to_camera,
    is_front_to_camera,
)
from stair_monitor.result_builder import ResultBuilderMixin
from stair_monitor.settings import (
    ANALYZING_COLOR,
    DEMO_MODE,
    DIRECTION_HISTORY_LEN,
    DIRECTION_MIN_FRAMES,
    DIRECTION_PIXEL_THRESHOLD,
    DIRECTION_SIGN_NORMAL,
    ENABLE_PERF_LOG,
    LANE_SIGN_NORMAL,
    OUTSIDE_COLOR,
    SAFE_COLOR,
    STAIRS_LEFT_EXPAND_BOTTOM_PX,
    STAIRS_LEFT_EXPAND_TOP_PX,
    UNKNOWN_COLOR,
)

HAND_CLAIM_HOLD_HITS = 4
HAND_CLAIM_CARRY_HITS = 4
HAND_CLAIM_RESET_MISSES = 10


class BehaviorAnalyzer(
    BehaviorHistoryMixin,
    CarryAnalysisMixin,
    HandrailAnalysisMixin,
    ResultBuilderMixin,
):
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
            step_bottom[0][0] - STAIRS_LEFT_EXPAND_BOTTOM_PX,
            step_bottom[0][1],
        ]
        bottom_right = step_bottom[1]
        top_right = step_top[1]
        top_left = [
            step_top[0][0] - STAIRS_LEFT_EXPAND_TOP_PX,
            step_top[0][1],
        ]
        self.stairs_poly = np.array(
            [bottom_left, bottom_right, top_right, top_left], np.int32
        )
        self.track_history = {}
        self.lane_history = {}
        self.hold_status_history = {}
        self.last_valid_direction = {}
        self.hand_claim_state = {}
        self.front_carry_history = {}
        self.front_carry_one_arm_history = {}
        self.backward_history = {}
        self.standing_history = {}
        self.standing_motion_history = {}

    def is_inside_stairs(self, p_lane):
        if p_lane is None or len(self.stairs_poly) < 3:
            return False

        return cv2.pointPolygonTest(
            self.stairs_poly,
            (float(p_lane[0]), float(p_lane[1])),
            False,
        ) >= 0

    @staticmethod
    def _new_hand_claim_entry():
        return {
            "claim": None,
            "hold_hits": 0,
            "carry_hits": 0,
            "misses": 0,
        }

    def _get_hand_claim_state(self, track_id):
        if track_id not in self.hand_claim_state:
            self.hand_claim_state[track_id] = {
                "left": self._new_hand_claim_entry(),
                "right": self._new_hand_claim_entry(),
            }
        return self.hand_claim_state[track_id]

    def _update_hand_claim_state(
        self,
        track_id,
        left_hold_raw,
        right_hold_raw,
        left_carry_raw,
        right_carry_raw,
    ):
        # Mot tay khong nen vua duoc dung de ket luan vin lan can,
        # vua duoc dung de ket luan mang vac.
        # Trang thai on dinh truoc se claim tay do.
        claim_state = self._get_hand_claim_state(track_id)
        hand_raw_inputs = {
            "left": (left_hold_raw, left_carry_raw),
            "right": (right_hold_raw, right_carry_raw),
        }

        for hand_name, (hold_raw, carry_raw) in hand_raw_inputs.items():
            hand_state = claim_state[hand_name]
            hand_state["hold_hits"] = (
                min(hand_state["hold_hits"] + 1, HAND_CLAIM_HOLD_HITS)
                if hold_raw
                else max(hand_state["hold_hits"] - 1, 0)
            )
            hand_state["carry_hits"] = (
                min(hand_state["carry_hits"] + 1, HAND_CLAIM_CARRY_HITS)
                if carry_raw
                else max(hand_state["carry_hits"] - 1, 0)
            )

            if not hold_raw and not carry_raw:
                hand_state["misses"] += 1
            else:
                hand_state["misses"] = 0

            if hand_state["misses"] >= HAND_CLAIM_RESET_MISSES:
                claim_state[hand_name] = self._new_hand_claim_entry()
                continue

            if hand_state["claim"] is not None:
                continue

            hold_ready = hand_state["hold_hits"] >= HAND_CLAIM_HOLD_HITS
            carry_ready = hand_state["carry_hits"] >= HAND_CLAIM_CARRY_HITS
            if hold_ready and carry_ready:
                hand_state["claim"] = (
                    "HOLD"
                    if hand_state["hold_hits"] >= hand_state["carry_hits"]
                    else "CARRY"
                )
            elif hold_ready:
                hand_state["claim"] = "HOLD"
            elif carry_ready:
                hand_state["claim"] = "CARRY"

        return claim_state

    @staticmethod
    def _record_perf(perf, key, start_time):
        if perf is None or start_time is None:
            return
        perf[key] = perf.get(key, 0.0) + (time.perf_counter() - start_time) * 1000.0

    def _resolve_hold_and_claim_state(
        self,
        track_id,
        hold_direction,
        handrail_evidence,
        keypoints,
        holding_raw,
        best_wrist,
        features,
    ):
        hold_state_before_claim = self._evaluate_hold_state(
            hold_direction,
            handrail_evidence,
        )
        carry_pose = self._get_carry_pose(
            keypoints,
            holding_raw,
            best_wrist,
            features=features,
        )
        hand_claim_state = self._update_hand_claim_state(
            track_id,
            hold_state_before_claim["left_hold_raw"],
            hold_state_before_claim["right_hold_raw"],
            carry_pose["left_carry_raw_before_claim"],
            carry_pose["right_carry_raw_before_claim"],
        )
        hold_state = self._apply_hand_claim_to_hold_state(
            hold_state_before_claim,
            hand_claim_state,
        )
        carry_pose = self._apply_hand_claim_to_carry_pose(
            carry_pose,
            hand_claim_state,
        )
        return hold_state, carry_pose, hand_claim_state

    @staticmethod
    def _copy_hold_state_fields(hold_state):
        return (
            hold_state["holding_raw"],
            hold_state["hold_raw_status"],
            hold_state["holding_correct_raw"],
            hold_state["holding_wrong_raw"],
            hold_state["hold_status_correct"],
            hold_state["hold_status_wrong"],
            hold_state["dist_correct"],
            hold_state["dist_wrong"],
            hold_state["seg_dist_correct"],
            hold_state["seg_dist_wrong"],
            hold_state["t_correct"],
            hold_state["t_wrong"],
            hold_state["wrist_side_correct"],
            hold_state["wrist_side_wrong"],
            hold_state["best_wrist_correct"],
            hold_state["best_wrist_wrong"],
            hold_state["best_wrist_correct_point"],
            hold_state["best_wrist_wrong_point"],
            hold_state["correct_line_name"],
            hold_state["wrong_line_name"],
            hold_state["correct_rule"],
            hold_state["wrong_rule"],
            hold_state["best_wrist"],
            hold_state["best_wrist_point"],
            hold_state["dist_wrist"],
            hold_state["wrist_side"],
        )

    def analyze(self, track_id, p_lane, p_motion, keypoints, box=None, features=None):
        perf = {} if ENABLE_PERF_LOG else None
        analyze_start = time.perf_counter() if perf is not None else None

        features = features or extract_pose_features(keypoints, box)
        if p_lane is None:
            p_lane = features.get("feet_point")
        if p_motion is None:
            p_motion = features.get("motion_point")

        body_facing = features.get("body_facing", "UNKNOWN")
        arm_side_order = features.get("arm_side_order", "UNKNOWN")

        direction = "ANALYZING"
        dy = None
        v = None
        wrong_lane_raw = False
        wrong_lane = False
        lane_wrong_hits = 0
        inside_stairs = False

        holding = False
        holding_raw = False
        hold_raw_status = "UNKNOWN"
        hold_final_status = "ANALYZING"

        hold_correct_hits = 0
        hold_wrong_side_hits = 0
        hold_none_hits = 0
        hold_unknown_hits = 0
        hold_not_hold_evidence_hits = 0

        holding_correct_raw = False
        holding_wrong_raw = False
        hold_status_correct = "UNKNOWN"
        hold_status_wrong = "UNKNOWN"

        dist_correct = -999
        dist_wrong = -999
        seg_dist_correct = None
        seg_dist_wrong = None
        t_correct = None
        t_wrong = None
        wrist_side_correct = "UNKNOWN"
        wrist_side_wrong = "UNKNOWN"
        best_wrist_correct = "NONE"
        best_wrist_wrong = "NONE"
        best_wrist_correct_point = None
        best_wrist_wrong_point = None
        correct_line_name = "NONE"
        wrong_line_name = "NONE"
        correct_rule = "NA"
        wrong_rule = "NA"

        best_wrist = "NONE"
        best_wrist_point = None
        dist_wrist = -999
        wrist_side = "UNKNOWN"
        backward_raw = False
        backward_hits = 0
        backward_confirmed = False
        standing_raw = False
        standing_hits = 0
        standing_still_confirmed = False
        standing_motion_range = None
        standing_len = 0
        left_hand_claim = "NONE"
        right_hand_claim = "NONE"
        left_hold_claim_hits = 0
        right_hold_claim_hits = 0
        left_carry_claim_hits = 0
        right_carry_claim_hits = 0
        left_hold_raw_before_claim = False
        right_hold_raw_before_claim = False
        left_hold_raw_after_claim = False
        right_hold_raw_after_claim = False
        left_carry_raw_before_claim = False
        right_carry_raw_before_claim = False
        left_carry_raw_after_claim = False
        right_carry_raw_after_claim = False
        carry_info = None
        warnings = []

        if track_id not in self.track_history:
            self.track_history[track_id] = []
        if p_motion is not None:
            self.track_history[track_id].append(p_motion[1])

        standing_start = time.perf_counter() if perf is not None else None
        inside_stairs = self.is_inside_stairs(p_lane)
        if inside_stairs:
            (
                standing_raw,
                standing_hits,
                standing_still_confirmed,
                standing_motion_range,
                standing_len,
            ) = self.update_standing_still(track_id, p_motion)
        self._record_perf(perf, "standing", standing_start)

        def build_result(
            context,
            status,
            color,
            display_status="",
            carry_info_override=None,
            **overrides,
        ):
            self._record_perf(perf, "analyze", analyze_start)
            return self._build_result_from_context(
                context,
                status=status,
                display_status=display_status,
                color=color,
                perf=perf,
                carry_info=carry_info_override,
                **overrides,
            )

        if len(self.track_history[track_id]) < DIRECTION_MIN_FRAMES:
            if not inside_stairs:
                self._reset_behavior_histories(track_id)
                hold_raw_status = "OUTSIDE"
                hold_final_status = "OUTSIDE"
                standing_raw = False
                standing_hits = 0
                standing_still_confirmed = False
                standing_motion_range = None
                standing_len = 0
                return build_result(
                    locals(),
                    status="ANALYZING...",
                    display_status="" if DEMO_MODE else "ANALYZING...",
                    color=OUTSIDE_COLOR,
                )

            hold_direction = self.last_valid_direction.get(track_id)
            hold_start = time.perf_counter() if perf is not None else None
            handrail_evidence = compute_handrail_evidence(
                features,
                self.left_line,
                self.right_line,
            )
            hold_state, carry_pose, hand_claim_state = self._resolve_hold_and_claim_state(
                track_id,
                hold_direction,
                handrail_evidence,
                keypoints,
                holding_raw,
                best_wrist,
                features,
            )
            self._record_perf(perf, "hold", hold_start)
            left_hand_claim = hand_claim_state["left"]["claim"] or "NONE"
            right_hand_claim = hand_claim_state["right"]["claim"] or "NONE"
            left_hold_claim_hits = hand_claim_state["left"]["hold_hits"]
            right_hold_claim_hits = hand_claim_state["right"]["hold_hits"]
            left_carry_claim_hits = hand_claim_state["left"]["carry_hits"]
            right_carry_claim_hits = hand_claim_state["right"]["carry_hits"]
            left_hold_raw_before_claim = hold_state["left_hold_raw_before_claim"]
            right_hold_raw_before_claim = hold_state["right_hold_raw_before_claim"]
            left_hold_raw_after_claim = hold_state["left_hold_raw_after_claim"]
            right_hold_raw_after_claim = hold_state["right_hold_raw_after_claim"]
            left_carry_raw_before_claim = carry_pose["left_carry_raw_before_claim"]
            right_carry_raw_before_claim = carry_pose["right_carry_raw_before_claim"]
            left_carry_raw_after_claim = carry_pose["left_carry_raw_after_claim"]
            right_carry_raw_after_claim = carry_pose["right_carry_raw_after_claim"]
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
            (
                hold_correct_hits,
                hold_wrong_side_hits,
                hold_none_hits,
                hold_unknown_hits,
                hold_not_hold_evidence_hits,
                hold_final_status,
                holding,
            ) = self._update_hold_status_history(track_id, hold_raw_status)

            warnings = self._collect_warnings(
                False,
                hold_final_status,
                False,
                standing_still_confirmed,
            )
            if warnings:
                status, display_status, color = self._summarize_result(
                    direction,
                    warnings,
                    hold_final_status,
                    UNKNOWN_COLOR if DEMO_MODE else ANALYZING_COLOR,
                    "ANALYZING...",
                )
                return build_result(
                    locals(),
                    status=status,
                    display_status=display_status,
                    color=color,
                )

            return build_result(
                locals(),
                status="ANALYZING...",
                display_status="" if DEMO_MODE else "ANALYZING...",
                color=UNKNOWN_COLOR if DEMO_MODE else ANALYZING_COLOR,
            )

        direction_start = time.perf_counter() if perf is not None else None
        dy = self.track_history[track_id][-1] - self.track_history[track_id][0]
        self.track_history[track_id] = self.track_history[track_id][
            -DIRECTION_HISTORY_LEN:
        ]

        if DIRECTION_SIGN_NORMAL:
            direction = (
                "UP"
                if dy < -DIRECTION_PIXEL_THRESHOLD
                else "DOWN" if dy > DIRECTION_PIXEL_THRESHOLD else "IDLE"
            )
        else:
            direction = (
                "DOWN"
                if dy < -DIRECTION_PIXEL_THRESHOLD
                else "UP" if dy > DIRECTION_PIXEL_THRESHOLD else "IDLE"
            )
        self._record_perf(perf, "direction", direction_start)

        if not inside_stairs:
            self._reset_behavior_histories(track_id)
            standing_raw = False
            standing_hits = 0
            standing_still_confirmed = False
            standing_motion_range = None
            standing_len = 0
            return build_result(
                locals(),
                status=f"{direction} | Ngoai Vung",
                display_status="" if DEMO_MODE else f"{direction} | Ngoai Vung",
                color=OUTSIDE_COLOR,
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

        if direction in ("UP", "DOWN"):
            self.last_valid_direction[track_id] = direction

        backward_start = time.perf_counter() if perf is not None else None
        if direction == "DOWN" and is_front_to_camera(body_facing):
            backward_raw = True
        elif direction == "UP" and is_back_to_camera(body_facing):
            backward_raw = True

        backward_hits, backward_confirmed = self._update_backward_history(
            track_id, backward_raw
        )
        self._record_perf(perf, "backward", backward_start)

        lane_start = time.perf_counter() if perf is not None else None
        if len(self.center_line) >= 2:
            a, b = self.center_line[0], self.center_line[1]
            v = (b[0] - a[0]) * (p_lane[1] - a[1]) - (b[1] - a[1]) * (
                p_lane[0] - a[0]
            )

            if direction in ("UP", "DOWN"):
                if LANE_SIGN_NORMAL:
                    wrong_lane_raw = (direction == "UP" and v < 0) or (
                        direction == "DOWN" and v > 0
                    )
                else:
                    wrong_lane_raw = (direction == "UP" and v > 0) or (
                        direction == "DOWN" and v < 0
                    )
                lane_wrong_hits, wrong_lane = self._update_lane_history(
                    track_id, wrong_lane_raw
                )
            else:
                lane_wrong_hits, wrong_lane = self._get_lane_history_state(track_id)
        self._record_perf(perf, "lane", lane_start)

        # Khi chua biet chieu di, khong the xac dinh lan can dung/sai ben.
        # Nhung van co the kiem tra xem nguoi do co vin bat ky lan can nao khong.
        # Vi vay UNKNOWN/IDLE khong duoc tu dong coi la "Khong Vin".
        hold_direction = direction
        if direction not in ("UP", "DOWN"):
            # Neu track da tung co huong hop le, dung lai huong cu khi ho dung yen tam thoi.
            hold_direction = self.last_valid_direction.get(track_id)

        hold_start = time.perf_counter() if perf is not None else None
        handrail_evidence = compute_handrail_evidence(
            features,
            self.left_line,
            self.right_line,
        )
        hold_state, carry_pose, hand_claim_state = self._resolve_hold_and_claim_state(
            track_id,
            hold_direction,
            handrail_evidence,
            keypoints,
            holding_raw,
            best_wrist,
            features,
        )
        self._record_perf(perf, "hold", hold_start)
        left_hand_claim = hand_claim_state["left"]["claim"] or "NONE"
        right_hand_claim = hand_claim_state["right"]["claim"] or "NONE"
        left_hold_claim_hits = hand_claim_state["left"]["hold_hits"]
        right_hold_claim_hits = hand_claim_state["right"]["hold_hits"]
        left_carry_claim_hits = hand_claim_state["left"]["carry_hits"]
        right_carry_claim_hits = hand_claim_state["right"]["carry_hits"]
        left_hold_raw_before_claim = hold_state["left_hold_raw_before_claim"]
        right_hold_raw_before_claim = hold_state["right_hold_raw_before_claim"]
        left_hold_raw_after_claim = hold_state["left_hold_raw_after_claim"]
        right_hold_raw_after_claim = hold_state["right_hold_raw_after_claim"]
        left_carry_raw_before_claim = carry_pose["left_carry_raw_before_claim"]
        right_carry_raw_before_claim = carry_pose["right_carry_raw_before_claim"]
        left_carry_raw_after_claim = carry_pose["left_carry_raw_after_claim"]
        right_carry_raw_after_claim = carry_pose["right_carry_raw_after_claim"]
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

        (
            hold_correct_hits,
            hold_wrong_side_hits,
            hold_none_hits,
            hold_unknown_hits,
            hold_not_hold_evidence_hits,
            hold_final_status,
            holding,
        ) = self._update_hold_status_history(track_id, hold_raw_status)

        if direction == "IDLE" and not standing_still_confirmed:
            return build_result(
                locals(),
                status="IDLE",
                display_status="",
                color=SAFE_COLOR if DEMO_MODE else (200, 200, 200),
            )

        if direction == "IDLE":
            warnings = self._collect_warnings(
                wrong_lane,
                hold_final_status,
                backward_confirmed,
                standing_still_confirmed,
            )
            status, display_status, color = self._summarize_result(
                direction,
                warnings,
                hold_final_status,
                SAFE_COLOR if DEMO_MODE else (200, 200, 200),
                direction,
            )

            return build_result(
                locals(),
                status=status,
                display_status=display_status,
                color=color,
            )

        carry_start = time.perf_counter() if perf is not None else None
        carry_info = self._analyze_carry(
            track_id,
            keypoints,
            holding_raw,
            best_wrist,
            features=features,
            carry_pose=carry_pose,
        )
        self._record_perf(perf, "carry", carry_start)
        warnings = self._collect_warnings(
            wrong_lane,
            hold_final_status,
            backward_confirmed,
            standing_still_confirmed,
            is_carrying=carry_info["is_carrying"],
        )
        status, display_status, color = self._summarize_result(
            direction,
            warnings,
            hold_final_status,
            SAFE_COLOR,
            f"{direction} | An Toan",
        )

        return build_result(
            locals(),
            status=status,
            display_status=display_status,
            color=color,
            carry_info_override=carry_info,
        )

    def cleanup_inactive_tracks(self, active_track_ids):
        # Xoa history cua track da mat de tranh memory tang va tranh huong cu
        # dinh sang nguoi moi khi tracker tai su dung track_id.
        active_track_ids = set(active_track_ids)
        history_maps = [
            self.track_history,
            self.lane_history,
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
                track_id
                for track_id in history_map.keys()
                if track_id not in active_track_ids
            ]
            for track_id in inactive_ids:
                del history_map[track_id]
