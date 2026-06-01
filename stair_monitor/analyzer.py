import cv2
import numpy as np

from stair_monitor.behavior_history import BehaviorHistoryMixin
from stair_monitor.carry_analysis import CarryAnalysisMixin
from stair_monitor.geometry import estimate_arm_side_order, estimate_body_facing
from stair_monitor.handrail_analysis import (
    HandrailAnalysisMixin,
    get_best_wrist_for_handrail_by_rule,
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
    HANDRAIL_SEGMENT_MAX_DISTANCE,
    LANE_SIGN_NORMAL,
    OUTSIDE_COLOR,
    SAFE_COLOR,
    STAIRS_LEFT_EXPAND_BOTTOM_PX,
    STAIRS_LEFT_EXPAND_TOP_PX,
    UNKNOWN_COLOR,
    WRONG_SIDE_HANDRAIL_SEGMENT_MAX_DISTANCE,
)


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

    def analyze(self, track_id, p_lane, p_motion, keypoints):
        body_facing = estimate_body_facing(keypoints)
        arm_side_order = estimate_arm_side_order(keypoints)

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

        if track_id not in self.track_history:
            self.track_history[track_id] = []
        self.track_history[track_id].append(p_motion[1])

        inside_stairs = self.is_inside_stairs(p_lane)
        if inside_stairs:
            (
                standing_raw,
                standing_hits,
                standing_still_confirmed,
                standing_motion_range,
                standing_len,
            ) = self.update_standing_still(track_id, p_motion)

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
                return self._build_result_from_context(
                    locals(),
                    status="ANALYZING...",
                    display_status="" if DEMO_MODE else "ANALYZING...",
                    color=OUTSIDE_COLOR,
                )

            warnings = self._collect_warnings(
                False,
                "ANALYZING",
                False,
                standing_still_confirmed,
            )
            if warnings:
                status, display_status, color = self._summarize_result(
                    direction,
                    warnings,
                    "ANALYZING",
                    UNKNOWN_COLOR if DEMO_MODE else ANALYZING_COLOR,
                    "ANALYZING...",
                )
                return self._build_result_from_context(
                    locals(),
                    status=status,
                    display_status=display_status,
                    color=color,
                )

            return self._build_result_from_context(
                locals(),
                status="ANALYZING...",
                display_status="" if DEMO_MODE else "ANALYZING...",
                color=UNKNOWN_COLOR if DEMO_MODE else ANALYZING_COLOR,
            )

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

        if not inside_stairs:
            self._reset_behavior_histories(track_id)
            standing_raw = False
            standing_hits = 0
            standing_still_confirmed = False
            standing_motion_range = None
            standing_len = 0
            return self._build_result_from_context(
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

        if direction == "DOWN" and is_front_to_camera(body_facing):
            backward_raw = True
        elif direction == "UP" and is_back_to_camera(body_facing):
            backward_raw = True

        backward_hits, backward_confirmed = self._update_backward_history(
            track_id, backward_raw
        )

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

        (
            correct_line,
            correct_line_name,
            correct_rule,
            wrong_line,
            wrong_line_name,
            wrong_rule,
        ) = self._get_handrail_targets(direction, self.left_line, self.right_line)

        if correct_line is not None and len(correct_line) >= 2:
            (
                holding_correct_raw,
                dist_correct,
                wrist_side_correct,
                best_wrist_correct,
                best_wrist_correct_point,
                hold_status_correct,
                seg_dist_correct,
                t_correct,
            ) = get_best_wrist_for_handrail_by_rule(
                keypoints,
                correct_line,
                correct_rule,
                HANDRAIL_SEGMENT_MAX_DISTANCE,
            )

        if wrong_line is not None and len(wrong_line) >= 2:
            (
                holding_wrong_raw,
                dist_wrong,
                wrist_side_wrong,
                best_wrist_wrong,
                best_wrist_wrong_point,
                hold_status_wrong,
                seg_dist_wrong,
                t_wrong,
            ) = get_best_wrist_for_handrail_by_rule(
                keypoints,
                wrong_line,
                wrong_rule,
                WRONG_SIDE_HANDRAIL_SEGMENT_MAX_DISTANCE,
            )

        if holding_wrong_raw:
            hold_raw_status = "WRONG_SIDE"
            holding_raw = False
        elif holding_correct_raw:
            hold_raw_status = "CORRECT"
            holding_raw = True
        elif hold_status_correct == "UNKNOWN" and hold_status_wrong == "UNKNOWN":
            hold_raw_status = "UNKNOWN"
            holding_raw = False
        else:
            hold_raw_status = "NONE"
            holding_raw = False

        (
            hold_correct_hits,
            hold_wrong_side_hits,
            hold_none_hits,
            hold_unknown_hits,
            hold_final_status,
            holding,
        ) = self._update_hold_status_history(track_id, hold_raw_status)

        (
            best_wrist,
            best_wrist_point,
            dist_wrist,
            wrist_side,
        ) = self._select_primary_hold_debug(
            hold_raw_status,
            best_wrist_correct,
            best_wrist_correct_point,
            dist_correct,
            wrist_side_correct,
            best_wrist_wrong,
            best_wrist_wrong_point,
            dist_wrong,
            wrist_side_wrong,
        )

        if direction == "IDLE" and not standing_still_confirmed:
            return self._build_result_from_context(
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

            return self._build_result_from_context(
                locals(),
                status=status,
                display_status=display_status,
                color=color,
            )

        carry_info = self._analyze_carry(track_id, keypoints, holding_raw, best_wrist)
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

        return self._build_result_from_context(
            locals(),
            status=status,
            display_status=display_status,
            color=color,
            carry_info=carry_info,
        )
