import cv2
import numpy as np

from stair_monitor.carry import detect_carrying_pose
from stair_monitor.geometry import (
    estimate_arm_side_order,
    estimate_body_facing,
    get_side_name,
    point_to_segment_distance,
    signed_distance_to_line,
)
from stair_monitor.settings import (
    ANALYZING_COLOR,
    BACKWARD_HISTORY_LEN,
    BACKWARD_MIN_HITS,
    DEMO_MODE,
    DIRECTION_HISTORY_LEN,
    DIRECTION_MIN_FRAMES,
    DIRECTION_PIXEL_THRESHOLD,
    DIRECTION_SIGN_NORMAL,
    DRAW_SAFE_STATUS,
    FRONT_CARRY_HISTORY_LEN,
    FRONT_CARRY_MIN_HITS,
    FRONT_CARRY_ONE_ARM_HISTORY_LEN,
    FRONT_CARRY_ONE_ARM_MIN_HITS,
    HANDRAIL_SEGMENT_MAX_DISTANCE,
    HOLD_HISTORY_LEN,
    HOLD_MIN_NOT_HOLD_HITS,
    HOLD_MIN_WRONG_SIDE_HITS,
    LANE_HISTORY_LEN,
    LANE_MIN_WRONG_HITS,
    LANE_SIGN_NORMAL,
    LEFT_HANDRAIL_MAX_DISTANCE,
    OUTSIDE_COLOR,
    RIGHT_HANDRAIL_MAX_DISTANCE,
    WRONG_SIDE_HANDRAIL_SEGMENT_MAX_DISTANCE,
    SAFE_COLOR,
    STAIRS_LEFT_EXPAND_BOTTOM_PX,
    STAIRS_LEFT_EXPAND_TOP_PX,
    UNKNOWN_COLOR,
    VIOLATION_COLOR,
)

LEFT_HANDRAIL_RULE = "LEFT_HANDRAIL_RULE"
RIGHT_HANDRAIL_RULE = "RIGHT_HANDRAIL_RULE"
REAL_VIOLATION_LABELS = (
    "Sai Lan",
    "Khong Vin",
    "Vin Sai Ben",
    "Mang Vac",
    "Di Lui",
)


def is_front_to_camera(body_facing):
    return body_facing is not None and "FRONT_TO_CAMERA" in str(body_facing)


def is_back_to_camera(body_facing):
    return body_facing is not None and "BACK_TO_CAMERA" in str(body_facing)


def get_best_wrist_for_handrail_by_rule(keypoints, line, rule, segment_max_distance):
    """
    Check both wrists and choose the closest valid wrist to the handrail.

    Rule is fixed by physical handrail side, not motion direction.

    Returns:
    - holding: True/False
    - dist: signed distance to the handrail
    - side: LEFT_SIDE / RIGHT_SIDE / ON_LINE / UNKNOWN
    - wrist_name: LEFT_WRIST / RIGHT_WRIST / NONE
    - wrist_point: chosen wrist point or None
    - hold_status: TRUE / FALSE / UNKNOWN
    - segment_dist: distance to the handrail segment
    - projection_t: unclamped projection coefficient along the segment
    """
    if keypoints is None or len(keypoints) < 11 or len(line) < 2:
        return False, -999, "UNKNOWN", "NONE", None, "UNKNOWN", None, None

    candidates = []

    if len(keypoints[9]) > 2 and keypoints[9][2] > 0.5:
        wrist_point = (int(keypoints[9][0]), int(keypoints[9][1]))
        d = signed_distance_to_line(wrist_point, line)
        segment_dist, projection_t, _ = point_to_segment_distance(
            wrist_point, line[0], line[1]
        )
        valid_d = False
        if rule == LEFT_HANDRAIL_RULE:
            valid_d = -LEFT_HANDRAIL_MAX_DISTANCE <= d <= -20
        elif rule == RIGHT_HANDRAIL_RULE:
            valid_d = 0 <= d <= RIGHT_HANDRAIL_MAX_DISTANCE
        valid_segment = 0.0 <= projection_t <= 1.0 and (
            segment_dist <= segment_max_distance
        )
        candidates.append(
            {
                "name": "LEFT_WRIST",
                "point": wrist_point,
                "dist": d,
                "segment_dist": segment_dist,
                "projection_t": projection_t,
                "side": get_side_name(d),
                "valid_d": valid_d,
                "valid_segment": valid_segment,
                "valid": valid_d and valid_segment,
            }
        )

    if len(keypoints[10]) > 2 and keypoints[10][2] > 0.25:
        wrist_point = (int(keypoints[10][0]), int(keypoints[10][1]))
        d = signed_distance_to_line(wrist_point, line)
        segment_dist, projection_t, _ = point_to_segment_distance(
            wrist_point, line[0], line[1]
        )
        valid_d = False
        if rule == LEFT_HANDRAIL_RULE:
            valid_d = -LEFT_HANDRAIL_MAX_DISTANCE <= d <= -20
        elif rule == RIGHT_HANDRAIL_RULE:
            valid_d = 0 <= d <= RIGHT_HANDRAIL_MAX_DISTANCE
        valid_segment = 0.0 <= projection_t <= 1.0 and (
            segment_dist <= segment_max_distance
        )
        candidates.append(
            {
                "name": "RIGHT_WRIST",
                "point": wrist_point,
                "dist": d,
                "segment_dist": segment_dist,
                "projection_t": projection_t,
                "side": get_side_name(d),
                "valid_d": valid_d,
                "valid_segment": valid_segment,
                "valid": valid_d and valid_segment,
            }
        )

    if not candidates:
        return False, -999, "UNKNOWN", "NONE", None, "UNKNOWN", None, None

    valid_candidates = [c for c in candidates if c["valid"]]
    if valid_candidates:
        best = min(
            valid_candidates, key=lambda c: (c["segment_dist"], abs(c["dist"]))
        )
        return (
            True,
            best["dist"],
            best["side"],
            best["name"],
            best["point"],
            "TRUE",
            best["segment_dist"],
            best["projection_t"],
        )

    best = min(candidates, key=lambda c: (c["segment_dist"], abs(c["dist"])))
    return (
        False,
        best["dist"],
        best["side"],
        best["name"],
        best["point"],
        "FALSE",
        best["segment_dist"],
        best["projection_t"],
    )


class BehaviorAnalyzer:
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

    def is_inside_stairs(self, p_lane):
        if p_lane is None or len(self.stairs_poly) < 3:
            return False

        return cv2.pointPolygonTest(
            self.stairs_poly,
            (float(p_lane[0]), float(p_lane[1])),
            False,
        ) >= 0

    def _reset_behavior_histories(self, track_id):
        if track_id in self.lane_history:
            self.lane_history[track_id] = []
        if track_id in self.hold_status_history:
            self.hold_status_history[track_id] = []
        if track_id in self.front_carry_history:
            self.front_carry_history[track_id] = []
        if track_id in self.front_carry_one_arm_history:
            self.front_carry_one_arm_history[track_id] = []
        if track_id in self.backward_history:
            self.backward_history[track_id] = []
        if hasattr(self, "violation_history") and track_id in self.violation_history:
            self.violation_history[track_id] = []

    def _update_hold_status_history(self, track_id, hold_final_status_raw):
        if track_id not in self.hold_status_history:
            self.hold_status_history[track_id] = []

        self.hold_status_history[track_id].append(hold_final_status_raw)
        self.hold_status_history[track_id] = self.hold_status_history[track_id][
            -HOLD_HISTORY_LEN:
        ]

        history = self.hold_status_history[track_id]
        hold_correct_hits = sum(1 for s in history if s == "CORRECT")
        hold_wrong_side_hits = sum(1 for s in history if s == "WRONG_SIDE")
        hold_none_hits = sum(1 for s in history if s == "NONE")
        hold_unknown_hits = sum(1 for s in history if s == "UNKNOWN")

        hold_correct_confirmed = (
            hold_correct_hits > 0
            and hold_correct_hits >= max(4, HOLD_HISTORY_LEN // 4)
        )
        hold_wrong_side_confirmed = (
            len(history) >= HOLD_HISTORY_LEN
            and hold_wrong_side_hits >= HOLD_MIN_WRONG_SIDE_HITS
        )
        hold_none_confirmed = (
            len(history) >= HOLD_HISTORY_LEN
            and hold_none_hits >= HOLD_MIN_NOT_HOLD_HITS
        )

        if hold_wrong_side_confirmed:
            hold_final_status = "WRONG_SIDE"
            holding = False
        elif hold_correct_confirmed:
            hold_final_status = "CORRECT"
            holding = True
        elif hold_none_confirmed:
            hold_final_status = "NONE"
            holding = False
        elif (
            hold_unknown_hits > hold_correct_hits
            and hold_unknown_hits > hold_none_hits
        ):
            hold_final_status = "UNKNOWN"
            holding = False
        else:
            hold_final_status = "ANALYZING"
            holding = False

        return (
            hold_correct_hits,
            hold_wrong_side_hits,
            hold_none_hits,
            hold_unknown_hits,
            hold_final_status,
            holding,
        )

    def _get_lane_history_state(self, track_id):
        history = self.lane_history.get(track_id, [])
        lane_wrong_hits = sum(1 for is_wrong in history if is_wrong)
        wrong_lane_confirmed = (
            len(history) >= LANE_HISTORY_LEN
            and lane_wrong_hits >= LANE_MIN_WRONG_HITS
        )
        return lane_wrong_hits, wrong_lane_confirmed

    def _update_lane_history(self, track_id, wrong_lane_raw):
        if track_id not in self.lane_history:
            self.lane_history[track_id] = []

        self.lane_history[track_id].append(wrong_lane_raw)
        self.lane_history[track_id] = self.lane_history[track_id][-LANE_HISTORY_LEN:]

        return self._get_lane_history_state(track_id)

    def _update_backward_history(self, track_id, backward_raw):
        if track_id not in self.backward_history:
            self.backward_history[track_id] = []

        self.backward_history[track_id].append(backward_raw)
        self.backward_history[track_id] = self.backward_history[track_id][
            -BACKWARD_HISTORY_LEN:
        ]

        backward_hits = sum(self.backward_history[track_id])
        backward_confirmed = (
            len(self.backward_history[track_id]) >= BACKWARD_HISTORY_LEN
            and backward_hits >= BACKWARD_MIN_HITS
        )
        return backward_hits, backward_confirmed

    @staticmethod
    def _build_display_status(direction, warnings):
        if DEMO_MODE:
            return " - ".join(warnings)
        if warnings:
            return f"{direction} | {' - '.join(warnings)}"
        if DRAW_SAFE_STATUS:
            return f"{direction} | An Toan"
        return ""

    @staticmethod
    def _select_primary_hold_debug(
        hold_raw_status,
        best_wrist_correct,
        best_wrist_correct_point,
        dist_correct,
        wrist_side_correct,
        best_wrist_wrong,
        best_wrist_wrong_point,
        dist_wrong,
        wrist_side_wrong,
    ):
        if hold_raw_status == "CORRECT":
            return (
                best_wrist_correct,
                best_wrist_correct_point,
                dist_correct,
                wrist_side_correct,
            )
        if hold_raw_status == "WRONG_SIDE":
            return (
                best_wrist_wrong,
                best_wrist_wrong_point,
                dist_wrong,
                wrist_side_wrong,
            )

        correct_valid = dist_correct != -999
        wrong_valid = dist_wrong != -999

        if correct_valid and wrong_valid:
            if abs(dist_correct) <= abs(dist_wrong):
                return (
                    best_wrist_correct,
                    best_wrist_correct_point,
                    dist_correct,
                    wrist_side_correct,
                )
            return (
                best_wrist_wrong,
                best_wrist_wrong_point,
                dist_wrong,
                wrist_side_wrong,
            )
        if correct_valid:
            return (
                best_wrist_correct,
                best_wrist_correct_point,
                dist_correct,
                wrist_side_correct,
            )
        if wrong_valid:
            return (
                best_wrist_wrong,
                best_wrist_wrong_point,
                dist_wrong,
                wrist_side_wrong,
            )
        return "NONE", None, -999, "UNKNOWN"

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

        if track_id not in self.track_history:
            self.track_history[track_id] = []
        self.track_history[track_id].append(p_motion[1])

        inside_stairs = self.is_inside_stairs(p_lane)

        if len(self.track_history[track_id]) < DIRECTION_MIN_FRAMES:
            if not inside_stairs:
                self._reset_behavior_histories(track_id)
                hold_raw_status = "OUTSIDE"
                hold_final_status = "OUTSIDE"
            return self._build_result(
                status="ANALYZING...",
                display_status="" if DEMO_MODE else "ANALYZING...",
                color=OUTSIDE_COLOR
                if not inside_stairs
                else UNKNOWN_COLOR if DEMO_MODE else ANALYZING_COLOR,
                direction=direction,
                lane_raw=wrong_lane_raw,
                lane_hits=lane_wrong_hits,
                lane_conf=wrong_lane,
                inside_stairs=inside_stairs,
                holding=holding,
                holding_raw=holding_raw,
                hold_status=hold_final_status,
                hold_raw_status=hold_raw_status,
                hold_confirmed_status=hold_final_status,
                hold_correct_hits=hold_correct_hits,
                hold_wrong_side_hits=hold_wrong_side_hits,
                hold_none_hits=hold_none_hits,
                hold_unknown_hits=hold_unknown_hits,
                holding_correct_raw=holding_correct_raw,
                holding_wrong_raw=holding_wrong_raw,
                hold_status_correct=hold_status_correct,
                hold_status_wrong=hold_status_wrong,
                dist_wrist=dist_wrist,
                wrist_side=wrist_side,
                best_wrist=best_wrist,
                best_wrist_point=best_wrist_point,
                dist_correct=dist_correct,
                dist_wrong=dist_wrong,
                wrist_side_correct=wrist_side_correct,
                wrist_side_wrong=wrist_side_wrong,
                best_wrist_correct=best_wrist_correct,
                best_wrist_wrong=best_wrist_wrong,
                best_wrist_correct_point=best_wrist_correct_point,
                best_wrist_wrong_point=best_wrist_wrong_point,
                correct_line_name=correct_line_name,
                wrong_line_name=wrong_line_name,
                correct_rule=correct_rule,
                wrong_rule=wrong_rule,
                p_lane=p_lane,
                p_motion=p_motion,
                body_facing=body_facing,
                arm_side_order=arm_side_order,
                backward_raw=backward_raw,
                backward_hits=backward_hits,
                backward_confirmed=backward_confirmed,
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
            return self._build_result(
                status=f"{direction} | Ngoai Vung",
                display_status="" if DEMO_MODE else f"{direction} | Ngoai Vung",
                color=OUTSIDE_COLOR,
                dy=dy,
                lane_v=v,
                direction=direction,
                lane_raw=False,
                lane_hits=0,
                lane_conf=False,
                wrong_lane=wrong_lane,
                inside_stairs=False,
                holding=holding,
                holding_raw=holding_raw,
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
                p_lane=p_lane,
                p_motion=p_motion,
                body_facing=body_facing,
                arm_side_order=arm_side_order,
                backward_raw=backward_raw,
                backward_hits=backward_hits,
                backward_confirmed=backward_confirmed,
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

        if direction == "UP":
            correct_line = self.left_line
            correct_line_name = "LEFT_HANDRAIL"
            correct_rule = LEFT_HANDRAIL_RULE
            wrong_line = self.right_line
            wrong_line_name = "RIGHT_HANDRAIL"
            wrong_rule = RIGHT_HANDRAIL_RULE
        elif direction == "DOWN":
            correct_line = self.right_line
            correct_line_name = "RIGHT_HANDRAIL"
            correct_rule = RIGHT_HANDRAIL_RULE
            wrong_line = self.left_line
            wrong_line_name = "LEFT_HANDRAIL"
            wrong_rule = LEFT_HANDRAIL_RULE
        else:
            correct_line = None
            wrong_line = None

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

        if direction == "IDLE":
            return self._build_result(
                status="IDLE",
                display_status="",
                color=SAFE_COLOR if DEMO_MODE else (200, 200, 200),
                dy=dy,
                lane_v=v,
                direction=direction,
                lane_raw=wrong_lane_raw,
                lane_hits=lane_wrong_hits,
                lane_conf=wrong_lane,
                wrong_lane=wrong_lane,
                inside_stairs=True,
                holding=holding,
                holding_raw=holding_raw,
                hold_status=hold_final_status,
                hold_raw_status=hold_raw_status,
                hold_confirmed_status=hold_final_status,
                hold_correct_hits=hold_correct_hits,
                hold_wrong_side_hits=hold_wrong_side_hits,
                hold_none_hits=hold_none_hits,
                hold_unknown_hits=hold_unknown_hits,
                holding_correct_raw=holding_correct_raw,
                holding_wrong_raw=holding_wrong_raw,
                hold_status_correct=hold_status_correct,
                hold_status_wrong=hold_status_wrong,
                dist_wrist=dist_wrist,
                wrist_side=wrist_side,
                best_wrist=best_wrist,
                best_wrist_point=best_wrist_point,
                dist_correct=dist_correct,
                dist_wrong=dist_wrong,
                seg_dist_correct=seg_dist_correct,
                seg_dist_wrong=seg_dist_wrong,
                t_correct=t_correct,
                t_wrong=t_wrong,
                wrist_side_correct=wrist_side_correct,
                wrist_side_wrong=wrist_side_wrong,
                best_wrist_correct=best_wrist_correct,
                best_wrist_wrong=best_wrist_wrong,
                best_wrist_correct_point=best_wrist_correct_point,
                best_wrist_wrong_point=best_wrist_wrong_point,
                correct_line_name=correct_line_name,
                wrong_line_name=wrong_line_name,
                correct_rule=correct_rule,
                wrong_rule=wrong_rule,
                p_lane=p_lane,
                p_motion=p_motion,
                body_facing=body_facing,
                arm_side_order=arm_side_order,
                backward_raw=backward_raw,
                backward_hits=backward_hits,
                backward_confirmed=backward_confirmed,
            )

        carry_info = detect_carrying_pose(
            keypoints=keypoints,
            holding=holding_raw,
            best_wrist=best_wrist,
        )

        front_carry_raw = carry_info.get("front_carry_raw", carry_info["front_carry"])
        front_carry_two_hand_raw = carry_info.get(
            "front_carry_two_hand_raw", False
        )
        front_carry_one_arm_raw = carry_info.get("front_carry_one_arm_raw", False)

        if track_id not in self.front_carry_history:
            self.front_carry_history[track_id] = []
        self.front_carry_history[track_id].append(front_carry_two_hand_raw)
        self.front_carry_history[track_id] = self.front_carry_history[track_id][
            -FRONT_CARRY_HISTORY_LEN:
        ]

        if track_id not in self.front_carry_one_arm_history:
            self.front_carry_one_arm_history[track_id] = []
        self.front_carry_one_arm_history[track_id].append(front_carry_one_arm_raw)
        self.front_carry_one_arm_history[track_id] = self.front_carry_one_arm_history[
            track_id
        ][-FRONT_CARRY_ONE_ARM_HISTORY_LEN:]

        front_carry_hits = sum(self.front_carry_history[track_id])
        front_carry_one_arm_hits = sum(self.front_carry_one_arm_history[track_id])

        two_hand_confirmed = (
            len(self.front_carry_history[track_id]) >= FRONT_CARRY_HISTORY_LEN
            and front_carry_hits >= FRONT_CARRY_MIN_HITS
        )
        one_arm_confirmed = (
            len(self.front_carry_one_arm_history[track_id])
            >= FRONT_CARRY_ONE_ARM_HISTORY_LEN
            and front_carry_one_arm_hits >= FRONT_CARRY_ONE_ARM_MIN_HITS
        )
        front_carry_confirmed = two_hand_confirmed or one_arm_confirmed

        if two_hand_confirmed:
            left_carry = True
            right_carry = True
        elif one_arm_confirmed:
            left_carry = carry_info.get("strong_left_front", False)
            right_carry = carry_info.get("strong_right_front", False)
        else:
            left_carry = False
            right_carry = False

        is_carrying = left_carry or right_carry

        if left_carry and right_carry:
            carrying_arm = "BOTH"
        elif left_carry:
            carrying_arm = "LEFT_ARM"
        elif right_carry:
            carrying_arm = "RIGHT_ARM"
        else:
            carrying_arm = "NONE"

        carry_info["front_carry_raw"] = front_carry_raw
        carry_info["front_carry_hits"] = front_carry_hits
        carry_info["front_carry_two_hand_raw"] = front_carry_two_hand_raw
        carry_info["front_carry_two_hand_hits"] = front_carry_hits
        carry_info["front_carry_one_arm_raw"] = front_carry_one_arm_raw
        carry_info["front_carry_one_arm_hits"] = front_carry_one_arm_hits
        carry_info["front_carry_two_hand_confirmed"] = two_hand_confirmed
        carry_info["front_carry_one_arm_confirmed"] = one_arm_confirmed
        carry_info["front_carry_confirmed"] = front_carry_confirmed
        carry_info["front_carry"] = front_carry_confirmed
        carry_info["left_carry"] = left_carry
        carry_info["right_carry"] = right_carry
        carry_info["is_carrying"] = is_carrying
        carry_info["carrying_arm"] = carrying_arm
        carry_info["carry_type"] = "FRONT_CARRY" if front_carry_confirmed else "NONE"

        warnings = []
        if wrong_lane:
            warnings.append("Sai Lan")
        if hold_final_status == "NONE":
            warnings.append("Khong Vin")
        elif hold_final_status == "WRONG_SIDE":
            warnings.append("Vin Sai Ben")
        elif hold_final_status == "UNKNOWN" and not DEMO_MODE:
            warnings.append("Khong Xac Dinh")
        if is_carrying:
            warnings.append("Mang Vac")
        if backward_confirmed:
            warnings.append("Di Lui")

        real_warnings = [w for w in warnings if w in REAL_VIOLATION_LABELS]
        has_real_violation = len(real_warnings) > 0
        has_unknown = hold_final_status == "UNKNOWN" and not has_real_violation

        if has_real_violation:
            color = VIOLATION_COLOR
        elif has_unknown:
            color = UNKNOWN_COLOR
        else:
            color = SAFE_COLOR

        status_warnings = list(warnings)
        if hold_final_status == "UNKNOWN" and "Khong Xac Dinh" not in status_warnings:
            status_warnings.append("Khong Xac Dinh")

        status = (
            f"{direction} | {' - '.join(status_warnings)}"
            if status_warnings
            else f"{direction} | An Toan"
        )
        display_status = self._build_display_status(direction, real_warnings)

        return self._build_result(
            status=status,
            display_status=display_status,
            color=color,
            dy=dy,
            lane_v=v,
            direction=direction,
            lane_raw=wrong_lane_raw,
            lane_hits=lane_wrong_hits,
            lane_conf=wrong_lane,
            wrong_lane=wrong_lane,
            inside_stairs=True,
            holding=holding,
            holding_raw=holding_raw,
            hold_status=hold_final_status,
            hold_raw_status=hold_raw_status,
            hold_confirmed_status=hold_final_status,
            hold_correct_hits=hold_correct_hits,
            hold_wrong_side_hits=hold_wrong_side_hits,
            hold_none_hits=hold_none_hits,
            hold_unknown_hits=hold_unknown_hits,
            holding_correct_raw=holding_correct_raw,
            holding_wrong_raw=holding_wrong_raw,
            hold_status_correct=hold_status_correct,
            hold_status_wrong=hold_status_wrong,
            dist_wrist=dist_wrist,
            wrist_side=wrist_side,
            best_wrist=best_wrist,
            best_wrist_point=best_wrist_point,
            dist_correct=dist_correct,
            dist_wrong=dist_wrong,
            seg_dist_correct=seg_dist_correct,
            seg_dist_wrong=seg_dist_wrong,
            t_correct=t_correct,
            t_wrong=t_wrong,
            wrist_side_correct=wrist_side_correct,
            wrist_side_wrong=wrist_side_wrong,
            best_wrist_correct=best_wrist_correct,
            best_wrist_wrong=best_wrist_wrong,
            best_wrist_correct_point=best_wrist_correct_point,
            best_wrist_wrong_point=best_wrist_wrong_point,
            correct_line_name=correct_line_name,
            wrong_line_name=wrong_line_name,
            correct_rule=correct_rule,
            wrong_rule=wrong_rule,
            is_carrying=carry_info["is_carrying"],
            carry_type=carry_info["carry_type"],
            left_arm_angle=carry_info["left_arm_angle"],
            right_arm_angle=carry_info["right_arm_angle"],
            left_wrist_in_torso=carry_info["left_wrist_in_torso"],
            right_wrist_in_torso=carry_info["right_wrist_in_torso"],
            wrist_dx=carry_info["wrist_dx"],
            wrist_dy=carry_info["wrist_dy"],
            wrist_distance=carry_info["wrist_distance"],
            left_bent=carry_info["left_bent"],
            right_bent=carry_info["right_bent"],
            wrists_close=carry_info["wrists_close"],
            any_wrist_in_torso=carry_info["any_wrist_in_torso"],
            both_wrist_in_torso=carry_info["both_wrist_in_torso"],
            front_carry=carry_info["front_carry"],
            front_carry_raw=carry_info["front_carry_raw"],
            front_carry_hits=carry_info["front_carry_hits"],
            front_carry_two_hand_raw=carry_info["front_carry_two_hand_raw"],
            front_carry_two_hand_hits=carry_info["front_carry_two_hand_hits"],
            front_carry_one_arm_raw=carry_info["front_carry_one_arm_raw"],
            front_carry_one_arm_hits=carry_info["front_carry_one_arm_hits"],
            front_carry_confirmed=carry_info["front_carry_confirmed"],
            left_carry=carry_info["left_carry"],
            right_carry=carry_info["right_carry"],
            carrying_arm=carry_info["carrying_arm"],
            p_lane=p_lane,
            p_motion=p_motion,
            body_facing=body_facing,
            arm_side_order=arm_side_order,
            backward_raw=backward_raw,
            backward_hits=backward_hits,
            backward_confirmed=backward_confirmed,
        )

    @staticmethod
    def _build_result(
        status,
        color,
        display_status="",
        dy=None,
        lane_v=None,
        direction="NA",
        lane_raw=False,
        lane_hits=0,
        lane_conf=False,
        wrong_lane=False,
        inside_stairs=False,
        holding=False,
        holding_raw=False,
        hold_status="ANALYZING",
        hold_raw_status="UNKNOWN",
        hold_confirmed_status="ANALYZING",
        hold_correct_hits=0,
        hold_wrong_side_hits=0,
        hold_none_hits=0,
        hold_unknown_hits=0,
        holding_correct_raw=False,
        holding_wrong_raw=False,
        hold_status_correct="UNKNOWN",
        hold_status_wrong="UNKNOWN",
        dist_wrist=-999,
        wrist_side="UNKNOWN",
        best_wrist="NONE",
        best_wrist_point=None,
        dist_correct=-999,
        dist_wrong=-999,
        seg_dist_correct=None,
        seg_dist_wrong=None,
        t_correct=None,
        t_wrong=None,
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
        is_carrying=False,
        carry_type="NONE",
        left_arm_angle=None,
        right_arm_angle=None,
        left_wrist_in_torso=False,
        right_wrist_in_torso=False,
        wrist_dx=None,
        wrist_dy=None,
        wrist_distance=None,
        left_bent=False,
        right_bent=False,
        wrists_close=False,
        any_wrist_in_torso=False,
        both_wrist_in_torso=False,
        front_carry=False,
        front_carry_raw=False,
        front_carry_hits=0,
        front_carry_two_hand_raw=False,
        front_carry_two_hand_hits=0,
        front_carry_one_arm_raw=False,
        front_carry_one_arm_hits=0,
        front_carry_confirmed=False,
        left_carry=False,
        right_carry=False,
        carrying_arm="NONE",
        p_lane=None,
        p_motion=None,
        body_facing="UNKNOWN",
        arm_side_order="UNKNOWN",
        backward_raw=False,
        backward_hits=0,
        backward_confirmed=False,
    ):
        return {
            "status": status,
            "display_status": display_status,
            "color": color,
            "dy": dy,
            "lane_v": lane_v,
            "direction": direction,
            "lane_raw": lane_raw,
            "lane_hits": lane_hits,
            "lane_conf": lane_conf,
            "wrong_lane": wrong_lane,
            "inside_stairs": inside_stairs,
            "holding": holding,
            "holding_raw": holding_raw,
            "hold_status": hold_status,
            "hold_raw_status": hold_raw_status,
            "hold_confirmed_status": hold_confirmed_status,
            "hold_correct_hits": hold_correct_hits,
            "hold_wrong_side_hits": hold_wrong_side_hits,
            "hold_none_hits": hold_none_hits,
            "hold_unknown_hits": hold_unknown_hits,
            "holding_correct_raw": holding_correct_raw,
            "holding_wrong_raw": holding_wrong_raw,
            "hold_status_correct": hold_status_correct,
            "hold_status_wrong": hold_status_wrong,
            "dist_wrist": dist_wrist,
            "wrist_side": wrist_side,
            "best_wrist": best_wrist,
            "best_wrist_point": best_wrist_point,
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
            "is_carrying": is_carrying,
            "carry_type": carry_type,
            "left_arm_angle": left_arm_angle,
            "right_arm_angle": right_arm_angle,
            "left_wrist_in_torso": left_wrist_in_torso,
            "right_wrist_in_torso": right_wrist_in_torso,
            "wrist_dx": wrist_dx,
            "wrist_dy": wrist_dy,
            "wrist_distance": wrist_distance,
            "left_bent": left_bent,
            "right_bent": right_bent,
            "wrists_close": wrists_close,
            "any_wrist_in_torso": any_wrist_in_torso,
            "both_wrist_in_torso": both_wrist_in_torso,
            "front_carry": front_carry,
            "front_carry_raw": front_carry_raw,
            "front_carry_hits": front_carry_hits,
            "front_carry_two_hand_raw": front_carry_two_hand_raw,
            "front_carry_two_hand_hits": front_carry_two_hand_hits,
            "front_carry_one_arm_raw": front_carry_one_arm_raw,
            "front_carry_one_arm_hits": front_carry_one_arm_hits,
            "front_carry_confirmed": front_carry_confirmed,
            "left_carry": left_carry,
            "right_carry": right_carry,
            "carrying_arm": carrying_arm,
            "p_lane": p_lane,
            "p_motion": p_motion,
            "body_facing": body_facing,
            "arm_side_order": arm_side_order,
            "backward_raw": backward_raw,
            "backward_hits": backward_hits,
            "backward_confirmed": backward_confirmed,
        }
