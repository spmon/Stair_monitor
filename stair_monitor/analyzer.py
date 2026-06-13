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
    BACKWARD_MIN_VALID_EVIDENCE,
    DEMO_MODE,
    DOWN_Y_INCREASES,
    DIRECTION_AXIS_FAR_TO_NEAR_IS_DOWN,
    DIRECTION_FLIP_CONFIRM_FRAMES,
    DIRECTION_FLIP_MIN_DELTA,
    DIRECTION_KEEP_LAST_WHEN_UNSTABLE_FRAMES,
    DIRECTION_HISTORY_LEN,
    DIRECTION_MIN_FRAMES,
    DIRECTION_PIXEL_THRESHOLD,
    ENABLE_HEAD_ZONE,
    ENABLE_HEAD_LANE_FALLBACK,
    ENABLE_PERF_LOG,
    HEAD_ZONE_CONFIRM_FRAMES,
    HEAD_ZONE_GRACE_FRAMES,
    HEAD_LANE_SIGN_NORMAL,
    LANE_MISSING_FEET_GRACE_FRAMES,
    LANE_SIGN_NORMAL,
    OUTSIDE_COLOR,
    SAFE_COLOR,
    STAIRS_LEFT_EXPAND_BOTTOM_PX,
    STAIRS_LEFT_EXPAND_TOP_PX,
    UNKNOWN_COLOR,
    USE_DIRECTION_AXIS,
)

HAND_CLAIM_HOLD_HITS = 4
HAND_CLAIM_CARRY_HITS = 4
HAND_CLAIM_RESET_MISSES = 10


# Lop trung tam ghep toan bo logic direction, lane, handrail, carry, backward, standing.
# Dau vao chinh moi frame la track_id, p_lane, p_motion, keypoints va bbox.
# Dau ra la 1 dict tong hop trang thai, canh bao va du lieu debug/overlay.
class BehaviorAnalyzer(
    BehaviorHistoryMixin,
    CarryAnalysisMixin,
    HandrailAnalysisMixin,
    ResultBuilderMixin,
):
    def __init__(self, config):
        # Nap cac duong line/vung cau thang cho ban Windows/demo hien tai.
        self.center_line = config.get("CENTER_LINE", [[0, 0], [0, 0]])
        self.head_center_line = config.get("HEAD_CENTER_LINE", [])

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
        self.head_zone_poly = np.array(config.get("HEAD_ZONE_POLY", []), np.int32)
        self.use_direction_axis = bool(USE_DIRECTION_AXIS)
        self.down_y_increases = bool(DOWN_Y_INCREASES)
        self.direction_axis_far_to_near_is_down = bool(
            DIRECTION_AXIS_FAR_TO_NEAR_IS_DOWN
        )
        self.stair_direction_axis = config.get("STAIR_DIRECTION_AXIS", [])
        (
            self.direction_axis_start,
            self.direction_axis_end,
            self.direction_axis_valid,
        ) = self._load_direction_axis(self.stair_direction_axis)
        self.track_history = {}
        self.direction_axis_history = {}
        self.direction_flip_state = {}
        self.lane_history = {}
        self.head_lane_history = {}
        self.lane_last_state = {}
        self.lane_last_seen = {}
        self.hold_status_history = {}
        self.last_valid_direction = {}
        self.last_valid_direction_frame = {}
        self.hand_claim_state = {}
        self.front_carry_history = {}
        self.front_carry_one_arm_history = {}
        self.backward_history = {}
        self.standing_history = {}
        self.standing_motion_history = {}
        self.inside_last_seen = {}
        self.head_zone_hits = {}
        self.ever_confirmed_inside = {}
        self.frame_index = -1

    def begin_frame(self):
        # Gia tri nay chi tang 1 lan cho moi frame video de cac grace history khop theo frame.
        self.frame_index += 1

    @staticmethod
    def _normalize_axis_point(point):
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return None
        return (int(point[0]), int(point[1]))

    @classmethod
    def _load_direction_axis(cls, raw_axis):
        if not isinstance(raw_axis, list) or len(raw_axis) < 2:
            return None, None, False

        axis_start = cls._normalize_axis_point(raw_axis[0])
        axis_end = cls._normalize_axis_point(raw_axis[1])
        if axis_start is None or axis_end is None or axis_start == axis_end:
            return None, None, False
        return axis_start, axis_end, True

    @staticmethod
    def project_point_to_axis_scalar(point, axis_start, axis_end):
        if point is None or axis_start is None or axis_end is None:
            return None

        axis_vec = np.array(
            [axis_end[0] - axis_start[0], axis_end[1] - axis_start[1]],
            dtype=np.float32,
        )
        axis_norm = float(np.linalg.norm(axis_vec))
        if axis_norm <= 1e-6:
            return None

        axis_unit = axis_vec / axis_norm
        point_vec = np.array(
            [point[0] - axis_start[0], point[1] - axis_start[1]],
            dtype=np.float32,
        )
        return float(np.dot(point_vec, axis_unit))

    @staticmethod
    def _is_weak_motion_source(p_motion_source):
        return p_motion_source in {
            "BBOX_CENTER_FALLBACK",
            "BBOX_UPPER_CENTER_FALLBACK",
        }

    def _get_recent_last_valid_direction(self, track_id):
        last_direction = self.last_valid_direction.get(track_id)
        last_frame = self.last_valid_direction_frame.get(track_id)
        if last_direction not in ("UP", "DOWN") or last_frame is None:
            return None

        if self.frame_index - last_frame > DIRECTION_KEEP_LAST_WHEN_UNSTABLE_FRAMES:
            return None
        return last_direction

    def _update_last_valid_direction_from_evidence(
        self, track_id, direction_raw, direction_final
    ):
        if direction_final not in ("UP", "DOWN"):
            return
        if direction_raw not in ("UP", "DOWN"):
            return
        if direction_raw != direction_final:
            return

        self.last_valid_direction[track_id] = direction_final
        self.last_valid_direction_frame[track_id] = self.frame_index

    def _get_direction_history(self, track_id):
        if track_id not in self.track_history:
            self.track_history[track_id] = []
        if track_id not in self.direction_axis_history:
            self.direction_axis_history[track_id] = []
        if track_id not in self.direction_flip_state:
            self.direction_flip_state[track_id] = {"candidate": None, "hits": 0}
        return (
            self.track_history[track_id],
            self.direction_axis_history[track_id],
            self.direction_flip_state[track_id],
        )

    def _compute_direction_from_dy_history(self, y_history):
        y_start = int(y_history[0]) if y_history else None
        y_now = int(y_history[-1]) if y_history else None
        dy = (y_now - y_start) if len(y_history) >= 2 else None

        if len(y_history) < DIRECTION_MIN_FRAMES:
            return "ANALYZING", y_start, y_now, dy, "NOT_ENOUGH_HISTORY"

        if dy is None:
            return "ANALYZING", y_start, y_now, dy, "NOT_ENOUGH_HISTORY"

        if dy > DIRECTION_PIXEL_THRESHOLD:
            if self.down_y_increases:
                return "DOWN", y_start, y_now, dy, "Y_INCREASE_DOWN"
            return "UP", y_start, y_now, dy, "Y_INCREASE_UP"

        if dy < -DIRECTION_PIXEL_THRESHOLD:
            if self.down_y_increases:
                return "UP", y_start, y_now, dy, "Y_DECREASE_UP"
            return "DOWN", y_start, y_now, dy, "Y_DECREASE_DOWN"

        return "IDLE", y_start, y_now, dy, "Y_DELTA_NOT_ENOUGH"

    def _evaluate_direction(
        self,
        track_id,
        p_motion,
        p_motion_source,
        inside_stairs,
    ):
        y_history, axis_history, flip_state = self._get_direction_history(track_id)
        _ = axis_history
        if p_motion is not None:
            y_history.append(int(p_motion[1]))
            y_history[:] = y_history[-DIRECTION_HISTORY_LEN:]

        direction_y_start = int(y_history[0]) if y_history else None
        direction_y_now = int(y_history[-1]) if y_history else None
        dy = None
        direction_raw = "ANALYZING"
        direction_final = "ANALYZING"
        direction_reason = "NOT_ENOUGH_HISTORY"
        direction_axis_s_current = None
        direction_axis_s_start = None
        direction_axis_delta = None
        motion_axis = "NO_MOTION_POINT"
        direction_flip_allowed = False
        direction_flip_candidate = flip_state.get("candidate") or "NONE"
        direction_flip_hits = int(flip_state.get("hits", 0) or 0)

        weak_motion_source = self._is_weak_motion_source(p_motion_source)
        last_valid_direction_recent = self._get_recent_last_valid_direction(track_id)
        last_valid_direction_any = self.last_valid_direction.get(track_id)
        (
            direction_raw,
            direction_y_start,
            direction_y_now,
            dy,
            direction_reason,
        ) = self._compute_direction_from_dy_history(y_history)
        if len(y_history) >= DIRECTION_MIN_FRAMES:
            motion_axis = "Y_IMAGE_DELTA"
        elif y_history:
            motion_axis = "Y_IMAGE_WAIT_HISTORY"

        raw_delta_abs = abs(dy) if dy is not None else 0.0

        if direction_raw in ("UP", "DOWN"):
            if (
                last_valid_direction_any in ("UP", "DOWN")
                and direction_raw != last_valid_direction_any
            ):
                if flip_state.get("candidate") == direction_raw:
                    flip_state["hits"] = int(flip_state.get("hits", 0)) + 1
                else:
                    flip_state["candidate"] = direction_raw
                    flip_state["hits"] = 1

                direction_flip_candidate = flip_state.get("candidate") or "NONE"
                direction_flip_hits = int(flip_state.get("hits", 0) or 0)

                if weak_motion_source:
                    direction_final = last_valid_direction_any
                    direction_reason = "KEEP_LAST_DIRECTION_FLIP_GUARD"
                else:
                    direction_flip_allowed = (
                        direction_flip_hits >= DIRECTION_FLIP_CONFIRM_FRAMES
                        and raw_delta_abs >= DIRECTION_FLIP_MIN_DELTA
                    )
                    if direction_flip_allowed:
                        direction_final = direction_raw
                        direction_reason = "FLIP_CONFIRMED"
                        flip_state["candidate"] = None
                        flip_state["hits"] = 0
                    else:
                        direction_final = last_valid_direction_any
                        direction_reason = "KEEP_LAST_DIRECTION_FLIP_GUARD"
            else:
                direction_final = direction_raw
                flip_state["candidate"] = None
                flip_state["hits"] = 0
        else:
            flip_state["candidate"] = None
            flip_state["hits"] = 0
            if inside_stairs and last_valid_direction_recent in ("UP", "DOWN"):
                direction_final = last_valid_direction_recent
                direction_reason = "KEEP_LAST_DIRECTION_UNSTABLE"
            else:
                direction_final = direction_raw

        direction_flip_candidate = flip_state.get("candidate") or "NONE"
        direction_flip_hits = int(flip_state.get("hits", 0) or 0)
        self._update_last_valid_direction_from_evidence(
            track_id,
            direction_raw,
            direction_final,
        )
        last_valid_direction = self.last_valid_direction.get(track_id)

        dir_display = direction_final
        dir_used_for_hold = None
        hold_direction_source = "NONE"
        if direction_final in ("UP", "DOWN"):
            dir_used_for_hold = direction_final
            hold_direction_source = (
                "LAST_VALID_DIRECTION"
                if direction_final != direction_raw
                else "CURRENT_DIRECTION"
            )
        elif inside_stairs and last_valid_direction_recent in ("UP", "DOWN"):
            dir_used_for_hold = last_valid_direction_recent
            hold_direction_source = "LAST_VALID_DIRECTION"

        return {
            "dy": dy,
            "direction_y_start": direction_y_start,
            "direction_y_now": direction_y_now,
            "direction_dy": dy,
            "direction_raw": direction_raw,
            "direction_final": direction_final,
            "direction_reason": direction_reason,
            "direction_axis_s_current": direction_axis_s_current,
            "direction_axis_s_start": direction_axis_s_start,
            "direction_axis_delta": direction_axis_delta,
            "motion_axis": motion_axis,
            "direction_flip_candidate": direction_flip_candidate,
            "direction_flip_hits": direction_flip_hits,
            "direction_flip_allowed": direction_flip_allowed,
            "last_valid_direction": last_valid_direction,
            "dir_display": dir_display,
            "dir_used_for_hold": dir_used_for_hold,
            "hold_direction_source": hold_direction_source,
        }

    # Kiem tra p_lane co nam trong polygon cau thang hay khong.
    # Sai lan/hold/standing chi nen duoc ket luan khi nguoi dang o trong vung nay.
    def is_inside_stairs(self, p_lane):
        if p_lane is None or len(self.stairs_poly) < 3:
            return False

        return cv2.pointPolygonTest(
            self.stairs_poly,
            (float(p_lane[0]), float(p_lane[1])),
            False,
        ) >= 0

    def is_inside_head_zone(self, upper_point):
        if upper_point is None or len(self.head_zone_poly) < 3:
            return False

        return cv2.pointPolygonTest(
            self.head_zone_poly,
            (float(upper_point[0]), float(upper_point[1])),
            False,
        ) >= 0

    @staticmethod
    def _compute_lane_side(point, line_points):
        if point is None or line_points is None or len(line_points) < 2:
            return None

        a, b = line_points[0], line_points[1]
        return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (
            point[0] - a[0]
        )

    @staticmethod
    def _is_wrong_lane_side(direction, side_value, sign_normal):
        if side_value is None or direction not in ("UP", "DOWN"):
            return None

        if sign_normal:
            return (direction == "UP" and side_value < 0) or (
                direction == "DOWN" and side_value > 0
            )
        return (direction == "UP" and side_value > 0) or (
            direction == "DOWN" and side_value < 0
        )

    def _evaluate_inside_stairs(self, track_id, features, fallback_lane_point):
        _ = fallback_lane_point
        left_ankle = features.get("left_ankle")
        right_ankle = features.get("right_ankle")
        upper_point = features.get("upper_point")
        upper_point_source = features.get("upper_point_source", "NA")
        left_ankle_valid = left_ankle is not None
        right_ankle_valid = right_ankle is not None
        valid_foot_count = int(left_ankle_valid) + int(right_ankle_valid)
        ankle_valid_count = valid_foot_count
        # Grace chi duoc phep chay khi KHONG co ankle hop le.
        # Khong duoc tin co "feet_reliable" caller truyen vao neu no lech voi ankle count.
        feet_reliable = ankle_valid_count > 0
        head_zone_enabled = bool(ENABLE_HEAD_ZONE and len(self.head_zone_poly) >= 3)
        upper_point_in_head_zone = bool(
            head_zone_enabled and self.is_inside_head_zone(upper_point)
        )
        head_zone_hits = int(self.head_zone_hits.get(track_id, 0) or 0)
        ever_confirmed_inside = bool(
            self.ever_confirmed_inside.get(track_id, False)
        )
        inside_feet_point = None
        inside_raw_by_feet = None
        inside_reason = "NO_FEET_NO_HEAD_ZONE"
        inside_grace_left = 0
        left_foot_in = False
        right_foot_in = False
        track_zone_state = "OUTSIDE_NO_FEET_NO_HEAD_ZONE"

        if feet_reliable:
            self.head_zone_hits[track_id] = 0
            head_zone_hits = 0
            if left_ankle_valid:
                left_foot_in = self.is_inside_stairs(left_ankle)
            if right_ankle_valid:
                right_foot_in = self.is_inside_stairs(right_ankle)

            inside_raw_by_feet = left_foot_in or right_foot_in
            if left_foot_in:
                inside_feet_point = left_ankle
            elif right_foot_in:
                inside_feet_point = right_ankle
            else:
                inside_feet_point = left_ankle or right_ankle

            if inside_raw_by_feet:
                self.inside_last_seen[track_id] = self.frame_index
                self.ever_confirmed_inside[track_id] = True
                ever_confirmed_inside = True
                return (
                    True,
                    inside_feet_point,
                    ankle_valid_count,
                    feet_reliable,
                    inside_raw_by_feet,
                    "ENTERED_BY_FOOT",
                    inside_grace_left,
                    left_ankle_valid,
                    right_ankle_valid,
                    left_foot_in,
                    right_foot_in,
                    valid_foot_count,
                    head_zone_enabled,
                    upper_point,
                    upper_point_source,
                    upper_point_in_head_zone,
                    head_zone_hits,
                    "INSIDE_CONFIRMED_BY_FOOT",
                    ever_confirmed_inside,
                )

            self.inside_last_seen.pop(track_id, None)
            return (
                False,
                inside_feet_point,
                ankle_valid_count,
                feet_reliable,
                inside_raw_by_feet,
                "ALL_VISIBLE_FEET_OUT",
                inside_grace_left,
                left_ankle_valid,
                right_ankle_valid,
                left_foot_in,
                right_foot_in,
                valid_foot_count,
                head_zone_enabled,
                upper_point,
                upper_point_source,
                upper_point_in_head_zone,
                head_zone_hits,
                "OUTSIDE_CONFIRMED_BY_FOOT",
                ever_confirmed_inside,
            )

        if upper_point_in_head_zone:
            head_zone_hits += 1
            self.head_zone_hits[track_id] = head_zone_hits
        else:
            head_zone_hits = 0
            self.head_zone_hits[track_id] = 0

        last_seen = self.inside_last_seen.get(track_id)
        if upper_point_in_head_zone and head_zone_hits >= HEAD_ZONE_CONFIRM_FRAMES:
            self.inside_last_seen[track_id] = self.frame_index
            self.ever_confirmed_inside[track_id] = True
            ever_confirmed_inside = True
            return (
                True,
                inside_feet_point,
                ankle_valid_count,
                feet_reliable,
                inside_raw_by_feet,
                "ENTERED_BY_HEAD_ZONE_NO_FEET",
                inside_grace_left,
                left_ankle_valid,
                right_ankle_valid,
                left_foot_in,
                right_foot_in,
                valid_foot_count,
                head_zone_enabled,
                upper_point,
                upper_point_source,
                upper_point_in_head_zone,
                head_zone_hits,
                "INSIDE_CONFIRMED_BY_HEAD",
                ever_confirmed_inside,
            )

        if last_seen is not None:
            frame_gap = self.frame_index - last_seen
            if frame_gap <= HEAD_ZONE_GRACE_FRAMES:
                inside_grace_left = max(
                    0,
                    HEAD_ZONE_GRACE_FRAMES - frame_gap + 1,
                )
                return (
                    True,
                    inside_feet_point,
                    ankle_valid_count,
                    feet_reliable,
                    inside_raw_by_feet,
                    "INSIDE_OCCLUDED_GRACE",
                    inside_grace_left,
                    left_ankle_valid,
                    right_ankle_valid,
                    left_foot_in,
                    right_foot_in,
                    valid_foot_count,
                    head_zone_enabled,
                    upper_point,
                    upper_point_source,
                    upper_point_in_head_zone,
                    head_zone_hits,
                    "INSIDE_OCCLUDED_GRACE",
                    ever_confirmed_inside,
                )

            self.inside_last_seen.pop(track_id, None)

        if upper_point_in_head_zone:
            inside_reason = "HEAD_ZONE_WAITING_CONFIRM"
            track_zone_state = "HEAD_ZONE_WAITING_CONFIRM"

        return (
            False,
            inside_feet_point,
            ankle_valid_count,
            feet_reliable,
            inside_raw_by_feet,
            inside_reason,
            inside_grace_left,
            left_ankle_valid,
            right_ankle_valid,
            left_foot_in,
            right_foot_in,
            valid_foot_count,
            head_zone_enabled,
            upper_point,
            upper_point_source,
            upper_point_in_head_zone,
            head_zone_hits,
            track_zone_state,
            ever_confirmed_inside,
        )

    def _select_p_lane_for_lane(self, features):
        left_ankle = features.get("left_ankle")
        right_ankle = features.get("right_ankle")
        left_valid = left_ankle is not None
        right_valid = right_ankle is not None

        if not left_valid and not right_valid:
            return None, "NO_FOOT", "NO_VALID_FOOT_FOR_LANE"

        if left_valid and right_valid:
            return (
                (
                    int((left_ankle[0] + right_ankle[0]) / 2),
                    int((left_ankle[1] + right_ankle[1]) / 2),
                ),
                "FEET_MIDPOINT",
                "FEET_VISIBLE",
            )

        if left_valid:
            return left_ankle, "LEFT_FOOT_ONLY", "FEET_VISIBLE"
        return right_ankle, "RIGHT_FOOT_ONLY", "FEET_VISIBLE"

    @staticmethod
    def _select_upper_point_for_lane(features):
        upper_point = features.get("upper_point")
        upper_point_source = features.get("upper_point_source", "NA")
        if upper_point is None:
            return None, "NA"
        return upper_point, upper_point_source

    @staticmethod
    # Trang thai claim duoc luu rieng cho tung tay cua tung track.
    def _new_hand_claim_entry():
        return {
            "claim": None,
            "hold_hits": 0,
            "carry_hits": 0,
            "misses": 0,
        }

    # Khoi tao state claim tay theo track_id neu chua co.
    def _get_hand_claim_state(self, track_id):
        if track_id not in self.hand_claim_state:
            self.hand_claim_state[track_id] = {
                "left": self._new_hand_claim_entry(),
                "right": self._new_hand_claim_entry(),
            }
        return self.hand_claim_state[track_id]

    # Claim tay giup ngan 1 tay bi dung dong thoi cho 2 logic hold va carry.
    # Uu tien ben nao on dinh truoc qua nhieu frame se claim tay do.
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
    # Ghi nhan thoi gian tung block de log perf, khong anh huong logic nhan dien.
    def _record_perf(perf, key, start_time):
        if perf is None or start_time is None:
            return
        perf[key] = perf.get(key, 0.0) + (time.perf_counter() - start_time) * 1000.0

    # Giai doan ghep hold raw + carry raw + claim state trong cung 1 frame.
    # Hold va carry van la 2 logic doc lap; lop claim chi dung de giam xung dot bang chung.
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
    # Dua hold_state ve tuple co thu tu on dinh de ghim vao ket qua cuoi cung.
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
        # Ham phan tich 1 nguoi trong 1 frame.
        # p_motion dung cho direction/backward/standing.
        # p_lane dung cho sai lan va kiem tra trong vung cau thang.
        perf = {} if ENABLE_PERF_LOG else None
        analyze_start = time.perf_counter() if perf is not None else None
        if self.frame_index < 0:
            self.frame_index = 0

        features = features or extract_pose_features(keypoints, box)
        # Neu caller chua truyen san, lay diem dai dien tu pose feature da extract.
        if p_lane is None:
            p_lane = features.get("feet_point")
        if p_motion is None:
            p_motion = features.get("motion_point")
        p_motion_source = features.get("motion_point_source", "NA")
        p_motion_x = p_motion[0] if p_motion is not None else None
        p_motion_y = p_motion[1] if p_motion is not None else None

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

        direction = "ANALYZING"
        direction_raw = "ANALYZING"
        direction_final = "ANALYZING"
        direction_reason = "NOT_ENOUGH_HISTORY"
        use_direction_axis = bool(self.use_direction_axis)
        down_y_increases = bool(self.down_y_increases)
        direction_axis_far_to_near_is_down = bool(
            self.direction_axis_far_to_near_is_down
        )
        stair_direction_axis_valid = bool(self.direction_axis_valid)
        stair_direction_axis = (
            [self.direction_axis_start, self.direction_axis_end]
            if self.direction_axis_valid
            else []
        )
        direction_axis_s_current = None
        direction_axis_s_start = None
        direction_axis_delta = None
        motion_axis = "ANALYZING"
        direction_y_start = None
        direction_y_now = None
        direction_dy = None
        direction_flip_candidate = "NONE"
        direction_flip_hits = 0
        direction_flip_allowed = False
        last_valid_direction = self.last_valid_direction.get(track_id)
        dir_display = "ANALYZING"
        dir_used_for_hold = None
        hold_direction_source = "NONE"
        dy = None
        v = None
        wrong_lane_raw = False
        wrong_lane = False
        lane_wrong_hits = 0
        lane_status = "UNKNOWN"
        lane_reason = "NA"
        lane_source = "NO_LANE"
        lane_direction = "ANALYZING"
        p_lane_source = "NONE"
        lane_missing_feet_grace_left = 0
        foot_lane_side = None
        head_lane_side = None
        head_lane_raw = None
        head_lane_hits = 0
        upper_point = None
        upper_point_source = "NA"
        upper_body_in_head_zone = False
        upper_point_in_head_zone = False
        head_center_line_valid = len(self.head_center_line) >= 2
        head_lane_sign_normal = HEAD_LANE_SIGN_NORMAL
        inside_stairs = False
        inside_feet_point = features.get("inside_feet_point") or p_lane
        ankle_valid_count = int(features.get("ankle_valid_count", 0) or 0)
        feet_reliable = ankle_valid_count > 0
        inside_raw_by_feet = None
        inside_reason = "UNKNOWN"
        inside_grace_left = 0
        head_zone_enabled = bool(ENABLE_HEAD_ZONE and len(self.head_zone_poly) >= 3)
        head_zone_hits = 0
        track_zone_state = "UNKNOWN"
        ever_confirmed_inside = False
        left_ankle_valid = bool(features.get("left_ankle") is not None)
        right_ankle_valid = bool(features.get("right_ankle") is not None)
        left_foot_in = False
        right_foot_in = False
        valid_foot_count = ankle_valid_count

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
        backward_reason = "UNKNOWN_NOT_ENOUGH_EVIDENCE"
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

        # Standing still su dung lich su p_motion va doc lap voi direction.
        # Nguoi chua du dieu kien ket luan UP/DOWN van co the bi bao Dung Yen.
        standing_start = time.perf_counter() if perf is not None else None
        (
            inside_stairs,
            inside_feet_point,
            ankle_valid_count,
            feet_reliable,
            inside_raw_by_feet,
            inside_reason,
            inside_grace_left,
            left_ankle_valid,
            right_ankle_valid,
            left_foot_in,
            right_foot_in,
            valid_foot_count,
            head_zone_enabled,
            upper_point,
            upper_point_source,
            upper_point_in_head_zone,
            head_zone_hits,
            track_zone_state,
            ever_confirmed_inside,
        ) = self._evaluate_inside_stairs(track_id, features, p_lane)
        upper_body_in_head_zone = upper_point_in_head_zone
        if inside_stairs:
            (
                standing_raw,
                standing_hits,
                standing_still_confirmed,
                standing_motion_range,
                standing_len,
            ) = self.update_standing_still(track_id, p_motion)
        self._record_perf(perf, "standing", standing_start)

        # Dong bo 1 diem tra ket qua duy nhat de giu format ket qua nhat quan.
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

        direction_start = time.perf_counter() if perf is not None else None
        direction_info = self._evaluate_direction(
            track_id=track_id,
            p_motion=p_motion,
            p_motion_source=p_motion_source,
            inside_stairs=inside_stairs,
        )
        dy = direction_info["dy"]
        direction_raw = direction_info["direction_raw"]
        direction_final = direction_info["direction_final"]
        direction_reason = direction_info["direction_reason"]
        direction_y_start = direction_info["direction_y_start"]
        direction_y_now = direction_info["direction_y_now"]
        direction_dy = direction_info["direction_dy"]
        direction_axis_s_current = direction_info["direction_axis_s_current"]
        direction_axis_s_start = direction_info["direction_axis_s_start"]
        direction_axis_delta = direction_info["direction_axis_delta"]
        motion_axis = direction_info["motion_axis"]
        direction_flip_candidate = direction_info["direction_flip_candidate"]
        direction_flip_hits = direction_info["direction_flip_hits"]
        direction_flip_allowed = direction_info["direction_flip_allowed"]
        last_valid_direction = direction_info["last_valid_direction"]
        dir_display = direction_info["dir_display"]
        dir_used_for_hold = direction_info["dir_used_for_hold"]
        hold_direction_source = direction_info["hold_direction_source"]
        direction = direction_final
        self._record_perf(perf, "direction", direction_start)

        # Khi direction chua on dinh thi chua ket luan UP/DOWN,
        # nhung van co the dung huong gan nhat hop le cho hold neu con moi.
        if direction == "ANALYZING":
            if not inside_stairs:
                # Ra khoi vung thi reset history de track cu khong lam ban frame sau.
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

            hold_direction = dir_used_for_hold
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

        if not inside_stairs:
            # Ngoai vung thi khong ket luan sai lan/hold trong frame nay va xoa history hanh vi.
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

        # Di lui can ca direction va body facing cung on dinh.
        # DOWN + FRONT_TO_CAMERA va UP + BACK_TO_CAMERA duoc xem la di lui.
        backward_start = time.perf_counter() if perf is not None else None
        backward_history_value = False
        body_facing_reliable = (
            body_facing in ("FRONT_TO_CAMERA", "BACK_TO_CAMERA")
            and body_facing_confidence > 0.5
            and body_facing_evidence_count >= BACKWARD_MIN_VALID_EVIDENCE
        )
        upper_body_occluded = not (
            hip_pair_valid
            and shoulder_pair_valid
            and (ear_pair_valid or head_valid)
        )

        if not body_facing_reliable:
            backward_raw = False
            backward_history_value = None
            if upper_body_occluded or body_facing_evidence_count < BACKWARD_MIN_VALID_EVIDENCE:
                backward_reason = "UNKNOWN_OCCLUDED"
            else:
                backward_reason = "UNKNOWN_NOT_ENOUGH_EVIDENCE"
        elif direction == "DOWN" and is_front_to_camera(body_facing):
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

        backward_hits, backward_confirmed = self._update_backward_history(
            track_id, backward_history_value
        )
        self._record_perf(perf, "backward", backward_start)

        # p_lane dung de xac dinh nguoi dang dung ben nao cua vach giua.
        # Sai lan chi nen xet khi da co direction UP/DOWN; lane history giup chong nhieu pose.
        lane_start = time.perf_counter() if perf is not None else None
        lane_direction = direction
        upper_point, upper_point_source = self._select_upper_point_for_lane(features)
        upper_body_in_head_zone = self.is_inside_head_zone(upper_point)
        selected_p_lane, p_lane_source, _selection_reason = self._select_p_lane_for_lane(
            features
        )
        p_lane = selected_p_lane
        if p_lane is not None:
            lane_source = "FOOT_LANE"
            self.head_lane_history[track_id] = []
            if direction not in ("UP", "DOWN"):
                lane_wrong_hits, wrong_lane = self._get_lane_history_state(track_id)
                lane_status = "UNKNOWN"
                lane_reason = "LANE_BY_FEET"
            elif len(self.center_line) >= 2:
                foot_lane_side = self._compute_lane_side(p_lane, self.center_line)
                v = foot_lane_side
                wrong_lane_raw = self._is_wrong_lane_side(
                    direction,
                    foot_lane_side,
                    LANE_SIGN_NORMAL,
                )
                lane_wrong_hits, wrong_lane = self._update_lane_history(
                    track_id, wrong_lane_raw
                )
                lane_status = "EVALUATED"
                lane_reason = "LANE_BY_FEET"
            else:
                lane_status = "UNKNOWN"
                lane_reason = "CENTER_LINE_MISSING"
        elif (
            ENABLE_HEAD_LANE_FALLBACK
            and upper_point is not None
            and upper_body_in_head_zone
            and head_center_line_valid
        ):
            lane_source = "HEAD_LANE"
            if direction not in ("UP", "DOWN"):
                head_lane_hits, wrong_lane = self._get_head_lane_history_state(track_id)
                lane_wrong_hits = head_lane_hits
                lane_status = "UNKNOWN"
                lane_reason = "LANE_BY_HEAD_ZONE_NO_FEET"
            else:
                head_lane_side = self._compute_lane_side(
                    upper_point,
                    self.head_center_line,
                )
                v = head_lane_side
                head_lane_raw = self._is_wrong_lane_side(
                    direction,
                    head_lane_side,
                    HEAD_LANE_SIGN_NORMAL,
                )
                wrong_lane_raw = head_lane_raw
                head_lane_hits, wrong_lane = self._update_head_lane_history(
                    track_id, head_lane_raw
                )
                lane_wrong_hits = head_lane_hits
                lane_status = "EVALUATED"
                lane_reason = "LANE_BY_HEAD_ZONE_NO_FEET"
        else:
            last_seen = self.lane_last_seen.get(track_id)
            if (
                last_seen is not None
                and self.frame_index - last_seen <= LANE_MISSING_FEET_GRACE_FRAMES
            ):
                lane_missing_feet_grace_left = max(
                    0,
                    LANE_MISSING_FEET_GRACE_FRAMES - (self.frame_index - last_seen) + 1,
                )
                last_lane_state = self.lane_last_state.get(track_id, {})
                wrong_lane_raw = last_lane_state.get("wrong_lane_raw")
                wrong_lane = last_lane_state.get("wrong_lane", False)
                lane_wrong_hits = last_lane_state.get("lane_wrong_hits", 0)
                lane_status = last_lane_state.get("lane_status", "UNKNOWN")
                lane_direction = last_lane_state.get("lane_direction", lane_direction)
                v = last_lane_state.get("lane_v")
                lane_source = "LANE_KEEP_LAST"
                foot_lane_side = last_lane_state.get("foot_lane_side")
                head_lane_side = last_lane_state.get("head_lane_side")
                head_lane_raw = last_lane_state.get("head_lane_raw")
                head_lane_hits = last_lane_state.get("head_lane_hits", 0)
                lane_reason = "MISSING_FEET_KEEP_LAST"
                p_lane_source = "NO_FOOT"
            else:
                wrong_lane_raw = None
                wrong_lane = False
                lane_wrong_hits = 0
                lane_status = "UNKNOWN"
                lane_source = "NO_LANE"
                lane_reason = "MISSING_FEET_NO_DISPLAY"
                p_lane_source = "NO_FOOT"
        if lane_source in ("FOOT_LANE", "HEAD_LANE"):
            self.lane_last_seen[track_id] = self.frame_index
            self.lane_last_state[track_id] = {
                "wrong_lane_raw": wrong_lane_raw,
                "wrong_lane": wrong_lane,
                "lane_wrong_hits": lane_wrong_hits,
                "lane_status": lane_status,
                "lane_direction": lane_direction,
                "lane_v": v,
                "foot_lane_side": foot_lane_side,
                "head_lane_side": head_lane_side,
                "head_lane_raw": head_lane_raw,
                "head_lane_hits": head_lane_hits,
            }
        self._record_perf(perf, "lane", lane_start)

        # Khi chua biet chieu di, khong the xac dinh lan can dung/sai ben.
        # Nhung van co the kiem tra xem nguoi do co vin bat ky lan can nao khong.
        # Vi vay UNKNOWN/IDLE khong duoc tu dong coi la "Khong Vin".
        hold_direction = dir_used_for_hold

        # Hold/vin tay la logic doc lap.
        # Khong duoc de logic carry ghi de vao ket qua hold; carry chi di qua lop claim tay.
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

        # IDLE nghia la chua du chuyen dong de ket luan UP/DOWN.
        # Neu cung chua du bang chung dung yen thi tra ve IDLE trung tinh.
        if direction == "IDLE" and not standing_still_confirmed:
            return build_result(
                locals(),
                status="IDLE",
                display_status="",
                color=SAFE_COLOR if DEMO_MODE else (200, 200, 200),
            )

        # Neu IDLE nhung dung yen da duoc xac nhan thi van tong hop canh bao tu standing/hold/lane/backward.
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

        # Carry duoc phan tich sau khi da co direction va hold state cho frame.
        # Cac threshold carry chi phuc vu loi Mang Vac, khong duoc anh huong lane/hold.
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
            self.direction_axis_history,
            self.direction_flip_state,
            self.lane_history,
            self.head_lane_history,
            self.lane_last_state,
            self.lane_last_seen,
            self.hold_status_history,
            self.last_valid_direction,
            self.last_valid_direction_frame,
            self.inside_last_seen,
            self.head_zone_hits,
            self.ever_confirmed_inside,
            self.hand_claim_state,
            self.front_carry_history,
            self.front_carry_one_arm_history,
            self.backward_history,
            self.standing_history,
            self.standing_motion_history,
        ]

        # Duyet tung kho history va xoa cac track khong con xuat hien trong frame hien tai.
        for history_map in history_maps:
            inactive_ids = [
                track_id
                for track_id in history_map.keys()
                if track_id not in active_track_ids
            ]
            for track_id in inactive_ids:
                del history_map[track_id]
