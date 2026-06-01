import numpy as np

from stair_monitor.settings import (
    BACKWARD_HISTORY_LEN,
    BACKWARD_MIN_HITS,
    HOLD_HISTORY_LEN,
    HOLD_MIN_NOT_HOLD_HITS,
    HOLD_MIN_WRONG_SIDE_HITS,
    LANE_HISTORY_LEN,
    LANE_MIN_WRONG_HITS,
    STANDING_STILL_HISTORY_LEN,
    STANDING_STILL_MIN_HITS,
    STANDING_STILL_PIXEL_THRESHOLD,
)


class BehaviorHistoryMixin:
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
        self.standing_history[track_id] = []
        self.standing_motion_history[track_id] = []
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
        hold_correct_hits = sum(1 for status in history if status == "CORRECT")
        hold_wrong_side_hits = sum(1 for status in history if status == "WRONG_SIDE")
        hold_none_hits = sum(1 for status in history if status == "NONE")
        hold_unknown_hits = sum(1 for status in history if status == "UNKNOWN")

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

    def update_standing_still(self, track_id, p_motion):
        """
        Update standing still history independently from direction.

        Returns:
        - standing_raw
        - standing_hits
        - standing_confirmed
        - motion_range
        - standing_len
        """
        if track_id not in self.standing_motion_history:
            self.standing_motion_history[track_id] = []

        self.standing_motion_history[track_id].append(
            (int(p_motion[0]), int(p_motion[1]))
        )
        self.standing_motion_history[track_id] = self.standing_motion_history[track_id][
            -STANDING_STILL_HISTORY_LEN:
        ]

        points = self.standing_motion_history[track_id]
        motion_range = None
        if len(points) >= 2:
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            dx = max(xs) - min(xs)
            dy = max(ys) - min(ys)
            motion_range = float(np.hypot(dx, dy))

        standing_raw = (
            len(points) >= STANDING_STILL_HISTORY_LEN
            and motion_range is not None
            and motion_range <= STANDING_STILL_PIXEL_THRESHOLD
        )

        if track_id not in self.standing_history:
            self.standing_history[track_id] = []

        self.standing_history[track_id].append(standing_raw)
        self.standing_history[track_id] = self.standing_history[track_id][
            -STANDING_STILL_HISTORY_LEN:
        ]

        standing_hits = sum(self.standing_history[track_id])
        standing_still_confirmed = (
            len(self.standing_history[track_id]) >= STANDING_STILL_HISTORY_LEN
            and standing_hits >= STANDING_STILL_MIN_HITS
        )

        return (
            standing_raw,
            standing_hits,
            standing_still_confirmed,
            motion_range,
            len(points),
        )

    def _update_standing_history(self, track_id, p_motion):
        return self.update_standing_still(track_id, p_motion)
