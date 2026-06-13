import numpy as np

from stair_monitor.settings import (
    BACKWARD_HISTORY_LEN,
    BACKWARD_MIN_HITS,
    HEAD_LANE_CONFIRM_FRAMES,
    HOLD_HISTORY_LEN,
    HOLD_MIN_NOT_HOLD_EVIDENCE_HITS,
    HOLD_MIN_WRONG_SIDE_HITS,
    LANE_HISTORY_LEN,
    LANE_MIN_WRONG_HITS,
    STANDING_STILL_HISTORY_LEN,
    STANDING_STILL_MIN_HITS,
    STANDING_STILL_PIXEL_THRESHOLD,
)


class BehaviorHistoryMixin:
    # Reset history theo tung track_id khi nguoi ra khoi vung hoac mat track.
    # Moi track co bo history rieng de tranh lay ket qua frame cu cua nguoi nay gan sang nguoi khac.
    def _reset_behavior_histories(self, track_id):
        if hasattr(self, "track_history") and track_id in self.track_history:
            self.track_history[track_id] = []
        if (
            hasattr(self, "direction_axis_history")
            and track_id in self.direction_axis_history
        ):
            self.direction_axis_history[track_id] = []
        if (
            hasattr(self, "direction_flip_state")
            and track_id in self.direction_flip_state
        ):
            self.direction_flip_state[track_id] = {
                "candidate": None,
                "hits": 0,
            }
        if track_id in self.lane_history:
            self.lane_history[track_id] = []
        if hasattr(self, "head_lane_history") and track_id in self.head_lane_history:
            self.head_lane_history[track_id] = []
        if hasattr(self, "head_zone_hits") and track_id in self.head_zone_hits:
            self.head_zone_hits[track_id] = 0
        if (
            hasattr(self, "ever_confirmed_inside")
            and track_id in self.ever_confirmed_inside
        ):
            self.ever_confirmed_inside[track_id] = False
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
        if hasattr(self, "hand_claim_state") and track_id in self.hand_claim_state:
            del self.hand_claim_state[track_id]
        if hasattr(self, "violation_history") and track_id in self.violation_history:
            self.violation_history[track_id] = []

    # hold_raw_status la ket qua cua tung frame.
    # hold_final_status la ket qua sau khi da qua bo loc history de chong nhieu keypoint/YOLO.
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
        hold_not_hold_evidence_hits = hold_none_hits + hold_unknown_hits

        hold_correct_confirmed = (
            hold_correct_hits > 0
            and hold_correct_hits >= max(4, HOLD_HISTORY_LEN // 4)
        )
        hold_wrong_side_confirmed = (
            len(history) >= HOLD_HISTORY_LEN
            and hold_wrong_side_hits >= HOLD_MIN_WRONG_SIDE_HITS
        )
        hold_not_hold_evidence_confirmed = (
            hold_not_hold_evidence_hits >= HOLD_MIN_NOT_HOLD_EVIDENCE_HITS
        )

        if hold_wrong_side_confirmed:
            hold_final_status = "WRONG_SIDE"
            holding = False
        elif hold_correct_confirmed:
            hold_final_status = "CORRECT"
            holding = True
        elif hold_not_hold_evidence_confirmed:
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
            hold_not_hold_evidence_hits,
            hold_final_status,
            holding,
        )

    # Doc lai trang thai sai lan da tich luy truoc do khi frame hien tai chua du dieu kien cap nhat.
    def _get_lane_history_state(self, track_id):
        history = self.lane_history.get(track_id, [])
        lane_wrong_hits = sum(1 for is_wrong in history if is_wrong)
        wrong_lane_confirmed = (
            len(history) >= LANE_HISTORY_LEN
            and lane_wrong_hits >= LANE_MIN_WRONG_HITS
        )
        return lane_wrong_hits, wrong_lane_confirmed

    # Lane history giup tranh bao sai chi vi 1 vai frame pose rung.
    def _update_lane_history(self, track_id, wrong_lane_raw):
        if track_id not in self.lane_history:
            self.lane_history[track_id] = []

        self.lane_history[track_id].append(wrong_lane_raw)
        self.lane_history[track_id] = self.lane_history[track_id][-LANE_HISTORY_LEN:]

        return self._get_lane_history_state(track_id)

    def _get_head_lane_history_state(self, track_id):
        history = self.head_lane_history.get(track_id, [])
        head_lane_hits = sum(1 for is_wrong in history if is_wrong)
        head_lane_confirmed = (
            len(history) >= HEAD_LANE_CONFIRM_FRAMES
            and head_lane_hits >= HEAD_LANE_CONFIRM_FRAMES
        )
        return head_lane_hits, head_lane_confirmed

    def _update_head_lane_history(self, track_id, wrong_lane_raw):
        if track_id not in self.head_lane_history:
            self.head_lane_history[track_id] = []

        self.head_lane_history[track_id].append(bool(wrong_lane_raw))
        self.head_lane_history[track_id] = self.head_lane_history[track_id][
            -HEAD_LANE_CONFIRM_FRAMES:
        ]
        return self._get_head_lane_history_state(track_id)

    # Di lui chi duoc xac nhan khi direction va body facing on dinh trong nhieu frame lien tiep.
    def _update_backward_history(self, track_id, backward_raw):
        if track_id not in self.backward_history:
            self.backward_history[track_id] = []

        if backward_raw is not None:
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
        # Dung lich su p_motion de kiem tra do dao dong tong the cua nguoi trong vung cau thang.
        # Logic dung yen doc lap voi direction: nguoi chua co UP/DOWN van co the bi xet dung yen.
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

        # standing_raw la ket qua frame-level, con standing_still_confirmed o duoi la ket qua sau history.
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
