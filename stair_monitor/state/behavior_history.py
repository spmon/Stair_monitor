import numpy as np

from stair_monitor.config.settings import SETTINGS


class BehaviorHistoryMixin:
    """Mixin gom cac bo history theo track_id cho analyzer.

    Moi track_id giu bo dem rieng de bo loc nhieu frame va tranh lay nham
    trang thai cua nguoi nay sang nguoi khac khi tracker van dang con song.
    """

    # Reset history theo tung track_id khi nguoi ra khoi vung hoac mat track.
    # Moi track co bo history rieng de tranh lay ket qua frame cu cua nguoi nay gan sang nguoi khac.
    def _reset_behavior_histories(self, track_id):
        """Reset cac history hanh vi cho mot track_id.

        Args:
            track_id: ID do tracker gan cho nguoi dang bi reset.

        Returns:
            None

        Notes:
            Reset nay chi xoa cac history hanh vi da duoc mixin nay quan ly.
            Muc tieu la tranh du lieu cu cua mot nguoi lam anh huong frame sau.
        """
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
        if hasattr(self, "hand_claim_state") and track_id in self.hand_claim_state:
            del self.hand_claim_state[track_id]
        if hasattr(self, "violation_history") and track_id in self.violation_history:
            self.violation_history[track_id] = []

    # hold_raw_status la ket qua cua tung frame.
    # hold_final_status la ket qua sau khi da qua bo loc history de chong nhieu keypoint/YOLO.
    def _update_hold_status_history(self, track_id, hold_final_status_raw):
        """Cap nhat history vin tay va tra ra trang thai da duoc bo loc.

        Args:
            track_id: ID cua nguoi dang duoc cap nhat history.
            hold_final_status_raw: Ket qua hold raw cua frame hien tai.

        Returns:
            tuple: So hit tung loai hold va trang thai final sau history.

        Notes:
            Raw status la bang chung frame-level. Final status chi duoc xac nhan
            sau khi history du hit de chong nhieu keypoint va tracker.
        """
        if track_id not in self.hold_status_history:
            self.hold_status_history[track_id] = []

        self.hold_status_history[track_id].append(hold_final_status_raw)
        self.hold_status_history[track_id] = self.hold_status_history[track_id][
            -SETTINGS.handrail.hold_history_len:
        ]

        history = self.hold_status_history[track_id]
        hold_correct_hits = sum(1 for status in history if status == "CORRECT")
        hold_wrong_side_hits = sum(1 for status in history if status == "WRONG_SIDE")
        hold_none_hits = sum(1 for status in history if status == "NONE")
        hold_unknown_hits = sum(1 for status in history if status == "UNKNOWN")
        hold_not_hold_evidence_hits = hold_none_hits + hold_unknown_hits

        hold_correct_confirmed = (
            hold_correct_hits > 0
            and hold_correct_hits
            >= max(4, SETTINGS.handrail.hold_history_len // 4)
        )
        hold_wrong_side_confirmed = (
            len(history) >= SETTINGS.handrail.hold_history_len
            and hold_wrong_side_hits >= SETTINGS.handrail.hold_min_wrong_side_hits
        )
        hold_not_hold_evidence_confirmed = (
            hold_not_hold_evidence_hits
            >= SETTINGS.handrail.hold_min_not_hold_evidence_hits
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
        """Lay tong hop lane history hien co cua mot track_id.

        Args:
            track_id: ID cua nguoi can doc lane history.

        Returns:
            tuple: (lane_wrong_hits, wrong_lane_confirmed)

        Notes:
            Ham nay dung khi frame hien tai chua du du lieu lane nhung overlay
            van can trang thai lane da tich luy truoc do.
        """
        history = self.lane_history.get(track_id, [])
        lane_wrong_hits = sum(1 for is_wrong in history if is_wrong)
        wrong_lane_confirmed = (
            len(history) >= SETTINGS.lane.history_len
            and lane_wrong_hits >= SETTINGS.lane.min_wrong_hits
        )
        return lane_wrong_hits, wrong_lane_confirmed

    # Lane history giup tranh bao sai chi vi 1 vai frame pose rung.
    def _update_lane_history(self, track_id, wrong_lane_raw):
        """Them 1 mau lane raw vao history cua track hien tai.

        Args:
            track_id: ID cua nguoi can cap nhat lane history.
            wrong_lane_raw: Ket qua sai lane frame-level.

        Returns:
            tuple: (lane_wrong_hits, wrong_lane_confirmed)

        Notes:
            Moi track_id can history rieng de bo loc rung pose/doc sai tam thoi.
        """
        if track_id not in self.lane_history:
            self.lane_history[track_id] = []

        self.lane_history[track_id].append(wrong_lane_raw)
        self.lane_history[track_id] = self.lane_history[track_id][
            -SETTINGS.lane.history_len:
        ]

        return self._get_lane_history_state(track_id)

    # Di lui chi duoc xac nhan khi direction va body facing on dinh trong nhieu frame lien tiep.
    def _update_backward_history(self, track_id, backward_raw):
        """Cap nhat history Di Lui theo track_id.

        Args:
            track_id: ID cua nguoi dang duoc cap nhat.
            backward_raw: Bang chung backward raw cua frame hien tai.

        Returns:
            tuple: (backward_hits, backward_confirmed)

        Notes:
            backward_raw co the la None khi body_facing chua du tin cay; luc do
            history khong them mau moi de tranh xac nhan sai.
        """
        if track_id not in self.backward_history:
            self.backward_history[track_id] = []

        if backward_raw is not None:
            self.backward_history[track_id].append(backward_raw)
            self.backward_history[track_id] = self.backward_history[track_id][
                -SETTINGS.backward.history_len:
            ]

        backward_hits = sum(self.backward_history[track_id])
        backward_confirmed = (
            len(self.backward_history[track_id]) >= SETTINGS.backward.history_len
            and backward_hits >= SETTINGS.backward.min_hits
        )
        return backward_hits, backward_confirmed

    def update_standing_still(self, track_id, p_motion):
        """
        Cap nhat history Dung Yen doc lap voi direction.

        Args:
            track_id: ID cua nguoi dang duoc cap nhat.
            p_motion: Motion point dai dien cho chuyen dong tong the.

        Returns:
            tuple:
                - standing_raw
                - standing_hits
                - standing_confirmed
                - motion_range
                - standing_len

        Notes:
            History p_motion duoc luu rieng cho moi track_id de do bien do dao
            dong tong the, khong phu thuoc viec direction da co UP/DOWN hay chua.
        """
        # Dung lich su p_motion de kiem tra do dao dong tong the cua nguoi trong vung cau thang.
        # Logic dung yen doc lap voi direction: nguoi chua co UP/DOWN van co the bi xet dung yen.
        if track_id not in self.standing_motion_history:
            self.standing_motion_history[track_id] = []

        self.standing_motion_history[track_id].append(
            (int(p_motion[0]), int(p_motion[1]))
        )
        self.standing_motion_history[track_id] = self.standing_motion_history[track_id][
            -SETTINGS.standing.still_history_len:
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
            len(points) >= SETTINGS.standing.still_history_len
            and motion_range is not None
            and motion_range <= SETTINGS.standing.still_pixel_threshold
        )

        if track_id not in self.standing_history:
            self.standing_history[track_id] = []

        self.standing_history[track_id].append(standing_raw)
        self.standing_history[track_id] = self.standing_history[track_id][
            -SETTINGS.standing.still_history_len:
        ]

        standing_hits = sum(self.standing_history[track_id])
        standing_still_confirmed = (
            len(self.standing_history[track_id]) >= SETTINGS.standing.still_history_len
            and standing_hits >= SETTINGS.standing.still_min_hits
        )

        return (
            standing_raw,
            standing_hits,
            standing_still_confirmed,
            motion_range,
            len(points),
        )

    def _update_standing_history(self, track_id, p_motion):
        """Alias giu ten cu cho logic standing history.

        Args:
            track_id: ID cua nguoi dang duoc cap nhat.
            p_motion: Motion point dai dien cho chuyen dong tong the.

        Returns:
            tuple: Ket qua tu update_standing_still.

        Notes:
            Ham nay ton tai de giu API hien tai, logic that nam o
            update_standing_still.
        """
        return self.update_standing_still(track_id, p_motion)
