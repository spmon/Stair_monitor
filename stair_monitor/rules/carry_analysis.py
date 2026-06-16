from stair_monitor.config.settings import SETTINGS
from stair_monitor.rules.carry_rule import detect_carrying_pose


class CarryAnalysisMixin:
    """Mixin gom logic history va hand-claim cho Mang Vac."""

    # Lay carry raw cua frame hien tai truoc khi claim tay co hieu luc.
    # Muc dich la giu rieng "bang chung carry" voi "bang chung hold" de debug cho ro.
    def _get_carry_pose(self, keypoints, holding_raw, best_wrist, features=None):
        """Lay carry raw frame-level truoc khi hand claim duoc ap dung.

        Args:
            keypoints: Mang keypoint YOLO pose.
            holding_raw: Bang chung hold raw cua frame hien tai.
            best_wrist: Wrist duoc hold logic chon ra de debug.
            features: Dict feature da extract neu caller co san.

        Returns:
            dict: Carry raw va cac field before/after claim khoi tao ban dau.

        Notes:
            Ham nay giu tach rieng bang chung carry raw voi claim state de de
            debug viec 1 tay bi chia vai tro giua hold va carry.
        """
        carry_info = detect_carrying_pose(
            keypoints=keypoints,
            holding=holding_raw,
            best_wrist=best_wrist,
            features=features,
        )
        left_carry_raw = carry_info.get("left_carry_raw", carry_info.get("left_carry", False))
        right_carry_raw = carry_info.get(
            "right_carry_raw",
            carry_info.get("right_carry", False),
        )
        carry_info["left_carry_raw_before_claim"] = left_carry_raw
        carry_info["right_carry_raw_before_claim"] = right_carry_raw
        carry_info["left_carry_raw_after_claim"] = left_carry_raw
        carry_info["right_carry_raw_after_claim"] = right_carry_raw
        carry_info["front_carry_raw_before_claim"] = carry_info.get(
            "front_carry_raw",
            carry_info.get("front_carry", False),
        )
        carry_info["front_carry_two_hand_raw_before_claim"] = carry_info.get(
            "front_carry_two_hand_raw",
            False,
        )
        carry_info["front_carry_one_arm_raw_before_claim"] = carry_info.get(
            "front_carry_one_arm_raw",
            False,
        )
        return carry_info

    # Neu mot tay da duoc claim cho HOLD thi chan tay do khoi logic carry.
    # Khong duoc de carry ghi de len ket qua hold; hold va carry la 2 logic doc lap.
    def _apply_hand_claim_to_carry_pose(self, carry_info, hand_claim_state):
        """Ap hand claim de loai tay da duoc xac nhan HOLD khoi carry raw.

        Args:
            carry_info: Dict carry raw cua frame hien tai.
            hand_claim_state: Claim state hien tai cua tung tay.

        Returns:
            dict: Carry info sau khi da loai bo tay bi claim HOLD.

        Notes:
            Claim nay chi loc bang chung. Hold final va carry final van duoc
            tinh rieng, khong ghi de vao nhau.
        """
        if hand_claim_state is None:
            hand_claim_state = {}

        carry_info = dict(carry_info)
        left_carry_raw_before_claim = carry_info.get(
            "left_carry_raw_before_claim",
            carry_info.get("left_carry_raw", carry_info.get("left_carry", False)),
        )
        right_carry_raw_before_claim = carry_info.get(
            "right_carry_raw_before_claim",
            carry_info.get("right_carry_raw", carry_info.get("right_carry", False)),
        )
        carry_info["left_carry_raw_before_claim"] = left_carry_raw_before_claim
        carry_info["right_carry_raw_before_claim"] = right_carry_raw_before_claim

        left_claim = hand_claim_state.get("left", {}).get("claim")
        right_claim = hand_claim_state.get("right", {}).get("claim")

        # HOLD claim thi chan carry tren dung tay da duoc xac nhan vin lan can.
        left_carry_raw_after_claim = (
            False if left_claim == "HOLD" else left_carry_raw_before_claim
        )
        right_carry_raw_after_claim = (
            False if right_claim == "HOLD" else right_carry_raw_before_claim
        )

        strong_left_after_claim = (
            carry_info.get("strong_left_front", False) and left_carry_raw_after_claim
        )
        strong_right_after_claim = (
            carry_info.get("strong_right_front", False) and right_carry_raw_after_claim
        )
        front_carry_two_hand_before_claim = carry_info.get(
            "front_carry_two_hand_raw_before_claim",
            carry_info.get("front_carry_two_hand_raw", False),
        )
        front_carry_two_hand_after_claim = (
            front_carry_two_hand_before_claim
            and left_carry_raw_after_claim
            and right_carry_raw_after_claim
        )
        front_carry_one_arm_after_claim = (
            strong_left_after_claim or strong_right_after_claim
        )
        front_carry_raw_after_claim = (
            front_carry_two_hand_after_claim or front_carry_one_arm_after_claim
        )

        carry_info["left_carry_raw_after_claim"] = left_carry_raw_after_claim
        carry_info["right_carry_raw_after_claim"] = right_carry_raw_after_claim
        carry_info["strong_left_front_after_claim"] = strong_left_after_claim
        carry_info["strong_right_front_after_claim"] = strong_right_after_claim
        carry_info["front_carry_two_hand_raw"] = front_carry_two_hand_after_claim
        carry_info["front_carry_one_arm_raw"] = front_carry_one_arm_after_claim
        carry_info["front_carry_raw"] = front_carry_raw_after_claim
        carry_info["front_carry"] = front_carry_raw_after_claim
        carry_info["left_carry_raw"] = left_carry_raw_after_claim
        carry_info["right_carry_raw"] = right_carry_raw_after_claim
        return carry_info

    # Carry history giup tranh bao Mang Vac chi vi 1 frame pose nhieu.
    # Raw la ket qua tung frame, confirmed la ket qua sau khi du hit qua nhieu frame.
    def _analyze_carry(
        self,
        track_id,
        keypoints,
        holding_raw,
        best_wrist,
        features=None,
        carry_pose=None,
    ):
        """Phan tich Mang Vac sau khi da co carry raw va claim state.

        Args:
            track_id: ID cua nguoi dang duoc cap nhat history.
            keypoints: Mang keypoint YOLO pose.
            holding_raw: Bang chung hold raw cua frame hien tai.
            best_wrist: Wrist duoc hold logic chon ra de debug.
            features: Dict feature da extract neu caller co san.
            carry_pose: Carry raw da tinh san neu caller da co.

        Returns:
            dict: Carry info sau khi da qua history va xac nhan final.

        Notes:
            front_carry_raw la ket qua tung frame. front_carry_confirmed la ket
            qua sau history theo track_id de chong nhieu YOLO/keypoint.
        """
        carry_info = (
            dict(carry_pose)
            if carry_pose is not None
            else self._get_carry_pose(
                keypoints,
                holding_raw,
                best_wrist,
                features=features,
            )
        )

        front_carry_raw = carry_info.get("front_carry_raw", carry_info["front_carry"])
        front_carry_two_hand_raw = carry_info.get(
            "front_carry_two_hand_raw", False
        )
        front_carry_one_arm_raw = carry_info.get("front_carry_one_arm_raw", False)

        # Lich su 2 tay va 1 tay duoc luu rieng de giu nguyen logic nguong hien tai.
        if track_id not in self.front_carry_history:
            self.front_carry_history[track_id] = []
        self.front_carry_history[track_id].append(front_carry_two_hand_raw)
        self.front_carry_history[track_id] = self.front_carry_history[track_id][
            -SETTINGS.carry.front_carry_history_len:
        ]

        if track_id not in self.front_carry_one_arm_history:
            self.front_carry_one_arm_history[track_id] = []
        self.front_carry_one_arm_history[track_id].append(front_carry_one_arm_raw)
        self.front_carry_one_arm_history[track_id] = self.front_carry_one_arm_history[
            track_id
        ][-SETTINGS.carry.front_carry_one_arm_history_len:]

        front_carry_hits = sum(self.front_carry_history[track_id])
        front_carry_one_arm_hits = sum(self.front_carry_one_arm_history[track_id])

        two_hand_confirmed = (
            len(self.front_carry_history[track_id])
            >= SETTINGS.carry.front_carry_history_len
            and front_carry_hits >= SETTINGS.carry.front_carry_min_hits
        )
        one_arm_confirmed = (
            len(self.front_carry_one_arm_history[track_id])
            >= SETTINGS.carry.front_carry_one_arm_history_len
            and front_carry_one_arm_hits >= SETTINGS.carry.front_carry_one_arm_min_hits
        )
        front_carry_confirmed = two_hand_confirmed or one_arm_confirmed

        if two_hand_confirmed:
            left_carry = True
            right_carry = True
        elif one_arm_confirmed:
            left_carry = carry_info.get(
                "strong_left_front_after_claim",
                carry_info.get("strong_left_front", False),
            )
            right_carry = carry_info.get(
                "strong_right_front_after_claim",
                carry_info.get("strong_right_front", False),
            )
        else:
            left_carry = False
            right_carry = False

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
        carry_info["is_carrying"] = left_carry or right_carry
        carry_info["carrying_arm"] = carrying_arm
        carry_info["carry_type"] = "FRONT_CARRY" if front_carry_confirmed else "NONE"
        return carry_info


def evaluate_carry(
    analyzer,
    track_id,
    keypoints,
    holding_raw,
    best_wrist,
    features=None,
    carry_pose=None,
):
    return analyzer._analyze_carry(
        track_id,
        keypoints,
        holding_raw,
        best_wrist,
        features=features,
        carry_pose=carry_pose,
    )
