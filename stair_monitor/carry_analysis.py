from stair_monitor.carry import detect_carrying_pose
from stair_monitor.settings import (
    FRONT_CARRY_HISTORY_LEN,
    FRONT_CARRY_MIN_HITS,
    FRONT_CARRY_ONE_ARM_HISTORY_LEN,
    FRONT_CARRY_ONE_ARM_MIN_HITS,
)


class CarryAnalysisMixin:
    # Lay carry raw cua frame hien tai truoc khi claim tay co hieu luc.
    # Muc dich la giu rieng "bang chung carry" voi "bang chung hold" de debug cho ro.
    def _get_carry_pose(self, keypoints, holding_raw, best_wrist, features=None):
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
