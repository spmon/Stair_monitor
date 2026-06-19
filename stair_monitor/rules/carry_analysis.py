from stair_monitor.config.settings import SETTINGS
from stair_monitor.rules.carry_rule import detect_carrying_pose


class CarryAnalysisMixin:
    """Mixin gom logic history va hand-claim cho Mang Vac."""

    @staticmethod
    def _build_carry_handrail_state(handrail_state):
        handrail_state = handrail_state or {}
        left_hit = bool(
            handrail_state.get(
                "left_wrist_hit",
                handrail_state.get("left_hit", False),
            )
        )
        right_hit = bool(
            handrail_state.get(
                "right_wrist_hit",
                handrail_state.get("right_hit", False),
            )
        )
        left_holding = bool(
            handrail_state.get(
                "left_hold_raw",
                handrail_state.get("left_holding", False),
            )
        )
        right_holding = bool(
            handrail_state.get(
                "right_hold_raw",
                handrail_state.get("right_holding", False),
            )
        )
        return {
            "left_carry_allowed": not (left_hit or left_holding),
            "right_carry_allowed": not (right_hit or right_holding),
        }

    @staticmethod
    def _resolve_one_hand_carry_side(
        left_carry_evidence,
        right_carry_evidence,
        left_carry_score,
        right_carry_score,
        front_carry_two_hand=False,
    ):
        if front_carry_two_hand:
            return "NONE"
        if left_carry_evidence and not right_carry_evidence:
            return "LEFT"
        if right_carry_evidence and not left_carry_evidence:
            return "RIGHT"
        if left_carry_evidence and right_carry_evidence:
            left_score = (
                float(left_carry_score)
                if isinstance(left_carry_score, (int, float))
                else float("-inf")
            )
            right_score = (
                float(right_carry_score)
                if isinstance(right_carry_score, (int, float))
                else float("-inf")
            )
            return "LEFT" if left_score >= right_score else "RIGHT"
        return "NONE"

    @staticmethod
    def _derive_carry_reason(
        *,
        left_carry_allowed,
        right_carry_allowed,
        front_carry_active,
        front_carry_confirmed,
        front_carry_two_hand,
        one_hand_carry_side,
    ):
        if not left_carry_allowed and not right_carry_allowed:
            return "CARRY_SUPPRESSED_BY_HANDRAIL_HOLD"
        if front_carry_active and front_carry_two_hand:
            return (
                "FRONT_CARRY_TWO_HAND_CONFIRMED"
                if front_carry_confirmed
                else "FRONT_CARRY_TWO_HAND_RAW"
            )
        if front_carry_active and one_hand_carry_side == "LEFT":
            return (
                "FRONT_CARRY_LEFT_CONFIRMED"
                if front_carry_confirmed
                else "FRONT_CARRY_LEFT_RAW"
            )
        if front_carry_active and one_hand_carry_side == "RIGHT":
            return (
                "FRONT_CARRY_RIGHT_CONFIRMED"
                if front_carry_confirmed
                else "FRONT_CARRY_RIGHT_RAW"
            )
        if front_carry_confirmed and not left_carry_allowed and right_carry_allowed:
            return "LEFT_CARRY_SUPPRESSED_BY_HANDRAIL_HOLD"
        if front_carry_confirmed and not right_carry_allowed and left_carry_allowed:
            return "RIGHT_CARRY_SUPPRESSED_BY_HANDRAIL_HOLD"
        if front_carry_confirmed:
            return "CARRY_CONFIRMED_NO_ALLOWED_SIDE"
        return "NO_CARRY"

    # Lay carry raw cua frame hien tai truoc khi claim tay co hieu luc.
    # Muc dich la giu rieng "bang chung carry" voi "bang chung hold" de debug cho ro.
    def _get_carry_pose(
        self,
        keypoints,
        holding_raw,
        features=None,
        handrail_state=None,
    ):
        """Lay carry raw frame-level truoc khi hand claim duoc ap dung.

        Args:
            keypoints: Mang keypoint YOLO pose.
            holding_raw: Bang chung hold raw cua frame hien tai.
            features: Dict feature da extract neu caller co san.
            handrail_state: Raw handrail state cua frame hien tai.

        Returns:
            dict: Carry raw va cac field before/after claim khoi tao ban dau.

        Notes:
            Ham nay giu tach rieng bang chung carry raw voi claim state de de
            debug viec 1 tay bi chia vai tro giua hold va carry.
        """
        carry_info = detect_carrying_pose(
            keypoints=keypoints,
            holding=holding_raw,
            features=features,
        )
        handrail_gate = self._build_carry_handrail_state(handrail_state)
        left_carry_allowed = handrail_gate["left_carry_allowed"]
        right_carry_allowed = handrail_gate["right_carry_allowed"]

        strong_left_front = (
            bool(carry_info.get("strong_left_front", False))
            and left_carry_allowed
        )
        strong_right_front = (
            bool(carry_info.get("strong_right_front", False))
            and right_carry_allowed
        )
        left_carry_raw = (
            bool(carry_info.get("left_carry_raw", carry_info.get("left_carry", False)))
            and left_carry_allowed
        )
        right_carry_raw = (
            bool(
                carry_info.get(
                    "right_carry_raw",
                    carry_info.get("right_carry", False),
                )
            )
            and right_carry_allowed
        )
        front_carry_two_hand_raw = (
            bool(carry_info.get("front_carry_two_hand_raw", False))
            and left_carry_raw
            and right_carry_raw
        )
        front_carry_one_arm_raw = strong_left_front or strong_right_front
        front_carry_raw = front_carry_two_hand_raw or front_carry_one_arm_raw
        left_carry_score = carry_info.get("left_carry_score")
        right_carry_score = carry_info.get("right_carry_score")
        left_carry_score = left_carry_score if strong_left_front else None
        right_carry_score = right_carry_score if strong_right_front else None
        one_hand_carry_side = self._resolve_one_hand_carry_side(
            strong_left_front,
            strong_right_front,
            left_carry_score,
            right_carry_score,
            front_carry_two_hand=front_carry_two_hand_raw,
        )

        carry_info["left_carry_allowed"] = left_carry_allowed
        carry_info["right_carry_allowed"] = right_carry_allowed
        carry_info["strong_left_front"] = strong_left_front
        carry_info["strong_right_front"] = strong_right_front
        carry_info["front_carry_two_hand"] = front_carry_two_hand_raw
        carry_info["front_carry_strong_one_arm"] = front_carry_one_arm_raw
        carry_info["front_carry_two_hand_raw"] = front_carry_two_hand_raw
        carry_info["front_carry_one_arm_raw"] = front_carry_one_arm_raw
        carry_info["front_carry_raw"] = front_carry_raw
        carry_info["front_carry"] = front_carry_raw
        carry_info["left_carry_raw"] = left_carry_raw
        carry_info["right_carry_raw"] = right_carry_raw
        carry_info["left_carry_score"] = left_carry_score
        carry_info["right_carry_score"] = right_carry_score
        carry_info["left_carry_evidence"] = strong_left_front
        carry_info["right_carry_evidence"] = strong_right_front
        carry_info["one_hand_carry_side"] = one_hand_carry_side
        carry_info["carry_reason"] = self._derive_carry_reason(
            left_carry_allowed=left_carry_allowed,
            right_carry_allowed=right_carry_allowed,
            front_carry_active=front_carry_raw,
            front_carry_confirmed=False,
            front_carry_two_hand=front_carry_two_hand_raw,
            one_hand_carry_side=one_hand_carry_side,
        )
        carry_info["left_carry_raw_before_claim"] = left_carry_raw
        carry_info["right_carry_raw_before_claim"] = right_carry_raw
        carry_info["left_carry_raw_after_claim"] = left_carry_raw
        carry_info["right_carry_raw_after_claim"] = right_carry_raw
        carry_info["front_carry_raw_before_claim"] = front_carry_raw
        carry_info["front_carry_two_hand_raw_before_claim"] = front_carry_two_hand_raw
        carry_info["front_carry_one_arm_raw_before_claim"] = front_carry_one_arm_raw
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
        left_carry_allowed = bool(carry_info.get("left_carry_allowed", True))
        right_carry_allowed = bool(carry_info.get("right_carry_allowed", True))
        strong_left_after_claim = (
            bool(carry_info.get("strong_left_front", False))
            and left_carry_raw_after_claim
        )
        strong_right_after_claim = (
            bool(carry_info.get("strong_right_front", False))
            and right_carry_raw_after_claim
        )
        front_carry_two_hand_before_claim = bool(
            carry_info.get(
                "front_carry_two_hand_raw_before_claim",
                carry_info.get("front_carry_two_hand_raw", False),
            )
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
        left_carry_score = carry_info.get("left_carry_score")
        right_carry_score = carry_info.get("right_carry_score")
        left_carry_score = left_carry_score if strong_left_after_claim else None
        right_carry_score = right_carry_score if strong_right_after_claim else None
        one_hand_carry_side = self._resolve_one_hand_carry_side(
            strong_left_after_claim,
            strong_right_after_claim,
            left_carry_score,
            right_carry_score,
            front_carry_two_hand=front_carry_two_hand_after_claim,
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
        carry_info["left_carry_evidence"] = strong_left_after_claim
        carry_info["right_carry_evidence"] = strong_right_after_claim
        carry_info["left_carry_score"] = left_carry_score
        carry_info["right_carry_score"] = right_carry_score
        carry_info["one_hand_carry_side"] = one_hand_carry_side
        carry_info["carry_reason"] = self._derive_carry_reason(
            left_carry_allowed=left_carry_allowed,
            right_carry_allowed=right_carry_allowed,
            front_carry_active=front_carry_raw_after_claim,
            front_carry_confirmed=False,
            front_carry_two_hand=front_carry_two_hand_after_claim,
            one_hand_carry_side=one_hand_carry_side,
        )
        return carry_info

    # Carry history giup tranh bao Mang Vac chi vi 1 frame pose nhieu.
    # Raw la ket qua tung frame, confirmed la ket qua sau khi du hit qua nhieu frame.
    def _analyze_carry(
        self,
        track_id,
        keypoints,
        holding_raw,
        features=None,
        carry_pose=None,
        handrail_state=None,
    ):
        """Phan tich Mang Vac sau khi da co carry raw va claim state.

        Args:
            track_id: ID cua nguoi dang duoc cap nhat history.
            keypoints: Mang keypoint YOLO pose.
            holding_raw: Bang chung hold raw cua frame hien tai.
            features: Dict feature da extract neu caller co san.
            carry_pose: Carry raw da tinh san neu caller da co.
            handrail_state: Handrail state hien tai/confirmed cua frame.

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
                features=features,
                handrail_state=handrail_state,
            )
        )

        front_carry_raw = carry_info.get("front_carry_raw", carry_info["front_carry"])
        front_carry_two_hand_raw = bool(
            carry_info.get("front_carry_two_hand_raw", False)
        )
        front_carry_one_arm_raw = bool(carry_info.get("front_carry_one_arm_raw", False))
        handrail_gate = self._build_carry_handrail_state(handrail_state)
        left_carry_allowed = (
            bool(carry_info.get("left_carry_allowed", True))
            and handrail_gate["left_carry_allowed"]
        )
        right_carry_allowed = (
            bool(carry_info.get("right_carry_allowed", True))
            and handrail_gate["right_carry_allowed"]
        )
        left_carry_evidence = (
            bool(
                carry_info.get(
                    "strong_left_front_after_claim",
                    carry_info.get("strong_left_front", False),
                )
            )
            and left_carry_allowed
        )
        right_carry_evidence = (
            bool(
                carry_info.get(
                    "strong_right_front_after_claim",
                    carry_info.get("strong_right_front", False),
                )
            )
            and right_carry_allowed
        )
        left_carry_score = carry_info.get("left_carry_score")
        right_carry_score = carry_info.get("right_carry_score")
        left_carry_score = left_carry_score if left_carry_evidence else None
        right_carry_score = right_carry_score if right_carry_evidence else None

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

        if two_hand_confirmed and left_carry_allowed and right_carry_allowed:
            left_carry = True
            right_carry = True
            one_hand_carry_side = "NONE"
        else:
            one_hand_carry_side = self._resolve_one_hand_carry_side(
                left_carry_evidence,
                right_carry_evidence,
                left_carry_score,
                right_carry_score,
                front_carry_two_hand=False,
            )
            left_carry = front_carry_confirmed and one_hand_carry_side == "LEFT"
            right_carry = front_carry_confirmed and one_hand_carry_side == "RIGHT"
            if not front_carry_confirmed:
                one_hand_carry_side = "NONE"

        if left_carry and right_carry:
            carrying_arm = "BOTH"
        elif left_carry:
            carrying_arm = "LEFT_ARM"
        elif right_carry:
            carrying_arm = "RIGHT_ARM"
        else:
            carrying_arm = "NONE"
            one_hand_carry_side = "NONE"

        carry_info["front_carry_raw"] = front_carry_raw
        carry_info["front_carry_hits"] = front_carry_hits
        carry_info["front_carry_two_hand_raw"] = front_carry_two_hand_raw
        carry_info["front_carry_two_hand_hits"] = front_carry_hits
        carry_info["front_carry_one_arm_raw"] = front_carry_one_arm_raw
        carry_info["front_carry_one_arm_hits"] = front_carry_one_arm_hits
        carry_info["front_carry_two_hand_confirmed"] = two_hand_confirmed
        carry_info["front_carry_one_arm_confirmed"] = one_arm_confirmed
        carry_info["front_carry_confirmed"] = front_carry_confirmed
        carry_info["front_carry"] = left_carry or right_carry
        carry_info["left_carry"] = left_carry
        carry_info["right_carry"] = right_carry
        carry_info["left_carry_allowed"] = left_carry_allowed
        carry_info["right_carry_allowed"] = right_carry_allowed
        carry_info["left_carry_evidence"] = left_carry_evidence
        carry_info["right_carry_evidence"] = right_carry_evidence
        carry_info["left_carry_score"] = left_carry_score
        carry_info["right_carry_score"] = right_carry_score
        carry_info["one_hand_carry_side"] = one_hand_carry_side
        carry_info["is_carrying"] = left_carry or right_carry
        carry_info["carrying_arm"] = carrying_arm
        carry_info["carry_type"] = "FRONT_CARRY" if (left_carry or right_carry) else "NONE"
        carry_info["carry_reason"] = self._derive_carry_reason(
            left_carry_allowed=left_carry_allowed,
            right_carry_allowed=right_carry_allowed,
            front_carry_active=left_carry or right_carry,
            front_carry_confirmed=front_carry_confirmed,
            front_carry_two_hand=left_carry and right_carry,
            one_hand_carry_side=one_hand_carry_side,
        )
        return carry_info


def evaluate_carry(
    analyzer,
    track_id,
    keypoints,
    holding_raw,
    features=None,
    carry_pose=None,
    handrail_state=None,
):
    return analyzer._analyze_carry(
        track_id,
        keypoints,
        holding_raw,
        features=features,
        carry_pose=carry_pose,
        handrail_state=handrail_state,
    )
