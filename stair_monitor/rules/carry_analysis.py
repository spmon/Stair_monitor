from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from stair_monitor.common.types import AnalysisSubjectID, KeypointsArray, Numeric, PoseFeatures
from stair_monitor.config.settings import SETTINGS
from stair_monitor.rules.carry_rule import detect_carrying_pose

CarryInfo = dict[str, object]
HandClaimEntry = dict[str, int | str | None]
HandClaimState = dict[str, HandClaimEntry]
HandrailState = dict[str, object]
HandSide = Literal["left", "right"]
HandClaimValue = Literal["HOLD", "CARRY"]


@dataclass(frozen=True, slots=True)
class CarryConfirmationInput:
    track_id: AnalysisSubjectID
    front_carry_raw: bool
    front_carry_two_hand_raw: bool
    front_carry_one_arm_raw: bool
    left_carry_allowed: bool
    right_carry_allowed: bool
    left_carry_evidence: bool
    right_carry_evidence: bool
    left_carry_score: Numeric | None
    right_carry_score: Numeric | None


@dataclass(frozen=True, slots=True)
class CarryConfirmationResult:
    front_carry_hits: int
    front_carry_one_arm_hits: int
    front_carry_confirmed: bool
    front_carry_two_hand_confirmed: bool
    front_carry_one_arm_confirmed: bool
    left_carry: bool
    right_carry: bool
    one_hand_carry_side: str
    carrying_arm: str
    is_carrying: bool
    carry_type: str


@dataclass(frozen=True, slots=True)
class PerHandCarryRawState:
    left: bool
    right: bool

    def to_legacy_fields(
        self,
        *,
        left_key: str,
        right_key: str,
    ) -> dict[str, bool]:
        return {
            left_key: self.left,
            right_key: self.right,
        }


@dataclass(frozen=True, slots=True)
class PerHandClaimSelection:
    left: HandClaimValue | None
    right: HandClaimValue | None


@dataclass(frozen=True, slots=True)
class CarryClaimApplicationResult:
    carry_raw_before_claim: PerHandCarryRawState
    carry_raw_after_claim: PerHandCarryRawState

# File nay tong hop logic Mang Vac cho ban Windows/demo.
# FLOW: carry raw tu pose -> gate/claim voi handrail -> history nhieu frame -> warning final `Mang Vac`.
# WHY: Carry can bo loc nhieu frame vi pose tay rung rat de nham trong tung frame le.

class CarryAnalysisMixin:
    """Mixin gom logic history va hand-claim cho Mang Vac.

    WARNING:
        - Carry chi duoc sinh warning `Mang Vac`.
        - Carry khong duoc phep ghi de len handrail final; no chi giao tiep voi
          handrail qua gate/claim de tranh xung dot du lieu tay.
    """

    @staticmethod
    def _build_carry_handrail_state(
        handrail_state: HandrailState | None,
    ) -> dict[str, bool]:
        """Rut gon handrail state thanh gate duoc/khong duoc tinh carry tren moi tay."""
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
        left_carry_evidence: bool,
        right_carry_evidence: bool,
        left_carry_score: Numeric | None,
        right_carry_score: Numeric | None,
        front_carry_two_hand: bool = False,
    ) -> str:
        """Chon ben dai dien neu carry 1 tay co bang chung o ca hai phia."""
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
        left_carry_allowed: bool,
        right_carry_allowed: bool,
        front_carry_active: bool,
        front_carry_confirmed: bool,
        front_carry_two_hand: bool,
        one_hand_carry_side: str,
    ) -> str:
        """Sinh reason debug de giai thich carry dang o raw, confirmed hay dang bi suppress."""
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
        keypoints: KeypointsArray | None,
        holding_raw: bool,
        features: PoseFeatures | None = None,
        handrail_state: HandrailState | None = None,
    ) -> CarryInfo:
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
        # FLOW: Handrail gate chan som nhung tay da qua gan/dang vin rail,
        # tranh de raw carry "an" nham vao tay dang duoc dung cho vin.
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
    @staticmethod
    def _get_hand_claim_label(
        hand_claim_state: HandClaimState | None,
        hand_side: HandSide,
    ) -> HandClaimValue | None:
        if hand_claim_state is None:
            return None
        claim_value = hand_claim_state.get(hand_side, {}).get("claim")
        if claim_value in ("HOLD", "CARRY"):
            return claim_value
        return None

    @classmethod
    def _read_per_hand_claim_selection(
        cls,
        hand_claim_state: HandClaimState | None,
    ) -> PerHandClaimSelection:
        """Doc claim per-hand va chuan hoa ve boundary typed noi bo cua carry."""
        return PerHandClaimSelection(
            left=cls._get_hand_claim_label(hand_claim_state, "left"),
            right=cls._get_hand_claim_label(hand_claim_state, "right"),
        )

    @staticmethod
    def _read_carry_raw_before_claim(
        carry_info: CarryInfo,
    ) -> PerHandCarryRawState:
        """Doc carry raw before-claim tu dict legacy va giu fallback field cu."""
        return PerHandCarryRawState(
            left=bool(
                carry_info.get(
                    "left_carry_raw_before_claim",
                    carry_info.get(
                        "left_carry_raw",
                        carry_info.get("left_carry", False),
                    ),
                )
            ),
            right=bool(
                carry_info.get(
                    "right_carry_raw_before_claim",
                    carry_info.get(
                        "right_carry_raw",
                        carry_info.get("right_carry", False),
                    ),
                )
            ),
        )

    @staticmethod
    def _apply_hold_claim_to_carry_raw_state(
        carry_raw_before_claim: PerHandCarryRawState,
        claim_selection: PerHandClaimSelection,
    ) -> PerHandCarryRawState:
        """Chi claim HOLD da confirmed moi duoc suppress carry raw cung tay."""
        return PerHandCarryRawState(
            left=(
                False
                if claim_selection.left == "HOLD"
                else carry_raw_before_claim.left
            ),
            right=(
                False
                if claim_selection.right == "HOLD"
                else carry_raw_before_claim.right
            ),
        )

    @classmethod
    def _build_carry_claim_application_result(
        cls,
        carry_info: CarryInfo,
        hand_claim_state: HandClaimState | None,
    ) -> CarryClaimApplicationResult:
        """Tach ro carry raw truoc claim va sau claim cho tung tay."""
        carry_raw_before_claim = cls._read_carry_raw_before_claim(carry_info)
        claim_selection = cls._read_per_hand_claim_selection(hand_claim_state)
        carry_raw_after_claim = cls._apply_hold_claim_to_carry_raw_state(
            carry_raw_before_claim,
            claim_selection,
        )
        return CarryClaimApplicationResult(
            carry_raw_before_claim=carry_raw_before_claim,
            carry_raw_after_claim=carry_raw_after_claim,
        )

    def _apply_hand_claim_to_carry_pose(
        self,
        carry_info: CarryInfo,
        hand_claim_state: HandClaimState | None,
    ) -> CarryInfo:
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
        claim_application = self._build_carry_claim_application_result(
            carry_info,
            hand_claim_state,
        )
        carry_info.update(
            claim_application.carry_raw_before_claim.to_legacy_fields(
                left_key="left_carry_raw_before_claim",
                right_key="right_carry_raw_before_claim",
            )
        )
        left_carry_raw_after_claim = claim_application.carry_raw_after_claim.left
        right_carry_raw_after_claim = claim_application.carry_raw_after_claim.right
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

        carry_info.update(
            claim_application.carry_raw_after_claim.to_legacy_fields(
                left_key="left_carry_raw_after_claim",
                right_key="right_carry_raw_after_claim",
            )
        )
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
    @staticmethod
    def _build_carry_confirmation_input(
        *,
        track_id: AnalysisSubjectID,
        front_carry_raw: bool,
        front_carry_two_hand_raw: bool,
        front_carry_one_arm_raw: bool,
        left_carry_allowed: bool,
        right_carry_allowed: bool,
        left_carry_evidence: bool,
        right_carry_evidence: bool,
        left_carry_score: Numeric | None,
        right_carry_score: Numeric | None,
    ) -> CarryConfirmationInput:
        """Chuan hoa input confirmation sau khi raw carry da qua gate/claim."""
        return CarryConfirmationInput(
            track_id=track_id,
            front_carry_raw=front_carry_raw,
            front_carry_two_hand_raw=front_carry_two_hand_raw,
            front_carry_one_arm_raw=front_carry_one_arm_raw,
            left_carry_allowed=left_carry_allowed,
            right_carry_allowed=right_carry_allowed,
            left_carry_evidence=left_carry_evidence,
            right_carry_evidence=right_carry_evidence,
            left_carry_score=left_carry_score,
            right_carry_score=right_carry_score,
        )

    def _update_carry_confirmation_history(
        self,
        confirmation_input: CarryConfirmationInput,
    ) -> tuple[int, int, bool, bool, bool]:
        """Cap nhat history two-hand va one-hand, giu nguyen hit/miss policy hien tai."""
        track_id = confirmation_input.track_id
        if track_id not in self.front_carry_history:
            self.front_carry_history[track_id] = []
        self.front_carry_history[track_id].append(
            confirmation_input.front_carry_two_hand_raw
        )
        self.front_carry_history[track_id] = self.front_carry_history[track_id][
            -SETTINGS.carry.front_carry_history_len:
        ]

        if track_id not in self.front_carry_one_arm_history:
            self.front_carry_one_arm_history[track_id] = []
        self.front_carry_one_arm_history[track_id].append(
            confirmation_input.front_carry_one_arm_raw
        )
        self.front_carry_one_arm_history[track_id] = (
            self.front_carry_one_arm_history[track_id][
                -SETTINGS.carry.front_carry_one_arm_history_len:
            ]
        )

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
        return (
            front_carry_hits,
            front_carry_one_arm_hits,
            two_hand_confirmed,
            one_arm_confirmed,
            front_carry_confirmed,
        )

    @staticmethod
    def _resolve_carrying_arm(
        left_carry: bool,
        right_carry: bool,
    ) -> tuple[str, str]:
        if left_carry and right_carry:
            return "BOTH", "NONE"
        if left_carry:
            return "LEFT_ARM", "LEFT"
        if right_carry:
            return "RIGHT_ARM", "RIGHT"
        return "NONE", "NONE"

    def _resolve_carry_confirmation_result(
        self,
        confirmation_input: CarryConfirmationInput,
        front_carry_hits: int,
        front_carry_one_arm_hits: int,
        two_hand_confirmed: bool,
        one_arm_confirmed: bool,
        front_carry_confirmed: bool,
    ) -> CarryConfirmationResult:
        """Tach phan confirm/final carry state khoi raw evidence input."""
        if (
            two_hand_confirmed
            and confirmation_input.left_carry_allowed
            and confirmation_input.right_carry_allowed
        ):
            left_carry = True
            right_carry = True
            one_hand_carry_side = "NONE"
        else:
            one_hand_carry_side = self._resolve_one_hand_carry_side(
                confirmation_input.left_carry_evidence,
                confirmation_input.right_carry_evidence,
                confirmation_input.left_carry_score,
                confirmation_input.right_carry_score,
                front_carry_two_hand=False,
            )
            left_carry = (
                front_carry_confirmed and one_hand_carry_side == "LEFT"
            )
            right_carry = (
                front_carry_confirmed and one_hand_carry_side == "RIGHT"
            )
            if not front_carry_confirmed:
                one_hand_carry_side = "NONE"

        carrying_arm, normalized_one_hand_side = self._resolve_carrying_arm(
            left_carry,
            right_carry,
        )
        return CarryConfirmationResult(
            front_carry_hits=front_carry_hits,
            front_carry_one_arm_hits=front_carry_one_arm_hits,
            front_carry_confirmed=front_carry_confirmed,
            front_carry_two_hand_confirmed=two_hand_confirmed,
            front_carry_one_arm_confirmed=one_arm_confirmed,
            left_carry=left_carry,
            right_carry=right_carry,
            one_hand_carry_side=normalized_one_hand_side,
            carrying_arm=carrying_arm,
            is_carrying=left_carry or right_carry,
            carry_type="FRONT_CARRY" if (left_carry or right_carry) else "NONE",
        )

    def _apply_carry_confirmation_result(
        self,
        carry_info: CarryInfo,
        confirmation_input: CarryConfirmationInput,
        confirmation_result: CarryConfirmationResult,
    ) -> CarryInfo:
        """Map ket qua confirmation typed ve dict CarryInfo legacy cho downstream."""
        carry_info["front_carry_raw"] = confirmation_input.front_carry_raw
        carry_info["front_carry_hits"] = confirmation_result.front_carry_hits
        carry_info["front_carry_two_hand_raw"] = (
            confirmation_input.front_carry_two_hand_raw
        )
        carry_info["front_carry_two_hand_hits"] = confirmation_result.front_carry_hits
        carry_info["front_carry_one_arm_raw"] = (
            confirmation_input.front_carry_one_arm_raw
        )
        carry_info["front_carry_one_arm_hits"] = (
            confirmation_result.front_carry_one_arm_hits
        )
        carry_info["front_carry_two_hand_confirmed"] = (
            confirmation_result.front_carry_two_hand_confirmed
        )
        carry_info["front_carry_one_arm_confirmed"] = (
            confirmation_result.front_carry_one_arm_confirmed
        )
        carry_info["front_carry_confirmed"] = (
            confirmation_result.front_carry_confirmed
        )
        carry_info["front_carry"] = confirmation_result.is_carrying
        carry_info["left_carry"] = confirmation_result.left_carry
        carry_info["right_carry"] = confirmation_result.right_carry
        carry_info["left_carry_allowed"] = confirmation_input.left_carry_allowed
        carry_info["right_carry_allowed"] = confirmation_input.right_carry_allowed
        carry_info["left_carry_evidence"] = confirmation_input.left_carry_evidence
        carry_info["right_carry_evidence"] = confirmation_input.right_carry_evidence
        carry_info["left_carry_score"] = confirmation_input.left_carry_score
        carry_info["right_carry_score"] = confirmation_input.right_carry_score
        carry_info["one_hand_carry_side"] = confirmation_result.one_hand_carry_side
        carry_info["is_carrying"] = confirmation_result.is_carrying
        carry_info["carrying_arm"] = confirmation_result.carrying_arm
        carry_info["carry_type"] = confirmation_result.carry_type
        carry_info["carry_reason"] = self._derive_carry_reason(
            left_carry_allowed=confirmation_input.left_carry_allowed,
            right_carry_allowed=confirmation_input.right_carry_allowed,
            front_carry_active=confirmation_result.is_carrying,
            front_carry_confirmed=confirmation_result.front_carry_confirmed,
            front_carry_two_hand=(
                confirmation_result.left_carry and confirmation_result.right_carry
            ),
            one_hand_carry_side=confirmation_result.one_hand_carry_side,
        )
        return carry_info

    def _analyze_carry(
        self,
        track_id: AnalysisSubjectID,
        keypoints: KeypointsArray | None,
        holding_raw: bool,
        features: PoseFeatures | None = None,
        carry_pose: CarryInfo | None = None,
        handrail_state: HandrailState | None = None,
    ) -> CarryInfo:
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
        # INPUT: `carry_pose` co the da duoc handrail branch tinh san trong cung frame.
        # WHY: Tai su dung no giup carry va handrail doc chung mot bo bang chung, tranh lech debug.
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
        confirmation_input = self._build_carry_confirmation_input(
            track_id=track_id,
            front_carry_raw=bool(front_carry_raw),
            front_carry_two_hand_raw=front_carry_two_hand_raw,
            front_carry_one_arm_raw=front_carry_one_arm_raw,
            left_carry_allowed=left_carry_allowed,
            right_carry_allowed=right_carry_allowed,
            left_carry_evidence=left_carry_evidence,
            right_carry_evidence=right_carry_evidence,
            left_carry_score=left_carry_score,
            right_carry_score=right_carry_score,
        )
        (
            front_carry_hits,
            front_carry_one_arm_hits,
            two_hand_confirmed,
            one_arm_confirmed,
            front_carry_confirmed,
        ) = self._update_carry_confirmation_history(confirmation_input)
        confirmation_result = self._resolve_carry_confirmation_result(
            confirmation_input,
            front_carry_hits,
            front_carry_one_arm_hits,
            two_hand_confirmed,
            one_arm_confirmed,
            front_carry_confirmed,
        )
        return self._apply_carry_confirmation_result(
            carry_info,
            confirmation_input,
            confirmation_result,
        )


def evaluate_carry(
    analyzer: CarryAnalysisMixin,
    track_id: AnalysisSubjectID,
    keypoints: KeypointsArray | None,
    holding_raw: bool,
    features: PoseFeatures | None = None,
    carry_pose: CarryInfo | None = None,
    handrail_state: HandrailState | None = None,
) -> CarryInfo:
    """Wrapper public de analyzer goi carry analysis.

    OUTPUT:
        - dict carry final sau khi da qua history va gate voi handrail.
    """
    return analyzer._analyze_carry(
        track_id,
        keypoints,
        holding_raw,
        features=features,
        carry_pose=carry_pose,
        handrail_state=handrail_state,
    )
