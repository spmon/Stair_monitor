from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from stair_monitor.common.types import (
    AnalysisSubjectID,
    KeypointsArray,
    LinePoints,
    Point,
    PoseFeatures,
)
from stair_monitor.config.settings import SETTINGS
from stair_monitor.vision.geometry import (
    extract_pose_features,
    get_side_name,
    point_to_segment_distance,
    signed_distance_to_line,
)

# File nay phan tich bang chung vin tay tren ban Windows/demo.
# FLOW: wrist keypoints + handrail line -> hit raw tung tay -> hand claim -> hold final.
# WHY: Can tach ro `hit raw` voi `hold final` vi 1 frame de bi rung keypoint neu ket luan ngay.

LEFT_HANDRAIL_RULE = "LEFT_HANDRAIL_RULE"
RIGHT_HANDRAIL_RULE = "RIGHT_HANDRAIL_RULE"
# WHY: Claim can hit count rieng de tranh 1 tay bi flip lien tuc giua HOLD va CARRY.
HAND_CLAIM_HOLD_HITS = 4
HAND_CLAIM_CARRY_HITS = 4
HAND_CLAIM_RESET_MISSES = 10

HandrailEvidence = dict[str, object]
HoldState = dict[str, object]
HandClaimEntry = dict[str, int | str | None]
HandClaimState = dict[str, HandClaimEntry]
WristDebugPayload = dict[str, object]
PairKey = tuple[str, str]
PairLegacyEvidence = dict[str, object]
PairLegacyMap = dict[PairKey, PairLegacyEvidence | None]
HandSide = Literal["left", "right"]
HandClaimValue = Literal["HOLD", "CARRY"]


@dataclass(frozen=True, slots=True)
class HandrailRawEvidenceInput:
    features: PoseFeatures | None
    left_line: LinePoints | None
    right_line: LinePoints | None


@dataclass(frozen=True, slots=True)
class WristRailEvidence:
    wrist_point: Point
    signed_distance: float
    segment_distance: float
    projection_t: float
    closest_point: Point
    side: str

    def to_legacy_dict(self) -> PairLegacyEvidence:
        return {
            "point": self.wrist_point,
            "dist": self.signed_distance,
            "segment_dist": self.segment_distance,
            "projection_t": self.projection_t,
            "closest_point": self.closest_point,
            "side": self.side,
        }


@dataclass(frozen=True, slots=True)
class HandrailRawEvidenceResult:
    left_line: LinePoints | None
    right_line: LinePoints | None
    pairs: dict[PairKey, WristRailEvidence | None]
    left_wrist_valid: bool
    right_wrist_valid: bool
    holding_left_hand: bool
    holding_right_hand: bool
    left_hand_on_left_rail: bool | str
    left_hand_on_right_rail: bool | str
    right_hand_on_left_rail: bool | str
    right_hand_on_right_rail: bool | str

    def to_legacy_dict(self) -> HandrailEvidence:
        legacy_pairs: PairLegacyMap = {
            pair_key: (
                pair_evidence.to_legacy_dict()
                if pair_evidence is not None
                else None
            )
            for pair_key, pair_evidence in self.pairs.items()
        }
        return {
            "left_line": self.left_line,
            "right_line": self.right_line,
            "pairs": legacy_pairs,
            "left_wrist_valid": self.left_wrist_valid,
            "right_wrist_valid": self.right_wrist_valid,
            "holding_left_hand": self.holding_left_hand,
            "holding_right_hand": self.holding_right_hand,
            "left_hand_on_left_rail": self.left_hand_on_left_rail,
            "left_hand_on_right_rail": self.left_hand_on_right_rail,
            "right_hand_on_left_rail": self.right_hand_on_left_rail,
            "right_hand_on_right_rail": self.right_hand_on_right_rail,
        }


@dataclass(frozen=True, slots=True)
class PerHandBoolState:
    left: bool
    right: bool

    def for_side(self, hand_side: HandSide) -> bool:
        return self.left if hand_side == "left" else self.right

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
class HoldRawClaimPhaseState:
    before_claim: PerHandBoolState
    after_claim: PerHandBoolState


@dataclass(frozen=True, slots=True)
class PerHandClaimSelection:
    left: HandClaimValue | None
    right: HandClaimValue | None

    def for_side(self, hand_side: HandSide) -> HandClaimValue | None:
        return self.left if hand_side == "left" else self.right


@dataclass(frozen=True, slots=True)
class HandClaimUpdateInput:
    hold_raw: PerHandBoolState
    carry_raw: PerHandBoolState
    handrail_active: PerHandBoolState


def build_handrail_raw_evidence_input(
    features: PoseFeatures | None,
    left_line: LinePoints | None,
    right_line: LinePoints | None,
) -> HandrailRawEvidenceInput:
    """Chuan hoa input raw handrail evidence cho boundary typed noi bo."""
    return HandrailRawEvidenceInput(
        features=features,
        left_line=left_line,
        right_line=right_line,
    )


def _read_per_hand_bool_state(
    values: dict[str, object],
    *,
    left_key: str,
    right_key: str,
) -> PerHandBoolState:
    """Doc 2 field trai/phai tu dict legacy va chuyen thanh state typed noi bo."""
    return PerHandBoolState(
        left=bool(values.get(left_key, False)),
        right=bool(values.get(right_key, False)),
    )


def _read_hold_raw_claim_phase_state(
    hold_state: HoldState,
) -> HoldRawClaimPhaseState:
    """Tach ro hold raw truoc claim va sau claim tu hold_state legacy."""
    before_claim = PerHandBoolState(
        left=bool(
            hold_state.get(
                "left_hold_raw_before_claim",
                hold_state.get("left_hold_raw", False),
            )
        ),
        right=bool(
            hold_state.get(
                "right_hold_raw_before_claim",
                hold_state.get("right_hold_raw", False),
            )
        ),
    )
    after_claim = PerHandBoolState(
        left=bool(
            hold_state.get(
                "left_hold_raw_after_claim",
                hold_state.get("left_hold_raw", False),
            )
        ),
        right=bool(
            hold_state.get(
                "right_hold_raw_after_claim",
                hold_state.get("right_hold_raw", False),
            )
        ),
    )
    return HoldRawClaimPhaseState(
        before_claim=before_claim,
        after_claim=after_claim,
    )


def _get_hand_claim_label(
    hand_claim_state: HandClaimState | None,
    hand_side: HandSide,
) -> HandClaimValue | None:
    """Lay claim hien tai cua 1 tay neu da duoc gan HOLD/CARRY."""
    if hand_claim_state is None:
        return None
    claim_value = hand_claim_state.get(hand_side, {}).get("claim")
    if claim_value in ("HOLD", "CARRY"):
        return claim_value
    return None


def _read_per_hand_claim_selection(
    hand_claim_state: HandClaimState | None,
) -> PerHandClaimSelection:
    """Doc claim per-hand tu state dict va chuan hoa ve boundary typed noi bo."""
    return PerHandClaimSelection(
        left=_get_hand_claim_label(hand_claim_state, "left"),
        right=_get_hand_claim_label(hand_claim_state, "right"),
    )


def _apply_carry_claim_to_hold_raw_state(
    raw_hold_state: PerHandBoolState,
    claim_selection: PerHandClaimSelection,
) -> PerHandBoolState:
    """Chi claim CARRY da confirmed moi duoc suppress hold raw cung tay."""
    return PerHandBoolState(
        left=False if claim_selection.left == "CARRY" else raw_hold_state.left,
        right=False if claim_selection.right == "CARRY" else raw_hold_state.right,
    )


def _build_hand_claim_update_input(
    *,
    left_hold_raw: bool,
    right_hold_raw: bool,
    left_carry_raw: bool,
    right_carry_raw: bool,
    left_handrail_active: bool,
    right_handrail_active: bool,
) -> HandClaimUpdateInput:
    """Gom raw hold/raw carry/handrail hit theo tung tay cho claim transition."""
    return HandClaimUpdateInput(
        hold_raw=PerHandBoolState(
            left=left_hold_raw,
            right=right_hold_raw,
        ),
        carry_raw=PerHandBoolState(
            left=left_carry_raw,
            right=right_carry_raw,
        ),
        handrail_active=PerHandBoolState(
            left=left_handrail_active,
            right=right_handrail_active,
        ),
    )


# Body facing phuc vu logic di lui.
def is_front_to_camera(body_facing: str | None) -> bool:
    return body_facing is not None and "FRONT_TO_CAMERA" in str(body_facing)


# Body facing phuc vu logic di lui.
def is_back_to_camera(body_facing: str | None) -> bool:
    return body_facing is not None and "BACK_TO_CAMERA" in str(body_facing)


# Danh gia 1 co tay so voi 1 line lan can.
# Vua luu signed distance de biet dung phia nao, vua luu segment distance de tranh bat nham phan keo dai vo han.
def _evaluate_wrist_against_line(
    wrist_point: Point | None,
    line: LinePoints | None,
) -> HandrailEvidence | None:
    evidence = _evaluate_wrist_against_line_typed(wrist_point, line)
    if evidence is None:
        return None
    return evidence.to_legacy_dict()


def _evaluate_wrist_against_line_typed(
    wrist_point: Point | None,
    line: LinePoints | None,
) -> WristRailEvidence | None:
    """Danh gia 1 wrist voi 1 line lan can.

    Args:
        wrist_point: Toa do wrist dang xet.
        line: Hai diem dau mut cua line lan can.

    Returns:
        dict | None: dist, segment_dist, projection_t, side va point.

    Notes:
        dist la signed distance den line vo han. segment_dist la khoang cach den
        chinh doan line. projection_t cho biet hinh chieu roi trong doan hay khong.
    """
    if wrist_point is None or line is None or len(line) < 2:
        return None

    dist = signed_distance_to_line(wrist_point, line)
    segment_dist, projection_t, closest = point_to_segment_distance(
        wrist_point,
        line[0],
        line[1],
    )
    return WristRailEvidence(
        wrist_point=wrist_point,
        signed_distance=dist,
        segment_distance=segment_dist,
        projection_t=projection_t,
        closest_point=(
            int(round(float(closest[0]))),
            int(round(float(closest[1]))),
        ),
        side=get_side_name(dist),
    )


def _pair_rule_for_rail(rail_name: str) -> str:
    if rail_name == "LEFT_HANDRAIL":
        return LEFT_HANDRAIL_RULE
    return RIGHT_HANDRAIL_RULE


def _pair_holds_rail_from_evidence(
    pair_results: dict[PairKey, WristRailEvidence | None],
    wrist_name: str,
    rail_name: str,
    rule: str,
) -> bool | str:
    pair = pair_results.get((wrist_name, rail_name))
    if pair is None:
        return "UNKNOWN"
    if rule == LEFT_HANDRAIL_RULE:
        return bool(
            -SETTINGS.handrail.left_max_distance <= pair.signed_distance <= -10
        )
    if rule == RIGHT_HANDRAIL_RULE:
        return bool(10 <= pair.signed_distance <= 60)
    return "UNKNOWN"


def compute_handrail_evidence_typed(
    rule_input: HandrailRawEvidenceInput,
) -> HandrailRawEvidenceResult:
    """Tinh raw wrist/rail evidence voi contract typed, sau do caller co the convert ve dict cu."""
    left_wrist = (
        rule_input.features.get("left_wrist")
        if rule_input.features is not None
        else None
    )
    right_wrist = (
        rule_input.features.get("right_wrist")
        if rule_input.features is not None
        else None
    )

    pair_results = {
        ("LEFT_WRIST", "LEFT_HANDRAIL"): _evaluate_wrist_against_line_typed(
            left_wrist,
            rule_input.left_line,
        ),
        ("LEFT_WRIST", "RIGHT_HANDRAIL"): _evaluate_wrist_against_line_typed(
            left_wrist,
            rule_input.right_line,
        ),
        ("RIGHT_WRIST", "LEFT_HANDRAIL"): _evaluate_wrist_against_line_typed(
            right_wrist,
            rule_input.left_line,
        ),
        ("RIGHT_WRIST", "RIGHT_HANDRAIL"): _evaluate_wrist_against_line_typed(
            right_wrist,
            rule_input.right_line,
        ),
    }

    left_holding = any(
        pair_results.get((wrist_name, rail_name)) is not None
        and _pair_holds_rail_from_evidence(
            pair_results,
            wrist_name,
            rail_name,
            _pair_rule_for_rail(rail_name),
        )
        is True
        for wrist_name, rail_name in pair_results
        if wrist_name == "LEFT_WRIST"
    )
    right_holding = any(
        pair_results.get((wrist_name, rail_name)) is not None
        and _pair_holds_rail_from_evidence(
            pair_results,
            wrist_name,
            rail_name,
            _pair_rule_for_rail(rail_name),
        )
        is True
        for wrist_name, rail_name in pair_results
        if wrist_name == "RIGHT_WRIST"
    )

    return HandrailRawEvidenceResult(
        left_line=rule_input.left_line,
        right_line=rule_input.right_line,
        pairs=pair_results,
        left_wrist_valid=left_wrist is not None,
        right_wrist_valid=right_wrist is not None,
        holding_left_hand=left_holding,
        holding_right_hand=right_holding,
        left_hand_on_left_rail=_pair_holds_rail_from_evidence(
            pair_results,
            "LEFT_WRIST",
            "LEFT_HANDRAIL",
            LEFT_HANDRAIL_RULE,
        ),
        left_hand_on_right_rail=_pair_holds_rail_from_evidence(
            pair_results,
            "LEFT_WRIST",
            "RIGHT_HANDRAIL",
            RIGHT_HANDRAIL_RULE,
        ),
        right_hand_on_left_rail=_pair_holds_rail_from_evidence(
            pair_results,
            "RIGHT_WRIST",
            "LEFT_HANDRAIL",
            LEFT_HANDRAIL_RULE,
        ),
        right_hand_on_right_rail=_pair_holds_rail_from_evidence(
            pair_results,
            "RIGHT_WRIST",
            "RIGHT_HANDRAIL",
            RIGHT_HANDRAIL_RULE,
        ),
    )


# Gom bang chung cho tung cap co tay - lan can trong 1 frame.
# Bang chung nay se duoc tai su dung cho hold logic va debug, khong sua doi ket qua nhan dien.
def compute_handrail_evidence(
    features: PoseFeatures | None,
    left_line: LinePoints | None,
    right_line: LinePoints | None,
    config: object | None = None,
) -> HandrailEvidence:
    """Tinh bang chung handrail frame-level cho ca 2 wrist va 2 line.

    Args:
        features: Dict pose feature da extract.
        left_line: Line lan can trai.
        right_line: Line lan can phai.
        config: Bien de giu API, hien khong can dung them.

    Returns:
        dict: Tat ca pair wrist-line va cac field debug tong hop.

    Notes:
        LEFT_HANDRAIL_RULE / RIGHT_HANDRAIL_RULE chi mo ta nguong vat ly de xem
        wrist co dang bam vao rail hay khong. Runtime decision cuoi cung se chi
        nhin truc tiep left_holding / right_holding.
    """
    _ = config
    rule_input = build_handrail_raw_evidence_input(
        features=features,
        left_line=left_line,
        right_line=right_line,
    )
    return compute_handrail_evidence_typed(rule_input).to_legacy_dict()


def evaluate_holding_status(
    analyzer: HandrailAnalysisMixin,
    hold_direction: str | None,
    handrail_evidence: HandrailEvidence,
) -> HoldState:
    """Wrapper goi analyzer de lay hold state frame-level.

    INPUT:
        - `hold_direction`: direction dang dung cho frame hien tai.
        - `handrail_evidence`: bang chung wrist-line da tinh san.

    OUTPUT:
        - dict hold raw cho frame hien tai.
    """
    return analyzer._evaluate_hold_state(hold_direction, handrail_evidence)


def evaluate_handrail(
    analyzer: HandrailAnalysisMixin,
    track_id: AnalysisSubjectID,
    hold_direction: str | None,
    keypoints: KeypointsArray | None,
    features: PoseFeatures,
) -> tuple[HoldState, dict[str, object], HandClaimState]:
    """Tinh handrail final cho 1 subject trong 1 frame.

    INPUT:
        - Wrist/keypoint da extract trong `features`.
        - 2 line lan can cua camera.

    OUTPUT:
        - `hold_state`, `carry_pose`, `hand_claim_state`.

    WHY:
        - Handrail va carry dung chung du lieu tay, nen phai tra them claim
          state de analyzer giai xung dot giua 2 rule.
    """
    handrail_evidence = compute_handrail_evidence(
        features,
        analyzer.left_line,
        analyzer.right_line,
    )
    return analyzer._resolve_hold_and_claim_state(
        track_id,
        hold_direction,
        handrail_evidence,
        keypoints,
        False,
        features,
    )


class HandrailAnalysisMixin:
    """Mixin gom logic hold raw trai/phai va hand-claim.

    WARNING:
        - Quy uoc nghiep vu hien tai xem vin ben phai la dung, ben trai la sai ben.
        - Ket luan final chi duoc xuat ra sau khi bang chung da qua history/hit count.
    """

    @staticmethod
    def _format_hit_confirm_debug_reason(
        reason_code: str,
        hit_count: int,
        confirm_required: int,
    ) -> str:
        return (
            f"{reason_code} "
            f"hit_count={int(hit_count)}/{int(confirm_required)}"
        )

    @classmethod
    def _build_confirmed_handrail_decision(
        cls,
        *,
        left_wrist_valid: bool,
        right_wrist_valid: bool,
        left_hit: bool,
        right_hit: bool,
        left_hit_count: int,
        right_hit_count: int,
        left_confirm_required: int,
        right_confirm_required: int,
        left_holding: bool,
        right_holding: bool,
        left_hand_claim: str,
        right_hand_claim: str,
    ) -> dict[str, str]:
        # WARNING: Day la noi doi hold history thanh nhan nghiep vu cuoi cung:
        # `OK`, `VIN_SAI_BEN`, `KHONG_VIN` hoac `WAIT_HOLD_CONFIRM`.
        if left_holding:
            return {
                "handrail_status": "VIN_SAI_BEN",
                "handrail_reason": "LEFT_HAND_HOLDING_IS_WRONG",
                "handrail_debug_reason": "LEFT_HAND_HOLDING_IS_WRONG",
            }

        if right_holding:
            return {
                "handrail_status": "OK",
                "handrail_reason": "RIGHT_HAND_HOLDING_OK",
                "handrail_debug_reason": "RIGHT_HAND_HOLDING_OK",
            }

        if right_hit:
            if right_hand_claim == "CARRY":
                return {
                    "handrail_status": "WAIT_HOLD_CONFIRM",
                    "handrail_reason": "RIGHT_HIT_SUPPRESSED_BY_CARRY",
                    "handrail_debug_reason": cls._format_hit_confirm_debug_reason(
                        "RIGHT_HIT_SUPPRESSED_BY_CARRY",
                        right_hit_count,
                        right_confirm_required,
                    ),
                }
            return {
                "handrail_status": "WAIT_HOLD_CONFIRM",
                "handrail_reason": "RIGHT_HIT_NOT_CONFIRMED",
                "handrail_debug_reason": cls._format_hit_confirm_debug_reason(
                    "RIGHT_HIT_NOT_CONFIRMED",
                    right_hit_count,
                    right_confirm_required,
                ),
            }

        if left_hit:
            if left_hand_claim == "CARRY":
                return {
                    "handrail_status": "WAIT_HOLD_CONFIRM",
                    "handrail_reason": "LEFT_HIT_SUPPRESSED_BY_CARRY",
                    "handrail_debug_reason": cls._format_hit_confirm_debug_reason(
                        "LEFT_HIT_SUPPRESSED_BY_CARRY",
                        left_hit_count,
                        left_confirm_required,
                    ),
                }
            return {
                "handrail_status": "WAIT_HOLD_CONFIRM",
                "handrail_reason": "LEFT_HIT_NOT_CONFIRMED",
                "handrail_debug_reason": cls._format_hit_confirm_debug_reason(
                    "LEFT_HIT_NOT_CONFIRMED",
                    left_hit_count,
                    left_confirm_required,
                ),
            }

        if not left_wrist_valid and not right_wrist_valid:
            return {
                "handrail_status": "KHONG_VIN",
                "handrail_reason": "WRIST_LOW_CONF",
                "handrail_debug_reason": "WRIST_LOW_CONF",
            }

        return {
            "handrail_status": "KHONG_VIN",
            "handrail_reason": "NO_HAND_HOLDING",
            "handrail_debug_reason": "NO_HAND_HOLDING",
        }

    @staticmethod
    def _build_wrist_debug_payload(
        handrail_evidence: HandrailEvidence | None,
        wrist_name: str,
        wrist_valid: bool,
        wrist_hit: bool,
    ) -> WristDebugPayload:
        pair_results = handrail_evidence.get("pairs", {}) if handrail_evidence else {}
        left_pair = pair_results.get((wrist_name, "LEFT_HANDRAIL"))
        right_pair = pair_results.get((wrist_name, "RIGHT_HANDRAIL"))

        nearest_rail = "NONE"
        nearest_distance = None
        nearest_point = None
        pair_candidates = []
        if left_pair is not None:
            pair_candidates.append(("LEFT_HANDRAIL", left_pair))
        if right_pair is not None:
            pair_candidates.append(("RIGHT_HANDRAIL", right_pair))
        if pair_candidates:
            nearest_rail, nearest_pair = min(
                pair_candidates,
                key=lambda candidate: candidate[1]["segment_dist"],
            )
            nearest_distance = nearest_pair.get("segment_dist")
            nearest_point = nearest_pair.get("closest_point")

        return {
            "valid": wrist_valid,
            "hit": wrist_hit,
            "distance_to_left_rail": (
                left_pair.get("segment_dist") if left_pair is not None else None
            ),
            "distance_to_right_rail": (
                right_pair.get("segment_dist") if right_pair is not None else None
            ),
            "nearest_rail": nearest_rail,
            "nearest_distance": nearest_distance,
            "nearest_point": nearest_point,
        }

    @staticmethod
    def _extract_wrist_debug_payload_from_hold_state(
        hold_state: HoldState,
        wrist_side: HandSide,
    ) -> WristDebugPayload:
        if wrist_side == "left":
            return {
                "valid": bool(hold_state.get("left_wrist_valid", False)),
                "hit": bool(hold_state.get("left_wrist_hit", False)),
                "distance_to_left_rail": hold_state.get(
                    "left_wrist_distance_to_left_rail"
                ),
                "distance_to_right_rail": hold_state.get(
                    "left_wrist_distance_to_right_rail"
                ),
                "nearest_rail": str(
                    hold_state.get("left_wrist_nearest_rail", "NONE")
                ),
                "nearest_distance": hold_state.get("left_wrist_nearest_distance"),
                "nearest_point": hold_state.get("left_wrist_nearest_point"),
            }
        return {
            "valid": bool(hold_state.get("right_wrist_valid", False)),
            "hit": bool(hold_state.get("right_wrist_hit", False)),
            "distance_to_left_rail": hold_state.get(
                "right_wrist_distance_to_left_rail"
            ),
            "distance_to_right_rail": hold_state.get(
                "right_wrist_distance_to_right_rail"
            ),
            "nearest_rail": str(hold_state.get("right_wrist_nearest_rail", "NONE")),
            "nearest_distance": hold_state.get("right_wrist_nearest_distance"),
            "nearest_point": hold_state.get("right_wrist_nearest_point"),
        }

    def _build_left_right_hold_state(
        self,
        left_wrist_valid: bool,
        right_wrist_valid: bool,
        raw_hold_state: PerHandBoolState,
        left_debug_payload: WristDebugPayload | None = None,
        right_debug_payload: WristDebugPayload | None = None,
        hold_raw_claim_phase: HoldRawClaimPhaseState | None = None,
    ) -> HoldState:
        if left_debug_payload is None:
            left_debug_payload = {}
        if right_debug_payload is None:
            right_debug_payload = {}
        if hold_raw_claim_phase is None:
            hold_raw_claim_phase = HoldRawClaimPhaseState(
                before_claim=raw_hold_state,
                after_claim=raw_hold_state,
            )

        hold_info = {
            "holding_raw": raw_hold_state.left or raw_hold_state.right,
            "hold_raw_status": "NONE",
            "holding_correct_raw": False,
            "holding_wrong_raw": False,
            "left_wrist_valid": left_wrist_valid,
            "right_wrist_valid": right_wrist_valid,
            "left_wrist_hit": bool(
                left_debug_payload.get("hit", raw_hold_state.left)
            ),
            "right_wrist_hit": bool(
                right_debug_payload.get("hit", raw_hold_state.right)
            ),
            "left_wrist_distance_to_left_rail": left_debug_payload.get(
                "distance_to_left_rail"
            ),
            "left_wrist_distance_to_right_rail": left_debug_payload.get(
                "distance_to_right_rail"
            ),
            "right_wrist_distance_to_left_rail": right_debug_payload.get(
                "distance_to_left_rail"
            ),
            "right_wrist_distance_to_right_rail": right_debug_payload.get(
                "distance_to_right_rail"
            ),
            "left_wrist_nearest_distance": left_debug_payload.get(
                "nearest_distance"
            ),
            "right_wrist_nearest_distance": right_debug_payload.get(
                "nearest_distance"
            ),
            "left_wrist_nearest_rail": str(
                left_debug_payload.get("nearest_rail", "NONE")
            ),
            "right_wrist_nearest_rail": str(
                right_debug_payload.get("nearest_rail", "NONE")
            ),
            "left_wrist_nearest_point": left_debug_payload.get("nearest_point"),
            "right_wrist_nearest_point": right_debug_payload.get("nearest_point"),
            "handrail_status": "KHONG_VIN",
            "handrail_reason": "NO_HAND_HOLDING",
            "handrail_debug_reason": "NO_HAND_HOLDING",
        }
        hold_info.update(
            raw_hold_state.to_legacy_fields(
                left_key="left_holding",
                right_key="right_holding",
            )
        )
        hold_info.update(
            raw_hold_state.to_legacy_fields(
                left_key="left_hold_raw",
                right_key="right_hold_raw",
            )
        )
        hold_info.update(
            hold_raw_claim_phase.before_claim.to_legacy_fields(
                left_key="left_hold_raw_before_claim",
                right_key="right_hold_raw_before_claim",
            )
        )
        hold_info.update(
            hold_raw_claim_phase.after_claim.to_legacy_fields(
                left_key="left_hold_raw_after_claim",
                right_key="right_hold_raw_after_claim",
            )
        )

        if raw_hold_state.left:
            hold_info["hold_raw_status"] = "WRONG_SIDE"
            hold_info["holding_wrong_raw"] = True
            hold_info["handrail_status"] = "VIN_SAI_BEN"
            hold_info["handrail_reason"] = "LEFT_HAND_HOLDING_IS_WRONG"
            hold_info["handrail_debug_reason"] = "LEFT_HAND_HOLDING_IS_WRONG"
        elif raw_hold_state.right:
            hold_info["hold_raw_status"] = "CORRECT"
            hold_info["holding_correct_raw"] = True
            hold_info["handrail_status"] = "OK"
            hold_info["handrail_reason"] = "RIGHT_HAND_HOLDING_OK"
            hold_info["handrail_debug_reason"] = "RIGHT_HAND_HOLDING_OK"
        elif not left_wrist_valid and not right_wrist_valid:
            hold_info["handrail_reason"] = "WRIST_LOW_CONF"
            hold_info["handrail_debug_reason"] = "WRIST_LOW_CONF"
        elif hold_info["right_wrist_hit"]:
            hold_info["handrail_status"] = "WAIT_HOLD_CONFIRM"
            hold_info["handrail_reason"] = "RIGHT_HIT_NOT_CONFIRMED"
            hold_info["handrail_debug_reason"] = "RIGHT_HIT_NOT_CONFIRMED"
        elif hold_info["left_wrist_hit"]:
            hold_info["handrail_status"] = "WAIT_HOLD_CONFIRM"
            hold_info["handrail_reason"] = "LEFT_HIT_NOT_CONFIRMED"
            hold_info["handrail_debug_reason"] = "LEFT_HIT_NOT_CONFIRMED"

        return hold_info

    def _evaluate_hold_state(self, *args: object) -> HoldState:
        """Danh gia hold raw theo direction neu co, hoac theo any-rail neu chua co.

        Args:
            *args: Ho tro 2 dang goi:
                - (hold_direction, handrail_evidence)
                - (keypoints, hold_direction, left_line, right_line)

        Returns:
            dict: Hold state frame-level.

        Notes:
            UNKNOWN/IDLE khong duoc tu dong suy ra "Khong Vin". Khi chua biet
            direction, ham nay chi duoc phep ket luan co/khong co vin mot rail.
        """
        if len(args) == 2:
            _hold_direction, handrail_evidence = args
        elif len(args) == 4:
            keypoints, _hold_direction, left_line, right_line = args
            features = extract_pose_features(keypoints, None)
            handrail_evidence = compute_handrail_evidence(
                features,
                left_line,
                right_line,
            )
        else:
            raise TypeError(
                "_evaluate_hold_state expects either "
                "(hold_direction, handrail_evidence) or "
                "(keypoints, hold_direction, left_line, right_line)"
            )

        if handrail_evidence is None:
            handrail_evidence = {}
        left_wrist_valid = bool(handrail_evidence.get("left_wrist_valid", False))
        right_wrist_valid = bool(handrail_evidence.get("right_wrist_valid", False))
        raw_hold_state = PerHandBoolState(
            left=bool(handrail_evidence.get("holding_left_hand", False)),
            right=bool(handrail_evidence.get("holding_right_hand", False)),
        )
        return self._build_left_right_hold_state(
            left_wrist_valid,
            right_wrist_valid,
            raw_hold_state,
            left_debug_payload=self._build_wrist_debug_payload(
                handrail_evidence,
                "LEFT_WRIST",
                left_wrist_valid,
                raw_hold_state.left,
            ),
            right_debug_payload=self._build_wrist_debug_payload(
                handrail_evidence,
                "RIGHT_WRIST",
                right_wrist_valid,
                raw_hold_state.right,
            ),
        )

    # Claim CARRY loai bo tay do khoi hold de tranh 1 tay vua "vin" vua "mang vac".
    # Hold la logic doc lap, carry chi duoc anh huong qua lop claim nay.
    def _apply_hand_claim_to_hold_state(
        self,
        hold_state: HoldState,
        hand_claim_state: HandClaimState | None,
    ) -> HoldState:
        """Loai candidate hold o tay da duoc claim CARRY.

        Args:
            hold_state: Hold raw state truoc khi ap claim.
            hand_claim_state: Claim state hien tai cua tung tay.

        Returns:
            dict: Hold state sau khi da bo candidate bi claim CARRY.

        Notes:
            Claim nay chi giam xung dot bang chung. Viec canh bao final van do
            history hold cua analyzer xac nhan.
        """
        if hand_claim_state is None:
            hand_claim_state = {}

        raw_hold_before_claim = _read_per_hand_bool_state(
            hold_state,
            left_key="left_holding",
            right_key="right_holding",
        )
        claim_selection = _read_per_hand_claim_selection(hand_claim_state)
        effective_hold_after_claim = _apply_carry_claim_to_hold_raw_state(
            raw_hold_before_claim,
            claim_selection,
        )
        hold_raw_claim_phase = _read_hold_raw_claim_phase_state(hold_state)

        claimed_hold_state = self._build_left_right_hold_state(
            bool(hold_state.get("left_wrist_valid", False)),
            bool(hold_state.get("right_wrist_valid", False)),
            effective_hold_after_claim,
            left_debug_payload=self._extract_wrist_debug_payload_from_hold_state(
                hold_state,
                "left",
            ),
            right_debug_payload=self._extract_wrist_debug_payload_from_hold_state(
                hold_state,
                "right",
            ),
            hold_raw_claim_phase=HoldRawClaimPhaseState(
                before_claim=hold_raw_claim_phase.before_claim,
                after_claim=effective_hold_after_claim,
            ),
        )
        return claimed_hold_state

    @staticmethod
    def _new_hand_claim_entry() -> HandClaimEntry:
        return {
            "claim": None,
            "hold_hits": 0,
            "carry_hits": 0,
            "misses": 0,
        }

    def _get_hand_claim_state(self, track_id: AnalysisSubjectID) -> HandClaimState:
        if track_id not in self.hand_claim_state:
            self.hand_claim_state[track_id] = {
                "left": self._new_hand_claim_entry(),
                "right": self._new_hand_claim_entry(),
            }
        return self.hand_claim_state[track_id]

    def _update_hand_claim_state(
        self,
        track_id: AnalysisSubjectID,
        left_hold_raw: bool,
        right_hold_raw: bool,
        left_carry_raw: bool,
        right_carry_raw: bool,
        left_handrail_active: bool = False,
        right_handrail_active: bool = False,
    ) -> HandClaimState:
        # WHY: Mot tay khong nen vua duoc tinh la HOLD vua duoc tinh la CARRY trong cung mot giai doan.
        # Claim state la lop trung gian de khoa vai tro cua tung tay sau khi bang chung da du hit.
        claim_state = self._get_hand_claim_state(track_id)
        claim_update_input = _build_hand_claim_update_input(
            left_hold_raw=left_hold_raw,
            right_hold_raw=right_hold_raw,
            left_carry_raw=left_carry_raw,
            right_carry_raw=right_carry_raw,
            left_handrail_active=left_handrail_active,
            right_handrail_active=right_handrail_active,
        )

        for hand_side in ("left", "right"):
            hand_state = claim_state[hand_side]
            hold_raw_active = claim_update_input.hold_raw.for_side(hand_side)
            carry_raw_active = claim_update_input.carry_raw.for_side(hand_side)
            handrail_hit_active = claim_update_input.handrail_active.for_side(
                hand_side
            )
            hand_state["hold_hits"] = (
                min(hand_state["hold_hits"] + 1, HAND_CLAIM_HOLD_HITS)
                if hold_raw_active
                else max(hand_state["hold_hits"] - 1, 0)
            )
            hand_state["carry_hits"] = (
                min(hand_state["carry_hits"] + 1, HAND_CLAIM_CARRY_HITS)
                if carry_raw_active
                else max(hand_state["carry_hits"] - 1, 0)
            )

            if not hold_raw_active and not carry_raw_active:
                hand_state["misses"] += 1
            else:
                hand_state["misses"] = 0

            if handrail_hit_active:
                hand_state["carry_hits"] = 0
                if hand_state["claim"] == "CARRY":
                    hand_state["claim"] = None

            if hand_state["misses"] >= HAND_CLAIM_RESET_MISSES:
                claim_state[hand_side] = self._new_hand_claim_entry()
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

    def _resolve_hold_and_claim_state(
        self,
        track_id: AnalysisSubjectID,
        hold_direction: str | None,
        handrail_evidence: HandrailEvidence,
        keypoints: KeypointsArray | None,
        holding_raw: bool,
        features: PoseFeatures | None,
    ) -> tuple[HoldState, dict[str, object], HandClaimState]:
        """Ghep handrail raw voi carry raw de ra trang thai tay cuoi cung.

        FLOW:
            1. Tinh hold raw tu wrist/rail.
            2. Tinh carry raw tu pose tay/than.
            3. Cap nhat claim HOLD/CARRY cho tung tay.
            4. Ap claim nguoc lai vao hold va carry de tranh xung dot.

        OUTPUT:
            - hold_state sau claim
            - carry_pose sau claim
            - hand_claim_state
        """
        hold_state_before_claim = self._evaluate_hold_state(
            hold_direction,
            handrail_evidence,
        )
        carry_pose = self._get_carry_pose(
            keypoints,
            holding_raw,
            features=features,
            handrail_state=hold_state_before_claim,
        )
        hand_claim_state = self._update_hand_claim_state(
            track_id,
            hold_state_before_claim["left_hold_raw"],
            hold_state_before_claim["right_hold_raw"],
            carry_pose["left_carry_raw_before_claim"],
            carry_pose["right_carry_raw_before_claim"],
            left_handrail_active=bool(hold_state_before_claim["left_wrist_hit"]),
            right_handrail_active=bool(hold_state_before_claim["right_wrist_hit"]),
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
    def _copy_hold_state_fields(hold_state: HoldState) -> tuple[object, ...]:
        return (
            hold_state["holding_raw"],
            hold_state["hold_raw_status"],
            hold_state["holding_correct_raw"],
            hold_state["holding_wrong_raw"],
            hold_state["left_wrist_valid"],
            hold_state["right_wrist_valid"],
            hold_state["left_holding"],
            hold_state["right_holding"],
            hold_state["handrail_status"],
            hold_state["handrail_reason"],
        )
