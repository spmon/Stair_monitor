from stair_monitor.config.settings import SETTINGS
from stair_monitor.vision.geometry import (
    extract_pose_features,
    get_side_name,
    point_to_segment_distance,
    signed_distance_to_line,
)

LEFT_HANDRAIL_RULE = "LEFT_HANDRAIL_RULE"
RIGHT_HANDRAIL_RULE = "RIGHT_HANDRAIL_RULE"
HAND_CLAIM_HOLD_HITS = 4
HAND_CLAIM_CARRY_HITS = 4
HAND_CLAIM_RESET_MISSES = 10


# Body facing phuc vu logic di lui.
def is_front_to_camera(body_facing):
    return body_facing is not None and "FRONT_TO_CAMERA" in str(body_facing)


# Body facing phuc vu logic di lui.
def is_back_to_camera(body_facing):
    return body_facing is not None and "BACK_TO_CAMERA" in str(body_facing)


# Danh gia 1 co tay so voi 1 line lan can.
# Vua luu signed distance de biet dung phia nao, vua luu segment distance de tranh bat nham phan keo dai vo han.
def _evaluate_wrist_against_line(wrist_point, line):
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
    return {
        "point": wrist_point,
        "dist": dist,
        "segment_dist": segment_dist,
        "projection_t": projection_t,
        "closest_point": (
            int(round(float(closest[0]))),
            int(round(float(closest[1]))),
        ),
        "side": get_side_name(dist),
    }


# Gom bang chung cho tung cap co tay - lan can trong 1 frame.
# Bang chung nay se duoc tai su dung cho hold logic va debug, khong sua doi ket qua nhan dien.
def compute_handrail_evidence(features, left_line, right_line, config=None):
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
    left_wrist = features.get("left_wrist") if features is not None else None
    right_wrist = features.get("right_wrist") if features is not None else None

    pair_results = {
        ("LEFT_WRIST", "LEFT_HANDRAIL"): _evaluate_wrist_against_line(
            left_wrist,
            left_line,
        ),
        ("LEFT_WRIST", "RIGHT_HANDRAIL"): _evaluate_wrist_against_line(
            left_wrist,
            right_line,
        ),
        ("RIGHT_WRIST", "LEFT_HANDRAIL"): _evaluate_wrist_against_line(
            right_wrist,
            left_line,
        ),
        ("RIGHT_WRIST", "RIGHT_HANDRAIL"): _evaluate_wrist_against_line(
            right_wrist,
            right_line,
        ),
    }

    def _pair_holds_rail(wrist_name, rail_name, rule):
        pair = pair_results.get((wrist_name, rail_name))
        if pair is None:
            return "UNKNOWN"
        if rule == LEFT_HANDRAIL_RULE:
            return bool(
                -SETTINGS.handrail.left_max_distance <= pair["dist"] <= -10
            )
        if rule == RIGHT_HANDRAIL_RULE:
            return bool(0 <= pair["dist"] <= 60)
        return "UNKNOWN"

    # Ket qua o day chi la bang chung frame-level cho tung co tay/tung rail.
    # Runtime hold decision se quy ve 2 bien left_holding / right_holding.
    return {
        "left_line": left_line,
        "right_line": right_line,
        "pairs": pair_results,
        "left_wrist_valid": left_wrist is not None,
        "right_wrist_valid": right_wrist is not None,
        "holding_left_hand": any(
            pair_results.get((wrist_name, rail_name)) is not None
            and _pair_holds_rail(
                wrist_name,
                rail_name,
                LEFT_HANDRAIL_RULE if rail_name == "LEFT_HANDRAIL" else RIGHT_HANDRAIL_RULE,
            )
            is True
            for wrist_name, rail_name in pair_results
            if wrist_name == "LEFT_WRIST"
        ),
        "holding_right_hand": any(
            pair_results.get((wrist_name, rail_name)) is not None
            and _pair_holds_rail(
                wrist_name,
                rail_name,
                LEFT_HANDRAIL_RULE if rail_name == "LEFT_HANDRAIL" else RIGHT_HANDRAIL_RULE,
            )
            is True
            for wrist_name, rail_name in pair_results
            if wrist_name == "RIGHT_WRIST"
        ),
        "left_hand_on_left_rail": _pair_holds_rail(
            "LEFT_WRIST",
            "LEFT_HANDRAIL",
            LEFT_HANDRAIL_RULE,
        ),
        "left_hand_on_right_rail": _pair_holds_rail(
            "LEFT_WRIST",
            "RIGHT_HANDRAIL",
            RIGHT_HANDRAIL_RULE,
        ),
        "right_hand_on_left_rail": _pair_holds_rail(
            "RIGHT_WRIST",
            "LEFT_HANDRAIL",
            LEFT_HANDRAIL_RULE,
        ),
        "right_hand_on_right_rail": _pair_holds_rail(
            "RIGHT_WRIST",
            "RIGHT_HANDRAIL",
            RIGHT_HANDRAIL_RULE,
        ),
}


def evaluate_holding_status(analyzer, hold_direction, handrail_evidence):
    return analyzer._evaluate_hold_state(hold_direction, handrail_evidence)


def evaluate_handrail(
    analyzer,
    track_id,
    hold_direction,
    keypoints,
    features,
):
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
    """Mixin gom logic hold raw trai/phai va hand-claim."""

    @staticmethod
    def _format_hit_confirm_debug_reason(
        reason_code,
        hit_count,
        confirm_required,
    ) -> str:
        return (
            f"{reason_code} "
            f"hit_count={int(hit_count)}/{int(confirm_required)}"
        )

    @classmethod
    def _build_confirmed_handrail_decision(
        cls,
        *,
        left_wrist_valid,
        right_wrist_valid,
        left_hit,
        right_hit,
        left_hit_count,
        right_hit_count,
        left_confirm_required,
        right_confirm_required,
        left_holding,
        right_holding,
        left_hand_claim,
        right_hand_claim,
    ):
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
        handrail_evidence,
        wrist_name,
        wrist_valid,
        wrist_hit,
    ):
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
    def _extract_wrist_debug_payload_from_hold_state(hold_state, wrist_side):
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
        left_wrist_valid,
        right_wrist_valid,
        left_holding,
        right_holding,
        left_debug_payload=None,
        right_debug_payload=None,
    ):
        if left_debug_payload is None:
            left_debug_payload = {}
        if right_debug_payload is None:
            right_debug_payload = {}

        hold_info = {
            "holding_raw": left_holding or right_holding,
            "hold_raw_status": "NONE",
            "holding_correct_raw": False,
            "holding_wrong_raw": False,
            "left_wrist_valid": left_wrist_valid,
            "right_wrist_valid": right_wrist_valid,
            "left_wrist_hit": bool(left_debug_payload.get("hit", left_holding)),
            "right_wrist_hit": bool(right_debug_payload.get("hit", right_holding)),
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
            "left_holding": left_holding,
            "right_holding": right_holding,
            "left_hold_raw": left_holding,
            "right_hold_raw": right_holding,
            "left_hold_raw_before_claim": left_holding,
            "right_hold_raw_before_claim": right_holding,
            "left_hold_raw_after_claim": left_holding,
            "right_hold_raw_after_claim": right_holding,
            "handrail_status": "KHONG_VIN",
            "handrail_reason": "NO_HAND_HOLDING",
            "handrail_debug_reason": "NO_HAND_HOLDING",
        }

        if left_holding:
            hold_info["hold_raw_status"] = "WRONG_SIDE"
            hold_info["holding_wrong_raw"] = True
            hold_info["handrail_status"] = "VIN_SAI_BEN"
            hold_info["handrail_reason"] = "LEFT_HAND_HOLDING_IS_WRONG"
            hold_info["handrail_debug_reason"] = "LEFT_HAND_HOLDING_IS_WRONG"
        elif right_holding:
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

    def _evaluate_hold_state(self, *args):
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
        left_holding_raw = bool(handrail_evidence.get("holding_left_hand", False))
        right_holding_raw = bool(handrail_evidence.get("holding_right_hand", False))
        return self._build_left_right_hold_state(
            left_wrist_valid,
            right_wrist_valid,
            left_holding_raw,
            right_holding_raw,
            left_debug_payload=self._build_wrist_debug_payload(
                handrail_evidence,
                "LEFT_WRIST",
                left_wrist_valid,
                left_holding_raw,
            ),
            right_debug_payload=self._build_wrist_debug_payload(
                handrail_evidence,
                "RIGHT_WRIST",
                right_wrist_valid,
                right_holding_raw,
            ),
        )

    # Claim CARRY loai bo tay do khoi hold de tranh 1 tay vua "vin" vua "mang vac".
    # Hold la logic doc lap, carry chi duoc anh huong qua lop claim nay.
    def _apply_hand_claim_to_hold_state(self, hold_state, hand_claim_state):
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

        left_holding = bool(hold_state.get("left_holding", False))
        right_holding = bool(hold_state.get("right_holding", False))
        if hand_claim_state.get("left", {}).get("claim") == "CARRY":
            left_holding = False
        if hand_claim_state.get("right", {}).get("claim") == "CARRY":
            right_holding = False

        claimed_hold_state = self._build_left_right_hold_state(
            bool(hold_state.get("left_wrist_valid", False)),
            bool(hold_state.get("right_wrist_valid", False)),
            left_holding,
            right_holding,
            left_debug_payload=self._extract_wrist_debug_payload_from_hold_state(
                hold_state,
                "left",
            ),
            right_debug_payload=self._extract_wrist_debug_payload_from_hold_state(
                hold_state,
                "right",
            ),
        )
        claimed_hold_state["left_hold_raw_before_claim"] = hold_state.get(
            "left_hold_raw_before_claim",
            hold_state.get("left_hold_raw", False),
        )
        claimed_hold_state["right_hold_raw_before_claim"] = hold_state.get(
            "right_hold_raw_before_claim",
            hold_state.get("right_hold_raw", False),
        )
        claimed_hold_state["left_hold_raw_after_claim"] = claimed_hold_state.get(
            "left_hold_raw",
            False,
        )
        claimed_hold_state["right_hold_raw_after_claim"] = claimed_hold_state.get(
            "right_hold_raw",
            False,
        )
        return claimed_hold_state

    @staticmethod
    def _new_hand_claim_entry():
        return {
            "claim": None,
            "hold_hits": 0,
            "carry_hits": 0,
            "misses": 0,
        }

    def _get_hand_claim_state(self, track_id):
        if track_id not in self.hand_claim_state:
            self.hand_claim_state[track_id] = {
                "left": self._new_hand_claim_entry(),
                "right": self._new_hand_claim_entry(),
            }
        return self.hand_claim_state[track_id]

    def _update_hand_claim_state(
        self,
        track_id,
        left_hold_raw,
        right_hold_raw,
        left_carry_raw,
        right_carry_raw,
        left_handrail_active=False,
        right_handrail_active=False,
    ):
        claim_state = self._get_hand_claim_state(track_id)
        hand_raw_inputs = {
            "left": (left_hold_raw, left_carry_raw, left_handrail_active),
            "right": (right_hold_raw, right_carry_raw, right_handrail_active),
        }

        for hand_name, (hold_raw, carry_raw, handrail_active) in hand_raw_inputs.items():
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

            if handrail_active:
                hand_state["carry_hits"] = 0
                if hand_state["claim"] == "CARRY":
                    hand_state["claim"] = None

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

    def _resolve_hold_and_claim_state(
        self,
        track_id,
        hold_direction,
        handrail_evidence,
        keypoints,
        holding_raw,
        features,
    ):
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
    def _copy_hold_state_fields(hold_state):
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
