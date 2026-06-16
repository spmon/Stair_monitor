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
    segment_dist, projection_t, _ = point_to_segment_distance(
        wrist_point,
        line[0],
        line[1],
    )
    return {
        "point": wrist_point,
        "dist": dist,
        "segment_dist": segment_dist,
        "projection_t": projection_t,
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
        LEFT_HANDRAIL_RULE / RIGHT_HANDRAIL_RULE chi mo ta quy tac vat ly cua
        tung ben line. Viec ket luan dung ben hay sai ben theo direction se duoc
        analyzer xu ly o buoc hold.
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
            return bool(10 <= pair["dist"] <= 40)
        return "UNKNOWN"

    # Ket qua o day chi la bang chung frame-level cho tung co tay/tung rail.
    # Viec ket luan dung ben/sai ben con phai doi direction xu ly o analyzer.
    return {
        "left_line": left_line,
        "right_line": right_line,
        "pairs": pair_results,
        "best_left": pair_results.get(("LEFT_WRIST", "LEFT_HANDRAIL")),
        "best_right": pair_results.get(("RIGHT_WRIST", "RIGHT_HANDRAIL")),
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
        "NONE",
        features,
    )


# Chon co tay hop le gan lan can nhat theo rule cua chinh ben lan can do.
# Rule o day la thuoc tinh vat ly cua line trai/phai, khong phai huong di chuyen UP/DOWN.
def get_best_wrist_for_handrail_by_rule(keypoints, line, rule, segment_max_distance):
    """Chon wrist hop le gan line nhat theo rule cua chinh line do.

    Args:
        keypoints: Mang keypoint YOLO pose.
        line: Hai diem dau mut cua line lan can.
        rule: LEFT_HANDRAIL_RULE hoac RIGHT_HANDRAIL_RULE.
        segment_max_distance: Gioi han khoang cach toi chinh doan line.

    Returns:
        tuple:
            - holding: True/False
            - dist: Signed distance toi line
            - side: LEFT_SIDE / RIGHT_SIDE / ON_LINE / UNKNOWN
            - wrist_name: LEFT_WRIST / RIGHT_WRIST / NONE
            - wrist_point: Wrist duoc chon hoac None
            - hold_status: TRUE / FALSE / UNKNOWN
            - segment_dist: Khoang cach toi doan line
            - projection_t: He so hinh chieu truoc khi clamp

    Notes:
        valid_d va valid_segment la 2 lop loc khac nhau:
        - valid_d: wrist nam dung phia cua line va trong cua so dist.
        - valid_segment: wrist co hinh chieu roi vao doan line va du gan doan.
        Rule o day la quy tac vat ly cua line trai/phai, khong phai huong di.
    """
    if keypoints is None or len(keypoints) < 11 or len(line) < 2:
        return False, -999, "UNKNOWN", "NONE", None, "UNKNOWN", None, None

    candidates = []

    if len(keypoints[9]) > 2 and keypoints[9][2] > 0.5:
        wrist_point = (int(keypoints[9][0]), int(keypoints[9][1]))
        d = signed_distance_to_line(wrist_point, line)
        segment_dist, projection_t, _ = point_to_segment_distance(
            wrist_point, line[0], line[1]
        )
        valid_d = False
        if rule == LEFT_HANDRAIL_RULE:
            valid_d = -SETTINGS.handrail.left_max_distance <= d <= -10
        elif rule == RIGHT_HANDRAIL_RULE:
            valid_d = 0 <= d <= SETTINGS.handrail.right_max_distance
        # projection_t trong [0, 1] nghia la hinh chieu roi vao trong chinh doan line.
        valid_segment = 0.0 <= projection_t <= 1.0 and (
            segment_dist <= segment_max_distance
        )
        candidates.append(
            {
                "name": "LEFT_WRIST",
                "point": wrist_point,
                "dist": d,
                "segment_dist": segment_dist,
                "projection_t": projection_t,
                "side": get_side_name(d),
                "valid_d": valid_d,
                "valid_segment": valid_segment,
                "valid": valid_d and valid_segment,
            }
        )

    if len(keypoints[10]) > 2 and keypoints[10][2] > 0.25:
        wrist_point = (int(keypoints[10][0]), int(keypoints[10][1]))
        d = signed_distance_to_line(wrist_point, line)
        segment_dist, projection_t, _ = point_to_segment_distance(
            wrist_point, line[0], line[1]
        )
        valid_d = False
        if rule == LEFT_HANDRAIL_RULE:
            valid_d = -SETTINGS.handrail.left_max_distance <= d <= -10
        elif rule == RIGHT_HANDRAIL_RULE:
            valid_d = 0 <= d <= SETTINGS.handrail.right_max_distance
        valid_segment = 0.0 <= projection_t <= 1.0 and (
            segment_dist <= segment_max_distance
        )
        candidates.append(
            {
                "name": "RIGHT_WRIST",
                "point": wrist_point,
                "dist": d,
                "segment_dist": segment_dist,
                "projection_t": projection_t,
                "side": get_side_name(d),
                "valid_d": valid_d,
                "valid_segment": valid_segment,
                "valid": valid_d and valid_segment,
            }
        )

    if not candidates:
        return False, -999, "UNKNOWN", "NONE", None, "UNKNOWN", None, None

    # Uu tien ung vien vua dung phia cua line vua co hinh chieu roi vao chinh doan lan can.
    valid_candidates = [candidate for candidate in candidates if candidate["valid"]]
    if valid_candidates:
        best = min(
            valid_candidates,
            key=lambda candidate: (candidate["segment_dist"], abs(candidate["dist"])),
        )
        return (
            True,
            best["dist"],
            best["side"],
            best["name"],
            best["point"],
            "TRUE",
            best["segment_dist"],
            best["projection_t"],
        )

    best = min(
        candidates,
        key=lambda candidate: (candidate["segment_dist"], abs(candidate["dist"])),
    )
    return (
        False,
        best["dist"],
        best["side"],
        best["name"],
        best["point"],
        "FALSE",
        best["segment_dist"],
        best["projection_t"],
    )


def get_best_wrist_for_handrail_by_rule_from_evidence(
    handrail_evidence,
    line_name,
    rule,
    segment_max_distance,
):
    """Dung lai handrail evidence da tinh san de chon wrist tot nhat.

    Args:
        handrail_evidence: Dict bang chung da tinh boi compute_handrail_evidence.
        line_name: LEFT_HANDRAIL hoac RIGHT_HANDRAIL.
        rule: LEFT_HANDRAIL_RULE hoac RIGHT_HANDRAIL_RULE.
        segment_max_distance: Gioi han khoang cach toi doan line.

    Returns:
        tuple: Cung format voi get_best_wrist_for_handrail_by_rule.

    Notes:
        Ham nay tranh tinh lai signed distance va segment distance nhieu lan
        trong cung 1 frame.
    """
    # Ban dung lai evidence da tinh san de tranh tinh signed distance/segment distance lap lai.
    if handrail_evidence is None:
        return False, -999, "UNKNOWN", "NONE", None, "UNKNOWN", None, None

    pair_results = handrail_evidence.get("pairs", {})
    candidates = []
    for wrist_name in ("LEFT_WRIST", "RIGHT_WRIST"):
        pair = pair_results.get((wrist_name, line_name))
        if pair is None:
            continue

        dist = pair["dist"]
        projection_t = pair["projection_t"]
        segment_dist = pair["segment_dist"]
        valid_d = False
        if rule == LEFT_HANDRAIL_RULE:
            valid_d = -SETTINGS.handrail.left_max_distance <= dist <= -10
        elif rule == RIGHT_HANDRAIL_RULE:
            valid_d = 0 <= dist <= SETTINGS.handrail.right_max_distance
        valid_segment = 0.0 <= projection_t <= 1.0 and (
            segment_dist <= segment_max_distance
        )
        candidates.append(
            {
                "name": wrist_name,
                "point": pair["point"],
                "dist": dist,
                "segment_dist": segment_dist,
                "projection_t": projection_t,
                "side": pair["side"],
                "valid_d": valid_d,
                "valid_segment": valid_segment,
                "valid": valid_d and valid_segment,
            }
        )

    if not candidates:
        return False, -999, "UNKNOWN", "NONE", None, "UNKNOWN", None, None

    valid_candidates = [candidate for candidate in candidates if candidate["valid"]]
    if valid_candidates:
        best = min(
            valid_candidates,
            key=lambda candidate: (candidate["segment_dist"], abs(candidate["dist"])),
        )
        return (
            True,
            best["dist"],
            best["side"],
            best["name"],
            best["point"],
            "TRUE",
            best["segment_dist"],
            best["projection_t"],
        )

    best = min(
        candidates,
        key=lambda candidate: (candidate["segment_dist"], abs(candidate["dist"])),
    )
    return (
        False,
        best["dist"],
        best["side"],
        best["name"],
        best["point"],
        "FALSE",
        best["segment_dist"],
        best["projection_t"],
    )


class HandrailAnalysisMixin:
    """Mixin gom logic hold raw, directional handrail va hand-claim."""

    @staticmethod
        # Mapping lan can dung/sai ben duoc chon theo profile camera.
        # - Current camera: giu nguyen logic cu
        # - Bottom stair camera: DOWN dung LEFT, UP dung RIGHT
    def _get_handrail_targets(direction, left_line, right_line):
        if SETTINGS.camera.use_current_camera_angle:
            if direction == "UP":
                return (
                    left_line,
                    "LEFT_HANDRAIL",
                    LEFT_HANDRAIL_RULE,
                    right_line,
                    "RIGHT_HANDRAIL",
                    RIGHT_HANDRAIL_RULE,
                )
            if direction == "DOWN":
                return (
                    right_line,
                    "RIGHT_HANDRAIL",
                    RIGHT_HANDRAIL_RULE,
                    left_line,
                    "LEFT_HANDRAIL",
                    LEFT_HANDRAIL_RULE,
                )
            return None, "NONE", "NA", None, "NONE", "NA"

        if direction == "DOWN":
            return (
                left_line,
                "LEFT_HANDRAIL",
                LEFT_HANDRAIL_RULE,
                right_line,
                "RIGHT_HANDRAIL",
                RIGHT_HANDRAIL_RULE,
            )
        if direction == "UP":
            return (
                right_line,
                "RIGHT_HANDRAIL",
                RIGHT_HANDRAIL_RULE,
                left_line,
                "LEFT_HANDRAIL",
                LEFT_HANDRAIL_RULE,
            )
        return None, "NONE", "NA", None, "NONE", "NA"

    @staticmethod
    # Quy doi ten wrist sang key trai/phai de xu ly claim tay.
    def _get_hand_key_from_wrist_name(wrist_name):
        """Quy doi ten wrist debug sang key trai/phai.

        Args:
            wrist_name: LEFT_WRIST, RIGHT_WRIST hoac NONE.

        Returns:
            str | None: left, right hoac None.

        Notes:
            hand_claim_state luu theo left/right, con evidence debug luu theo
            LEFT_WRIST/RIGHT_WRIST.
        """
        if wrist_name == "LEFT_WRIST":
            return "left"
        if wrist_name == "RIGHT_WRIST":
            return "right"
        return None

    # Tao ung vien hold cho tung line muc tieu.
    # Ket qua giu lai ca raw status, signed distance va segment distance de debug ro "vi sao".
    def _build_hold_candidate(
        self,
        handrail_evidence,
        line,
        line_name,
        rule,
        role,
        segment_max_distance,
    ):
        """Tao 1 ung vien hold cho 1 line muc tieu.

        Args:
            handrail_evidence: Dict bang chung wrist-line da tinh san.
            line: Line lan can dang xet.
            line_name: LEFT_HANDRAIL hoac RIGHT_HANDRAIL.
            rule: Rule vat ly cua line do.
            role: CORRECT, WRONG hoac ANY.
            segment_max_distance: Gioi han khoang cach toi doan line.

        Returns:
            dict | None: Ung vien hold cho line nay.

        Notes:
            holding_raw o day van chi la bang chung frame-level. Hold final status
            se duoc history trong analyzer/behavior_history xac nhan sau.
        """
        if line is None or len(line) < 2:
            return None

        (
            holding_raw,
            dist,
            wrist_side,
            best_wrist,
            best_wrist_point,
            hold_status,
            seg_dist,
            projection_t,
        ) = get_best_wrist_for_handrail_by_rule_from_evidence(
            handrail_evidence,
            line_name,
            rule,
            segment_max_distance,
        )
        return {
            "role": role,
            "line_name": line_name,
            "rule": rule,
            "holding_raw": holding_raw,
            "dist": dist,
            "wrist_side": wrist_side,
            "best_wrist": best_wrist,
            "best_wrist_point": best_wrist_point,
            "hold_status": hold_status,
            "seg_dist": seg_dist,
            "t": projection_t,
            "hand_key": self._get_hand_key_from_wrist_name(best_wrist),
        }

    @staticmethod
    # Gia tri rong de giu shape ket qua on dinh khi line hoac keypoint khong du.
    def _empty_hold_candidate(role):
        """Tra ve 1 candidate rong de giu shape ket qua on dinh.

        Args:
            role: Vai tro candidate trong hold state.

        Returns:
            dict: Candidate rong.

        Notes:
            Shape on dinh giup result_builder/debug overlay de truy cap field.
        """
        return {
            "role": role,
            "line_name": "NONE",
            "rule": "NA",
            "holding_raw": False,
            "dist": -999,
            "wrist_side": "UNKNOWN",
            "best_wrist": "NONE",
            "best_wrist_point": None,
            "hold_status": "UNKNOWN",
            "seg_dist": None,
            "t": None,
            "hand_key": None,
        }

    # Tong hop cac ung vien hold thanh 1 ket luan frame-level.
    # UNKNOWN = du lieu chua du chac; NONE = co du lieu nhung khong thay bang chung vin.
    # WRONG_SIDE = co vin nhung vin nham ben; CORRECT = co bang chung vin dung ben.
    def _summarize_hold_candidates(self, candidates, directional_hold):
        """Tong hop nhieu candidate thanh 1 ket luan hold raw cua frame.

        Args:
            candidates: Danh sach candidate da tinh cho cac line lien quan.
            directional_hold: True neu da biet line nao dung ben/sai ben.

        Returns:
            dict: Hold info frame-level va du lieu debug chi tiet.

        Notes:
            correct/wrong handrail chi co y nghia day du khi da biet direction.
            Truoc do chi nen ket luan co vin bat ky lan can nao hay khong.
        """
        hold_info = {
            "holding_raw": False,
            "hold_raw_status": "UNKNOWN",
            "holding_correct_raw": False,
            "holding_wrong_raw": False,
            "hold_status_correct": "UNKNOWN",
            "hold_status_wrong": "UNKNOWN",
            "dist_correct": -999,
            "dist_wrong": -999,
            "seg_dist_correct": None,
            "seg_dist_wrong": None,
            "t_correct": None,
            "t_wrong": None,
            "wrist_side_correct": "UNKNOWN",
            "wrist_side_wrong": "UNKNOWN",
            "best_wrist_correct": "NONE",
            "best_wrist_wrong": "NONE",
            "best_wrist_correct_point": None,
            "best_wrist_wrong_point": None,
            "correct_line_name": "NONE",
            "wrong_line_name": "NONE",
            "correct_rule": "NA",
            "wrong_rule": "NA",
            "best_wrist": "NONE",
            "best_wrist_point": None,
            "dist_wrist": -999,
            "wrist_side": "UNKNOWN",
            "left_hold_raw": False,
            "right_hold_raw": False,
            "left_hold_raw_before_claim": False,
            "right_hold_raw_before_claim": False,
            "left_hold_raw_after_claim": False,
            "right_hold_raw_after_claim": False,
            "hold_candidates": [candidate.copy() for candidate in candidates],
            "directional_hold": directional_hold,
        }

        hold_info["left_hold_raw"] = any(
            candidate["holding_raw"] and candidate["hand_key"] == "left"
            for candidate in candidates
        )
        hold_info["right_hold_raw"] = any(
            candidate["holding_raw"] and candidate["hand_key"] == "right"
            for candidate in candidates
        )
        hold_info["left_hold_raw_before_claim"] = hold_info["left_hold_raw"]
        hold_info["right_hold_raw_before_claim"] = hold_info["right_hold_raw"]
        hold_info["left_hold_raw_after_claim"] = hold_info["left_hold_raw"]
        hold_info["right_hold_raw_after_claim"] = hold_info["right_hold_raw"]

        if directional_hold:
            # Khi da biet direction, tach ro lan can dung ben va sai ben theo huong di.
            correct_candidate = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.get("role") == "CORRECT"
                ),
                self._empty_hold_candidate("CORRECT"),
            )
            wrong_candidate = next(
                (
                    candidate for candidate in candidates if candidate.get("role") == "WRONG"
                ),
                self._empty_hold_candidate("WRONG"),
            )

            hold_info["holding_correct_raw"] = correct_candidate["holding_raw"]
            hold_info["holding_wrong_raw"] = wrong_candidate["holding_raw"]
            hold_info["hold_status_correct"] = correct_candidate["hold_status"]
            hold_info["hold_status_wrong"] = wrong_candidate["hold_status"]
            hold_info["dist_correct"] = correct_candidate["dist"]
            hold_info["dist_wrong"] = wrong_candidate["dist"]
            hold_info["seg_dist_correct"] = correct_candidate["seg_dist"]
            hold_info["seg_dist_wrong"] = wrong_candidate["seg_dist"]
            hold_info["t_correct"] = correct_candidate["t"]
            hold_info["t_wrong"] = wrong_candidate["t"]
            hold_info["wrist_side_correct"] = correct_candidate["wrist_side"]
            hold_info["wrist_side_wrong"] = wrong_candidate["wrist_side"]
            hold_info["best_wrist_correct"] = correct_candidate["best_wrist"]
            hold_info["best_wrist_wrong"] = wrong_candidate["best_wrist"]
            hold_info["best_wrist_correct_point"] = correct_candidate["best_wrist_point"]
            hold_info["best_wrist_wrong_point"] = wrong_candidate["best_wrist_point"]
            hold_info["correct_line_name"] = correct_candidate["line_name"]
            hold_info["wrong_line_name"] = wrong_candidate["line_name"]
            hold_info["correct_rule"] = correct_candidate["rule"]
            hold_info["wrong_rule"] = wrong_candidate["rule"]

            if wrong_candidate["holding_raw"]:
                hold_info["hold_raw_status"] = "WRONG_SIDE"
            elif correct_candidate["holding_raw"]:
                hold_info["hold_raw_status"] = "CORRECT"
                hold_info["holding_raw"] = True
            elif (
                correct_candidate["hold_status"] == "UNKNOWN"
                and wrong_candidate["hold_status"] == "UNKNOWN"
            ):
                hold_info["hold_raw_status"] = "UNKNOWN"
            else:
                hold_info["hold_raw_status"] = "NONE"

            (
                hold_info["best_wrist"],
                hold_info["best_wrist_point"],
                hold_info["dist_wrist"],
                hold_info["wrist_side"],
            ) = self._select_primary_hold_debug(
                hold_info["hold_raw_status"],
                hold_info["best_wrist_correct"],
                hold_info["best_wrist_correct_point"],
                hold_info["dist_correct"],
                hold_info["wrist_side_correct"],
                hold_info["best_wrist_wrong"],
                hold_info["best_wrist_wrong_point"],
                hold_info["dist_wrong"],
                hold_info["wrist_side_wrong"],
            )
            return hold_info

        debug_candidates = [
            candidate for candidate in candidates if candidate["holding_raw"]
        ] or [candidate for candidate in candidates if candidate["dist"] != -999]
        if debug_candidates:
            primary = min(
                debug_candidates,
                key=lambda candidate: (
                    candidate["seg_dist"]
                    if candidate["seg_dist"] is not None
                    else float("inf"),
                    abs(candidate["dist"])
                    if candidate["dist"] != -999
                    else float("inf"),
                ),
            )
            hold_info["dist_correct"] = primary["dist"]
            hold_info["seg_dist_correct"] = primary["seg_dist"]
            hold_info["t_correct"] = primary["t"]
            hold_info["wrist_side_correct"] = primary["wrist_side"]
            hold_info["best_wrist_correct"] = primary["best_wrist"]
            hold_info["best_wrist_correct_point"] = primary["best_wrist_point"]
            hold_info["hold_status_correct"] = primary["hold_status"]
            hold_info["correct_line_name"] = primary["line_name"]
            hold_info["correct_rule"] = primary["rule"]
            hold_info["best_wrist"] = primary["best_wrist"]
            hold_info["best_wrist_point"] = primary["best_wrist_point"]
            hold_info["dist_wrist"] = primary["dist"]
            hold_info["wrist_side"] = primary["wrist_side"]

        if any(candidate["holding_raw"] for candidate in candidates):
            # CORRECT o day chi co nghia la "co vin mot lan can nao do",
            # khong co nghia la da dung ben theo chieu di.
            hold_info["hold_raw_status"] = "CORRECT"
            hold_info["holding_raw"] = True
            hold_info["holding_correct_raw"] = True
        elif not candidates or all(
            candidate["hold_status"] == "UNKNOWN" for candidate in candidates
        ):
            hold_info["hold_raw_status"] = "UNKNOWN"
        else:
            hold_info["hold_raw_status"] = "NONE"

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
        # Ham nay danh gia hold theo 2 che do:
        # - Co direction: phan biet ro lan can dung/sai ben.
        # - Chua co direction: chi duoc ket luan co/khong co vin bat ky lan can nao.
        if len(args) == 2:
            hold_direction, handrail_evidence = args
            left_line = handrail_evidence.get("left_line") if handrail_evidence else None
            right_line = (
                handrail_evidence.get("right_line") if handrail_evidence else None
            )
        elif len(args) == 4:
            keypoints, hold_direction, left_line, right_line = args
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

        candidates = []
        if hold_direction in ("UP", "DOWN"):
            (
                correct_line,
                correct_line_name,
                correct_rule,
                wrong_line,
                wrong_line_name,
                wrong_rule,
            ) = self._get_handrail_targets(hold_direction, left_line, right_line)

            if correct_line is not None and len(correct_line) >= 2:
                candidates.append(
                    self._build_hold_candidate(
                        handrail_evidence,
                        correct_line,
                        correct_line_name,
                        correct_rule,
                        "CORRECT",
                        SETTINGS.handrail.segment_max_distance,
                    )
                )

            if wrong_line is not None and len(wrong_line) >= 2:
                candidates.append(
                    self._build_hold_candidate(
                        handrail_evidence,
                        wrong_line,
                        wrong_line_name,
                        wrong_rule,
                        "WRONG",
                        SETTINGS.handrail.wrong_side_segment_max_distance,
                    )
                )

            return self._summarize_hold_candidates(
                [candidate for candidate in candidates if candidate is not None],
                directional_hold=True,
            )

        # Khi chua biet chieu di, khong the ket luan lan can dung hay sai ben.
        # Tuy vay van phai kiem tra xem nguoi do co dang vin bat ky lan can nao khong.
        if left_line is not None and len(left_line) >= 2:
            candidates.append(
                self._build_hold_candidate(
                    handrail_evidence,
                    left_line,
                    "LEFT_HANDRAIL",
                    LEFT_HANDRAIL_RULE,
                    "ANY",
                    SETTINGS.handrail.segment_max_distance,
                )
            )

        if right_line is not None and len(right_line) >= 2:
            candidates.append(
                self._build_hold_candidate(
                    handrail_evidence,
                    right_line,
                    "RIGHT_HANDRAIL",
                    RIGHT_HANDRAIL_RULE,
                    "ANY",
                    SETTINGS.handrail.segment_max_distance,
                )
            )
        return self._summarize_hold_candidates(
            [candidate for candidate in candidates if candidate is not None],
            directional_hold=False,
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

        filtered_candidates = []
        for candidate in hold_state.get("hold_candidates", []):
            hand_key = candidate.get("hand_key")
            hand_claim = hand_claim_state.get(hand_key, {}).get("claim")
            # CARRY claim thi khong duoc dung tay nay de ket luan hold nua.
            if hand_claim == "CARRY":
                continue
            filtered_candidates.append(candidate.copy())

        claimed_hold_state = self._summarize_hold_candidates(
            filtered_candidates,
            hold_state.get("directional_hold", False),
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
    ):
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

    @staticmethod
    # Chon diem debug chinh de ve overlay va in thong tin khoang cach cho nguoi de doc.
    def _select_primary_hold_debug(
        hold_raw_status,
        best_wrist_correct,
        best_wrist_correct_point,
        dist_correct,
        wrist_side_correct,
        best_wrist_wrong,
        best_wrist_wrong_point,
        dist_wrong,
        wrist_side_wrong,
    ):
        """Chon wrist/chi so debug chinh de ve overlay.

        Args:
            hold_raw_status: Trang thai hold raw cua frame hien tai.
            best_wrist_correct: Wrist tot nhat cho line dung ben.
            best_wrist_correct_point: Toa do wrist tot nhat dung ben.
            dist_correct: Signed distance cho line dung ben.
            wrist_side_correct: Nhan side cua wrist voi line dung ben.
            best_wrist_wrong: Wrist tot nhat cho line sai ben.
            best_wrist_wrong_point: Toa do wrist tot nhat sai ben.
            dist_wrong: Signed distance cho line sai ben.
            wrist_side_wrong: Nhan side cua wrist voi line sai ben.

        Returns:
            tuple: (best_wrist, best_point, dist, side)

        Notes:
            Ham nay chi chon diem de debug/overlay, khong doi ket qua nhan dien.
        """
        if hold_raw_status == "CORRECT":
            return (
                best_wrist_correct,
                best_wrist_correct_point,
                dist_correct,
                wrist_side_correct,
            )
        if hold_raw_status == "WRONG_SIDE":
            return (
                best_wrist_wrong,
                best_wrist_wrong_point,
                dist_wrong,
                wrist_side_wrong,
            )

        correct_valid = dist_correct != -999
        wrong_valid = dist_wrong != -999

        if correct_valid and wrong_valid:
            if abs(dist_correct) <= abs(dist_wrong):
                return (
                    best_wrist_correct,
                    best_wrist_correct_point,
                    dist_correct,
                    wrist_side_correct,
                )
            return (
                best_wrist_wrong,
                best_wrist_wrong_point,
                dist_wrong,
                wrist_side_wrong,
            )
        if correct_valid:
            return (
                best_wrist_correct,
                best_wrist_correct_point,
                dist_correct,
                wrist_side_correct,
            )
        if wrong_valid:
            return (
                best_wrist_wrong,
                best_wrist_wrong_point,
                dist_wrong,
                wrist_side_wrong,
            )
        return "NONE", None, -999, "UNKNOWN"
