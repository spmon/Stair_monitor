from stair_monitor.settings import (
    DEMO_MODE,
    DRAW_SAFE_STATUS,
    UNKNOWN_COLOR,
    VIOLATION_COLOR,
)

REAL_VIOLATION_LABELS = (
    "Sai Lan",
    "Khong Vin",
    "Vin Sai Ben",
    "Mang Vac",
    "Di Lui",
    "Dung Yen",
)
RESULT_CONTEXT_FIELDS = (
    ("dy", "dy"),
    ("lane_v", "v"),
    ("direction", "direction"),
    ("lane_raw", "wrong_lane_raw"),
    ("lane_hits", "lane_wrong_hits"),
    ("lane_conf", "wrong_lane"),
    ("wrong_lane", "wrong_lane"),
    ("inside_stairs", "inside_stairs"),
    ("holding", "holding"),
    ("holding_raw", "holding_raw"),
    ("hold_status", "hold_final_status"),
    ("hold_raw_status", "hold_raw_status"),
    ("hold_confirmed_status", "hold_final_status"),
    ("hold_correct_hits", "hold_correct_hits"),
    ("hold_wrong_side_hits", "hold_wrong_side_hits"),
    ("hold_none_hits", "hold_none_hits"),
    ("hold_unknown_hits", "hold_unknown_hits"),
    ("holding_correct_raw", "holding_correct_raw"),
    ("holding_wrong_raw", "holding_wrong_raw"),
    ("hold_status_correct", "hold_status_correct"),
    ("hold_status_wrong", "hold_status_wrong"),
    ("dist_wrist", "dist_wrist"),
    ("wrist_side", "wrist_side"),
    ("best_wrist", "best_wrist"),
    ("best_wrist_point", "best_wrist_point"),
    ("dist_correct", "dist_correct"),
    ("dist_wrong", "dist_wrong"),
    ("seg_dist_correct", "seg_dist_correct"),
    ("seg_dist_wrong", "seg_dist_wrong"),
    ("t_correct", "t_correct"),
    ("t_wrong", "t_wrong"),
    ("wrist_side_correct", "wrist_side_correct"),
    ("wrist_side_wrong", "wrist_side_wrong"),
    ("best_wrist_correct", "best_wrist_correct"),
    ("best_wrist_wrong", "best_wrist_wrong"),
    ("best_wrist_correct_point", "best_wrist_correct_point"),
    ("best_wrist_wrong_point", "best_wrist_wrong_point"),
    ("correct_line_name", "correct_line_name"),
    ("wrong_line_name", "wrong_line_name"),
    ("correct_rule", "correct_rule"),
    ("wrong_rule", "wrong_rule"),
    ("p_lane", "p_lane"),
    ("p_motion", "p_motion"),
    ("body_facing", "body_facing"),
    ("arm_side_order", "arm_side_order"),
    ("backward_raw", "backward_raw"),
    ("backward_hits", "backward_hits"),
    ("backward_confirmed", "backward_confirmed"),
    ("standing_raw", "standing_raw"),
    ("standing_hits", "standing_hits"),
    ("standing_still_confirmed", "standing_still_confirmed"),
    ("standing_motion_range", "standing_motion_range"),
    ("standing_len", "standing_len"),
)
CARRY_RESULT_FIELDS = (
    "is_carrying",
    "carry_type",
    "left_arm_angle",
    "right_arm_angle",
    "left_wrist_in_torso",
    "right_wrist_in_torso",
    "wrist_dx",
    "wrist_dy",
    "wrist_distance",
    "left_bent",
    "right_bent",
    "wrists_close",
    "any_wrist_in_torso",
    "both_wrist_in_torso",
    "front_carry",
    "front_carry_raw",
    "front_carry_hits",
    "front_carry_two_hand_raw",
    "front_carry_two_hand_hits",
    "front_carry_one_arm_raw",
    "front_carry_one_arm_hits",
    "front_carry_confirmed",
    "left_carry",
    "right_carry",
    "carrying_arm",
)


class ResultBuilderMixin:
    @staticmethod
    def _build_display_status(direction, warnings):
        if DEMO_MODE:
            return " - ".join(warnings)
        if warnings:
            return f"{direction} | {' - '.join(warnings)}"
        if DRAW_SAFE_STATUS:
            return f"{direction} | An Toan"
        return ""

    @staticmethod
    def _collect_warnings(
        wrong_lane,
        hold_final_status,
        backward_confirmed,
        standing_still_confirmed,
        is_carrying=False,
    ):
        warnings = []
        if wrong_lane:
            warnings.append("Sai Lan")
        if hold_final_status == "NONE":
            warnings.append("Khong Vin")
        elif hold_final_status == "WRONG_SIDE":
            warnings.append("Vin Sai Ben")
        elif hold_final_status == "UNKNOWN" and not DEMO_MODE:
            warnings.append("Khong Xac Dinh")
        if is_carrying:
            warnings.append("Mang Vac")
        if backward_confirmed:
            warnings.append("Di Lui")
        if standing_still_confirmed:
            warnings.append("Dung Yen")
        return warnings

    def _summarize_result(
        self,
        direction,
        warnings,
        hold_final_status,
        safe_color,
        safe_status,
    ):
        real_warnings = [
            warning for warning in warnings if warning in REAL_VIOLATION_LABELS
        ]
        has_real_violation = len(real_warnings) > 0
        has_unknown = hold_final_status == "UNKNOWN" and not has_real_violation

        if has_real_violation:
            color = VIOLATION_COLOR
        elif has_unknown:
            color = UNKNOWN_COLOR
        else:
            color = safe_color

        status_warnings = list(warnings)
        if hold_final_status == "UNKNOWN" and "Khong Xac Dinh" not in status_warnings:
            status_warnings.append("Khong Xac Dinh")

        status = (
            f"{direction} | {' - '.join(status_warnings)}"
            if status_warnings
            else safe_status
        )
        display_status = self._build_display_status(direction, real_warnings)
        return status, display_status, color

    def _build_result_from_context(
        self,
        context,
        status,
        color,
        display_status="",
        carry_info=None,
        **overrides,
    ):
        result_kwargs = {
            result_key: context[context_key]
            for result_key, context_key in RESULT_CONTEXT_FIELDS
            if context_key in context
        }
        if carry_info is not None:
            result_kwargs.update(
                {
                    result_key: carry_info[result_key]
                    for result_key in CARRY_RESULT_FIELDS
                    if result_key in carry_info
                }
            )
        result_kwargs.update(overrides)
        return self._build_result(
            status=status,
            display_status=display_status,
            color=color,
            **result_kwargs,
        )

    @staticmethod
    def _build_result(
        status,
        color,
        display_status="",
        dy=None,
        lane_v=None,
        direction="NA",
        lane_raw=False,
        lane_hits=0,
        lane_conf=False,
        wrong_lane=False,
        inside_stairs=False,
        holding=False,
        holding_raw=False,
        hold_status="ANALYZING",
        hold_raw_status="UNKNOWN",
        hold_confirmed_status="ANALYZING",
        hold_correct_hits=0,
        hold_wrong_side_hits=0,
        hold_none_hits=0,
        hold_unknown_hits=0,
        holding_correct_raw=False,
        holding_wrong_raw=False,
        hold_status_correct="UNKNOWN",
        hold_status_wrong="UNKNOWN",
        dist_wrist=-999,
        wrist_side="UNKNOWN",
        best_wrist="NONE",
        best_wrist_point=None,
        dist_correct=-999,
        dist_wrong=-999,
        seg_dist_correct=None,
        seg_dist_wrong=None,
        t_correct=None,
        t_wrong=None,
        wrist_side_correct="UNKNOWN",
        wrist_side_wrong="UNKNOWN",
        best_wrist_correct="NONE",
        best_wrist_wrong="NONE",
        best_wrist_correct_point=None,
        best_wrist_wrong_point=None,
        correct_line_name="NONE",
        wrong_line_name="NONE",
        correct_rule="NA",
        wrong_rule="NA",
        is_carrying=False,
        carry_type="NONE",
        left_arm_angle=None,
        right_arm_angle=None,
        left_wrist_in_torso=False,
        right_wrist_in_torso=False,
        wrist_dx=None,
        wrist_dy=None,
        wrist_distance=None,
        left_bent=False,
        right_bent=False,
        wrists_close=False,
        any_wrist_in_torso=False,
        both_wrist_in_torso=False,
        front_carry=False,
        front_carry_raw=False,
        front_carry_hits=0,
        front_carry_two_hand_raw=False,
        front_carry_two_hand_hits=0,
        front_carry_one_arm_raw=False,
        front_carry_one_arm_hits=0,
        front_carry_confirmed=False,
        left_carry=False,
        right_carry=False,
        carrying_arm="NONE",
        p_lane=None,
        p_motion=None,
        body_facing="UNKNOWN",
        arm_side_order="UNKNOWN",
        backward_raw=False,
        backward_hits=0,
        backward_confirmed=False,
        standing_raw=False,
        standing_hits=0,
        standing_still_confirmed=False,
        standing_motion_range=None,
        standing_len=0,
    ):
        return {
            "status": status,
            "display_status": display_status,
            "color": color,
            "dy": dy,
            "lane_v": lane_v,
            "direction": direction,
            "lane_raw": lane_raw,
            "lane_hits": lane_hits,
            "lane_conf": lane_conf,
            "wrong_lane": wrong_lane,
            "inside_stairs": inside_stairs,
            "holding": holding,
            "holding_raw": holding_raw,
            "hold_status": hold_status,
            "hold_raw_status": hold_raw_status,
            "hold_confirmed_status": hold_confirmed_status,
            "hold_correct_hits": hold_correct_hits,
            "hold_wrong_side_hits": hold_wrong_side_hits,
            "hold_none_hits": hold_none_hits,
            "hold_unknown_hits": hold_unknown_hits,
            "holding_correct_raw": holding_correct_raw,
            "holding_wrong_raw": holding_wrong_raw,
            "hold_status_correct": hold_status_correct,
            "hold_status_wrong": hold_status_wrong,
            "dist_wrist": dist_wrist,
            "wrist_side": wrist_side,
            "best_wrist": best_wrist,
            "best_wrist_point": best_wrist_point,
            "dist_correct": dist_correct,
            "dist_wrong": dist_wrong,
            "seg_dist_correct": seg_dist_correct,
            "seg_dist_wrong": seg_dist_wrong,
            "t_correct": t_correct,
            "t_wrong": t_wrong,
            "wrist_side_correct": wrist_side_correct,
            "wrist_side_wrong": wrist_side_wrong,
            "best_wrist_correct": best_wrist_correct,
            "best_wrist_wrong": best_wrist_wrong,
            "best_wrist_correct_point": best_wrist_correct_point,
            "best_wrist_wrong_point": best_wrist_wrong_point,
            "correct_line_name": correct_line_name,
            "wrong_line_name": wrong_line_name,
            "correct_rule": correct_rule,
            "wrong_rule": wrong_rule,
            "is_carrying": is_carrying,
            "carry_type": carry_type,
            "left_arm_angle": left_arm_angle,
            "right_arm_angle": right_arm_angle,
            "left_wrist_in_torso": left_wrist_in_torso,
            "right_wrist_in_torso": right_wrist_in_torso,
            "wrist_dx": wrist_dx,
            "wrist_dy": wrist_dy,
            "wrist_distance": wrist_distance,
            "left_bent": left_bent,
            "right_bent": right_bent,
            "wrists_close": wrists_close,
            "any_wrist_in_torso": any_wrist_in_torso,
            "both_wrist_in_torso": both_wrist_in_torso,
            "front_carry": front_carry,
            "front_carry_raw": front_carry_raw,
            "front_carry_hits": front_carry_hits,
            "front_carry_two_hand_raw": front_carry_two_hand_raw,
            "front_carry_two_hand_hits": front_carry_two_hand_hits,
            "front_carry_one_arm_raw": front_carry_one_arm_raw,
            "front_carry_one_arm_hits": front_carry_one_arm_hits,
            "front_carry_confirmed": front_carry_confirmed,
            "left_carry": left_carry,
            "right_carry": right_carry,
            "carrying_arm": carrying_arm,
            "p_lane": p_lane,
            "p_motion": p_motion,
            "body_facing": body_facing,
            "arm_side_order": arm_side_order,
            "backward_raw": backward_raw,
            "backward_hits": backward_hits,
            "backward_confirmed": backward_confirmed,
            "standing_raw": standing_raw,
            "standing_hits": standing_hits,
            "standing_still_confirmed": standing_still_confirmed,
            "standing_motion_range": standing_motion_range,
            "standing_len": standing_len,
        }
