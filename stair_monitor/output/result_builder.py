from __future__ import annotations

from stair_monitor.common.types import (
    AnalysisResult,
    ColorBGR,
    DebugInfoDict,
    PerfStats,
    WarningList,
)
from stair_monitor.config.settings import SETTINGS

# File nay dong goi ket qua cuoi cung de overlay/log doc de dang.
# FLOW: context tam cua analyzer -> warnings/status/color/display_status -> result dict on dinh.
# WARNING: Day la noi quyet dinh nhan canh bao cuoi cung, nhung khong tu tinh rule nghiep vu moi.

REAL_VIOLATION_LABELS = list(SETTINGS.violation.count_labels)
LANE_SUPPRESSED_HOLD_WARNINGS = frozenset(
    {"Khong Vin", "Vin Sai Ben", "Khong Xac Dinh"}
)
DISPLAY_TEXT_REPLACEMENTS = {
    **SETTINGS.violation.display_names,
    "Khong Xac Dinh": "Không xác định",
    "Ngoai Vung": "Ngoài vùng",
    "An Toan": "An toàn",
}
# Bang mapping nay giu result dict on dinh va tranh phai truyen tay tung field o analyzer.
# RESULT_CONTEXT_FIELDS map ten field public trong result dict
# voi ten bien trong locals()/context cua analyzer.
RESULT_CONTEXT_FIELDS = (
    ("track_id", "track_id"),
    ("dy", "dy"),
    ("direction_dy", "direction_dy"),
    ("lane_v", "v"),
    ("direction", "direction"),
    ("final_direction", "final_direction"),
    ("hip_direction", "hip_direction"),
    ("shoulder_direction", "shoulder_direction"),
    ("direction_source", "direction_source"),
    ("direction_reason", "direction_reason"),
    ("monitor_point_hip", "monitor_point_hip"),
    ("monitor_point_hip_source", "monitor_point_hip_source"),
    ("monitor_point_shoulder", "monitor_point_shoulder"),
    ("monitor_point_shoulder_source", "monitor_point_shoulder_source"),
    ("use_current_camera_angle", "use_current_camera_angle"),
    ("camera_angle_profile", "camera_angle_profile"),
    ("lane_raw", "wrong_lane_raw"),
    ("lane_hits", "lane_wrong_hits"),
    ("lane_conf", "wrong_lane"),
    ("wrong_lane", "wrong_lane"),
    ("lane_status", "lane_status"),
    ("lane_reason", "lane_reason"),
    ("lane_source", "lane_source"),
    ("lane_direction", "lane_direction"),
    ("lane_side_value", "lane_side_value"),
    ("lane_side_label", "lane_side_label"),
    ("correct_lane_side", "correct_lane_side"),
    ("lane_mapping_source", "lane_mapping_source"),
    ("p_lane_source", "p_lane_source"),
    ("lane_missing_feet_grace_left", "lane_missing_feet_grace_left"),
    ("foot_lane_side", "foot_lane_side"),
    ("inside_stairs", "inside_stairs"),
    ("inside_feet_point", "inside_feet_point"),
    ("feet_point_source", "feet_point_source"),
    ("inside_feet_point_source", "inside_feet_point_source"),
    ("feet_available", "feet_available"),
    ("feet_unavailable_reason", "feet_unavailable_reason"),
    ("sh_hip_visible_shoulder_count", "sh_hip_visible_shoulder_count"),
    ("sh_hip_visible_hip_count", "sh_hip_visible_hip_count"),
    ("sh_hip_selected_pair", "sh_hip_selected_pair"),
    ("sh_hip_virtual_feet_point", "sh_hip_virtual_feet_point"),
    ("sh_hip_virtual_feet_source", "sh_hip_virtual_feet_source"),
    ("ankle_valid_count", "ankle_valid_count"),
    ("feet_reliable", "feet_reliable"),
    ("bbox_height", "bbox_height"),
    ("left_ankle_valid", "left_ankle_valid"),
    ("right_ankle_valid", "right_ankle_valid"),
    ("left_ankle_point", "left_ankle_point"),
    ("right_ankle_point", "right_ankle_point"),
    ("left_ankle_raw_point", "left_ankle_raw_point"),
    ("right_ankle_raw_point", "right_ankle_raw_point"),
    ("left_ankle_step_point", "left_ankle_step_point"),
    ("right_ankle_step_point", "right_ankle_step_point"),
    ("left_ankle_conf", "left_ankle_conf"),
    ("right_ankle_conf", "right_ankle_conf"),
    ("left_current_ankle_step", "left_current_ankle_step"),
    ("right_current_ankle_step", "right_current_ankle_step"),
    ("left_raw_ankle_step", "left_raw_ankle_step"),
    ("right_raw_ankle_step", "right_raw_ankle_step"),
    ("left_raw_step", "left_raw_step"),
    ("right_raw_step", "right_raw_step"),
    ("left_filtered_step", "left_filtered_step"),
    ("right_filtered_step", "right_filtered_step"),
    ("left_last_valid_step", "left_last_valid_step"),
    ("right_last_valid_step", "right_last_valid_step"),
    ("left_step_filter_reason", "left_step_filter_reason"),
    ("right_step_filter_reason", "right_step_filter_reason"),
    ("left_adjusted_step", "left_adjusted_step"),
    ("right_adjusted_step", "right_adjusted_step"),
    ("ankle_step_offset_x", "ankle_step_offset_x"),
    ("ankle_step_offset_y", "ankle_step_offset_y"),
    ("ankle_step_offset_direction", "ankle_step_offset_direction"),
    ("left_step_reason", "left_step_reason"),
    ("right_step_reason", "right_step_reason"),
    ("left_step_nearest_band_id", "left_step_nearest_band_id"),
    ("right_step_nearest_band_id", "right_step_nearest_band_id"),
    ("left_step_nearest_distance", "left_step_nearest_distance"),
    ("right_step_nearest_distance", "right_step_nearest_distance"),
    ("left_step_is_inside", "left_step_is_inside"),
    ("right_step_is_inside", "right_step_is_inside"),
    ("left_step_is_near_boundary", "left_step_is_near_boundary"),
    ("right_step_is_near_boundary", "right_step_is_near_boundary"),
    ("left_step_index", "left_step_index"),
    ("right_step_index", "right_step_index"),
    ("foot_gap", "foot_gap"),
    ("step_gap", "step_gap"),
    ("left_planted_step", "left_planted_step"),
    ("right_planted_step", "right_planted_step"),
    ("planted_step_gap", "planted_step_gap"),
    ("left_is_planted", "left_is_planted"),
    ("right_is_planted", "right_is_planted"),
    ("left_ankle_speed", "left_ankle_speed"),
    ("right_ankle_speed", "right_ankle_speed"),
    ("left_foot_step_reason", "left_foot_step_reason"),
    ("right_foot_step_reason", "right_foot_step_reason"),
    ("left_foot_state_label", "left_foot_state_label"),
    ("right_foot_state_label", "right_foot_state_label"),
    ("left_foot_landed", "left_foot_landed"),
    ("right_foot_landed", "right_foot_landed"),
    ("two_step_skip_check_available", "two_step_skip_check_available"),
    ("two_step_skip_status", "two_step_skip_status"),
    ("two_step_skip_reason", "two_step_skip_reason"),
    ("two_step_skip_confirmed", "two_step_skip_confirmed"),
    ("left_foot_in", "left_foot_in"),
    ("right_foot_in", "right_foot_in"),
    ("valid_foot_count", "valid_foot_count"),
    ("track_zone_state", "track_zone_state"),
    ("inside_raw_by_feet", "inside_raw_by_feet"),
    ("inside_reason", "inside_reason"),
    ("inside_grace_left", "inside_grace_left"),
    ("holding", "holding"),
    ("holding_raw", "holding_raw"),
    ("hold_status", "hold_final_status"),
    ("hold_raw_status", "hold_raw_status"),
    ("hold_confirmed_status", "hold_final_status"),
    ("hold_correct_hits", "hold_correct_hits"),
    ("hold_wrong_side_hits", "hold_wrong_side_hits"),
    ("hold_none_hits", "hold_none_hits"),
    ("hold_unknown_hits", "hold_unknown_hits"),
    ("hold_not_hold_evidence_hits", "hold_not_hold_evidence_hits"),
    ("holding_correct_raw", "holding_correct_raw"),
    ("holding_wrong_raw", "holding_wrong_raw"),
    ("left_hand_claim", "left_hand_claim"),
    ("right_hand_claim", "right_hand_claim"),
    ("left_hold_claim_hits", "left_hold_claim_hits"),
    ("right_hold_claim_hits", "right_hold_claim_hits"),
    ("left_carry_claim_hits", "left_carry_claim_hits"),
    ("right_carry_claim_hits", "right_carry_claim_hits"),
    ("left_hold_raw_before_claim", "left_hold_raw_before_claim"),
    ("right_hold_raw_before_claim", "right_hold_raw_before_claim"),
    ("left_hold_raw_after_claim", "left_hold_raw_after_claim"),
    ("right_hold_raw_after_claim", "right_hold_raw_after_claim"),
    ("left_carry_raw_before_claim", "left_carry_raw_before_claim"),
    ("right_carry_raw_before_claim", "right_carry_raw_before_claim"),
    ("left_carry_raw_after_claim", "left_carry_raw_after_claim"),
    ("right_carry_raw_after_claim", "right_carry_raw_after_claim"),
    ("left_wrist_valid", "left_wrist_valid"),
    ("right_wrist_valid", "right_wrist_valid"),
    ("left_wrist_hit", "left_wrist_hit"),
    ("right_wrist_hit", "right_wrist_hit"),
    ("left_wrist_hit_count", "left_wrist_hit_count"),
    ("right_wrist_hit_count", "right_wrist_hit_count"),
    ("left_wrist_miss_count", "left_wrist_miss_count"),
    ("right_wrist_miss_count", "right_wrist_miss_count"),
    ("left_wrist_confirm_required", "left_wrist_confirm_required"),
    ("right_wrist_confirm_required", "right_wrist_confirm_required"),
    ("left_wrist_distance_to_left_rail", "left_wrist_distance_to_left_rail"),
    ("left_wrist_distance_to_right_rail", "left_wrist_distance_to_right_rail"),
    ("right_wrist_distance_to_left_rail", "right_wrist_distance_to_left_rail"),
    ("right_wrist_distance_to_right_rail", "right_wrist_distance_to_right_rail"),
    ("left_wrist_nearest_distance", "left_wrist_nearest_distance"),
    ("right_wrist_nearest_distance", "right_wrist_nearest_distance"),
    ("left_wrist_nearest_rail", "left_wrist_nearest_rail"),
    ("right_wrist_nearest_rail", "right_wrist_nearest_rail"),
    ("left_wrist_nearest_point", "left_wrist_nearest_point"),
    ("right_wrist_nearest_point", "right_wrist_nearest_point"),
    ("left_holding", "left_holding"),
    ("right_holding", "right_holding"),
    ("handrail_status", "handrail_status"),
    ("handrail_reason", "handrail_reason"),
    ("handrail_debug_reason", "handrail_debug_reason"),
    ("p_lane", "p_lane"),
    ("p_motion", "p_motion"),
    ("body_facing", "body_facing"),
    ("body_facing_confidence", "body_facing_confidence"),
    ("body_facing_evidence_count", "body_facing_evidence_count"),
    ("body_facing_front_votes", "body_facing_front_votes"),
    ("body_facing_back_votes", "body_facing_back_votes"),
    ("body_facing_reason", "body_facing_reason"),
    ("hip_pair_valid", "hip_pair_valid"),
    ("shoulder_pair_valid", "shoulder_pair_valid"),
    ("ear_pair_valid", "ear_pair_valid"),
    ("head_valid", "head_valid"),
    ("arm_side_order", "arm_side_order"),
    ("backward_raw", "backward_raw"),
    ("backward_hits", "backward_hits"),
    ("backward_confirmed", "backward_confirmed"),
    ("backward_mapping_source", "backward_mapping_source"),
    ("handrail_mapping_source", "handrail_mapping_source"),
    ("backward_reason", "backward_reason"),
    ("standing_raw", "standing_raw"),
    ("standing_hits", "standing_hits"),
    ("standing_still_confirmed", "standing_still_confirmed"),
    ("standing_motion_range", "standing_motion_range"),
    ("standing_len", "standing_len"),
    ("warnings", "warnings"),
)
# CARRY_RESULT_FIELDS la cac field duoc carry module bo sung vao result dict cuoi cung.
CARRY_RESULT_FIELDS = (
    "is_carrying",
    "carry_type",
    "left_arm_angle",
    "right_arm_angle",
    "left_wrist_in_torso",
    "right_wrist_in_torso",
    "body_scale",
    "shoulder_width",
    "torso_height",
    "wrist_dx",
    "wrist_dx_threshold",
    "wrist_dy",
    "wrist_dy_threshold",
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
    "left_carry_allowed",
    "right_carry_allowed",
    "left_carry_evidence",
    "right_carry_evidence",
    "left_carry_score",
    "right_carry_score",
    "one_hand_carry_side",
    "carry_reason",
    "carrying_arm",
    "left_carry_raw_before_claim",
    "right_carry_raw_before_claim",
    "left_carry_raw_after_claim",
    "right_carry_raw_after_claim",
)
# DEBUG_INFO_FIELDS quy dinh tap field se duoc sao chep vao debug_info
# de overlay/log doc chung 1 cau truc ma khong phai giu nguyen ca locals().
DEBUG_INFO_FIELDS = (
    "status",
    "display_status",
    "track_id",
    "dy",
    "direction_dy",
    "lane_v",
    "direction",
    "final_direction",
    "hip_direction",
    "shoulder_direction",
    "direction_source",
    "direction_reason",
    "monitor_point_hip",
    "monitor_point_hip_source",
    "monitor_point_shoulder",
    "monitor_point_shoulder_source",
    "use_current_camera_angle",
    "camera_angle_profile",
    "lane_raw",
    "lane_hits",
    "lane_conf",
    "wrong_lane",
    "lane_status",
    "lane_reason",
    "lane_source",
    "lane_direction",
    "lane_side_value",
    "lane_side_label",
    "correct_lane_side",
    "lane_mapping_source",
    "p_lane_source",
    "lane_missing_feet_grace_left",
    "foot_lane_side",
    "inside_stairs",
    "inside_feet_point",
    "feet_point_source",
    "inside_feet_point_source",
    "feet_available",
    "feet_unavailable_reason",
    "sh_hip_visible_shoulder_count",
    "sh_hip_visible_hip_count",
    "sh_hip_selected_pair",
    "sh_hip_virtual_feet_point",
    "sh_hip_virtual_feet_source",
    "ankle_valid_count",
    "feet_reliable",
    "bbox_height",
    "left_ankle_valid",
    "right_ankle_valid",
    "left_ankle_point",
    "right_ankle_point",
    "left_ankle_raw_point",
    "right_ankle_raw_point",
    "left_ankle_step_point",
    "right_ankle_step_point",
    "left_ankle_conf",
    "right_ankle_conf",
    "left_current_ankle_step",
    "right_current_ankle_step",
    "left_raw_ankle_step",
    "right_raw_ankle_step",
    "left_raw_step",
    "right_raw_step",
    "left_filtered_step",
    "right_filtered_step",
    "left_last_valid_step",
    "right_last_valid_step",
    "left_step_filter_reason",
    "right_step_filter_reason",
    "left_adjusted_step",
    "right_adjusted_step",
    "ankle_step_offset_x",
    "ankle_step_offset_y",
    "ankle_step_offset_direction",
    "left_step_reason",
    "right_step_reason",
    "left_step_nearest_band_id",
    "right_step_nearest_band_id",
    "left_step_nearest_distance",
    "right_step_nearest_distance",
    "left_step_is_inside",
    "right_step_is_inside",
    "left_step_is_near_boundary",
    "right_step_is_near_boundary",
    "left_step_index",
    "right_step_index",
    "foot_gap",
    "step_gap",
    "left_planted_step",
    "right_planted_step",
    "planted_step_gap",
    "left_is_planted",
    "right_is_planted",
    "left_ankle_speed",
    "right_ankle_speed",
    "left_foot_step_reason",
    "right_foot_step_reason",
    "left_foot_state_label",
    "right_foot_state_label",
    "left_foot_landed",
    "right_foot_landed",
    "two_step_skip_check_available",
    "two_step_skip_status",
    "two_step_skip_reason",
    "two_step_skip_confirmed",
    "left_foot_in",
    "right_foot_in",
    "valid_foot_count",
    "track_zone_state",
    "inside_raw_by_feet",
    "inside_reason",
    "inside_grace_left",
    "holding",
    "holding_raw",
    "hold_status",
    "hold_raw_status",
    "hold_confirmed_status",
    "hold_correct_hits",
    "hold_wrong_side_hits",
    "hold_none_hits",
    "hold_unknown_hits",
    "hold_not_hold_evidence_hits",
    "not_hold_by_evidence",
    "holding_correct_raw",
    "holding_wrong_raw",
    "left_hand_claim",
    "right_hand_claim",
    "left_hold_claim_hits",
    "right_hold_claim_hits",
    "left_carry_claim_hits",
    "right_carry_claim_hits",
    "left_hold_raw_before_claim",
    "right_hold_raw_before_claim",
    "left_hold_raw_after_claim",
    "right_hold_raw_after_claim",
    "left_carry_raw_before_claim",
    "right_carry_raw_before_claim",
    "left_carry_raw_after_claim",
    "right_carry_raw_after_claim",
    "left_wrist_valid",
    "right_wrist_valid",
    "left_wrist_hit",
    "right_wrist_hit",
    "left_wrist_hit_count",
    "right_wrist_hit_count",
    "left_wrist_miss_count",
    "right_wrist_miss_count",
    "left_wrist_confirm_required",
    "right_wrist_confirm_required",
    "left_wrist_distance_to_left_rail",
    "left_wrist_distance_to_right_rail",
    "right_wrist_distance_to_left_rail",
    "right_wrist_distance_to_right_rail",
    "left_wrist_nearest_distance",
    "right_wrist_nearest_distance",
    "left_wrist_nearest_rail",
    "right_wrist_nearest_rail",
    "left_wrist_nearest_point",
    "right_wrist_nearest_point",
    "left_holding",
    "right_holding",
    "handrail_status",
    "handrail_reason",
    "handrail_debug_reason",
    "is_carrying",
    "carry_type",
    "left_arm_angle",
    "right_arm_angle",
    "left_wrist_in_torso",
    "right_wrist_in_torso",
    "body_scale",
    "shoulder_width",
    "torso_height",
    "wrist_dx",
    "wrist_dx_threshold",
    "wrist_dy",
    "wrist_dy_threshold",
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
    "left_carry_allowed",
    "right_carry_allowed",
    "left_carry_evidence",
    "right_carry_evidence",
    "left_carry_score",
    "right_carry_score",
    "one_hand_carry_side",
    "carry_reason",
    "carrying_arm",
    "p_lane",
    "p_motion",
    "body_facing",
    "body_facing_confidence",
    "body_facing_evidence_count",
    "body_facing_front_votes",
    "body_facing_back_votes",
    "body_facing_reason",
    "hip_pair_valid",
    "shoulder_pair_valid",
    "ear_pair_valid",
    "head_valid",
    "arm_side_order",
    "backward_raw",
    "backward_hits",
    "backward_confirmed",
    "backward_mapping_source",
    "handrail_mapping_source",
    "backward_reason",
    "standing_raw",
    "standing_hits",
    "standing_still_confirmed",
    "standing_motion_range",
    "standing_len",
)


class ResultBuilderMixin:
    """Mixin gom warning, status va dong goi result dict cuoi cung."""

    @staticmethod
    def _lane_suppresses_hold_display(warnings: WarningList) -> bool:
        """Cho biet Sai Lan co dang an bot warning hold tren overlay hay khong."""
        return "Sai Lan" in warnings

    @staticmethod
    def _apply_lane_warning_priority(warnings: WarningList) -> WarningList:
        """Ap quy tac uu tien warning khi Sai Lan da xuat hien."""
        # WHY: Khi da co `Sai Lan`, overlay demo uu tien nhan nay de tranh roi text hold phu.
        if not ResultBuilderMixin._lane_suppresses_hold_display(warnings):
            return warnings
        return [
            warning
            for warning in warnings
            if warning not in LANE_SUPPRESSED_HOLD_WARNINGS
        ]

    @staticmethod
    # Doi nhan noi bo sang chuoi hien thi de overlay doc de hon.
    def _translate_display_text(text: str) -> str:
        """Doi nhan noi bo sang chuoi hien thi de overlay doc de hon.

        Args:
            text: Chuoi noi bo can quy doi.

        Returns:
            str: Chuoi da thay nhan hien thi.

        Notes:
            Ham nay chi anh huong text overlay, khong doi warning/noi dung logic.
        """
        translated = text or ""
        for internal_label, display_label in DISPLAY_TEXT_REPLACEMENTS.items():
            translated = translated.replace(internal_label, display_label)
        return translated

    @staticmethod
    # display_status la chuoi gon de ve demo.
    # status day du van duoc giu lai trong result de debug/log khi can.
    def _build_display_status(direction: str, warnings: WarningList) -> str:
        """Tao chuoi status gon de ve tren bbox/alert demo.

        Args:
            direction: Direction hien tai cua track.
            warnings: Danh sach warning noi bo.

        Returns:
            str: Chuoi ngan gon de overlay.

        Notes:
            display_status ngan hon status day du de khong lam roi khung nguoi.
        """
        if SETTINGS.demo.demo_mode:
            return ResultBuilderMixin._translate_display_text(" - ".join(warnings))
        if warnings:
            return ResultBuilderMixin._translate_display_text(
                f"{direction} | {' - '.join(warnings)}"
            )
        if SETTINGS.demo.draw_safe_status:
            return ResultBuilderMixin._translate_display_text(f"{direction} | An Toan")
        return ""

    @staticmethod
    # Gom cac loai loi thuc su tu tung module.
    # Danh sach nay duoc dung cho overlay hien tai, alert demo va log.
    def _collect_warnings(
        direction: str,
        wrong_lane: bool,
        hold_final_status: str,
        backward_confirmed: bool,
        standing_still_confirmed: bool,
        is_carrying: bool = False,
        two_step_skip_confirmed: bool = False,
    ) -> WarningList:
        """Gom cac warning final tu tung module hanh vi.

        Args:
            direction: Direction hien tai cua subject.
            wrong_lane: Ket qua Sai Lan final.
            hold_final_status: Ket qua hold final sau history.
            backward_confirmed: Ket qua Di Lui final.
            standing_still_confirmed: Ket qua Dung Yen final.
            is_carrying: Ket qua Mang Vac final.

        Returns:
            list[str]: Danh sach warning final.

        Notes:
            Danh sach nay duoc dung cho ca overlay, alert demo va log.
        """
        # WARNING: `Di Lui` duoc uu tien hon hold/no-hold trong runtime hien tai.
        # WARNING: `Mang Vac`, `Sai Lan`, `Dung Yen`, `Buoc 2 Bac` la warning doc lap cua tung rule rieng.
        warnings = []
        if wrong_lane:
            warnings.append("Sai Lan")
        if not backward_confirmed:
            if hold_final_status == "WRONG_SIDE":
                warnings.append("Vin Sai Ben")
            elif hold_final_status == "NONE":
                warnings.append("Khong Vin")
            elif hold_final_status == "UNKNOWN" and not SETTINGS.demo.demo_mode:
                warnings.append("Khong Xac Dinh")
        if is_carrying:
            warnings.append("Mang Vac")
        if backward_confirmed:
            warnings.append("Di Lui")
        if standing_still_confirmed:
            warnings.append("Dung Yen")
        if direction == "UP" and two_step_skip_confirmed:
            warnings.append("Buoc 2 Bac")
        return ResultBuilderMixin._apply_lane_warning_priority(warnings)

    def _summarize_result(
        self,
        direction: str,
        warnings: WarningList,
        hold_final_status: str,
        safe_color: ColorBGR,
        safe_status: str,
    ) -> tuple[str, str, ColorBGR]:
        """Tong hop mau sac va status cuoi cung cho 1 track.

        Args:
            direction: Direction hien tai.
            warnings: Danh sach warning final.
            hold_final_status: Hold final sau history.
            safe_color: Mau dung khi khong co vi pham.
            safe_status: Chuoi fallback khi khong co warning.

        Returns:
            tuple: (status, display_status, color)

        Notes:
            debug overlay co the dung status day du, con bbox overlay thuong dung
            display_status gon hon de de doc.
        """
        # WHY: Mau sac va text display duoc tinh sau cung o day de moi rule khong phai tu quyet dinh UI.
        # Tach "real violation" ra khoi UNKNOWN de alert demo va mau sac khong bi nham.
        real_warnings = [
            warning for warning in warnings if warning in REAL_VIOLATION_LABELS
        ]
        has_real_violation = len(real_warnings) > 0
        has_unknown = hold_final_status == "UNKNOWN" and not has_real_violation

        if has_real_violation:
            color = SETTINGS.violation.violation_color
        elif has_unknown:
            color = SETTINGS.violation.unknown_color
        else:
            color = safe_color

        lane_suppresses_hold_display = self._lane_suppresses_hold_display(warnings)
        status_warnings = list(warnings)
        if (
            hold_final_status == "UNKNOWN"
            and not SETTINGS.demo.demo_mode
            and not lane_suppresses_hold_display
            and "Khong Xac Dinh" not in status_warnings
        ):
            status_warnings.append("Khong Xac Dinh")

        status = (
            f"{direction} | {' - '.join(status_warnings)}"
            if status_warnings
            else safe_status
        )
        status = self._translate_display_text(status)
        display_status = self._build_display_status(direction, real_warnings)
        return status, display_status, color

    def _build_result_from_context(
        self,
        context: dict[str, object],
        status: str,
        color: ColorBGR,
        display_status: str = "",
        carry_info: dict[str, object] | None = None,
        perf: PerfStats | None = None,
        **overrides: object,
    ) -> AnalysisResult:
        """Dong goi ket qua analyzer thanh 1 result dict on dinh.

        Args:
            context: locals()/context cua analyzer tai thoi diem tra ket qua.
            status: Chuoi status day du.
            color: Mau overlay final.
            display_status: Chuoi gon de ve tren bbox.
            carry_info: Dict carry info neu co.
            perf: Thong tin perf theo block.
            **overrides: Field bo sung hoac ghi de them vao result.

        Returns:
            dict: Result dict cuoi cung.

        Notes:
            File nay co vai tro gom ket qua tu analyzer thanh 1 dict duy nhat de
            rendering va log co the doc ma khong can biet noi bo analyzer.
        """
        # FLOW: Context trong analyzer co nhieu bien tam. RESULT_CONTEXT_FIELDS chon ra nhung field public duoc phep lo ra ngoai.
        # Rut field can thiet tu locals()/context cua analyzer de dong goi ra 1 result dict duy nhat.
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
            perf=perf,
            **result_kwargs,
        )

    @staticmethod
    def _build_result(
        status: str,
        color: ColorBGR,
        display_status: str = "",
        **result_fields: object,
    ) -> AnalysisResult:
        """Tao result dict cuoi cung va debug_info cho overlay/log.

        Args:
            status: Chuoi status day du.
            color: Mau overlay final.
            display_status: Chuoi gon de ve tren bbox.
            ...: Cac field behavior/debug duoc analyzer truyen vao.

        Returns:
            dict: Result dict final cho 1 nguoi trong 1 frame.

        Notes:
            warnings la danh sach canh bao cuoi cung. debug_info chi phuc vu
            overlay/log, khong duoc de logic nhan dien phu thuoc vao no.
        """
        fields: DebugInfoDict = {
            "track_id": None,
            "dy": None,
            "direction_dy": None,
            "lane_v": None,
            "direction": "NA",
            "final_direction": "NA",
            "hip_direction": "UNKNOWN",
            "shoulder_direction": "UNKNOWN",
            "direction_source": "NO_VALID_MONITOR_DIRECTION",
            "direction_reason": "UNKNOWN",
            "monitor_point_hip": None,
            "monitor_point_hip_source": "NO_HIP_CENTER",
            "monitor_point_shoulder": None,
            "monitor_point_shoulder_source": "NO_SHOULDER_CENTER",
            "use_current_camera_angle": True,
            "camera_angle_profile": "CURRENT_CAMERA",
            "lane_raw": False,
            "lane_hits": 0,
            "lane_conf": False,
            "wrong_lane": False,
            "lane_status": "UNKNOWN",
            "lane_reason": "NA",
            "lane_source": "NO_LANE",
            "lane_direction": "ANALYZING",
            "lane_side_value": None,
            "lane_side_label": "UNKNOWN",
            "correct_lane_side": "UNKNOWN",
            "lane_mapping_source": "CURRENT_CAMERA",
            "p_lane_source": "NONE",
            "lane_missing_feet_grace_left": 0,
            "foot_lane_side": None,
            "inside_stairs": False,
            "inside_feet_point": None,
            "feet_point_source": "FEET_UNAVAILABLE",
            "inside_feet_point_source": "FEET_UNAVAILABLE",
            "feet_available": False,
            "feet_unavailable_reason": "NO_SHOULDER_NO_HIP",
            "sh_hip_visible_shoulder_count": 0,
            "sh_hip_visible_hip_count": 0,
            "sh_hip_selected_pair": "NONE",
            "sh_hip_virtual_feet_point": None,
            "sh_hip_virtual_feet_source": "FEET_UNAVAILABLE",
            "ankle_valid_count": 0,
            "feet_reliable": False,
            "bbox_height": None,
            "left_ankle_valid": False,
            "right_ankle_valid": False,
            "left_ankle_point": None,
            "right_ankle_point": None,
            "left_ankle_raw_point": None,
            "right_ankle_raw_point": None,
            "left_ankle_step_point": None,
            "right_ankle_step_point": None,
            "left_ankle_conf": 0.0,
            "right_ankle_conf": 0.0,
            "left_current_ankle_step": None,
            "right_current_ankle_step": None,
            "left_raw_ankle_step": None,
            "right_raw_ankle_step": None,
            "left_raw_step": None,
            "right_raw_step": None,
            "left_filtered_step": None,
            "right_filtered_step": None,
            "left_last_valid_step": None,
            "right_last_valid_step": None,
            "left_step_filter_reason": "NOT_EVALUATED",
            "right_step_filter_reason": "NOT_EVALUATED",
            "left_adjusted_step": None,
            "right_adjusted_step": None,
            "ankle_step_offset_x": 0,
            "ankle_step_offset_y": 0,
            "ankle_step_offset_direction": "UNKNOWN",
            "left_step_reason": "NOT_EVALUATED",
            "right_step_reason": "NOT_EVALUATED",
            "left_step_nearest_band_id": None,
            "right_step_nearest_band_id": None,
            "left_step_nearest_distance": None,
            "right_step_nearest_distance": None,
            "left_step_is_inside": False,
            "right_step_is_inside": False,
            "left_step_is_near_boundary": False,
            "right_step_is_near_boundary": False,
            "left_step_index": None,
            "right_step_index": None,
            "foot_gap": None,
            "step_gap": None,
            "left_planted_step": None,
            "right_planted_step": None,
            "planted_step_gap": None,
            "left_is_planted": False,
            "right_is_planted": False,
            "left_ankle_speed": None,
            "right_ankle_speed": None,
            "left_foot_step_reason": "NOT_EVALUATED",
            "right_foot_step_reason": "NOT_EVALUATED",
            "left_foot_state_label": "WAIT",
            "right_foot_state_label": "WAIT",
            "left_foot_landed": False,
            "right_foot_landed": False,
            "two_step_skip_check_available": False,
            "two_step_skip_status": "NOT_EVALUATED",
            "two_step_skip_reason": "NOT_EVALUATED",
            "two_step_skip_confirmed": False,
            "left_foot_in": False,
            "right_foot_in": False,
            "valid_foot_count": 0,
            "track_zone_state": "UNKNOWN",
            "inside_raw_by_feet": None,
            "inside_reason": "UNKNOWN",
            "inside_grace_left": 0,
            "holding": False,
            "holding_raw": False,
            "hold_status": "ANALYZING",
            "hold_raw_status": "UNKNOWN",
            "hold_confirmed_status": "ANALYZING",
            "hold_correct_hits": 0,
            "hold_wrong_side_hits": 0,
            "hold_none_hits": 0,
            "hold_unknown_hits": 0,
            "hold_not_hold_evidence_hits": 0,
            "holding_correct_raw": False,
            "holding_wrong_raw": False,
            "left_hand_claim": "NONE",
            "right_hand_claim": "NONE",
            "left_hold_claim_hits": 0,
            "right_hold_claim_hits": 0,
            "left_carry_claim_hits": 0,
            "right_carry_claim_hits": 0,
            "left_hold_raw_before_claim": False,
            "right_hold_raw_before_claim": False,
            "left_hold_raw_after_claim": False,
            "right_hold_raw_after_claim": False,
            "left_wrist_valid": False,
            "right_wrist_valid": False,
            "left_wrist_hit": False,
            "right_wrist_hit": False,
            "left_wrist_hit_count": 0,
            "right_wrist_hit_count": 0,
            "left_wrist_miss_count": 0,
            "right_wrist_miss_count": 0,
            "left_wrist_confirm_required": 0,
            "right_wrist_confirm_required": 0,
            "left_wrist_distance_to_left_rail": None,
            "left_wrist_distance_to_right_rail": None,
            "right_wrist_distance_to_left_rail": None,
            "right_wrist_distance_to_right_rail": None,
            "left_wrist_nearest_distance": None,
            "right_wrist_nearest_distance": None,
            "left_wrist_nearest_rail": "NONE",
            "right_wrist_nearest_rail": "NONE",
            "left_wrist_nearest_point": None,
            "right_wrist_nearest_point": None,
            "left_holding": False,
            "right_holding": False,
            "handrail_status": "NOT_EVALUATED",
            "handrail_reason": "NOT_EVALUATED",
            "handrail_debug_reason": "NOT_EVALUATED",
            "is_carrying": False,
            "carry_type": "NONE",
            "left_arm_angle": None,
            "right_arm_angle": None,
            "left_wrist_in_torso": False,
            "right_wrist_in_torso": False,
            "body_scale": None,
            "shoulder_width": None,
            "torso_height": None,
            "wrist_dx": None,
            "wrist_dx_threshold": None,
            "wrist_dy": None,
            "wrist_dy_threshold": None,
            "wrist_distance": None,
            "left_bent": False,
            "right_bent": False,
            "wrists_close": False,
            "any_wrist_in_torso": False,
            "both_wrist_in_torso": False,
            "front_carry": False,
            "front_carry_raw": False,
            "front_carry_hits": 0,
            "front_carry_two_hand_raw": False,
            "front_carry_two_hand_hits": 0,
            "front_carry_one_arm_raw": False,
            "front_carry_one_arm_hits": 0,
            "front_carry_confirmed": False,
            "left_carry": False,
            "right_carry": False,
            "left_carry_allowed": True,
            "right_carry_allowed": True,
            "left_carry_evidence": False,
            "right_carry_evidence": False,
            "left_carry_score": None,
            "right_carry_score": None,
            "one_hand_carry_side": "NONE",
            "carry_reason": "NO_CARRY",
            "carrying_arm": "NONE",
            "left_carry_raw_before_claim": False,
            "right_carry_raw_before_claim": False,
            "left_carry_raw_after_claim": False,
            "right_carry_raw_after_claim": False,
            "p_lane": None,
            "p_motion": None,
            "body_facing": "UNKNOWN",
            "body_facing_confidence": 0.0,
            "body_facing_evidence_count": 0,
            "body_facing_front_votes": 0,
            "body_facing_back_votes": 0,
            "body_facing_reason": "UNKNOWN",
            "hip_pair_valid": False,
            "shoulder_pair_valid": False,
            "ear_pair_valid": False,
            "head_valid": False,
            "arm_side_order": "UNKNOWN",
            "backward_raw": False,
            "backward_hits": 0,
            "backward_confirmed": False,
            "backward_mapping_source": "CURRENT_CAMERA",
            "handrail_mapping_source": "CURRENT_CAMERA",
            "backward_reason": "UNKNOWN_NOT_ENOUGH_EVIDENCE",
            "standing_raw": False,
            "standing_hits": 0,
            "standing_still_confirmed": False,
            "standing_motion_range": None,
            "standing_len": 0,
            "warnings": [],
            "perf": None,
        }
        fields.update(result_fields)

        hold_status = str(fields.get("hold_status", "ANALYZING"))
        hold_not_hold_evidence_hits = fields.get("hold_not_hold_evidence_hits", 0)
        warning_values = fields.get("warnings", [])
        perf_value = fields.get("perf")

        not_hold_by_evidence = (
            hold_status == "NONE"
            and isinstance(hold_not_hold_evidence_hits, int)
            and hold_not_hold_evidence_hits
            >= SETTINGS.handrail.hold_min_not_hold_evidence_hits
        )

        result: AnalysisResult = {
            "status": status,
            "display_status": display_status,
            "color": color,
            **fields,
            "warnings": list(warning_values) if isinstance(warning_values, list) else [],
            "perf": perf_value if isinstance(perf_value, dict) or perf_value is None else None,
            "not_hold_by_evidence": not_hold_by_evidence,
        }
        # debug_info chi phuc vu overlay/log debug.
        # Logic nhan dien khong duoc phu thuoc vao viec co bat debug overlay hay khong.
        result["debug_info"] = (
            {key: result.get(key) for key in DEBUG_INFO_FIELDS}
            if SETTINGS.demo.enable_debug_overlay
            else None
        )
        return result
