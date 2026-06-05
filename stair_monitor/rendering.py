import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from stair_monitor.geometry import extract_pose_features
from stair_monitor.settings import (
    DEMO_MODE,
    DRAW_DEBUG_DETAIL,
    DRAW_KEYPOINTS,
    DRAW_SKELETON,
    ENABLE_DEBUG_OVERLAY,
    ENABLE_SUMMARY_PANEL,
    ENABLE_VERBOSE_PERSON_DEBUG,
    ENABLE_VIETNAMESE_TEXT,
    SHOW_ONLY_VIOLATIONS,
    VIOLATION_COUNT_LABELS,
    VIOLATION_COLOR,
    VIOLATION_DISPLAY_NAMES,
)

FONT_CACHE = {}
FONT_CANDIDATES = {
    False: [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ],
    True: [
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/tahomabd.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
    ],
}


def get_vietnamese_font(font_size=28, bold=False):
    cache_key = (font_size, bold)
    if cache_key in FONT_CACHE:
        return FONT_CACHE[cache_key]

    for font_path in FONT_CANDIDATES[bold]:
        if os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
            FONT_CACHE[cache_key] = font
            return font

    font = ImageFont.load_default()
    FONT_CACHE[cache_key] = font
    return font


def _bgr_to_rgb(color):
    return (color[2], color[1], color[0])


def _frame_to_pil(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def _pil_to_frame(pil_image, frame):
    frame[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def _measure_vietnamese_lines(lines, font_size=28, padding=14, line_gap=8):
    if not lines:
        return 0, 0

    probe_image = Image.new("RGB", (1, 1))
    probe_draw = ImageDraw.Draw(probe_image)

    max_width = 0
    total_height = padding * 2
    for idx, line in enumerate(lines):
        font = get_vietnamese_font(font_size, bold=(idx == 0))
        bbox = probe_draw.textbbox((0, 0), line, font=font)
        line_width = bbox[2] - bbox[0]
        line_height = bbox[3] - bbox[1]
        max_width = max(max_width, line_width)
        total_height += line_height
        if idx < len(lines) - 1:
            total_height += line_gap

    return max_width + padding * 2, total_height


class VietnameseTextDrawer:
    def __init__(self, frame, enabled=True):
        self.frame = frame
        self.enabled = enabled and ENABLE_VIETNAMESE_TEXT
        self.operations = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.enabled or not self.operations:
            self.operations = []
            return

        pil_image = _frame_to_pil(self.frame)
        draw = ImageDraw.Draw(pil_image, "RGBA")

        for op in self.operations:
            if op["type"] == "text":
                draw.text(
                    (int(op["position"][0]), int(op["position"][1])),
                    op["text"],
                    font=get_vietnamese_font(op["font_size"], bold=op["bold"]),
                    fill=_bgr_to_rgb(op["color"]),
                )
                continue

            if op["type"] != "panel":
                continue

            x = op["x"]
            y = op["y"]
            lines = op["lines"]
            alpha = op["alpha"]
            font_size = op["font_size"]
            text_color = op["text_color"]
            panel_color = op["panel_color"]
            padding = op["padding"]
            line_gap = op["line_gap"]
            anchor = op["anchor"]

            panel_width, panel_height = _measure_vietnamese_lines(
                lines,
                font_size=font_size,
                padding=padding,
                line_gap=line_gap,
            )
            frame_h, frame_w = self.frame.shape[:2]

            x1 = int(x - panel_width) if anchor == "right" else int(x)
            y1 = int(y)
            x1 = max(0, min(x1, max(0, frame_w - panel_width)))
            y1 = max(0, min(y1, max(0, frame_h - panel_height)))
            x2 = x1 + panel_width
            y2 = y1 + panel_height

            draw.rounded_rectangle(
                [(x1, y1), (x2, y2)],
                radius=18,
                fill=(*_bgr_to_rgb(panel_color), int(255 * alpha)),
            )

            cursor_y = y1 + padding
            for idx, line in enumerate(lines):
                is_title = idx == 0
                font = get_vietnamese_font(font_size, bold=is_title)
                bbox = draw.textbbox((0, 0), line, font=font)
                line_height = bbox[3] - bbox[1]
                draw.text(
                    (x1 + padding, cursor_y),
                    line,
                    font=font,
                    fill=(*_bgr_to_rgb(text_color), 255),
                )
                cursor_y += line_height + line_gap

        _pil_to_frame(pil_image, self.frame)
        self.operations = []

    def text(self, text, position, font_size=28, color=(255, 255, 255), bold=False):
        if not text:
            return
        if not self.enabled:
            cv2.putText(
                self.frame,
                text,
                (int(position[0]), int(position[1] + font_size)),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(font_size / 36.0, 0.5),
                color,
                2 if bold else 1,
            )
            return

        self.operations.append(
            {
                "type": "text",
                "text": text,
                "position": (int(position[0]), int(position[1])),
                "font_size": font_size,
                "color": color,
                "bold": bold,
            }
        )

    def transparent_panel(
        self,
        x,
        y,
        lines,
        alpha=0.45,
        font_size=28,
        text_color=(255, 255, 255),
        panel_color=(28, 36, 48),
        padding=14,
        line_gap=8,
        anchor="left",
    ):
        if not lines:
            return 0, 0

        panel_width, panel_height = _measure_vietnamese_lines(
            lines,
            font_size=font_size,
            padding=padding,
            line_gap=line_gap,
        )
        frame_h, frame_w = self.frame.shape[:2]

        x1 = int(x - panel_width) if anchor == "right" else int(x)
        y1 = int(y)
        x1 = max(0, min(x1, max(0, frame_w - panel_width)))
        y1 = max(0, min(y1, max(0, frame_h - panel_height)))
        x2 = x1 + panel_width
        y2 = y1 + panel_height

        if not self.enabled:
            overlay = self.frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), panel_color, -1)
            cv2.addWeighted(
                overlay,
                alpha,
                self.frame,
                1.0 - alpha,
                0,
                dst=self.frame,
            )
            cursor_y = y1 + padding + font_size
            for line in lines:
                cv2.putText(
                    self.frame,
                    line,
                    (x1 + padding, cursor_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    max(font_size / 36.0, 0.5),
                    text_color,
                    1,
                )
                cursor_y += font_size + line_gap
            return panel_width, panel_height

        self.operations.append(
            {
                "type": "panel",
                "x": x,
                "y": y,
                "lines": list(lines),
                "alpha": alpha,
                "font_size": font_size,
                "text_color": text_color,
                "panel_color": panel_color,
                "padding": padding,
                "line_gap": line_gap,
                "anchor": anchor,
            }
        )
        return panel_width, panel_height


def draw_vietnamese_text(
    frame,
    text,
    position,
    font_size=28,
    color=(255, 255, 255),
    bold=False,
    text_drawer=None,
):
    if not text:
        return

    if text_drawer is not None:
        text_drawer.text(
            text,
            position,
            font_size=font_size,
            color=color,
            bold=bold,
        )
        return

    with VietnameseTextDrawer(frame) as drawer:
        drawer.text(
            text,
            position,
            font_size=font_size,
            color=color,
            bold=bold,
        )


def draw_transparent_panel_with_vietnamese_text(
    frame,
    x,
    y,
    lines,
    alpha=0.45,
    font_size=28,
    text_color=(255, 255, 255),
    panel_color=(28, 36, 48),
    padding=14,
    line_gap=8,
    anchor="left",
    text_drawer=None,
):
    if not lines:
        return 0, 0

    if text_drawer is not None:
        return text_drawer.transparent_panel(
            x,
            y,
            lines,
            alpha=alpha,
            font_size=font_size,
            text_color=text_color,
            panel_color=panel_color,
            padding=padding,
            line_gap=line_gap,
            anchor=anchor,
        )

    with VietnameseTextDrawer(frame) as drawer:
        return drawer.transparent_panel(
            x,
            y,
            lines,
            alpha=alpha,
            font_size=font_size,
            text_color=text_color,
            panel_color=panel_color,
            padding=padding,
            line_gap=line_gap,
            anchor=anchor,
        )


def draw_label_with_background(
    frame,
    text,
    x,
    y,
    font_scale=0.9,
    thickness=3,
    text_color=(255, 255, 255),
    bg_color=(0, 0, 255),
    padding=8,
    text_drawer=None,
):
    if not text:
        return

    frame_h, frame_w = frame.shape[:2]
    font_size = max(20, int(30 * font_scale))
    box_w, box_h = _measure_vietnamese_lines(
        [text],
        font_size=font_size,
        padding=padding,
        line_gap=0,
    )

    x1 = max(0, min(int(x), max(0, frame_w - box_w)))
    y1 = max(0, min(int(y - box_h), max(0, frame_h - box_h)))
    draw_transparent_panel_with_vietnamese_text(
        frame,
        x1,
        y1,
        [text],
        alpha=0.65,
        font_size=font_size,
        text_color=text_color,
        panel_color=bg_color,
        padding=padding,
        line_gap=0,
        text_drawer=text_drawer,
    )


def draw_scene_guides(frame, config, analyzer):
    if "CENTER_LINE" in config:
        cv2.line(
            frame,
            tuple(config["CENTER_LINE"][0]),
            tuple(config["CENTER_LINE"][1]),
            (0, 255, 255),
            2,
        )
    if len(analyzer.stairs_poly) > 2:
        cv2.polylines(
            frame, [analyzer.stairs_poly.reshape((-1, 1, 2))], True, (255, 0, 0), 2
        )

    if len(analyzer.left_line) >= 2:
        cv2.line(
            frame,
            tuple(analyzer.left_line[0]),
            tuple(analyzer.left_line[1]),
            (0, 165, 255),
            3,
        )
    if len(analyzer.right_line) >= 2:
        cv2.line(
            frame,
            tuple(analyzer.right_line[0]),
            tuple(analyzer.right_line[1]),
            (255, 0, 255),
            3,
        )


def get_feet_point(box, keypoints):
    features = extract_pose_features(keypoints, box)
    return features.get("feet_point")


def get_motion_point(box, keypoints):
    features = extract_pose_features(keypoints, box)
    return features.get("motion_point")


def draw_person_overlay(
    frame,
    box,
    keypoints,
    lane_point,
    motion_point,
    analysis,
    text_drawer=None,
):
    display_status = analysis.get("display_status", analysis.get("status", ""))
    if DEMO_MODE and SHOW_ONLY_VIOLATIONS:
        if not display_status:
            return

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), VIOLATION_COLOR, 3)

        label_y = y2 + 42
        if label_y > frame.shape[0] - 5:
            label_y = y1 - 8

        draw_label_with_background(
            frame,
            display_status,
            x1,
            label_y,
            font_scale=0.9,
            thickness=3,
            text_color=(255, 255, 255),
            bg_color=VIOLATION_COLOR,
            padding=8,
            text_drawer=text_drawer,
        )
        return

    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), analysis["color"], 2)

    if display_status:
        draw_label_with_background(
            frame,
            display_status,
            x1,
            y2 + 38,
            font_scale=0.75,
            thickness=2,
            text_color=(255, 255, 255),
            bg_color=analysis["color"],
            padding=8,
            text_drawer=text_drawer,
        )

    if keypoints is not None and DRAW_KEYPOINTS:
        cv2.circle(frame, lane_point, 6, (0, 0, 255), -1)
        cv2.circle(frame, motion_point, 6, (255, 0, 0), -1)

    if keypoints is not None and DRAW_SKELETON:
        if (
            len(keypoints) > 9
            and keypoints[5][2] > 0.5
            and keypoints[7][2] > 0.5
            and keypoints[9][2] > 0.5
        ):
            p_s = (int(keypoints[5][0]), int(keypoints[5][1]))
            p_e = (int(keypoints[7][0]), int(keypoints[7][1]))
            p_w = (int(keypoints[9][0]), int(keypoints[9][1]))
            cv2.line(frame, p_s, p_e, (255, 255, 0), 2)
            cv2.line(frame, p_e, p_w, (255, 255, 0), 2)
            cv2.circle(frame, p_s, 4, (255, 255, 0), -1)
            cv2.circle(frame, p_e, 4, (255, 255, 0), -1)
            cv2.circle(frame, p_w, 4, (255, 255, 0), -1)

        if (
            len(keypoints) > 10
            and keypoints[6][2] > 0.5
            and keypoints[8][2] > 0.5
            and keypoints[10][2] > 0.5
        ):
            p_s = (int(keypoints[6][0]), int(keypoints[6][1]))
            p_e = (int(keypoints[8][0]), int(keypoints[8][1]))
            p_w = (int(keypoints[10][0]), int(keypoints[10][1]))
            cv2.line(frame, p_s, p_e, (0, 255, 255), 2)
            cv2.line(frame, p_e, p_w, (0, 255, 255), 2)
            cv2.circle(frame, p_s, 4, (0, 255, 255), -1)
            cv2.circle(frame, p_e, 4, (0, 255, 255), -1)
            cv2.circle(frame, p_w, 4, (0, 255, 255), -1)

    if DRAW_KEYPOINTS and analysis.get("best_wrist_point") is not None:
        cv2.circle(frame, analysis["best_wrist_point"], 8, (0, 255, 255), -1)

    if ENABLE_DEBUG_OVERLAY and ENABLE_VERBOSE_PERSON_DEBUG and analysis.get(
        "carry_type",
        "NONE",
    ) != "NONE":
        cv2.putText(
            frame,
            f"CARRY:{analysis['carry_type']}",
            (x1, y2 + 47),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    if ENABLE_DEBUG_OVERLAY and ENABLE_VERBOSE_PERSON_DEBUG:
        for idx, debug_text in enumerate(build_debug_lines(analysis)):
            cv2.putText(
                frame,
                debug_text,
                (x2 + 5, y1 + 15 + idx * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )


def draw_people_count(frame, current_inside_count, text_drawer=None):
    if not ENABLE_SUMMARY_PANEL:
        return

    draw_transparent_panel_with_vietnamese_text(
        frame,
        20,
        20,
        [
            "ĐANG TRONG VÙNG",
            f"Người trong vùng: {current_inside_count}",
        ],
        alpha=0.45,
        font_size=24,
        panel_color=(44, 56, 82),
        text_drawer=text_drawer,
    )


def _build_violation_count_lines(title, summary_lines, violation_counts):
    lines = [title, *summary_lines]
    for label in VIOLATION_COUNT_LABELS:
        count = violation_counts.get(label, 0)
        if count > 0:
            lines.append(f"{VIOLATION_DISPLAY_NAMES[label]}: {count}")
    return lines


def draw_violation_summary(
    frame,
    current_people_count,
    current_violation_people_count,
    current_violation_counts,
    total_violation_people_count,
    total_violation_counts,
    text_drawer=None,
):
    _ = current_violation_people_count
    if not ENABLE_SUMMARY_PANEL:
        return

    font_size = 26 if frame.shape[1] >= 1400 else 22
    margin = 20

    current_lines = _build_violation_count_lines(
        "ĐANG VI PHẠM",
        [
            f"Người trong vùng: {current_people_count}",
            
        ],
        current_violation_counts,
    )
    total_lines = _build_violation_count_lines(
        "TỔNG TỪ ĐẦU VIDEO",
        [f"Tổng người từng vi phạm: {total_violation_people_count}"],
        total_violation_counts,
    )

    draw_transparent_panel_with_vietnamese_text(
        frame,
        margin,
        margin,
        current_lines,
        alpha=0.45,
        font_size=font_size,
        panel_color=(48, 64, 96),
        text_drawer=text_drawer,
    )
    draw_transparent_panel_with_vietnamese_text(
        frame,
        frame.shape[1] - margin,
        margin,
        total_lines,
        alpha=0.45,
        font_size=font_size,
        panel_color=(34, 82, 70),
        anchor="right",
        text_drawer=text_drawer,
    )


def build_debug_lines(analysis):
    analysis = analysis.get("debug_info") or analysis
    return [
        f"BEST_WRIST:{analysis.get('best_wrist', 'NONE')}",
        f"BEST_WRIST_CORRECT:{analysis.get('best_wrist_correct', 'NONE')}",
        f"BEST_WRIST_WRONG:{analysis.get('best_wrist_wrong', 'NONE')}",
        f"D:{int(analysis['dist_wrist'])}"
        if analysis.get("dist_wrist", -999) != -999
        else "D:NA",
        f"D_CORRECT:{int(analysis['dist_correct'])}"
        if analysis.get("dist_correct", -999) != -999
        else "D_CORRECT:NA",
        f"SEG_D_CORRECT:{int(analysis['seg_dist_correct'])}"
        if analysis.get("seg_dist_correct") is not None
        else "SEG_D_CORRECT:NA",
        f"T_CORRECT:{analysis['t_correct']:.2f}"
        if analysis.get("t_correct") is not None
        else "T_CORRECT:NA",
        f"D_WRONG:{int(analysis['dist_wrong'])}"
        if analysis.get("dist_wrong", -999) != -999
        else "D_WRONG:NA",
        f"SEG_D_WRONG:{int(analysis['seg_dist_wrong'])}"
        if analysis.get("seg_dist_wrong") is not None
        else "SEG_D_WRONG:NA",
        f"T_WRONG:{analysis['t_wrong']:.2f}"
        if analysis.get("t_wrong") is not None
        else "T_WRONG:NA",
        f"W_SIDE:{analysis.get('wrist_side', 'UNKNOWN')}",
        f"W_SIDE_CORRECT:{analysis.get('wrist_side_correct', 'UNKNOWN')}",
        f"W_SIDE_WRONG:{analysis.get('wrist_side_wrong', 'UNKNOWN')}",
        f"CORRECT_LINE:{analysis.get('correct_line_name', 'NONE')}",
        f"CORRECT_RULE:{analysis.get('correct_rule', 'NA')}",
        f"WRONG_LINE:{analysis.get('wrong_line_name', 'NONE')}",
        f"WRONG_RULE:{analysis.get('wrong_rule', 'NA')}",
        f"DY:{int(analysis['dy'])}" if analysis.get("dy") is not None else "DY:NA",
        f"LANE_V:{int(analysis['lane_v'])}"
        if analysis.get("lane_v") is not None
        else "LANE_V:NA",
        f"DIR:{analysis.get('direction', 'NA')}",
        f"INSIDE_STAIRS:{analysis.get('inside_stairs', False)}",
        f"LANE_RAW:{analysis.get('lane_raw', False)}",
        f"LANE_HITS:{analysis.get('lane_hits', 0)}",
        f"LANE_CONF:{analysis.get('lane_conf', False)}",
        f"BODY_FACE:{analysis.get('body_facing', 'UNKNOWN')}",
        f"BACKWARD_RAW:{analysis.get('backward_raw', False)}",
        f"BACKWARD_HITS:{analysis.get('backward_hits', 0)}",
        f"BACKWARD_CONF:{analysis.get('backward_confirmed', False)}",
        f"STAND_RAW:{analysis.get('standing_raw', False)}",
        f"STAND_HITS:{analysis.get('standing_hits', 0)}",
        f"STAND_CONF:{analysis.get('standing_still_confirmed', False)}",
        f"STAND_RANGE:{analysis['standing_motion_range']:.1f}"
        if analysis.get("standing_motion_range") is not None
        else "STAND_RANGE:NA",
        f"STAND_LEN:{analysis.get('standing_len', 0)}",
        f"ARM_ORDER:{analysis.get('arm_side_order', 'UNKNOWN')}",
        f"HOLD_CORRECT_RAW:{analysis.get('holding_correct_raw', False)}",
        f"HOLD_WRONG_RAW:{analysis.get('holding_wrong_raw', False)}",
        f"HOLD_RAW:{analysis.get('holding_raw', False)}",
        f"HOLD_RAW_STATUS:{analysis.get('hold_raw_status', 'UNKNOWN')}",
        f"HOLD:{analysis.get('holding', False)}",
        f"HOLD_FINAL_STATUS:{analysis.get('hold_status', 'UNKNOWN')}",
        f"HOLD_CORRECT_HITS:{analysis.get('hold_correct_hits', 0)}",
        f"HOLD_WRONG_HITS:{analysis.get('hold_wrong_side_hits', 0)}",
        f"HOLD_NONE_HITS:{analysis.get('hold_none_hits', 0)}",
        f"HOLD_UNKNOWN_HITS:{analysis.get('hold_unknown_hits', 0)}",
        f"HOLD_NOT_HOLD_EVIDENCE_HITS:{analysis.get('hold_not_hold_evidence_hits', 0)}",
        f"NOT_HOLD_BY_EVIDENCE:{analysis.get('not_hold_by_evidence', False)}",
        f"HOLD_CONF:{analysis.get('hold_confirmed_status', 'UNKNOWN')}",
        f"L_CLAIM:{analysis.get('left_hand_claim', 'NONE')}",
        f"R_CLAIM:{analysis.get('right_hand_claim', 'NONE')}",
        f"L_HOLD_CLAIM_HITS:{analysis.get('left_hold_claim_hits', 0)}",
        f"R_HOLD_CLAIM_HITS:{analysis.get('right_hold_claim_hits', 0)}",
        f"L_CARRY_CLAIM_HITS:{analysis.get('left_carry_claim_hits', 0)}",
        f"R_CARRY_CLAIM_HITS:{analysis.get('right_carry_claim_hits', 0)}",
        f"L_HOLD_RAW_B:{analysis.get('left_hold_raw_before_claim', False)}",
        f"R_HOLD_RAW_B:{analysis.get('right_hold_raw_before_claim', False)}",
        f"L_HOLD_RAW_A:{analysis.get('left_hold_raw_after_claim', False)}",
        f"R_HOLD_RAW_A:{analysis.get('right_hold_raw_after_claim', False)}",
        f"L_CARRY_RAW_B:{analysis.get('left_carry_raw_before_claim', False)}",
        f"R_CARRY_RAW_B:{analysis.get('right_carry_raw_before_claim', False)}",
        f"L_CARRY_RAW_A:{analysis.get('left_carry_raw_after_claim', False)}",
        f"R_CARRY_RAW_A:{analysis.get('right_carry_raw_after_claim', False)}",
        f"L_CARRY:{analysis.get('left_carry', False)}",
        f"R_CARRY:{analysis.get('right_carry', False)}",
        f"CARRY:{analysis.get('is_carrying', False)}",
        f"CARRY_TYPE:{analysis.get('carry_type', 'NONE')}",
        f"CARRY_ARM:{analysis.get('carrying_arm', 'NONE')}",
        f"L_ANG:{int(analysis['left_arm_angle'])}"
        if analysis.get("left_arm_angle") is not None
        else "L_ANG:NA",
        f"R_ANG:{int(analysis['right_arm_angle'])}"
        if analysis.get("right_arm_angle") is not None
        else "R_ANG:NA",
        f"L_TORSO:{analysis.get('left_wrist_in_torso', False)}",
        f"R_TORSO:{analysis.get('right_wrist_in_torso', False)}",
        f"BODY_SCALE:{analysis['body_scale']:.1f}"
        if analysis.get("body_scale") is not None
        else "BODY_SCALE:NA",
        f"SHOULDER_W:{analysis['shoulder_width']:.1f}"
        if analysis.get("shoulder_width") is not None
        else "SHOULDER_W:NA",
        f"TORSO_H:{analysis['torso_height']:.1f}"
        if analysis.get("torso_height") is not None
        else "TORSO_H:NA",
        f"WRIST_DX:{int(analysis['wrist_dx'])}"
        if analysis.get("wrist_dx") is not None
        else "WRIST_DX:NA",
        f"WRIST_DX_TH:{int(analysis['wrist_dx_threshold'])}"
        if analysis.get("wrist_dx_threshold") is not None
        else "WRIST_DX_TH:NA",
        f"WRIST_DY:{int(analysis['wrist_dy'])}"
        if analysis.get("wrist_dy") is not None
        else "WRIST_DY:NA",
        f"WRIST_DY_TH:{int(analysis['wrist_dy_threshold'])}"
        if analysis.get("wrist_dy_threshold") is not None
        else "WRIST_DY_TH:NA",
        f"W_DX:{int(analysis['wrist_dx'])}"
        if analysis.get("wrist_dx") is not None
        else "W_DX:NA",
        f"W_DY:{int(analysis['wrist_dy'])}"
        if analysis.get("wrist_dy") is not None
        else "W_DY:NA",
        f"W_DIST:{int(analysis['wrist_distance'])}"
        if analysis.get("wrist_distance") is not None
        else "W_DIST:NA",
        f"L_BENT:{analysis.get('left_bent', False)}",
        f"R_BENT:{analysis.get('right_bent', False)}",
        f"W_CLOSE:{analysis.get('wrists_close', False)}",
        f"BOTH_TORSO:{analysis.get('both_wrist_in_torso', False)}",
        f"FRONT:{analysis.get('front_carry', False)}",
        f"FRONT_2H_RAW:{analysis.get('front_carry_two_hand_raw', False)}",
        f"FRONT_2H_HITS:{analysis.get('front_carry_two_hand_hits', 0)}",
        f"FRONT_1H_RAW:{analysis.get('front_carry_one_arm_raw', False)}",
        f"FRONT_1H_HITS:{analysis.get('front_carry_one_arm_hits', 0)}",
        f"FRONT_RAW:{analysis.get('front_carry_raw', False)}",
        f"FRONT_HITS:{analysis.get('front_carry_hits', 0)}",
        f"FRONT_CONF:{analysis.get('front_carry_confirmed', False)}",
        f"LANE_ERR:{analysis.get('wrong_lane', False)}",
        f"P_LANE:{analysis['p_lane'][0]},{analysis['p_lane'][1]}"
        if analysis.get("p_lane") is not None
        else "P_LANE:NA",
        f"MOTION:{analysis['p_motion'][0]},{analysis['p_motion'][1]}"
        if analysis.get("p_motion") is not None
        else "MOTION:NA",
    ]
