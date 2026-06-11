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
    VIOLATION_COUNT_LABELS,
    VIOLATION_COLOR,
    VIOLATION_DISPLAY_ORDER,
    VIOLATION_DISPLAY_NAMES,
)

# Font cache de tranh moi frame lai load font tieng Viet mot lan.
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

DEMO_ALERT_PADDING_X = 28
DEMO_ALERT_PADDING_Y = 18
DEMO_ALERT_LINE_GAP = 10
DEMO_ALERT_CORNER_RADIUS = 12
DEMO_ALERT_BOTTOM_MARGIN = 30


# Chon font co the ve tieng Viet co dau tren Windows/demo.
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


# Quy doi BGR cua OpenCV sang RGB de PIL ve dung mau.
def _bgr_to_rgb(color):
    return (color[2], color[1], color[0])


# Chuyen frame qua PIL de ve text tieng Viet co dau.
def _frame_to_pil(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# Ghi noi dung PIL ve lai frame OpenCV.
def _pil_to_frame(pil_image, frame):
    frame[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


# Do kich thuoc panel truoc khi ve de canh le va tranh text bi cat.
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
    # Helper nay chi phuc vu overlay/demo.
    # Logic nhan dien khong duoc phu thuoc vao viec co ve text hay khong.
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

        # Dung PIL/ImageDraw de giu dau tieng Viet khi ve overlay.
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
            # Fallback cv2.putText khi khong can giu text co dau.
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
            # Fallback nay nhanh hon, nhung se khong giu duoc tieng Viet co dau.
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


# Ham boc de ve 1 dong text tieng Viet tren frame.
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


# Ve panel nen trong suot + text tieng Viet.
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


# Ve nhan trang thai cho tung nguoi, tu dong canh lai de khong bi cat mep frame.
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


# Ve cac guide debug cua scene: vach giua, polygon cau thang, 2 line lan can.
# Phan nay chi de quan sat/demo, khong duoc anh huong logic nhan dien.
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


# Lay lai feet_point da chuan hoa tu pose feature.
def get_feet_point(box, keypoints):
    features = extract_pose_features(keypoints, box)
    return features.get("feet_point")


# Lay lai motion_point da chuan hoa tu pose feature.
def get_motion_point(box, keypoints):
    features = extract_pose_features(keypoints, box)
    return features.get("motion_point")


# Ve overlay cho tung nguoi sau khi analyzer da tra ket qua.
# Ham nay chi hien thi demo, khong duoc can du vao logic nhan dien.
def draw_person_overlay(
    frame,
    box,
    keypoints,
    lane_point,
    motion_point,
    analysis,
    text_drawer=None,
):
    clean_demo_mode = DEMO_MODE and not ENABLE_DEBUG_OVERLAY
    if clean_demo_mode:
        # Demo mode chi giu giao dien tong hop, khong ve bat ky overlay theo tung nguoi nao.
        return

    x1, y1, x2, y2 = map(int, box)
    display_status = analysis.get("display_status", analysis.get("status", ""))
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

    # Ve 2 diem dai dien de tranh nham:
    # - lane_point cho sai lan
    # - motion_point cho direction/backward/standing
    if keypoints is not None and DRAW_KEYPOINTS:
        cv2.circle(frame, lane_point, 6, (0, 0, 255), -1)
        cv2.circle(frame, motion_point, 6, (255, 0, 0), -1)

    # Skeleton nay chi de quan sat pose tay, khong lam thay doi ket qua phan tich.
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

    # Debug overlay can flag bat/tat vi ve nhieu text se anh huong FPS demo.
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
        debug_lines = build_debug_lines(analysis)
        debug_font = cv2.FONT_HERSHEY_SIMPLEX
        debug_font_scale = 0.5
        debug_thickness = 2
        debug_line_gap = 18
        debug_top_margin = 20
        _, frame_w = frame.shape[:2]

        max_line_width = 0
        for debug_text in debug_lines:
            (line_width, _), _ = cv2.getTextSize(
                debug_text,
                debug_font,
                debug_font_scale,
                debug_thickness,
            )
            max_line_width = max(max_line_width, line_width)

        debug_x = x2 + 5
        if debug_x + max_line_width > frame_w - 5:
            debug_x = max(5, x1 - max_line_width - 5)

        debug_x = max(5, min(debug_x, max(5, frame_w - max_line_width - 5)))
        debug_base_y = debug_top_margin

        for idx, debug_text in enumerate(debug_lines):
            cv2.putText(
                frame,
                debug_text,
                (debug_x, debug_base_y + idx * debug_line_gap),
                debug_font,
                debug_font_scale,
                (0, 255, 255),
                debug_thickness,
            )


# Ve panel tong so nguoi dang nam trong vung cau thang.
def draw_people_count(frame, current_inside_count, text_drawer=None):
    if not ENABLE_SUMMARY_PANEL:
        return

    if DEMO_MODE and not ENABLE_DEBUG_OVERLAY:
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


# Tao danh sach dong text cho panel thong ke.
def _build_violation_count_lines(title, summary_lines, violation_counts):
    lines = [title, *summary_lines]
    for label in VIOLATION_COUNT_LABELS:
        count = violation_counts.get(label, 0)
        if count > 0:
            lines.append(f"{VIOLATION_DISPLAY_NAMES[label]}: {count}")
    return lines


def _build_active_violation_lines(current_violation_counts):
    return [
        VIOLATION_DISPLAY_NAMES[label]
        for label in VIOLATION_DISPLAY_ORDER
        if current_violation_counts.get(label, 0) > 0
    ]


def draw_demo_violation_panel(frame, current_violation_counts, text_drawer=None):
    lines = _build_active_violation_lines(current_violation_counts)
    if not lines:
        return

    frame_h, frame_w = frame.shape[:2]
    font_size = 60 if frame_w >= 1600 else 51 if frame_w >= 1200 else 42
    alpha = 0.9
    badge_gap = DEMO_ALERT_LINE_GAP

    # Ve rieng bang PIL/textbbox de canh dung ascent/descent cua font,
    # tranh chu co dau bi sat day hoac bi cat trong badge DEMO MODE.
    pil_image = _frame_to_pil(frame)
    draw = ImageDraw.Draw(pil_image, "RGBA")
    font = get_vietnamese_font(font_size, bold=True)

    badge_layouts = []
    total_height = 0
    for line in lines:
        line_bboxes = [draw.textbbox((0, 0), line, font=font)]
        line_heights = [bbox[3] - bbox[1] for bbox in line_bboxes]
        line_widths = [bbox[2] - bbox[0] for bbox in line_bboxes]
        panel_width = max(line_widths, default=0) + DEMO_ALERT_PADDING_X * 2
        panel_height = (
            sum(line_heights)
            + DEMO_ALERT_LINE_GAP * max(0, len(line_bboxes) - 1)
            + DEMO_ALERT_PADDING_Y * 2
        )
        badge_layouts.append(
            {
                "lines": [line],
                "line_bboxes": line_bboxes,
                "line_heights": line_heights,
                "panel_width": panel_width,
                "panel_height": panel_height,
            }
        )
        total_height += panel_height

    total_height += badge_gap * max(0, len(badge_layouts) - 1)
    start_y = int(frame_h * 0.80)
    if start_y + total_height > frame_h:
        start_y = frame_h - total_height - DEMO_ALERT_BOTTOM_MARGIN
    start_y = max(20, start_y)

    cursor_y = start_y
    for badge in badge_layouts:
        panel_width = badge["panel_width"]
        panel_height = badge["panel_height"]
        panel_x = frame_w // 2 - panel_width // 2
        panel_x = max(0, min(panel_x, max(0, frame_w - panel_width)))
        panel_y = cursor_y
        if panel_y + panel_height > frame_h:
            panel_y = max(0, frame_h - panel_height - DEMO_ALERT_BOTTOM_MARGIN)

        draw.rounded_rectangle(
            [(panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height)],
            radius=DEMO_ALERT_CORNER_RADIUS,
            fill=(*_bgr_to_rgb(VIOLATION_COLOR), int(255 * alpha)),
        )

        current_y = panel_y + DEMO_ALERT_PADDING_Y
        for line, bbox, line_height in zip(
            badge["lines"],
            badge["line_bboxes"],
            badge["line_heights"],
        ):
            text_x = panel_x + DEMO_ALERT_PADDING_X - bbox[0]
            text_y = current_y - bbox[1]
            draw.text(
                (text_x, text_y),
                line,
                font=font,
                fill=(255, 255, 255, 255),
            )
            current_y += line_height + DEMO_ALERT_LINE_GAP

        cursor_y = panel_y + panel_height + badge_gap

    _pil_to_frame(pil_image, frame)


# Ve thong ke hien tai va tong tu dau video.
# current violation = loi dang xuat hien trong frame hien tai.
# total violation = tong so track_id tung vi pham tu dau video den hien tai.
def draw_violation_summary(
    frame,
    current_people_count,
    current_violation_people_count,
    current_violation_counts,
    total_violation_people_count,
    total_violation_counts,
    demo_violation_counts=None,
    text_drawer=None,
):
    _ = current_violation_people_count
    if not ENABLE_SUMMARY_PANEL:
        return

    if DEMO_MODE and not ENABLE_DEBUG_OVERLAY:
        draw_demo_violation_panel(
            frame,
            demo_violation_counts or current_violation_counts,
            text_drawer=text_drawer,
        )
        return

    font_size = 26 if frame.shape[1] >= 1400 else 22
    margin = 20

    # Ben trai la thong ke hien tai trong frame.
    current_lines = _build_violation_count_lines(
        "ĐANG VI PHẠM",
        [
            f"Người trong vùng: {current_people_count}",
            
        ],
        current_violation_counts,
    )
    # Ben phai la thong ke tong hop tu dau video.
    # 1 track co the vi pham nhieu frame nhung trong tong hop chi tinh 1 lan moi loai loi.
    total_lines = _build_violation_count_lines(
        "TỔNG TỪ ĐẦU VIDEO",
        [f"Tổng lượt vi phạm: {total_violation_people_count}"],
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
        f"INSIDE_FINAL:{analysis.get('inside_stairs', False)}",
        f"FEET_RELIABLE:{analysis.get('feet_reliable', False)}",
        f"ANKLE_VALID_COUNT:{analysis.get('ankle_valid_count', 0)}",
        f"LEFT_ANKLE_VALID:{analysis.get('left_ankle_valid', False)}",
        f"RIGHT_ANKLE_VALID:{analysis.get('right_ankle_valid', False)}",
        f"LEFT_FOOT_IN:{analysis.get('left_foot_in', False)}",
        f"RIGHT_FOOT_IN:{analysis.get('right_foot_in', False)}",
        f"VALID_FOOT_COUNT:{analysis.get('valid_foot_count', 0)}",
        f"INSIDE_RAW_BY_FEET:{analysis.get('inside_raw_by_feet')}"
        if analysis.get("inside_raw_by_feet") is not None
        else "INSIDE_RAW_BY_FEET:NA",
        f"INSIDE_REASON:{analysis.get('inside_reason', 'UNKNOWN')}",
        f"INSIDE_GRACE_LEFT:{analysis.get('inside_grace_left', 0)}",
        f"INSIDE_FEET:{analysis['inside_feet_point'][0]},{analysis['inside_feet_point'][1]}"
        if analysis.get("inside_feet_point") is not None
        else "INSIDE_FEET:NA",
        f"LANE_RAW:{analysis.get('lane_raw', False)}",
        f"LANE_HITS:{analysis.get('lane_hits', 0)}",
        f"LANE_CONF:{analysis.get('lane_conf', False)}",
        f"LANE_DIRECTION:{analysis.get('lane_direction', 'ANALYZING')}",
        f"LANE_REASON:{analysis.get('lane_reason', 'NA')}",
        f"P_LANE_SOURCE:{analysis.get('p_lane_source', 'NONE')}",
        f"BODY_FACING:{analysis.get('body_facing', 'UNKNOWN')}",
        f"BODY_FACING_CONF:{analysis.get('body_facing_confidence', 0.0):.2f}",
        f"BODY_FACING_EVIDENCE_COUNT:{analysis.get('body_facing_evidence_count', 0)}",
        f"BODY_FACING_FRONT_VOTES:{analysis.get('body_facing_front_votes', 0)}",
        f"BODY_FACING_BACK_VOTES:{analysis.get('body_facing_back_votes', 0)}",
        f"HIP_PAIR_VALID:{analysis.get('hip_pair_valid', False)}",
        f"SHOULDER_PAIR_VALID:{analysis.get('shoulder_pair_valid', False)}",
        f"EAR_PAIR_VALID:{analysis.get('ear_pair_valid', False)}",
        f"HEAD_VALID:{analysis.get('head_valid', False)}",
        f"BACKWARD_RAW:{analysis.get('backward_raw', False)}",
        f"BACKWARD_HITS:{analysis.get('backward_hits', 0)}",
        f"BACKWARD_CONF:{analysis.get('backward_confirmed', False)}",
        f"BACKWARD_REASON:{analysis.get('backward_reason', 'UNKNOWN')}",
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
        f"P_LANE_X:{analysis['p_lane'][0]}"
        if analysis.get("p_lane") is not None
        else "P_LANE_X:NA",
        f"P_LANE_Y:{analysis['p_lane'][1]}"
        if analysis.get("p_lane") is not None
        else "P_LANE_Y:NA",
        f"P_LANE:{analysis['p_lane'][0]},{analysis['p_lane'][1]}"
        if analysis.get("p_lane") is not None
        else "P_LANE:NA",
        f"MOTION:{analysis['p_motion'][0]},{analysis['p_motion'][1]}"
        if analysis.get("p_motion") is not None
        else "MOTION:NA",
    ]
