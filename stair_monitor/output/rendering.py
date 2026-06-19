import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from stair_monitor.common.types import ColorBGR, Point, PoseFeatures
from stair_monitor.config.settings import SETTINGS
from stair_monitor.vision.geometry import extract_pose_features

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
LANE_SUPPRESSED_WARNING_LABELS = ("Khong Vin", "Vin Sai Ben")
LANE_SUPPRESSED_HOLD_DEBUG_PREFIXES = (
    "LW_VALID:",
    "RW_VALID:",
    "LW_HOLD:",
    "RW_HOLD:",
    "HANDRAIL:",
    "REASON:",
)
DEMO_ALERT_CORNER_RADIUS = 12
DEMO_ALERT_BOTTOM_MARGIN = 30


def _show_handrail_debug_only() -> bool:
    return SETTINGS.demo.handrail_debug_only


# Chon font co the ve tieng Viet co dau tren Windows/demo.
def get_vietnamese_font(font_size=28, bold=False):
    """Lay font co the ve tieng Viet co dau tren Windows/demo.

    Args:
        font_size: Co chu can ve.
        bold: Co dung font dam hay khong.

    Returns:
        ImageFont: Font PIL da duoc cache.

    Notes:
        File nay chi phuc vu rendering. Logic nhan dien khong duoc phu thuoc
        vao viec co load duoc font hay khong.
    """
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


# Do kich thuoc box text truoc khi ve de canh le va tranh text bi cat.
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
    """Helper de gom thao tac ve text/box nen tren 1 frame.

    Notes:
        Lop nay chi phuc vu overlay/demo. Toan bo logic nhan dien phai nam o
        analyzer va cac module phan tich, khong duoc phu thuoc rendering.
    """

    # Helper nay chi phuc vu overlay/demo.
    # Logic nhan dien khong duoc phu thuoc vao viec co ve text hay khong.
    def __init__(self, frame, enabled=True):
        self.frame = frame
        self.enabled = enabled and SETTINGS.demo.enable_vietnamese_text
        self.operations = []

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
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

            if op["type"] != "box":
                continue

            x = op["x"]
            y = op["y"]
            lines = op["lines"]
            alpha = op["alpha"]
            font_size = op["font_size"]
            text_color = op["text_color"]
            box_color = op["box_color"]
            padding = op["padding"]
            line_gap = op["line_gap"]
            anchor = op["anchor"]

            box_width, box_height = _measure_vietnamese_lines(
                lines,
                font_size=font_size,
                padding=padding,
                line_gap=line_gap,
            )
            frame_h, frame_w = self.frame.shape[:2]

            x1 = int(x - box_width) if anchor == "right" else int(x)
            y1 = int(y)
            x1 = max(0, min(x1, max(0, frame_w - box_width)))
            y1 = max(0, min(y1, max(0, frame_h - box_height)))
            x2 = x1 + box_width
            y2 = y1 + box_height

            draw.rounded_rectangle(
                [(x1, y1), (x2, y2)],
                radius=18,
                fill=(*_bgr_to_rgb(box_color), int(255 * alpha)),
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

    def transparent_box(
        self,
        x,
        y,
        lines,
        alpha=0.45,
        font_size=28,
        text_color=(255, 255, 255),
        box_color=(28, 36, 48),
        padding=14,
        line_gap=8,
        anchor="left",
    ):
        if not lines:
            return 0, 0

        box_width, box_height = _measure_vietnamese_lines(
            lines,
            font_size=font_size,
            padding=padding,
            line_gap=line_gap,
        )
        frame_h, frame_w = self.frame.shape[:2]

        x1 = int(x - box_width) if anchor == "right" else int(x)
        y1 = int(y)
        x1 = max(0, min(x1, max(0, frame_w - box_width)))
        y1 = max(0, min(y1, max(0, frame_h - box_height)))
        x2 = x1 + box_width
        y2 = y1 + box_height

        if not self.enabled:
            # Fallback nay nhanh hon, nhung se khong giu duoc tieng Viet co dau.
            overlay = self.frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, -1)
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
            return box_width, box_height

        self.operations.append(
            {
                "type": "box",
                "x": x,
                "y": y,
                "lines": list(lines),
                "alpha": alpha,
                "font_size": font_size,
                "text_color": text_color,
                "box_color": box_color,
                "padding": padding,
                "line_gap": line_gap,
                "anchor": anchor,
            }
        )
        return box_width, box_height


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
    """Ve 1 dong text tieng Viet len frame.

    Args:
        frame: Frame OpenCV can ve.
        text: Noi dung can ve.
        position: Vi tri goc text.
        font_size: Co chu.
        color: Mau BGR.
        bold: Co dam hay khong.
        text_drawer: Drawer dang duoc mo san neu co.

    Returns:
        None

    Notes:
        Ham boc nay chi phuc vu rendering va giu API gon cho caller.
    """
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

# Ve box nen trong suot + text tieng Viet.
def draw_transparent_text_box(
    frame,
    x,
    y,
    lines,
    alpha=0.45,
    font_size=28,
    text_color=(255, 255, 255),
    box_color=(28, 36, 48),
    padding=14,
    line_gap=8,
    anchor="left",
    text_drawer=None,
):
    """Ve box trong suot co text tieng Viet.

    Args:
        frame: Frame OpenCV can ve.
        x: Toa do x cua box.
        y: Toa do y cua box.
        lines: Danh sach dong text.
        alpha: Do trong suot cua nen.
        font_size: Co chu.
        text_color: Mau chu.
        box_color: Mau nen box.
        padding: Le trong box.
        line_gap: Khoang cach giua cac dong.
        anchor: left hoac right.
        text_drawer: Drawer dang duoc mo san neu co.

    Returns:
        tuple: (box_width, box_height)

    Notes:
        Day la helper render box text dung chung cho label va alert demo.
    """
    if not lines:
        return 0, 0

    if text_drawer is not None:
        return text_drawer.transparent_box(
            x,
            y,
            lines,
            alpha=alpha,
            font_size=font_size,
            text_color=text_color,
            box_color=box_color,
            padding=padding,
            line_gap=line_gap,
            anchor=anchor,
        )

    with VietnameseTextDrawer(frame) as drawer:
        return drawer.transparent_box(
            x,
            y,
            lines,
            alpha=alpha,
            font_size=font_size,
            text_color=text_color,
            box_color=box_color,
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
    text_color=(255, 255, 255),
    bg_color=(0, 0, 255),
    padding=8,
    text_drawer=None,
):
    """Ve label status co nen cho 1 nguoi.

    Args:
        frame: Frame OpenCV can ve.
        text: Chuoi status can ve.
        x: Toa do x.
        y: Toa do y.
        font_scale: Tile co chu.
        text_color: Mau chu.
        bg_color: Mau nen.
        padding: Le trong box text.
        text_drawer: Drawer dang duoc mo san neu co.

    Returns:
        None

    Notes:
        Ham nay chi hien thi status da duoc analyzer tong hop san.
    """
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
    draw_transparent_text_box(
        frame,
        x1,
        y1,
        [text],
        alpha=0.65,
        font_size=font_size,
        text_color=text_color,
        box_color=bg_color,
        padding=padding,
        line_gap=0,
        text_drawer=text_drawer,
    )


def _short_feet_source_label(feet_point_source: str) -> str:
    if feet_point_source == "REAL_BOTH_ANKLES":
        return "REAL"
    if feet_point_source == "REAL_LEFT_ANKLE":
        return "REAL_L"
    if feet_point_source == "REAL_RIGHT_ANKLE":
        return "REAL_R"
    if feet_point_source == "VIRTUAL_FROM_SHOULDER_HIP":
        return "SH-HIP"
    if feet_point_source == "FEET_UNAVAILABLE":
        return "LOST"
    if feet_point_source == "NO_TRUSTED_FEET":
        return "NO_TRUSTED"
    if feet_point_source == "NO_REAL_FEET":
        return "NO_REAL"
    return feet_point_source


def _format_session_lifecycle_label(session_lifecycle: str) -> str:
    if session_lifecycle.startswith("CANDIDATE_"):
        return "CANDIDATE " + session_lifecycle.removeprefix("CANDIDATE_")
    return session_lifecycle.replace("_", " ")


def _draw_feet_debug_marker(
    frame,
    point: Point | None,
    color: ColorBGR,
    label: str,
    text_drawer: VietnameseTextDrawer | None = None,
    label_dx: int = 10,
    label_dy: int = -26,
) -> None:
    if point is None:
        return

    cv2.circle(frame, point, 7, (0, 0, 0), -1)
    cv2.circle(frame, point, 5, color, -1)
    draw_transparent_text_box(
        frame,
        point[0] + label_dx,
        max(0, point[1] + label_dy),
        [label],
        alpha=0.65,
        font_size=14,
        text_color=color,
        box_color=(0, 0, 0),
        padding=6,
        line_gap=0,
        text_drawer=text_drawer,
    )


def _draw_feet_compare_line(
    frame,
    start_point: Point | None,
    end_point: Point | None,
    color: ColorBGR,
    label: str,
    text_drawer: VietnameseTextDrawer | None = None,
) -> None:
    if start_point is None or end_point is None:
        return

    cv2.line(frame, start_point, end_point, color, 2)
    label_x = int((start_point[0] + end_point[0]) / 2) + 8
    label_y = int((start_point[1] + end_point[1]) / 2) - 24
    draw_transparent_text_box(
        frame,
        label_x,
        max(0, label_y),
        [label],
        alpha=0.55,
        font_size=14,
        text_color=color,
        box_color=(0, 0, 0),
        padding=6,
        line_gap=0,
        text_drawer=text_drawer,
    )


def _format_feet_compare_line(
    prefix: str,
    dx: int | None,
    dy: int | None,
    distance: float | None,
    available: bool,
) -> str:
    if not available or dx is None or dy is None or distance is None:
        return f"{prefix}: NA"
    return f"{prefix}: dx={dx} dy={dy} d={int(round(distance))}"


def _format_virtual_scale_debug_line(
    prefix: str,
    scale: float | None,
) -> str:
    scale_text = f"{scale:.2f}" if scale is not None else "NA"
    return f"{prefix} scale={scale_text}"


def _short_direction_source_label(direction_source: str) -> str:
    if direction_source == "HIP_SHOULDER_AGREE":
        return "AGREE"
    if direction_source == "HIP_ONLY":
        return "HIP_ONLY"
    if direction_source == "SHOULDER_ONLY":
        return "SH_ONLY"
    if direction_source == "HIP_SHOULDER_CONFLICT":
        return "CONFLICT"
    if direction_source == "NO_VALID_MONITOR_DIRECTION":
        return "NONE"
    return direction_source


def _format_person_uid_label(analysis) -> str:
    person_uid_label = analysis.get("person_uid_label")
    if person_uid_label:
        return str(person_uid_label)

    person_uid = analysis.get("person_uid")
    if isinstance(person_uid, int) and person_uid > 0:
        return f"P{person_uid:04d}"

    track_id = analysis.get("track_id")
    if (
        isinstance(track_id, int)
        and track_id > 0
        and analysis.get("yolo_track_id") is None
    ):
        return f"P{track_id:04d}"
    return ""


def _format_step_debug_value(step_index) -> str:
    return str(step_index) if isinstance(step_index, int) else "?"


def _format_step_distance(distance_value) -> str:
    if isinstance(distance_value, (int, float)):
        return f"{float(distance_value):.1f}"
    return "NA"


def _format_offset_pair(offset_x, offset_y) -> str:
    if not isinstance(offset_x, (int, float)) or not isinstance(offset_y, (int, float)):
        return "(NA,NA)"
    return f"({int(offset_x):+d},{int(offset_y):+d})"


def _format_handrail_distance_value(distance_value) -> str:
    if isinstance(distance_value, (int, float)):
        return str(int(round(float(distance_value))))
    return "NA"


def _format_handrail_bool_text(value) -> str:
    return "True" if bool(value) else "False"


def _format_handrail_count_text(hit_count, confirm_required) -> str:
    hit_count_value = int(hit_count) if isinstance(hit_count, (int, float)) else 0
    confirm_required_value = (
        int(confirm_required)
        if isinstance(confirm_required, (int, float))
        else 0
    )
    return f"{hit_count_value}/{confirm_required_value}"


def _format_carry_score_value(score_value) -> str:
    if isinstance(score_value, (int, float)):
        return f"{float(score_value):.1f}"
    return "NA"


def _draw_handrail_joint_marker(frame, point, label, color) -> None:
    if point is None:
        return

    cv2.circle(frame, point, 6, (0, 0, 0), -1)
    cv2.circle(frame, point, 4, color, -1)
    cv2.putText(
        frame,
        label,
        (int(point[0]) + 6, max(14, int(point[1]) - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        2,
    )


def _draw_handrail_arm_segments(frame, features: PoseFeatures) -> None:
    left_shoulder = features.get("left_shoulder")
    left_elbow = features.get("left_elbow")
    left_wrist = features.get("left_wrist")
    right_shoulder = features.get("right_shoulder")
    right_elbow = features.get("right_elbow")
    right_wrist = features.get("right_wrist")

    if left_shoulder is not None and left_elbow is not None:
        cv2.line(frame, left_shoulder, left_elbow, (255, 255, 0), 2)
    if left_elbow is not None and left_wrist is not None:
        cv2.line(frame, left_elbow, left_wrist, (255, 255, 0), 2)
    if right_shoulder is not None and right_elbow is not None:
        cv2.line(frame, right_shoulder, right_elbow, (0, 255, 255), 2)
    if right_elbow is not None and right_wrist is not None:
        cv2.line(frame, right_elbow, right_wrist, (0, 255, 255), 2)

    _draw_handrail_joint_marker(frame, left_wrist, "LW", (0, 255, 0))
    _draw_handrail_joint_marker(frame, right_wrist, "RW", (0, 255, 0))
    _draw_handrail_joint_marker(frame, left_elbow, "LE", (255, 255, 0))
    _draw_handrail_joint_marker(frame, right_elbow, "RE", (0, 255, 255))
    _draw_handrail_joint_marker(frame, left_shoulder, "LS", (255, 200, 120))
    _draw_handrail_joint_marker(frame, right_shoulder, "RS", (120, 200, 255))


def _draw_handrail_distance_line(
    frame,
    wrist_point: Point | None,
    nearest_point: Point | None,
    distance_value,
    hit: bool,
    wrist_label: str,
) -> None:
    if wrist_point is None or nearest_point is None:
        return

    color = (0, 255, 0) if hit else (0, 0, 255)
    cv2.line(frame, wrist_point, nearest_point, color, 2)
    cv2.circle(frame, nearest_point, 5, color, -1)

    if not SETTINGS.demo.show_handrail_distance_text:
        return

    mid_x = int((wrist_point[0] + nearest_point[0]) / 2)
    mid_y = int((wrist_point[1] + nearest_point[1]) / 2) - 6
    cv2.putText(
        frame,
        f"{wrist_label} d={_format_handrail_distance_value(distance_value)}",
        (mid_x, max(16, mid_y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        2,
    )


def _build_handrail_debug_only_lines(analysis) -> list[str]:
    return [
        (
            "LW d="
            + _format_handrail_distance_value(
                analysis.get("left_wrist_nearest_distance")
            )
            + " HIT="
            + ("1" if analysis.get("left_wrist_hit", False) else "0")
            + " CNT="
            + _format_handrail_count_text(
                analysis.get("left_wrist_hit_count"),
                analysis.get("left_wrist_confirm_required"),
            )
            + " MISS="
            + str(int(analysis.get("left_wrist_miss_count", 0)))
            + " HOLD="
            + ("1" if analysis.get("left_holding", False) else "0")
        ),
        (
            "RW d="
            + _format_handrail_distance_value(
                analysis.get("right_wrist_nearest_distance")
            )
            + " HIT="
            + ("1" if analysis.get("right_wrist_hit", False) else "0")
            + " CNT="
            + _format_handrail_count_text(
                analysis.get("right_wrist_hit_count"),
                analysis.get("right_wrist_confirm_required"),
            )
            + " MISS="
            + str(int(analysis.get("right_wrist_miss_count", 0)))
            + " HOLD="
            + ("1" if analysis.get("right_holding", False) else "0")
        ),
        f"HANDRAIL={analysis.get('handrail_status', 'NOT_EVALUATED')}",
        (
            "CARRY_ALLOW L="
            + ("1" if analysis.get("left_carry_allowed", True) else "0")
            + " R="
            + ("1" if analysis.get("right_carry_allowed", True) else "0")
        ),
        (
            "CARRY_SCORE L="
            + _format_carry_score_value(analysis.get("left_carry_score"))
            + " R="
            + _format_carry_score_value(analysis.get("right_carry_score"))
        ),
        (
            "1H_SIDE="
            + str(analysis.get("one_hand_carry_side", "NONE"))
            + " CARRY="
            + ("1" if analysis.get("is_carrying", False) else "0")
        ),
        "CARRY_REASON=" + str(analysis.get("carry_reason", "NO_CARRY")),
        "REASON="
        + str(
            analysis.get(
                "handrail_debug_reason",
                analysis.get("handrail_reason", "NOT_EVALUATED"),
            )
        ),
    ]


def _draw_handrail_debug_only_overlay(
    frame,
    box,
    features: PoseFeatures,
    analysis,
    text_drawer=None,
) -> None:
    x1, _y1, _x2, y2 = map(int, box)
    _draw_handrail_arm_segments(frame, features)

    left_wrist = features.get("left_wrist")
    right_wrist = features.get("right_wrist")
    left_hit = bool(analysis.get("left_wrist_hit", False))
    right_hit = bool(analysis.get("right_wrist_hit", False))

    if SETTINGS.demo.show_handrail_distance_lines:
        _draw_handrail_distance_line(
            frame,
            left_wrist,
            analysis.get("left_wrist_nearest_point"),
            analysis.get("left_wrist_nearest_distance"),
            left_hit,
            "LW",
        )
        _draw_handrail_distance_line(
            frame,
            right_wrist,
            analysis.get("right_wrist_nearest_point"),
            analysis.get("right_wrist_nearest_distance"),
            right_hit,
            "RW",
        )

    if SETTINGS.demo.show_handrail_distance_text:
        if left_wrist is not None:
            cv2.putText(
                frame,
                (
                    "LW HIT "
                    + _format_handrail_count_text(
                        analysis.get("left_wrist_hit_count"),
                        analysis.get("left_wrist_confirm_required"),
                    )
                    if left_hit
                    else "LW MISS "
                    + _format_handrail_count_text(
                        analysis.get("left_wrist_hit_count"),
                        analysis.get("left_wrist_confirm_required"),
                    )
                ),
                (int(left_wrist[0]) + 8, int(left_wrist[1]) + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0) if left_hit else (0, 0, 255),
                2,
            )
        if right_wrist is not None:
            cv2.putText(
                frame,
                (
                    "RW HIT "
                    + _format_handrail_count_text(
                        analysis.get("right_wrist_hit_count"),
                        analysis.get("right_wrist_confirm_required"),
                    )
                    if right_hit
                    else "RW MISS "
                    + _format_handrail_count_text(
                        analysis.get("right_wrist_hit_count"),
                        analysis.get("right_wrist_confirm_required"),
                    )
                ),
                (int(right_wrist[0]) + 8, int(right_wrist[1]) + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0) if right_hit else (0, 0, 255),
                2,
            )

    draw_transparent_text_box(
        frame,
        x1,
        y2 + 42,
        _build_handrail_debug_only_lines(analysis),
        alpha=0.45,
        font_size=13,
        text_color=(255, 255, 255),
        box_color=(0, 0, 0),
        padding=8,
        line_gap=4,
        text_drawer=text_drawer,
    )


def _build_step_compare_overlay_line(
    side_label: str,
    raw_step_index,
    filtered_step_index,
    step_filter_reason: str,
    ankle_conf,
) -> str:
    reason_text = step_filter_reason or "NOT_EVALUATED"
    conf_text = (
        f"{float(ankle_conf):.2f}"
        if isinstance(ankle_conf, (int, float))
        else "NA"
    )
    return (
        f"{side_label}_RAW_STEP={_format_step_debug_value(raw_step_index)} "
        f"{side_label}_FILT_STEP={_format_step_debug_value(filtered_step_index)} "
        f"{side_label}_REASON={reason_text} "
        f"{side_label}_CONF={conf_text}"
    )


def _build_step_nearest_overlay_line(
    side_label: str,
    nearest_band_id,
    nearest_distance,
) -> str:
    nearest_band_text = str(nearest_band_id) if isinstance(nearest_band_id, int) else "?"
    return (
        f"{side_label}_NEAR={nearest_band_text} "
        f"{side_label}_DIST={_format_step_distance(nearest_distance)}"
    )


def _build_identity_overlay_lines(analysis) -> list[str]:
    person_uid_label = _format_person_uid_label(analysis)
    yolo_track_id = analysis.get("yolo_track_id")
    session_lifecycle = str(analysis.get("session_lifecycle", "UNKNOWN"))
    session_lifecycle_label = _format_session_lifecycle_label(session_lifecycle)
    feet_source = _short_feet_source_label(
        str(analysis.get("identity_feet_source", "FEET_UNAVAILABLE"))
    )
    feet_reason = str(analysis.get("identity_feet_reason", "NONE"))
    gate_reason = str(analysis.get("identity_gate_reason", "NA"))
    inside_test = str(analysis.get("identity_inside_test", "UNKNOWN"))
    outside_proof = bool(analysis.get("identity_outside_proof", False))
    enter_confirm_hits = int(analysis.get("identity_enter_confirm_hits", 0) or 0)
    enter_confirm_target = int(
        analysis.get("identity_enter_confirm_target", 0) or 0
    )

    if person_uid_label and session_lifecycle == "EXITED":
        title_line = person_uid_label
    elif person_uid_label and yolo_track_id is not None:
        title_line = f"{person_uid_label} / YOLO {yolo_track_id}"
    elif person_uid_label:
        title_line = f"{person_uid_label} / YOLO lost"
    elif analysis.get("analysis_subject_label"):
        title_line = str(analysis.get("analysis_subject_label"))
    elif yolo_track_id is not None:
        title_line = f"YOLO {yolo_track_id}"
    else:
        title_line = "YOLO NA"

    lines = [
        title_line,
        session_lifecycle_label,
        f"FEET_SRC={feet_source}",
        f"FEET_REASON={feet_reason}",
        f"INSIDE_TEST={inside_test}",
        f"OUTSIDE_PROOF={'YES' if outside_proof else 'NO'}",
    ]
    if enter_confirm_target > 0 and session_lifecycle in (
        "CANDIDATE_WAIT_ENTER",
        "CANDIDATE_INSIDE_NO_OUTSIDE_PROOF",
        "UNASSIGNED_INSIDE_CANDIDATE",
    ):
        lines.append(f"ENTER_CNT={enter_confirm_hits}/{enter_confirm_target}")
    if gate_reason not in ("", "NA", "NONE") and gate_reason != feet_reason:
        lines.append(f"GATE={gate_reason}")
    relink_state = str(analysis.get("relink_state", "NONE"))
    relink_best = str(analysis.get("relink_best_candidate", ""))
    relink_second = str(analysis.get("relink_second_candidate", ""))
    relink_score = analysis.get("relink_score")
    relink_gap = analysis.get("relink_score_gap")
    relink_frame_gap = analysis.get("relink_frame_gap", 0)
    score_text = (
        f"{float(relink_score):.2f}"
        if isinstance(relink_score, (int, float))
        else "NA"
    )
    gap_text = (
        f"{float(relink_gap):.2f}"
        if isinstance(relink_gap, (int, float))
        else "NA"
    )
    if analysis.get("identity_status") == "RELINKED":
        lines.append("RELINK_OK")
        lines.append(
            f"{relink_best or person_uid_label or 'NA'} <- YOLO "
            f"{yolo_track_id if yolo_track_id is not None else 'NA'}"
        )
        lines.append(
            f"score={score_text} gap={gap_text} lost={int(relink_frame_gap)}"
        )
    elif relink_state == "RELINK_WAIT_MORE_FRAMES":
        lines.append("REID_WAIT")
        lines.append(f"best={relink_best or 'NA'} score={score_text}")
        lines.append(
            f"second={relink_second or 'NA'} gap={gap_text} lost={int(relink_frame_gap)}"
        )
    elif relink_state == "RELINK_REJECTED_AMBIGUOUS":
        lines.append("AMBIGUOUS")
        lines.append(f"best={relink_best or 'NA'} score={score_text}")
        lines.append(
            f"second={relink_second or 'NA'} gap={gap_text} lost={int(relink_frame_gap)}"
        )
    elif relink_state == "RELINK_REJECTED_ORDER_CONFLICT":
        lines.append("ORDER_CONFLICT")
        lines.append(f"best={relink_best or 'NA'} score={score_text}")
    lines.append(
        "CNT "
        f"E={analysis.get('total_entered_count', analysis.get('entered_count', 0))} "
        f"X={analysis.get('total_exited_count', analysis.get('exited_count', 0))} "
        f"IN={analysis.get('active_or_lost_inside_count', 0)}"
    )
    left_raw_step = analysis.get("left_raw_step")
    right_raw_step = analysis.get("right_raw_step")
    left_filtered_step = analysis.get("left_filtered_step")
    right_filtered_step = analysis.get("right_filtered_step")
    left_last_valid_step = analysis.get("left_last_valid_step")
    right_last_valid_step = analysis.get("right_last_valid_step")
    left_planted_step = analysis.get("left_planted_step")
    right_planted_step = analysis.get("right_planted_step")
    foot_gap = analysis.get("foot_gap", analysis.get("step_gap"))
    planted_gap = analysis.get("planted_step_gap")
    offset_x = analysis.get("ankle_step_offset_x")
    offset_y = analysis.get("ankle_step_offset_y")
    offset_direction = str(analysis.get("ankle_step_offset_direction", "UNKNOWN"))
    left_planted_text = (
        str(left_planted_step) if isinstance(left_planted_step, int) else "?"
    )
    right_planted_text = (
        str(right_planted_step) if isinstance(right_planted_step, int) else "?"
    )
    foot_gap_text = str(foot_gap) if isinstance(foot_gap, int) else "?"
    planted_gap_text = str(planted_gap) if isinstance(planted_gap, int) else "?"
    lines.append(
        "DIR="
        f"{analysis.get('direction', 'UNKNOWN')} "
        "OFFSET="
        f"{_format_offset_pair(offset_x, offset_y)}"
    )
    lines.append(f"ANK_OFFSET_DIR={offset_direction}")
    if SETTINGS.step_band.debug_show_reason:
        lines.append(
            _build_step_compare_overlay_line(
                "L",
                left_raw_step,
                left_filtered_step,
                str(analysis.get("left_step_filter_reason", "NOT_EVALUATED")),
                analysis.get("left_ankle_conf"),
            )
        )
        if str(analysis.get("left_step_reason", "NOT_EVALUATED")) != "OK":
            lines.append(
                _build_step_nearest_overlay_line(
                    "L",
                    analysis.get("left_step_nearest_band_id"),
                    analysis.get("left_step_nearest_distance"),
                )
            )
        lines.append(
            _build_step_compare_overlay_line(
                "R",
                right_raw_step,
                right_filtered_step,
                str(analysis.get("right_step_filter_reason", "NOT_EVALUATED")),
                analysis.get("right_ankle_conf"),
            )
        )
        if str(analysis.get("right_step_reason", "NOT_EVALUATED")) != "OK":
            lines.append(
                _build_step_nearest_overlay_line(
                    "R",
                    analysis.get("right_step_nearest_band_id"),
                    analysis.get("right_step_nearest_distance"),
                )
            )
        lines.append(
            "LAST_VALID "
            f"L={_format_step_debug_value(left_last_valid_step)} "
            f"R={_format_step_debug_value(right_last_valid_step)}"
        )
        lines.append(f"FOOT_GAP={foot_gap_text}")
    else:
        lines.append(
            "L_RAW_STEP="
            f"{_format_step_debug_value(left_raw_step)} "
            "L_FILT_STEP="
            f"{_format_step_debug_value(left_filtered_step)}"
        )
        lines.append(
            "R_RAW_STEP="
            f"{_format_step_debug_value(right_raw_step)} "
            "R_FILT_STEP="
            f"{_format_step_debug_value(right_filtered_step)} "
            "FOOT_GAP="
            f"{foot_gap_text}"
        )
        lines.append(
            "LAST_VALID "
            f"L={_format_step_debug_value(left_last_valid_step)} "
            f"R={_format_step_debug_value(right_last_valid_step)}"
        )
    lines.append(
        "L_PLANT="
        f"{left_planted_text} "
        "R_PLANT="
        f"{right_planted_text} "
        "PLANT_GAP="
        f"{planted_gap_text}"
    )
    lines.append(
        "L_STATE="
        f"{analysis.get('left_foot_state_label', 'WAIT')} "
        "R_STATE="
        f"{analysis.get('right_foot_state_label', 'WAIT')}"
    )
    if bool(analysis.get("two_step_skip_confirmed", False)):
        lines.append("BUOC 2 BAC")
    else:
        lines.append(str(analysis.get("two_step_skip_reason", "NOT_EVALUATED")))
    return lines


def draw_feet_comparison_debug(
    frame,
    box,
    features: PoseFeatures,
    analysis=None,
    text_drawer: VietnameseTextDrawer | None = None,
) -> None:
    """Ve raw ankle va adjusted step point cho logic two-step debug."""
    real_feet_point = features.get("real_feet_point")
    virtual_shoulder_hip_feet = features.get("virtual_feet_from_shoulder_hip")
    selected_feet_point = features.get("selected_feet_point")
    selected_feet_source = str(features.get("feet_point_source", "FEET_UNAVAILABLE"))
    feet_unavailable_reason = str(
        features.get("feet_unavailable_reason", "NO_SHOULDER_NO_HIP")
    )
    sh_hip_selected_pair = str(features.get("sh_hip_selected_pair", "NONE"))
    sh_hip_virtual_feet_source = str(
        features.get("sh_hip_virtual_feet_source", "FEET_UNAVAILABLE")
    )
    sh_hip_visible_shoulder_count = int(
        features.get("sh_hip_visible_shoulder_count", 0) or 0
    )
    sh_hip_visible_hip_count = int(features.get("sh_hip_visible_hip_count", 0) or 0)
    sh_hip_anchor_shoulder_point = features.get("sh_hip_anchor_shoulder_point")
    sh_hip_anchor_hip_point = features.get("sh_hip_anchor_hip_point")
    left_ankle_point = (
        analysis.get("left_ankle_raw_point")
        if analysis is not None
        else features.get("left_ankle")
    )
    right_ankle_point = (
        analysis.get("right_ankle_raw_point")
        if analysis is not None
        else features.get("right_ankle")
    )
    left_step_point = analysis.get("left_ankle_step_point") if analysis is not None else None
    right_step_point = (
        analysis.get("right_ankle_step_point") if analysis is not None else None
    )

    _draw_feet_debug_marker(
        frame,
        real_feet_point,
        (60, 220, 60),
        "REAL",
        text_drawer=text_drawer,
        label_dx=12,
        label_dy=-12,
    )
    _draw_feet_debug_marker(
        frame,
        virtual_shoulder_hip_feet,
        (0, 255, 255),
        "V-SH-HIP",
        text_drawer=text_drawer,
        label_dx=12,
        label_dy=16,
    )
    _draw_feet_debug_marker(
        frame,
        selected_feet_point,
        (255, 255, 255),
        "SELECTED",
        text_drawer=text_drawer,
        label_dx=12,
        label_dy=-34,
    )
    _draw_feet_debug_marker(
        frame,
        sh_hip_anchor_shoulder_point,
        (255, 200, 0),
        "SH-HIP_SH",
        text_drawer=text_drawer,
        label_dx=12,
        label_dy=-18,
    )
    _draw_feet_debug_marker(
        frame,
        sh_hip_anchor_hip_point,
        (0, 200, 255),
        "SH-HIP_HIP",
        text_drawer=text_drawer,
        label_dx=12,
        label_dy=18,
    )
    if sh_hip_anchor_shoulder_point is not None and sh_hip_anchor_hip_point is not None:
        cv2.line(
            frame,
            sh_hip_anchor_shoulder_point,
            sh_hip_anchor_hip_point,
            (0, 180, 220),
            1,
        )
    if sh_hip_anchor_hip_point is not None and virtual_shoulder_hip_feet is not None:
        cv2.line(
            frame,
            sh_hip_anchor_hip_point,
            virtual_shoulder_hip_feet,
            (0, 220, 220),
            1,
        )

    _draw_feet_debug_marker(
        frame,
        left_ankle_point,
        (0, 255, 0),
        "L_ANK_RAW",
        text_drawer=text_drawer,
        label_dx=10,
        label_dy=-28,
    )
    _draw_feet_debug_marker(
        frame,
        right_ankle_point,
        (0, 165, 255),
        "R_ANK_RAW",
        text_drawer=text_drawer,
        label_dx=10,
        label_dy=10,
    )
    if left_ankle_point is not None and left_step_point is not None:
        cv2.line(frame, left_ankle_point, left_step_point, (255, 255, 0), 2)
    if right_ankle_point is not None and right_step_point is not None:
        cv2.line(frame, right_ankle_point, right_step_point, (255, 0, 255), 2)
    _draw_feet_debug_marker(
        frame,
        left_step_point,
        (255, 255, 0),
        "L_STEP_PT",
        text_drawer=text_drawer,
        label_dx=12,
        label_dy=-4,
    )
    _draw_feet_debug_marker(
        frame,
        right_step_point,
        (255, 0, 255),
        "R_STEP_PT",
        text_drawer=text_drawer,
        label_dx=12,
        label_dy=18,
    )
    x1, y1, _x2, _y2 = map(int, box)
    debug_lines = [
        (
            f"L_ANK_RAW={left_ankle_point[0]},{left_ankle_point[1]}"
            if left_ankle_point is not None
            else "L_ANK_RAW=NA"
        ),
        (
            f"R_ANK_RAW={right_ankle_point[0]},{right_ankle_point[1]}"
            if right_ankle_point is not None
            else "R_ANK_RAW=NA"
        ),
        (
            f"L_STEP_PT={left_step_point[0]},{left_step_point[1]}"
            if left_step_point is not None
            else "L_STEP_PT=NA"
        ),
        (
            f"R_STEP_PT={right_step_point[0]},{right_step_point[1]}"
            if right_step_point is not None
            else "R_STEP_PT=NA"
        ),
        (
            "OFF_DIR="
            + str(analysis.get("ankle_step_offset_direction", "UNKNOWN"))
            if analysis is not None
            else "OFF_DIR=UNKNOWN"
        ),
        (
            "OFFSET="
            + _format_offset_pair(
                analysis.get("ankle_step_offset_x") if analysis is not None else None,
                analysis.get("ankle_step_offset_y") if analysis is not None else None,
            )
            if analysis is not None
            else "OFFSET=(NA,NA)"
        ),
        (
            f"REAL_FEET={real_feet_point[0]},{real_feet_point[1]}"
            if real_feet_point is not None
            else "REAL_FEET=NA"
        ),
        (
            f"V_SH_HIP={virtual_shoulder_hip_feet[0]},{virtual_shoulder_hip_feet[1]}"
            if virtual_shoulder_hip_feet is not None
            else "V_SH_HIP=NA"
        ),
        (
            f"SELECTED_FEET={selected_feet_point[0]},{selected_feet_point[1]}"
            if selected_feet_point is not None
            else "SELECTED_FEET=NA"
        ),
        f"SH_HIP_SRC={sh_hip_virtual_feet_source}",
        f"SH_HIP_PAIR={sh_hip_selected_pair}",
        (
            "SH_HIP_COUNTS="
            f"S{sh_hip_visible_shoulder_count}/H{sh_hip_visible_hip_count}"
        ),
    ]
    if selected_feet_source == "VIRTUAL_FROM_SHOULDER_HIP":
        debug_lines.extend(
            [
                "FEET=SH-HIP",
                f"SRC={sh_hip_selected_pair}",
            ]
        )
    elif selected_feet_source.startswith("REAL_"):
        debug_lines.extend(
            [
                "FEET=REAL",
                f"SRC={selected_feet_source}",
            ]
        )
    else:
        debug_lines.extend(
            [
                "FEET=LOST",
                f"REASON={feet_unavailable_reason}",
            ]
        )
    debug_box_y = max(0, y1 - (len(debug_lines) * 22 + 8))
    draw_transparent_text_box(
        frame,
        x1 + 8,
        debug_box_y,
        debug_lines,
        alpha=0.55,
        font_size=14,
        text_color=(255, 255, 255),
        box_color=(0, 0, 0),
        padding=7,
        line_gap=4,
        text_drawer=text_drawer,
    )

# Ve cac guide debug cua scene: vach giua, polygon cau thang, 2 line lan can.
# Phan nay chi de quan sat/demo, khong duoc anh huong logic nhan dien.
def draw_scene_guides(frame, config, analyzer):
    handrail_debug_only = _show_handrail_debug_only()
    if "CENTER_LINE" in config and not handrail_debug_only:
        cv2.line(
            frame,
            tuple(config["CENTER_LINE"][0]),
            tuple(config["CENTER_LINE"][1]),
            (0, 255, 255),
            2,
        )
    if len(analyzer.stairs_poly) > 2 and not handrail_debug_only:
        cv2.polylines(
            frame, [analyzer.stairs_poly.reshape((-1, 1, 2))], True, (255, 0, 0), 2
        )
    if SETTINGS.step_band.debug_show_bands and not handrail_debug_only:
        for step_band in getattr(analyzer, "step_bands", []):
            band_contour = np.asarray(step_band.polygon, dtype=np.int32).reshape(
                (-1, 1, 2)
            )
            cv2.polylines(frame, [band_contour], True, (80, 200, 120), 1)

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
    features: PoseFeatures | None = None,
    text_drawer=None,
):
    """Ve overlay cho 1 nguoi tu result dict da co san.

    Args:
        frame: Frame OpenCV can ve.
        box: BBox cua nguoi.
        keypoints: Mang keypoint YOLO pose.
        lane_point: Diem p_lane da chuan hoa.
        motion_point: Diem p_motion da chuan hoa.
        analysis: Result dict do analyzer tra ve.
        text_drawer: Drawer dang duoc mo san neu co.

    Returns:
        None

    Notes:
        File nay chi ve overlay. Moi logic nhan dien phai duoc tinh xong o
        analyzer truoc khi ham nay duoc goi.
    """
    clean_demo_mode = (
        SETTINGS.demo.demo_mode and not SETTINGS.demo.draw_debug
    )
    if clean_demo_mode:
        # Demo mode chi giu giao dien tong hop, khong ve bat ky overlay theo tung nguoi nao.
        return
    handrail_debug_only = _show_handrail_debug_only()

    x1, y1, x2, y2 = map(int, box)
    display_status = analysis.get("display_status", analysis.get("status", ""))
    cv2.rectangle(frame, (x1, y1), (x2, y2), analysis["color"], 2)

    if display_status and not handrail_debug_only:
        draw_label_with_background(
            frame,
            display_status,
            x1,
            y2 + 38,
            font_scale=0.75,
            text_color=(255, 255, 255),
            bg_color=analysis["color"],
            padding=8,
            text_drawer=text_drawer,
        )

    if SETTINGS.demo.draw_debug and not handrail_debug_only:
        identity_lines = _build_identity_overlay_lines(analysis)
        identity_box_width, _identity_box_height = _measure_vietnamese_lines(
            identity_lines,
            font_size=13,
            padding=10,
            line_gap=4,
        )
        frame_h, frame_w = frame.shape[:2]
        identity_box_y = max(0, min(y1 + 8, max(0, frame_h - 20)))
        identity_side_margin = 16
        left_space = x1
        right_space = frame_w - x2

        if left_space >= identity_box_width + identity_side_margin:
            identity_box_x = x1 - identity_side_margin
            identity_box_anchor = "right"
        elif right_space >= identity_box_width + identity_side_margin:
            identity_box_x = x2 + identity_side_margin
            identity_box_anchor = "left"
        elif right_space >= left_space:
            identity_box_x = min(frame_w - 4, x2 + identity_side_margin)
            identity_box_anchor = "left"
        else:
            identity_box_x = max(identity_box_width + 4, x1 - identity_side_margin)
            identity_box_anchor = "right"

        draw_transparent_text_box(
            frame,
            identity_box_x,
            identity_box_y,
            identity_lines,
            alpha=0.55,
            font_size=13,
            text_color=(255, 255, 255),
            box_color=(24, 24, 24),
            padding=10,
            line_gap=4,
            anchor=identity_box_anchor,
            text_drawer=text_drawer,
        )

    if SETTINGS.demo.draw_debug and features is not None and not handrail_debug_only:
        draw_feet_comparison_debug(
            frame,
            box,
            features,
            analysis=analysis,
            text_drawer=text_drawer,
        )

    if handrail_debug_only and features is not None:
        _draw_handrail_debug_only_overlay(
            frame,
            box,
            features,
            analysis,
            text_drawer=text_drawer,
        )
    # Skeleton nay chi de quan sat pose tay, khong lam thay doi ket qua phan tich.
    elif keypoints is not None and SETTINGS.demo.draw_skeleton:
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

    # Debug overlay can flag bat/tat vi ve nhieu text se anh huong FPS demo.
    if (
        SETTINGS.demo.enable_debug_overlay
        and SETTINGS.demo.enable_verbose_person_debug
        and not handrail_debug_only
        and analysis.get(
        "carry_type",
        "NONE",
        ) != "NONE"
    ):
        cv2.putText(
            frame,
            f"CARRY:{analysis['carry_type']}",
            (x1, y2 + 47),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    if (
        SETTINGS.demo.enable_debug_overlay
        and SETTINGS.demo.enable_verbose_person_debug
        and not handrail_debug_only
    ):
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



def _build_active_violation_lines(active_alert_flags):
    return [
        SETTINGS.violation.display_names[label]
        for label in SETTINGS.violation.display_order
        if active_alert_flags.get(label, 0) > 0
    ]


def draw_demo_violation_alerts(frame, active_alert_flags):
    """Ve badge vi pham lon o demo mode sach.

    Args:
        frame: Frame OpenCV can ve.
        active_alert_flags: Tap co/dang giu alert cho tung loai vi pham.

    Returns:
        None

    Notes:
        Alert nay giu giao dien demo gon trong demo mode khi tat debug overlay.
    """
    lines = _build_active_violation_lines(active_alert_flags)
    if not lines:
        return

    frame_h, frame_w = frame.shape[:2]
    font_size = 80 if frame_w >= 1600 else 51 if frame_w >= 1200 else 42
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
        badge_width = max(line_widths, default=0) + DEMO_ALERT_PADDING_X * 2
        badge_height = (
            sum(line_heights)
            + DEMO_ALERT_LINE_GAP * max(0, len(line_bboxes) - 1)
            + DEMO_ALERT_PADDING_Y * 2
        )
        badge_layouts.append(
            {
                "lines": [line],
                "line_bboxes": line_bboxes,
                "line_heights": line_heights,
                "badge_width": badge_width,
                "badge_height": badge_height,
            }
        )
        total_height += badge_height

    total_height += badge_gap * max(0, len(badge_layouts) - 1)
    start_y = int(frame_h * 0.80)
    if start_y + total_height > frame_h:
        start_y = frame_h - total_height - DEMO_ALERT_BOTTOM_MARGIN
    start_y = max(20, start_y)

    cursor_y = start_y
    for badge in badge_layouts:
        badge_width = badge["badge_width"]
        badge_height = badge["badge_height"]
        badge_x = frame_w // 2 - badge_width // 2
        badge_x = max(0, min(badge_x, max(0, frame_w - badge_width)))
        badge_y = cursor_y
        if badge_y + badge_height > frame_h:
            badge_y = max(0, frame_h - badge_height - DEMO_ALERT_BOTTOM_MARGIN)

        draw.rounded_rectangle(
            [(badge_x, badge_y), (badge_x + badge_width, badge_y + badge_height)],
            radius=DEMO_ALERT_CORNER_RADIUS,
            fill=(
                *_bgr_to_rgb(SETTINGS.violation.violation_color),
                int(255 * alpha),
            ),
        )

        current_y = badge_y + DEMO_ALERT_PADDING_Y
        for line, bbox, line_height in zip(
            badge["lines"],
            badge["line_bboxes"],
            badge["line_heights"],
        ):
            text_x = badge_x + DEMO_ALERT_PADDING_X - bbox[0]
            text_y = current_y - bbox[1]
            draw.text(
                (text_x, text_y),
                line,
                font=font,
                fill=(255, 255, 255, 255),
            )
            current_y += line_height + DEMO_ALERT_LINE_GAP

        cursor_y = badge_y + badge_height + badge_gap

    _pil_to_frame(pil_image, frame)



def build_debug_lines(analysis):
    """Xay danh sach dong debug tu result dict cua 1 nguoi.

    Args:
        analysis: Result dict hoac debug_info dict.

    Returns:
        list[str]: Cac dong text debug cho overlay.

    Notes:
        Neu co Sai Lan thi mot so dong hold se bi suppress trong rendering de
        debug block gon hon. Logic suppress warning that nam o result_builder.
    """
    analysis = analysis.get("debug_info") or analysis
    person_uid_label = _format_person_uid_label(analysis) or "NA"
    debug_lines = [
        f"PERSON_UID:{person_uid_label}",
        f"YOLO_TRACK_ID:{analysis.get('yolo_track_id', 'NA')}",
        f"PREV_YOLO_TRACK_ID:{analysis.get('previous_yolo_track_id', 'NA')}",
        f"IDENTITY_STATUS:{analysis.get('identity_status', 'ACTIVE')}",
        f"HAS_ACTIVE_PERSON_ID:{analysis.get('has_active_person_id', False)}",
        f"SESSION_STATUS:{analysis.get('session_status', 'ACTIVE')}",
        f"SESSION_LIFECYCLE:{analysis.get('session_lifecycle', 'UNKNOWN')}",
        f"IDENTITY_FEET_SOURCE:{analysis.get('identity_feet_source', 'FEET_UNAVAILABLE')}",
        f"IDENTITY_FEET_REASON:{analysis.get('identity_feet_reason', 'NONE')}",
        f"IDENTITY_GATE_REASON:{analysis.get('identity_gate_reason', 'NA')}",
        f"IDENTITY_INSIDE_TEST:{analysis.get('identity_inside_test', 'UNKNOWN')}",
        f"IDENTITY_ENTRY_REASON:{analysis.get('identity_entry_reason', 'NONE')}",
        f"IDENTITY_OUTSIDE_PROOF:{analysis.get('identity_outside_proof', False)}",
        f"IDENTITY_ENTER_CONFIRM_HITS:{analysis.get('identity_enter_confirm_hits', 0)}",
        f"IDENTITY_ENTER_CONFIRM_TARGET:{analysis.get('identity_enter_confirm_target', 0)}",
        f"IDENTITY_DEBUG:{analysis.get('identity_debug', 'NA')}",
        f"RELINK_SCORE:{analysis['relink_score']:.2f}"
        if isinstance(analysis.get("relink_score"), (int, float))
        else "RELINK_SCORE:NA",
        f"RELINK_FRAME_GAP:{analysis.get('relink_frame_gap', 0)}",
        f"RELINK_SCORE_GAP:{analysis['relink_score_gap']:.2f}"
        if isinstance(analysis.get("relink_score_gap"), (int, float))
        else "RELINK_SCORE_GAP:NA",
        f"RELINK_BEST_CANDIDATE:{analysis.get('relink_best_candidate', 'NA') or 'NA'}",
        f"RELINK_SECOND_CANDIDATE:{analysis.get('relink_second_candidate', 'NA') or 'NA'}",
        f"RELINK_STATE:{analysis.get('relink_state', 'NONE')}",
        f"CURRENT_PERSON_ID:{analysis.get('current_person_id', 'NA') or 'NA'}",
        f"PERSON_LIFECYCLE_STATE:{analysis.get('person_lifecycle_state', 'UNKNOWN')}",
        f"HAS_COUNTED_ENTER:{analysis.get('has_counted_enter', False)}",
        f"HAS_COUNTED_EXIT:{analysis.get('has_counted_exit', False)}",
        f"COUNT_EVENT_REASON:{analysis.get('count_event_reason', 'NO_COUNT_EVENT')}",
        f"TOTAL_ENTERED_COUNT:{analysis.get('total_entered_count', analysis.get('entered_count', 0))}",
        f"TOTAL_EXITED_COUNT:{analysis.get('total_exited_count', analysis.get('exited_count', 0))}",
        f"ENTERED_COUNT:{analysis.get('entered_count', 0)}",
        f"EXITED_COUNT:{analysis.get('exited_count', 0)}",
        f"ACTIVE_INSIDE_COUNT:{analysis.get('active_inside_count', 0)}",
        f"LOST_INSIDE_COUNT:{analysis.get('lost_inside_count', 0)}",
        f"ACTIVE_OR_LOST_INSIDE_COUNT:{analysis.get('active_or_lost_inside_count', 0)}",
        f"USE_CURRENT_CAMERA_ANGLE:{analysis.get('use_current_camera_angle', True)}",
        f"CAMERA_ANGLE_PROFILE:{analysis.get('camera_angle_profile', 'CURRENT_CAMERA')}",
        f"HANDRAIL_MAPPING_SOURCE:{analysis.get('handrail_mapping_source', 'CURRENT_CAMERA')}",
        f"DY:{int(analysis['dy'])}" if analysis.get("dy") is not None else "DY:NA",
        f"DIRECTION_DY:{int(analysis['direction_dy'])}"
        if analysis.get("direction_dy") is not None
        else "DIRECTION_DY:NA",
        f"LANE_V:{int(analysis['lane_v'])}"
        if analysis.get("lane_v") is not None
        else "LANE_V:NA",
        f"HIP_MONITOR_POINT:{analysis['monitor_point_hip'][0]},{analysis['monitor_point_hip'][1]}"
        if analysis.get("monitor_point_hip") is not None
        else "HIP_MONITOR_POINT:NA",
        f"SHOULDER_MONITOR_POINT:{analysis['monitor_point_shoulder'][0]},{analysis['monitor_point_shoulder'][1]}"
        if analysis.get("monitor_point_shoulder") is not None
        else "SHOULDER_MONITOR_POINT:NA",
        f"HIP_MONITOR_SOURCE:{analysis.get('monitor_point_hip_source', 'NO_HIP_CENTER')}",
        f"SHOULDER_MONITOR_SOURCE:{analysis.get('monitor_point_shoulder_source', 'NO_SHOULDER_CENTER')}",
        f"HIP_DIRECTION:{analysis.get('hip_direction', 'UNKNOWN')}",
        f"SHOULDER_DIRECTION:{analysis.get('shoulder_direction', 'UNKNOWN')}",
        f"FINAL_DIRECTION:{analysis.get('final_direction', analysis.get('direction', 'ANALYZING'))}",
        f"DIRECTION_SOURCE:{analysis.get('direction_source', 'NO_VALID_MONITOR_DIRECTION')}",
        f"DIR:{analysis.get('direction', 'NA')}",
        f"DIRECTION_REASON:{analysis.get('direction_reason', 'UNKNOWN')}",
        f"INSIDE_STAIRS:{analysis.get('inside_stairs', False)}",
        f"INSIDE_FINAL:{analysis.get('inside_stairs', False)}",
        f"FEET_AVAILABLE:{analysis.get('feet_available', analysis.get('inside_feet_point') is not None)}",
        f"FEET_RELIABLE:{analysis.get('feet_reliable', False)}",
        f"ANKLE_VALID_COUNT:{analysis.get('ankle_valid_count', 0)}",
        f"FEET_POINT_SOURCE:{analysis.get('feet_point_source', 'FEET_UNAVAILABLE')}",
        f"INSIDE_FEET_POINT_SOURCE:{analysis.get('inside_feet_point_source', 'FEET_UNAVAILABLE')}",
        f"FEET_UNAVAILABLE_REASON:{analysis.get('feet_unavailable_reason', 'NO_SHOULDER_NO_HIP')}",
        f"SH_HIP_VISIBLE_SHOULDERS:{analysis.get('sh_hip_visible_shoulder_count', 0)}",
        f"SH_HIP_VISIBLE_HIPS:{analysis.get('sh_hip_visible_hip_count', 0)}",
        f"SH_HIP_PAIR:{analysis.get('sh_hip_selected_pair', 'NONE')}",
        f"SH_HIP_SOURCE:{analysis.get('sh_hip_virtual_feet_source', 'FEET_UNAVAILABLE')}",
        (
            f"SH_HIP_POINT:{analysis['sh_hip_virtual_feet_point'][0]},{analysis['sh_hip_virtual_feet_point'][1]}"
            if analysis.get("sh_hip_virtual_feet_point") is not None
            else "SH_HIP_POINT:NA"
        ),
        f"BBOX_HEIGHT:{analysis.get('bbox_height')}"
        if analysis.get("bbox_height") is not None
        else "BBOX_HEIGHT:NA",
        f"LEFT_ANKLE_VALID:{analysis.get('left_ankle_valid', False)}",
        f"RIGHT_ANKLE_VALID:{analysis.get('right_ankle_valid', False)}",
        f"LEFT_ANKLE_POINT:{analysis['left_ankle_point'][0]},{analysis['left_ankle_point'][1]}"
        if analysis.get("left_ankle_point") is not None
        else "LEFT_ANKLE_POINT:NA",
        f"RIGHT_ANKLE_POINT:{analysis['right_ankle_point'][0]},{analysis['right_ankle_point'][1]}"
        if analysis.get("right_ankle_point") is not None
        else "RIGHT_ANKLE_POINT:NA",
        f"LEFT_ANKLE_RAW_POINT:{analysis['left_ankle_raw_point'][0]},{analysis['left_ankle_raw_point'][1]}"
        if analysis.get("left_ankle_raw_point") is not None
        else "LEFT_ANKLE_RAW_POINT:NA",
        f"RIGHT_ANKLE_RAW_POINT:{analysis['right_ankle_raw_point'][0]},{analysis['right_ankle_raw_point'][1]}"
        if analysis.get("right_ankle_raw_point") is not None
        else "RIGHT_ANKLE_RAW_POINT:NA",
        f"LEFT_ANKLE_STEP_POINT:{analysis['left_ankle_step_point'][0]},{analysis['left_ankle_step_point'][1]}"
        if analysis.get("left_ankle_step_point") is not None
        else "LEFT_ANKLE_STEP_POINT:NA",
        f"RIGHT_ANKLE_STEP_POINT:{analysis['right_ankle_step_point'][0]},{analysis['right_ankle_step_point'][1]}"
        if analysis.get("right_ankle_step_point") is not None
        else "RIGHT_ANKLE_STEP_POINT:NA",
        f"LEFT_ANKLE_CONF:{analysis['left_ankle_conf']:.2f}"
        if isinstance(analysis.get("left_ankle_conf"), (int, float))
        else "LEFT_ANKLE_CONF:NA",
        f"RIGHT_ANKLE_CONF:{analysis['right_ankle_conf']:.2f}"
        if isinstance(analysis.get("right_ankle_conf"), (int, float))
        else "RIGHT_ANKLE_CONF:NA",
        f"LEFT_CURRENT_ANKLE_STEP:{analysis.get('left_current_ankle_step')}"
        if analysis.get("left_current_ankle_step") is not None
        else "LEFT_CURRENT_ANKLE_STEP:NA",
        f"RIGHT_CURRENT_ANKLE_STEP:{analysis.get('right_current_ankle_step')}"
        if analysis.get("right_current_ankle_step") is not None
        else "RIGHT_CURRENT_ANKLE_STEP:NA",
        f"LEFT_RAW_ANKLE_STEP:{analysis.get('left_raw_ankle_step')}"
        if analysis.get("left_raw_ankle_step") is not None
        else "LEFT_RAW_ANKLE_STEP:NA",
        f"RIGHT_RAW_ANKLE_STEP:{analysis.get('right_raw_ankle_step')}"
        if analysis.get("right_raw_ankle_step") is not None
        else "RIGHT_RAW_ANKLE_STEP:NA",
        f"LEFT_RAW_STEP:{analysis.get('left_raw_step')}"
        if analysis.get("left_raw_step") is not None
        else "LEFT_RAW_STEP:NA",
        f"RIGHT_RAW_STEP:{analysis.get('right_raw_step')}"
        if analysis.get("right_raw_step") is not None
        else "RIGHT_RAW_STEP:NA",
        f"LEFT_FILTERED_STEP:{analysis.get('left_filtered_step')}"
        if analysis.get("left_filtered_step") is not None
        else "LEFT_FILTERED_STEP:NA",
        f"RIGHT_FILTERED_STEP:{analysis.get('right_filtered_step')}"
        if analysis.get("right_filtered_step") is not None
        else "RIGHT_FILTERED_STEP:NA",
        f"LEFT_LAST_VALID_STEP:{analysis.get('left_last_valid_step')}"
        if analysis.get("left_last_valid_step") is not None
        else "LEFT_LAST_VALID_STEP:NA",
        f"RIGHT_LAST_VALID_STEP:{analysis.get('right_last_valid_step')}"
        if analysis.get("right_last_valid_step") is not None
        else "RIGHT_LAST_VALID_STEP:NA",
        f"LEFT_ADJUSTED_STEP:{analysis.get('left_adjusted_step')}"
        if analysis.get("left_adjusted_step") is not None
        else "LEFT_ADJUSTED_STEP:NA",
        f"RIGHT_ADJUSTED_STEP:{analysis.get('right_adjusted_step')}"
        if analysis.get("right_adjusted_step") is not None
        else "RIGHT_ADJUSTED_STEP:NA",
        f"ANKLE_STEP_OFFSET_X:{analysis.get('ankle_step_offset_x', 0)}",
        f"ANKLE_STEP_OFFSET_Y:{analysis.get('ankle_step_offset_y', 0)}",
        f"ANKLE_STEP_OFFSET_DIRECTION:{analysis.get('ankle_step_offset_direction', 'UNKNOWN')}",
        f"LEFT_STEP_REASON:{analysis.get('left_step_reason', 'NOT_EVALUATED')}",
        f"RIGHT_STEP_REASON:{analysis.get('right_step_reason', 'NOT_EVALUATED')}",
        f"LEFT_STEP_FILTER_REASON:{analysis.get('left_step_filter_reason', 'NOT_EVALUATED')}",
        f"RIGHT_STEP_FILTER_REASON:{analysis.get('right_step_filter_reason', 'NOT_EVALUATED')}",
        f"LEFT_STEP_NEAREST_BAND_ID:{analysis.get('left_step_nearest_band_id')}"
        if analysis.get("left_step_nearest_band_id") is not None
        else "LEFT_STEP_NEAREST_BAND_ID:NA",
        f"RIGHT_STEP_NEAREST_BAND_ID:{analysis.get('right_step_nearest_band_id')}"
        if analysis.get("right_step_nearest_band_id") is not None
        else "RIGHT_STEP_NEAREST_BAND_ID:NA",
        f"LEFT_STEP_NEAREST_DISTANCE:{analysis['left_step_nearest_distance']:.1f}"
        if isinstance(analysis.get("left_step_nearest_distance"), (int, float))
        else "LEFT_STEP_NEAREST_DISTANCE:NA",
        f"RIGHT_STEP_NEAREST_DISTANCE:{analysis['right_step_nearest_distance']:.1f}"
        if isinstance(analysis.get("right_step_nearest_distance"), (int, float))
        else "RIGHT_STEP_NEAREST_DISTANCE:NA",
        f"LEFT_STEP_IS_INSIDE:{analysis.get('left_step_is_inside', False)}",
        f"RIGHT_STEP_IS_INSIDE:{analysis.get('right_step_is_inside', False)}",
        f"LEFT_STEP_IS_NEAR_BOUNDARY:{analysis.get('left_step_is_near_boundary', False)}",
        f"RIGHT_STEP_IS_NEAR_BOUNDARY:{analysis.get('right_step_is_near_boundary', False)}",
        f"LEFT_STEP_INDEX:{analysis.get('left_step_index')}"
        if analysis.get("left_step_index") is not None
        else "LEFT_STEP_INDEX:NA",
        f"RIGHT_STEP_INDEX:{analysis.get('right_step_index')}"
        if analysis.get("right_step_index") is not None
        else "RIGHT_STEP_INDEX:NA",
        f"FOOT_GAP:{analysis.get('foot_gap', analysis.get('step_gap'))}"
        if analysis.get("foot_gap", analysis.get("step_gap")) is not None
        else "FOOT_GAP:NA",
        f"STEP_GAP:{analysis.get('step_gap')}"
        if analysis.get("step_gap") is not None
        else "STEP_GAP:NA",
        f"LEFT_PLANTED_STEP:{analysis.get('left_planted_step')}"
        if analysis.get("left_planted_step") is not None
        else "LEFT_PLANTED_STEP:NA",
        f"RIGHT_PLANTED_STEP:{analysis.get('right_planted_step')}"
        if analysis.get("right_planted_step") is not None
        else "RIGHT_PLANTED_STEP:NA",
        f"PLANTED_STEP_GAP:{analysis.get('planted_step_gap')}"
        if analysis.get("planted_step_gap") is not None
        else "PLANTED_STEP_GAP:NA",
        f"LEFT_IS_PLANTED:{analysis.get('left_is_planted', False)}",
        f"RIGHT_IS_PLANTED:{analysis.get('right_is_planted', False)}",
        f"LEFT_ANKLE_SPEED:{analysis['left_ankle_speed']:.1f}"
        if isinstance(analysis.get("left_ankle_speed"), (int, float))
        else "LEFT_ANKLE_SPEED:NA",
        f"RIGHT_ANKLE_SPEED:{analysis['right_ankle_speed']:.1f}"
        if isinstance(analysis.get("right_ankle_speed"), (int, float))
        else "RIGHT_ANKLE_SPEED:NA",
        (
            "FEET=LOST"
            if analysis.get("inside_feet_point") is None
            else (
                "FEET=SH-HIP"
                if analysis.get("feet_point_source") == "VIRTUAL_FROM_SHOULDER_HIP"
                else "FEET=REAL"
            )
        ),
        (
            f"REASON={analysis.get('feet_unavailable_reason', 'NO_SHOULDER_NO_HIP')}"
            if analysis.get("inside_feet_point") is None
            else (
                f"SRC={analysis.get('sh_hip_selected_pair', 'NONE')}"
                if analysis.get("feet_point_source") == "VIRTUAL_FROM_SHOULDER_HIP"
                else f"REASON={analysis.get('feet_point_source', 'FEET_UNAVAILABLE')}"
            )
        ),
        f"LEFT_FOOT_STEP_REASON:{analysis.get('left_foot_step_reason', 'NOT_EVALUATED')}",
        f"RIGHT_FOOT_STEP_REASON:{analysis.get('right_foot_step_reason', 'NOT_EVALUATED')}",
        f"LEFT_FOOT_STATE:{analysis.get('left_foot_state_label', 'WAIT')}",
        f"RIGHT_FOOT_STATE:{analysis.get('right_foot_state_label', 'WAIT')}",
        f"LEFT_FOOT_LANDED:{analysis.get('left_foot_landed', False)}",
        f"RIGHT_FOOT_LANDED:{analysis.get('right_foot_landed', False)}",
        f"TWO_STEP_SKIP_CHECK_AVAILABLE:{analysis.get('two_step_skip_check_available', False)}",
        f"TWO_STEP_SKIP_REASON:{analysis.get('two_step_skip_reason', 'NOT_EVALUATED')}",
        f"TWO_STEP_SKIP_CONFIRMED:{analysis.get('two_step_skip_confirmed', False)}",
        f"LEFT_FOOT_IN:{analysis.get('left_foot_in', False)}",
        f"RIGHT_FOOT_IN:{analysis.get('right_foot_in', False)}",
        f"VALID_FOOT_COUNT:{analysis.get('valid_foot_count', 0)}",
        f"INSIDE_RAW_BY_FEET:{analysis.get('inside_raw_by_feet')}"
        if analysis.get("inside_raw_by_feet") is not None
        else "INSIDE_RAW_BY_FEET:NA",
        f"INSIDE_REASON:{analysis.get('inside_reason', 'UNKNOWN')}",
        f"INSIDE_GRACE_LEFT:{analysis.get('inside_grace_left', 0)}",
        f"TRACK_ZONE_STATE:{analysis.get('track_zone_state', 'UNKNOWN')}",
        f"INSIDE_FEET:{analysis['inside_feet_point'][0]},{analysis['inside_feet_point'][1]}"
        if analysis.get("inside_feet_point") is not None
        else "INSIDE_FEET:NA",
        f"LANE_RAW:{analysis.get('lane_raw', False)}",
        f"LANE_HITS:{analysis.get('lane_hits', 0)}",
        f"LANE_CONF:{analysis.get('lane_conf', False)}",
        f"LANE_DIRECTION:{analysis.get('lane_direction', 'ANALYZING')}",
        f"LANE_SOURCE:{analysis.get('lane_source', 'NO_LANE')}",
        f"LANE_REASON:{analysis.get('lane_reason', 'NA')}",
        f"LANE_SIDE_VALUE:{int(analysis['lane_side_value'])}"
        if analysis.get("lane_side_value") is not None
        else "LANE_SIDE_VALUE:NA",
        f"LANE_SIDE_LABEL:{analysis.get('lane_side_label', 'UNKNOWN')}",
        f"CORRECT_LANE_SIDE:{analysis.get('correct_lane_side', 'UNKNOWN')}",
        f"LANE_MAPPING_SOURCE:{analysis.get('lane_mapping_source', 'CURRENT_CAMERA')}",
        f"P_LANE_SOURCE:{analysis.get('p_lane_source', 'NONE')}",
        f"FOOT_LANE_SIDE:{int(analysis['foot_lane_side'])}"
        if analysis.get("foot_lane_side") is not None
        else "FOOT_LANE_SIDE:NA",
        f"LANE_MISSING_FEET_GRACE_LEFT:{analysis.get('lane_missing_feet_grace_left', 0)}",
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
        f"BACKWARD_MAPPING_SOURCE:{analysis.get('backward_mapping_source', 'CURRENT_CAMERA')}",
        f"BACKWARD_REASON:{analysis.get('backward_reason', 'UNKNOWN')}",
        f"STAND_RAW:{analysis.get('standing_raw', False)}",
        f"STAND_HITS:{analysis.get('standing_hits', 0)}",
        f"STAND_CONF:{analysis.get('standing_still_confirmed', False)}",
        f"STAND_RANGE:{analysis['standing_motion_range']:.1f}"
        if analysis.get("standing_motion_range") is not None
        else "STAND_RANGE:NA",
        f"STAND_LEN:{analysis.get('standing_len', 0)}",
        f"ARM_ORDER:{analysis.get('arm_side_order', 'UNKNOWN')}",
        f"LW_VALID:{int(bool(analysis.get('left_wrist_valid', False)))}",
        f"RW_VALID:{int(bool(analysis.get('right_wrist_valid', False)))}",
        f"LW_HIT:{int(bool(analysis.get('left_wrist_hit', False)))}",
        f"RW_HIT:{int(bool(analysis.get('right_wrist_hit', False)))}",
        "LW_HIT_CNT:"
        + _format_handrail_count_text(
            analysis.get("left_wrist_hit_count"),
            analysis.get("left_wrist_confirm_required"),
        ),
        "RW_HIT_CNT:"
        + _format_handrail_count_text(
            analysis.get("right_wrist_hit_count"),
            analysis.get("right_wrist_confirm_required"),
        ),
        f"LW_MISS_CNT:{int(analysis.get('left_wrist_miss_count', 0))}",
        f"RW_MISS_CNT:{int(analysis.get('right_wrist_miss_count', 0))}",
        f"LW_CONFIRM_REQUIRED:{int(analysis.get('left_wrist_confirm_required', 0))}",
        f"RW_CONFIRM_REQUIRED:{int(analysis.get('right_wrist_confirm_required', 0))}",
        f"LW_HOLD:{int(bool(analysis.get('left_holding', False)))}",
        f"RW_HOLD:{int(bool(analysis.get('right_holding', False)))}",
        f"HANDRAIL:{analysis.get('handrail_status', 'NOT_EVALUATED')}",
        f"REASON:{analysis.get('handrail_debug_reason', analysis.get('handrail_reason', 'NOT_EVALUATED'))}",
        f"L_CARRY:{analysis.get('left_carry', False)}",
        f"R_CARRY:{analysis.get('right_carry', False)}",
        f"L_CARRY_ALLOWED:{analysis.get('left_carry_allowed', True)}",
        f"R_CARRY_ALLOWED:{analysis.get('right_carry_allowed', True)}",
        f"L_CARRY_EVIDENCE:{analysis.get('left_carry_evidence', False)}",
        f"R_CARRY_EVIDENCE:{analysis.get('right_carry_evidence', False)}",
        f"L_CARRY_SCORE:{_format_carry_score_value(analysis.get('left_carry_score'))}",
        f"R_CARRY_SCORE:{_format_carry_score_value(analysis.get('right_carry_score'))}",
        f"ONE_HAND_CARRY_SIDE:{analysis.get('one_hand_carry_side', 'NONE')}",
        f"CARRY_REASON:{analysis.get('carry_reason', 'NO_CARRY')}",
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
    if not analysis.get("wrong_lane", False):
        return debug_lines

    filtered_lines = [
        line
        for line in debug_lines
        if not line.startswith(LANE_SUPPRESSED_HOLD_DEBUG_PREFIXES)
    ]
    filtered_lines.extend(
        [
            "LANE_SUPPRESS_HOLD_DISPLAY:True",
            "SUPPRESSED_WARNINGS:" + ",".join(LANE_SUPPRESSED_WARNING_LABELS),
        ]
    )
    return filtered_lines
