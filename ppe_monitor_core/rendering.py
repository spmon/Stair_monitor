import cv2

SAFE_LABEL_BG = (0, 160, 0)
SAFE_TEXT_COLOR = (255, 255, 255)
SAFE_BBOX_COLOR = (0, 255, 0)

WARNING_LABEL_BG = (0, 0, 255)
WARNING_TEXT_COLOR = (255, 255, 255)
WARNING_BBOX_COLOR = (0, 0, 255)

CHECKING_LABEL_BG = (0, 180, 255)
CHECKING_TEXT_COLOR = (0, 0, 0)
CHECKING_BBOX_COLOR = (0, 180, 255)

PANEL_BG_COLOR = (32, 32, 32)
PANEL_TEXT_COLOR = (255, 255, 255)
PANEL_BORDER_COLOR = (90, 90, 90)


# All drawing helpers in this file must draw on draw_frame only.
# Never pass infer_frame into these functions.
def draw_raw_ppe(draw_frame, hat_bboxes, vest_bboxes):
    for hat in hat_bboxes:
        cv2.rectangle(
            draw_frame,
            (int(hat[0]), int(hat[1])),
            (int(hat[2]), int(hat[3])),
            (180, 100, 0),
            1,
        )
        cv2.putText(
            draw_frame,
            "Hat raw",
            (int(hat[0]), int(hat[1]) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 100, 0),
            1,
        )

    for vest in vest_bboxes:
        cv2.rectangle(
            draw_frame,
            (int(vest[0]), int(vest[1])),
            (int(vest[2]), int(vest[3])),
            (0, 200, 200),
            1,
        )
        cv2.putText(
            draw_frame,
            "Vest raw",
            (int(vest[0]), int(vest[1]) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 200, 200),
            1,
        )


def _get_status_style(status_type):
    if status_type == "safe":
        return {
            "label_bg": SAFE_LABEL_BG,
            "text_color": SAFE_TEXT_COLOR,
            "bbox_color": SAFE_BBOX_COLOR,
            "border_color": (0, 110, 0),
        }
    if status_type == "warning":
        return {
            "label_bg": WARNING_LABEL_BG,
            "text_color": WARNING_TEXT_COLOR,
            "bbox_color": WARNING_BBOX_COLOR,
            "border_color": (0, 0, 170),
        }
    return {
        "label_bg": CHECKING_LABEL_BG,
        "text_color": CHECKING_TEXT_COLOR,
        "bbox_color": CHECKING_BBOX_COLOR,
        "border_color": (0, 120, 190),
    }


def _measure_text_box(text, font_scale=0.85, thickness=2, pad_x=10, pad_y=8):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    box_w = text_w + pad_x * 2
    box_h = text_h + pad_y * 2 + baseline
    return {
        "font": font,
        "text_w": text_w,
        "text_h": text_h,
        "baseline": baseline,
        "box_w": box_w,
        "box_h": box_h,
    }


def draw_text_box(
    draw_frame,
    text,
    x,
    y,
    bg_color,
    text_color=(255, 255, 255),
    font_scale=0.85,
    thickness=2,
    pad_x=10,
    pad_y=8,
    border_color=None,
):
    metrics = _measure_text_box(
        text,
        font_scale=font_scale,
        thickness=thickness,
        pad_x=pad_x,
        pad_y=pad_y,
    )
    font = metrics["font"]
    text_h = metrics["text_h"]
    box_w = metrics["box_w"]
    box_h = metrics["box_h"]

    frame_h, frame_w = draw_frame.shape[:2]
    box_x = max(0, min(int(x), max(0, frame_w - box_w - 5)))
    box_y = max(0, min(int(y), max(0, frame_h - box_h - 5)))
    rect_x1 = box_x
    rect_y1 = box_y
    rect_x2 = box_x + box_w
    rect_y2 = box_y + box_h

    cv2.rectangle(
        draw_frame,
        (rect_x1, rect_y1),
        (rect_x2, rect_y2),
        bg_color,
        -1,
    )
    if border_color is not None:
        cv2.rectangle(
            draw_frame,
            (rect_x1, rect_y1),
            (rect_x2, rect_y2),
            border_color,
            1,
        )

    text_x = rect_x1 + pad_x
    text_y = rect_y1 + pad_y + text_h
    cv2.putText(
        draw_frame,
        text,
        (text_x, text_y),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )
    return box_w, box_h


def draw_person_label(draw_frame, box, text, status_type):
    x1, y1, x2, y2 = map(int, box)
    style = _get_status_style(status_type)
    metrics = _measure_text_box(text, font_scale=0.85, thickness=2, pad_x=10, pad_y=8)
    label_w = metrics["box_w"]
    label_h = metrics["box_h"]
    frame_h, frame_w = draw_frame.shape[:2]

    label_x = max(0, x1)
    label_y = y2 + 8

    if label_x + label_w > frame_w:
        label_x = max(0, frame_w - label_w - 5)

    if label_y + label_h > frame_h:
        label_y = y1 - label_h - 8

    if label_y < 0:
        label_y = 5

    draw_text_box(
        draw_frame,
        text,
        label_x,
        label_y,
        style["label_bg"],
        text_color=style["text_color"],
        font_scale=0.85,
        thickness=2,
        pad_x=10,
        pad_y=8,
        border_color=style["border_color"],
    )


def draw_person_status(draw_frame, person_box, label_text, status_type, status_color):
    x1, y1, x2, y2 = map(int, person_box)
    _ = status_color
    style = _get_status_style(status_type)
    cv2.rectangle(
        draw_frame,
        (x1, y1),
        (x2, y2),
        style["bbox_color"],
        3,
    )
    draw_person_label(draw_frame, person_box, label_text, status_type)


def draw_person_debug(
    draw_frame,
    person_box,
    label_text,
    status_color,
    head_box,
    torso_box,
    matched_hat,
    matched_vest,
):
    if head_box is not None:
        cv2.rectangle(
            draw_frame,
            (int(head_box[0]), int(head_box[1])),
            (int(head_box[2]), int(head_box[3])),
            (0, 255, 100),
            1,
        )

    if torso_box is not None:
        cv2.rectangle(
            draw_frame,
            (int(torso_box[0]), int(torso_box[1])),
            (int(torso_box[2]), int(torso_box[3])),
            (0, 165, 255),
            1,
        )

    if matched_hat is not None:
        cv2.rectangle(
            draw_frame,
            (int(matched_hat[0]), int(matched_hat[1])),
            (int(matched_hat[2]), int(matched_hat[3])),
            (255, 0, 0),
            2,
        )

    if matched_vest is not None:
        cv2.rectangle(
            draw_frame,
            (int(matched_vest[0]), int(matched_vest[1])),
            (int(matched_vest[2]), int(matched_vest[3])),
            (0, 255, 255),
            2,
        )


def draw_count_panel(
    draw_frame,
    total_people_count,
    checking_count,
    full_ppe_count,
    no_hat_count,
    no_vest_count,
    no_hat_no_vest_count,
):
    if total_people_count == 0:
        return

    lines = [("PEOPLE: " + str(total_people_count), PANEL_BG_COLOR, PANEL_TEXT_COLOR)]

    if full_ppe_count > 0:
        lines.append((f"SAFE: {full_ppe_count}", SAFE_LABEL_BG, SAFE_TEXT_COLOR))
    if no_hat_count > 0:
        lines.append((f"NO HAT: {no_hat_count}", WARNING_LABEL_BG, WARNING_TEXT_COLOR))
    if no_vest_count > 0:
        lines.append((f"NO VEST: {no_vest_count}", WARNING_LABEL_BG, WARNING_TEXT_COLOR))
    if no_hat_no_vest_count > 0:
        lines.append(
            (
                f"NO HAT + NO VEST: {no_hat_no_vest_count}",
                WARNING_LABEL_BG,
                WARNING_TEXT_COLOR,
            )
        )
    if checking_count > 0:
        lines.append((f"CHECKING: {checking_count}", CHECKING_LABEL_BG, CHECKING_TEXT_COLOR))

    font_scale = 0.85
    thickness = 2
    pad_x = 10
    pad_y = 8
    line_gap = 8
    inner_pad = 12
    frame_h, frame_w = draw_frame.shape[:2]
    line_sizes = [
        _measure_text_box(
            text,
            font_scale=font_scale,
            thickness=thickness,
            pad_x=pad_x,
            pad_y=pad_y,
        )
        for text, _, _ in lines
    ]
    max_line_w = max(size["box_w"] for size in line_sizes)
    total_line_h = sum(size["box_h"] for size in line_sizes) + line_gap * (len(line_sizes) - 1)
    panel_w = min(max_line_w + inner_pad * 2, max(220, frame_w - 60))
    panel_h = min(total_line_h + inner_pad * 2, max(80, frame_h - 60))
    x = 30
    y = max(30, frame_h - 30 - panel_h)

    overlay = draw_frame.copy()
    cv2.rectangle(
        overlay,
        (x, y),
        (x + panel_w, y + panel_h),
        (18, 18, 18),
        -1,
    )
    alpha = 0.45
    draw_frame[:] = cv2.addWeighted(overlay, alpha, draw_frame, 1 - alpha, 0)

    cv2.rectangle(
        draw_frame,
        (x, y),
        (x + panel_w, y + panel_h),
        PANEL_BORDER_COLOR,
        2,
    )

    current_y = y + inner_pad
    for (text, bg_color, text_color), size in zip(lines, line_sizes):
        draw_text_box(
            draw_frame,
            text,
            x + inner_pad,
            current_y,
            bg_color,
            text_color=text_color,
            font_scale=font_scale,
            thickness=thickness,
            pad_x=pad_x,
            pad_y=pad_y,
            border_color=None,
        )
        current_y += size["box_h"] + line_gap
