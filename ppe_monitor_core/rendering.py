import cv2


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

    cv2.rectangle(
        draw_frame,
        (int(person_box[0]), int(person_box[1])),
        (int(person_box[2]), int(person_box[3])),
        status_color,
        2,
    )
    cv2.putText(
        draw_frame,
        label_text,
        (int(person_box[0]), int(person_box[1]) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        status_color,
        2,
    )


def draw_count_panel(
    frame,
    total_people_count,
    checking_count,
    full_ppe_count,
    no_hat_count,
    no_vest_count,
    no_hat_no_vest_count,
):
    if total_people_count == 0:
        return

    violation_count = no_hat_count + no_vest_count + no_hat_no_vest_count

    if violation_count > 0:
        border_color = (0, 0, 255)
    elif checking_count > 0:
        border_color = (0, 255, 255)
    else:
        border_color = (0, 255, 0)

    lines = [f"PEOPLE: {total_people_count}"]

    if full_ppe_count > 0:
        lines.append(f"FULL PPE: {full_ppe_count}")
    if no_hat_count > 0:
        lines.append(f"NO HAT: {no_hat_count}")
    if no_vest_count > 0:
        lines.append(f"NO VEST: {no_vest_count}")
    if no_hat_no_vest_count > 0:
        lines.append(f"NO HAT + NO VEST: {no_hat_no_vest_count}")
    if checking_count > 0:
        lines.append(f"CHECKING: {checking_count}")

    line_height = 32
    frame_h, frame_w = frame.shape[:2]
    panel_w = min(520, max(220, frame_w - 60))
    panel_h = 25 + len(lines) * line_height
    panel_h = min(panel_h, max(80, frame_h - 60))
    x = 30
    y = max(30, frame_h - 30 - panel_h)

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x, y),
        (x + panel_w, y + panel_h),
        (0, 0, 0),
        -1,
    )
    alpha = 0.25
    frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    cv2.rectangle(frame, (x, y), (x + panel_w, y + panel_h), border_color, 3)

    for idx, text in enumerate(lines):
        line_color = (255, 255, 255)

        if text.startswith("FULL PPE"):
            line_color = (0, 255, 0)
        elif text.startswith("NO HAT") or text.startswith("NO VEST"):
            line_color = (0, 0, 255)
        elif text.startswith("CHECKING"):
            line_color = (0, 255, 255)

        cv2.putText(
            frame,
            text,
            (x + 20, y + 35 + idx * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            line_color,
            2,
        )
