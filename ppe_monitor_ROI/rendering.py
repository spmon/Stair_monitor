import cv2

from ppe_monitor_ROI.roi import is_foot_point_in_roi, is_point_valid


def draw_foot_debug(frame, keypoints):
    for foot_idx in [15, 16]:
        foot = keypoints[foot_idx]
        if not is_point_valid(foot):
            continue

        color = (0, 0, 255) if is_foot_point_in_roi(foot) else (0, 255, 0)
        cv2.circle(frame, (int(foot[0]), int(foot[1])), 6, color, -1)


def draw_raw_ppe(frame, hat_bboxes, vest_bboxes):
    for hat in hat_bboxes:
        cv2.rectangle(
            frame,
            (int(hat[0]), int(hat[1])),
            (int(hat[2]), int(hat[3])),
            (180, 100, 0),
            1,
        )
        cv2.putText(
            frame,
            "Hat raw",
            (int(hat[0]), int(hat[1]) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 100, 0),
            1,
        )

    for vest in vest_bboxes:
        cv2.rectangle(
            frame,
            (int(vest[0]), int(vest[1])),
            (int(vest[2]), int(vest[3])),
            (0, 200, 200),
            1,
        )
        cv2.putText(
            frame,
            "Vest raw",
            (int(vest[0]), int(vest[1]) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 200, 200),
            1,
        )


def draw_person_debug(
    frame,
    person_box,
    label_text,
    status_color,
    head_box=None,
    torso_box=None,
    matched_hat=None,
    matched_vest=None,
):
    if head_box is not None:
        cv2.rectangle(
            frame,
            (int(head_box[0]), int(head_box[1])),
            (int(head_box[2]), int(head_box[3])),
            (0, 255, 100),
            1,
        )

    if torso_box is not None:
        cv2.rectangle(
            frame,
            (int(torso_box[0]), int(torso_box[1])),
            (int(torso_box[2]), int(torso_box[3])),
            (0, 165, 255),
            1,
        )

    if matched_hat is not None:
        cv2.rectangle(
            frame,
            (int(matched_hat[0]), int(matched_hat[1])),
            (int(matched_hat[2]), int(matched_hat[3])),
            (255, 0, 0),
            2,
        )
        cv2.putText(
            frame,
            "Hat",
            (int(matched_hat[0]), int(matched_hat[1]) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )

    if matched_vest is not None:
        cv2.rectangle(
            frame,
            (int(matched_vest[0]), int(matched_vest[1])),
            (int(matched_vest[2]), int(matched_vest[3])),
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            "Vest",
            (int(matched_vest[0]), int(matched_vest[1]) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
        )

    cv2.rectangle(
        frame,
        (int(person_box[0]), int(person_box[1])),
        (int(person_box[2]), int(person_box[3])),
        status_color,
        2,
    )
    cv2.putText(
        frame,
        label_text,
        (int(person_box[0]), int(person_box[1]) - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        status_color,
        2,
    )


def draw_roi(frame, roi_pts, roi_alert):
    roi_color = (0, 0, 255) if roi_alert else (0, 255, 0)
    cv2.polylines(
        frame,
        [roi_pts],
        isClosed=True,
        color=roi_color,
        thickness=4,
    )


def get_overall_status(frame_statuses):
    if len(frame_statuses) == 0:
        return "ROI STATUS: CLEAR", "PPE STATUS: -", (0, 255, 0)

    worst = max(frame_statuses, key=lambda status: status["severity"])
    if worst["severity"] == 0:
        return "ROI STATUS: PERSON IN ROI", "PPE STATUS: SAFE: Full PPE", (0, 255, 0)
    if worst["severity"] == 1:
        return "ROI STATUS: PERSON IN ROI", "PPE STATUS: CHECKING...", (0, 255, 255)

    return (
        "ROI STATUS: PERSON IN ROI",
        f"PPE STATUS: {worst['label']}",
        (0, 0, 255),
    )


def draw_status_panel(frame, line1, line2, color):
    x, y = 30, 40
    width, height = 760, 120

    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 0), -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 3)

    cv2.putText(
        frame,
        line1,
        (x + 20, y + 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        3,
    )
    cv2.putText(
        frame,
        line2,
        (x + 20, y + 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        3,
    )
