import cv2
import numpy as np

from ppe_monitor_core.config import (
    HAT_CLASS_ID,
    HELMET_CONF_THRES,
    VEST_CLASS_ID,
    VEST_CONF_THRES,
)
from ppe_monitor_core.geometry import overlap_ratio


def parse_ppe_boxes(ppe_results):
    hat_bboxes = []
    vest_bboxes = []

    if ppe_results.boxes is None:
        return hat_bboxes, vest_bboxes

    for box in ppe_results.boxes:
        class_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        item = [x1, y1, x2, y2, conf]
        if class_id == HAT_CLASS_ID and conf >= HELMET_CONF_THRES:
            hat_bboxes.append(item)
        elif class_id == VEST_CLASS_ID and conf >= VEST_CONF_THRES:
            vest_bboxes.append(item)

    return hat_bboxes, vest_bboxes


def is_probably_hair_not_helmet(frame, box):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box

    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(0, min(width, int(x2)))
    y2 = max(0, min(height, int(y2)))

    if x2 <= x1 or y2 <= y1:
        return False

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    return mean_brightness < 60.0


def match_ppe_item(
    person_box,
    target_box,
    candidate_boxes,
    overlap_thres,
    used_indices,
    frame=None,
    reject_dark_hair=False,
):
    if target_box is None:
        return False, None

    best_index = None
    best_overlap = 0.0

    for index, candidate in enumerate(candidate_boxes):
        if index in used_indices:
            continue

        candidate_box = candidate[:4]
        center_x = (candidate_box[0] + candidate_box[2]) / 2
        center_y = (candidate_box[1] + candidate_box[3]) / 2

        if not (
            person_box[0] <= center_x <= person_box[2]
            and person_box[1] <= center_y <= person_box[3]
        ):
            continue

        if reject_dark_hair and frame is not None:
            if is_probably_hair_not_helmet(frame, candidate_box):
                continue

        current_overlap = overlap_ratio(target_box, candidate_box)
        if current_overlap > overlap_thres and current_overlap > best_overlap:
            best_overlap = current_overlap
            best_index = index

    if best_index is None:
        return False, None

    used_indices.add(best_index)
    return True, candidate_boxes[best_index]
