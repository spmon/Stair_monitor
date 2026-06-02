import numpy as np


def is_point_valid(point):
    return point[0] > 0 and point[1] > 0


def overlap_ratio(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    if area_a <= 0 or area_b <= 0:
        return 0.0

    return inter_area / min(area_a, area_b)


def get_head_bbox(keypoints, person_box=None):
    head_keypoints = keypoints[0:5]
    valid_keypoints = head_keypoints[
        [(point[0] > 0 and point[1] > 0) for point in head_keypoints]
    ]

    if len(valid_keypoints) >= 2:
        hx1, hy1 = np.min(valid_keypoints[:, :2], axis=0)
        hx2, hy2 = np.max(valid_keypoints[:, :2], axis=0)

        head_height = hy2 - hy1
        if head_height < 10:
            head_height = 20

        return [
            hx1 - head_height * 0.3,
            hy1 - head_height * 0.8,
            hx2 + head_height * 0.3,
            hy2 + head_height * 0.2,
        ]

    if person_box is not None:
        x1, y1, x2, y2 = person_box
        width = x2 - x1
        height = y2 - y1
        return [
            x1 + 0.20 * width,
            y1,
            x2 - 0.20 * width,
            y1 + 0.30 * height,
        ]

    return None


def get_torso_bbox(keypoints, person_box=None):
    left_shoulder, right_shoulder = keypoints[5], keypoints[6]
    left_hip, right_hip = keypoints[11], keypoints[12]

    torso_points = []
    for point in [left_shoulder, right_shoulder, left_hip, right_hip]:
        if is_point_valid(point):
            torso_points.append(point[:2])

    if len(torso_points) >= 3:
        torso_points = np.array(torso_points, dtype=np.float32)
        x1, y1 = np.min(torso_points, axis=0)
        x2, y2 = np.max(torso_points, axis=0)

        width = x2 - x1
        height = y2 - y1
        return [
            x1 - 0.2 * width,
            y1 - 0.1 * height,
            x2 + 0.2 * width,
            y2 + 0.2 * height,
        ]

    if person_box is not None:
        x1, y1, x2, y2 = person_box
        width = x2 - x1
        height = y2 - y1
        return [
            x1 + 0.15 * width,
            y1 + 0.20 * height,
            x2 - 0.15 * width,
            y1 + 0.70 * height,
        ]

    return None
