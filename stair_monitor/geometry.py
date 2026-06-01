import math

import numpy as np


def _keypoint_is_visible(keypoints, idx, conf_th):
    return (
        keypoints is not None
        and len(keypoints) > idx
        and len(keypoints[idx]) > 2
        and keypoints[idx][2] > conf_th
    )


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


def signed_distance_to_line(pt, line_pts):
    """
    Compute the signed distance from a point to a 2-point line.
    D < 0: point is on one side of the line, treated here as LEFT_SIDE.
    D > 0: point is on the other side of the line, treated here as RIGHT_SIDE.
    Note: the sign depends on the order of the 2 line points in JSON.
    """
    if len(line_pts) < 2:
        return 9999

    x1, y1 = line_pts[0]
    x2, y2 = line_pts[1]
    x0, y0 = pt

    A, B = y2 - y1, -(x2 - x1)
    C = x2 * y1 - y2 * x1

    hyp = math.hypot(A, B)
    if hyp == 0:
        return 9999
    return (A * x0 + B * y0 + C) / hyp


def point_to_segment_distance(p, a, b):
    """
    Compute the distance from point p to the line segment ab.

    Returns:
    - dist: Euclidean distance from p to the segment
    - t: projection coefficient before clamping
    - closest: closest point on the segment
    """
    p = np.array(p, dtype=np.float32)
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)

    ab = b - a
    ap = p - a

    denom = np.dot(ab, ab)
    if denom < 1e-6:
        return float(np.linalg.norm(p - a)), 0.0, a

    t = float(np.dot(ap, ab) / denom)
    t_clamped = max(0.0, min(1.0, t))
    closest = a + t_clamped * ab
    dist = float(np.linalg.norm(p - closest))
    return dist, t, closest


def get_side_name(d):
    if d == -999 or d == 9999:
        return "UNKNOWN"
    if d < 0:
        return "LEFT_SIDE"
    if d > 0:
        return "RIGHT_SIDE"
    return "ON_LINE"


def estimate_body_facing(keypoints, conf_th=0.5):
    """
    Estimate body facing direction from pose keypoints.

    Returns base labels such as FRONT_TO_CAMERA / BACK_TO_CAMERA /
    SIDE_OR_UNKNOWN / UNKNOWN, or a face-visibility-prefixed variant.
    """
    if keypoints is None or len(keypoints) <= 6:
        return "UNKNOWN"

    if not _keypoint_is_visible(keypoints, 5, conf_th) or not _keypoint_is_visible(
        keypoints, 6, conf_th
    ):
        return "UNKNOWN"

    left_shoulder_x = keypoints[5][0]
    right_shoulder_x = keypoints[6][0]
    shoulder_dx = left_shoulder_x - right_shoulder_x

    # Small x gaps are too unstable to treat as front/back.
    if abs(shoulder_dx) <= 10:
        shoulder_guess = "SIDE_OR_UNKNOWN"
    elif shoulder_dx > 0:
        shoulder_guess = "FRONT_TO_CAMERA"
    else:
        shoulder_guess = "BACK_TO_CAMERA"

    face_visible_count = sum(
        1 for idx in [0, 1, 2, 3, 4] if _keypoint_is_visible(keypoints, idx, conf_th)
    )

    if face_visible_count >= 3:
        return f"FACE_VISIBLE_{shoulder_guess}"
    if face_visible_count <= 1:
        return f"FACE_NOT_VISIBLE_{shoulder_guess}"
    return shoulder_guess


def estimate_arm_side_order(keypoints, conf_th=0.5):
    """
    Estimate whether the left arm appears on the left or right side of the image.
    """
    left_arm_x = [
        keypoints[idx][0]
        for idx in [5, 7, 9]
        if _keypoint_is_visible(keypoints, idx, conf_th)
    ]
    right_arm_x = [
        keypoints[idx][0]
        for idx in [6, 8, 10]
        if _keypoint_is_visible(keypoints, idx, conf_th)
    ]

    if not left_arm_x or not right_arm_x:
        return "UNKNOWN"

    left_avg_x = sum(left_arm_x) / len(left_arm_x)
    right_avg_x = sum(right_arm_x) / len(right_arm_x)

    if left_avg_x > right_avg_x:
        return "LEFT_ARM_ON_IMAGE_RIGHT"
    if left_avg_x < right_avg_x:
        return "LEFT_ARM_ON_IMAGE_LEFT"
    return "UNKNOWN"
