import math

import numpy as np


# Kiem tra keypoint co du tin cay de dung cho cac logic suy luan hay khong.
def _keypoint_is_visible(keypoints, idx, conf_th):
    return (
        keypoints is not None
        and len(keypoints) > idx
        and len(keypoints[idx]) > 2
        and keypoints[idx][2] > conf_th
    )


# Lay toa do pixel cua keypoint neu do tin cay vuot nguong.
def _get_keypoint_point(keypoints, idx, conf_th=0.5):
    if not _keypoint_is_visible(keypoints, idx, conf_th):
        return None
    return (int(keypoints[idx][0]), int(keypoints[idx][1]))


# Midpoint duoc dung de tao cac diem dai dien on dinh hon so voi dung 1 keypoint le.
def _midpoint(point_a, point_b):
    if point_a is None or point_b is None:
        return None
    return (
        int((point_a[0] + point_b[0]) / 2),
        int((point_a[1] + point_b[1]) / 2),
    )


# Tinh goc tai diem b de suy luan tu the tay trong logic carry.
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


def _calculate_arm_angle_from_points(shoulder, elbow, wrist):
    if shoulder is None or elbow is None or wrist is None:
        return None
    return calculate_angle(shoulder, elbow, wrist)


# signed distance dung de biet co tay dang nam dung phia nao cua line lan can.
# Dau am/duong phu thuoc thu tu 2 diem line trong file config, vi vay khong duoc dao tuy tien.
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


# Khoang cach toi doan thang dung de tranh bat nham phan keo dai vo han cua line.
# Gia tri t cho biet hinh chieu roi vao trong doan [0, 1] hay nam ngoai doan.
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


# Quy doi dau signed distance thanh nhan de debug de doc hon.
def get_side_name(d):
    if d == -999 or d == 9999:
        return "UNKNOWN"
    if d < 0:
        return "LEFT_SIDE"
    if d > 0:
        return "RIGHT_SIDE"
    return "ON_LINE"


# Uoc luong huong than/mat de phuc vu logic di lui, khong dung de tinh lane.
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


# Xac dinh tay trai dang nam ben trai hay ben phai anh de debug body orientation.
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


# Tao torso box tu shoulder va hip de carry check vung truoc nguc/bung.
def _get_torso_box_from_features(features, margin_x=40, margin_y=40, margin_bottom=90):
    torso_points = [
        features.get("left_shoulder"),
        features.get("right_shoulder"),
        features.get("left_hip"),
        features.get("right_hip"),
    ]
    if any(point is None for point in torso_points):
        return None

    shoulders = torso_points[:2]
    hips = torso_points[2:]
    xs = [point[0] for point in torso_points]
    return (
        int(min(xs) - margin_x),
        int(min(point[1] for point in shoulders) - margin_y),
        int(max(xs) + margin_x),
        int(max(point[1] for point in hips) + margin_y + margin_bottom),
    )


# Gom cac pose feature co the tai su dung o nhieu logic.
# p_lane uu tien midpoint hai mat ca chan; neu thieu thi fallback bbox bottom center.
# p_motion uu tien tam hong vi on dinh hon chan khi buoc cau thang; neu thieu moi fallback bbox center.
# Hai diem nay phuc vu 2 logic khac nhau: p_lane cho sai lan, p_motion cho direction/standing.
def extract_pose_features(keypoints, bbox):
    bbox_tuple = tuple(int(value) for value in bbox) if bbox is not None else None

    left_wrist = _get_keypoint_point(keypoints, 9)
    right_wrist = _get_keypoint_point(keypoints, 10)
    left_elbow = _get_keypoint_point(keypoints, 7)
    right_elbow = _get_keypoint_point(keypoints, 8)
    left_shoulder = _get_keypoint_point(keypoints, 5)
    right_shoulder = _get_keypoint_point(keypoints, 6)
    left_hip = _get_keypoint_point(keypoints, 11)
    right_hip = _get_keypoint_point(keypoints, 12)
    left_ankle = _get_keypoint_point(keypoints, 15)
    right_ankle = _get_keypoint_point(keypoints, 16)

    bbox_center = None
    bbox_bottom_center = None
    if bbox_tuple is not None and len(bbox_tuple) >= 4:
        bbox_center = (
            int((bbox_tuple[0] + bbox_tuple[2]) / 2),
            int((bbox_tuple[1] + bbox_tuple[3]) / 2),
        )
        bbox_bottom_center = (
            int((bbox_tuple[0] + bbox_tuple[2]) / 2),
            int(bbox_tuple[3]),
        )

    hip_center = _midpoint(left_hip, right_hip)
    shoulder_center = _midpoint(left_shoulder, right_shoulder)

    # feet_point dai dien vi tri nguoi so voi vach giua de xet sai lan.
    if left_ankle is not None and right_ankle is not None:
        feet_point = _midpoint(left_ankle, right_ankle)
    else:
        feet_point = left_ankle or right_ankle or bbox_bottom_center

    # motion_point dai dien cho chuyen dong tong the cua nguoi.
    # Khong uu tien chan vi chan de nhieu khi pose rung hoac buoc buoc tren cau thang.
    motion_point = hip_center or bbox_center or bbox_bottom_center

    features = {
        "bbox": bbox_tuple,
        "bbox_center": bbox_center,
        "bbox_bottom_center": bbox_bottom_center,
        "left_wrist": left_wrist,
        "right_wrist": right_wrist,
        "left_elbow": left_elbow,
        "right_elbow": right_elbow,
        "left_shoulder": left_shoulder,
        "right_shoulder": right_shoulder,
        "left_hip": left_hip,
        "right_hip": right_hip,
        "left_ankle": left_ankle,
        "right_ankle": right_ankle,
        "hip_center": hip_center,
        "shoulder_center": shoulder_center,
        "motion_point": motion_point,
        "feet_point": feet_point,
        "left_arm_angle": _calculate_arm_angle_from_points(
            left_shoulder,
            left_elbow,
            left_wrist,
        ),
        "right_arm_angle": _calculate_arm_angle_from_points(
            right_shoulder,
            right_elbow,
            right_wrist,
        ),
        "body_facing": estimate_body_facing(keypoints),
        "arm_side_order": estimate_arm_side_order(keypoints),
        "keypoint_valid": {
            "left_wrist": left_wrist is not None,
            "right_wrist": right_wrist is not None,
            "left_elbow": left_elbow is not None,
            "right_elbow": right_elbow is not None,
            "left_shoulder": left_shoulder is not None,
            "right_shoulder": right_shoulder is not None,
            "left_hip": left_hip is not None,
            "right_hip": right_hip is not None,
            "left_ankle": left_ankle is not None,
            "right_ankle": right_ankle is not None,
        },
    }
    features["torso_box"] = _get_torso_box_from_features(features)
    return features
