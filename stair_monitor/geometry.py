import math

import numpy as np

from stair_monitor.settings import BACKWARD_MIN_VALID_EVIDENCE


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


def _average_points(points):
    valid_points = [point for point in points if point is not None]
    if not valid_points:
        return None
    return (
        int(sum(point[0] for point in valid_points) / len(valid_points)),
        int(sum(point[1] for point in valid_points) / len(valid_points)),
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


def _build_pair_body_facing_evidence(
    keypoints,
    left_idx,
    right_idx,
    conf_th=0.5,
    min_abs_dx=10,
):
    left_point = _get_keypoint_point(keypoints, left_idx, conf_th)
    right_point = _get_keypoint_point(keypoints, right_idx, conf_th)
    center = _midpoint(left_point, right_point)

    if left_point is None or right_point is None:
        return {
            "valid": False,
            "label": "UNKNOWN",
            "center": center,
            "dx": None,
        }

    dx = int(left_point[0] - right_point[0])
    if abs(dx) <= min_abs_dx:
        return {
            "valid": False,
            "label": "UNKNOWN",
            "center": center,
            "dx": dx,
        }

    return {
        "valid": True,
        "label": "FRONT_TO_CAMERA" if dx > 0 else "BACK_TO_CAMERA",
        "center": center,
        "dx": dx,
    }


def _build_head_body_facing_evidence(keypoints, conf_th=0.5):
    nose = _get_keypoint_point(keypoints, 0, conf_th)
    left_eye = _get_keypoint_point(keypoints, 1, conf_th)
    right_eye = _get_keypoint_point(keypoints, 2, conf_th)
    left_ear = _get_keypoint_point(keypoints, 3, conf_th)
    right_ear = _get_keypoint_point(keypoints, 4, conf_th)

    visible_points = [nose, left_eye, right_eye, left_ear, right_ear]
    head_center = _average_points(visible_points)
    face_visible_count = sum(point is not None for point in visible_points)
    face_core_visible_count = sum(
        point is not None for point in [nose, left_eye, right_eye]
    )

    # Head evidence chi vote FRONT khi thay duoc mat dau du ro.
    # Mat mui/tai/doi mat bi mat thi khong duoc xem la bang chung BACK.
    head_is_front = (
        nose is not None
        or face_core_visible_count >= 2
        or face_visible_count >= 3
    )
    if head_center is None or not head_is_front:
        return {
            "valid": False,
            "label": "UNKNOWN",
            "center": head_center,
            "visible_count": face_visible_count,
        }

    return {
        "valid": True,
        "label": "FRONT_TO_CAMERA",
        "center": head_center,
        "visible_count": face_visible_count,
    }


def compute_body_facing_evidence(keypoints, conf_th=0.5):
    if keypoints is None:
        return {
            "body_facing": "UNKNOWN",
            "body_facing_confidence": 0.0,
            "body_facing_evidence_count": 0,
            "body_facing_front_votes": 0,
            "body_facing_back_votes": 0,
            "body_facing_reason": "NO_KEYPOINTS",
            "hip_pair_valid": False,
            "shoulder_pair_valid": False,
            "ear_pair_valid": False,
            "head_valid": False,
        }

    hip_evidence = _build_pair_body_facing_evidence(keypoints, 11, 12, conf_th)
    shoulder_evidence = _build_pair_body_facing_evidence(keypoints, 5, 6, conf_th)
    ear_evidence = _build_pair_body_facing_evidence(keypoints, 3, 4, conf_th)
    head_evidence = _build_head_body_facing_evidence(keypoints, conf_th)

    evidence_list = [
        hip_evidence,
        shoulder_evidence,
        ear_evidence,
        head_evidence,
    ]
    front_votes = sum(
        1
        for evidence in evidence_list
        if evidence["valid"] and evidence["label"] == "FRONT_TO_CAMERA"
    )
    back_votes = sum(
        1
        for evidence in evidence_list
        if evidence["valid"] and evidence["label"] == "BACK_TO_CAMERA"
    )
    valid_evidence_count = sum(
        1 for evidence in evidence_list if evidence["valid"]
    )

    if valid_evidence_count == 0:
        body_facing = "UNKNOWN"
        confidence = 0.0
        reason = "NO_VALID_EVIDENCE"
    elif valid_evidence_count < BACKWARD_MIN_VALID_EVIDENCE:
        body_facing = "UNKNOWN"
        confidence = 0.0
        reason = "NOT_ENOUGH_VALID_EVIDENCE"
    elif front_votes > back_votes:
        body_facing = "FRONT_TO_CAMERA"
        confidence = front_votes / valid_evidence_count
        reason = "MAJORITY_FRONT"
    elif back_votes > front_votes:
        body_facing = "BACK_TO_CAMERA"
        confidence = back_votes / valid_evidence_count
        reason = "MAJORITY_BACK"
    else:
        body_facing = "UNKNOWN"
        confidence = 0.0
        reason = "CONFLICTING_EVIDENCE"

    return {
        "body_facing": body_facing,
        "body_facing_confidence": float(confidence),
        "body_facing_evidence_count": int(valid_evidence_count),
        "body_facing_front_votes": int(front_votes),
        "body_facing_back_votes": int(back_votes),
        "body_facing_reason": reason,
        "hip_pair_valid": bool(hip_evidence["valid"]),
        "shoulder_pair_valid": bool(shoulder_evidence["valid"]),
        "ear_pair_valid": bool(ear_evidence["valid"]),
        "head_valid": bool(head_evidence["valid"]),
    }


# Uoc luong huong than/mat de phuc vu logic di lui, khong dung de tinh lane.
def estimate_body_facing(keypoints, conf_th=0.5):
    return compute_body_facing_evidence(keypoints, conf_th).get(
        "body_facing", "UNKNOWN"
    )


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
    body_facing_evidence = compute_body_facing_evidence(keypoints)

    ankle_valid_count = int(left_ankle is not None) + int(right_ankle is not None)

    # feet_point dai dien vi tri nguoi so voi vach giua de xet sai lan.
    if left_ankle is not None and right_ankle is not None:
        feet_point = _midpoint(left_ankle, right_ankle)
        inside_feet_point = feet_point
    else:
        feet_point = left_ankle or right_ankle or bbox_bottom_center
        inside_feet_point = left_ankle or right_ankle
    # Buoc dau chi xem chan "dang tin" khi thay duoc it nhat 1 ankle.
    feet_reliable = ankle_valid_count > 0

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
        "inside_feet_point": inside_feet_point,
        "ankle_valid_count": ankle_valid_count,
        "feet_reliable": feet_reliable,
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
        "body_facing": body_facing_evidence["body_facing"],
        "body_facing_confidence": body_facing_evidence[
            "body_facing_confidence"
        ],
        "body_facing_evidence_count": body_facing_evidence[
            "body_facing_evidence_count"
        ],
        "body_facing_front_votes": body_facing_evidence[
            "body_facing_front_votes"
        ],
        "body_facing_back_votes": body_facing_evidence["body_facing_back_votes"],
        "body_facing_reason": body_facing_evidence["body_facing_reason"],
        "hip_pair_valid": body_facing_evidence["hip_pair_valid"],
        "shoulder_pair_valid": body_facing_evidence["shoulder_pair_valid"],
        "ear_pair_valid": body_facing_evidence["ear_pair_valid"],
        "head_valid": body_facing_evidence["head_valid"],
        "arm_side_order": estimate_arm_side_order(keypoints),
        "keypoint_valid": {
            "nose": _get_keypoint_point(keypoints, 0) is not None,
            "left_eye": _get_keypoint_point(keypoints, 1) is not None,
            "right_eye": _get_keypoint_point(keypoints, 2) is not None,
            "left_ear": _get_keypoint_point(keypoints, 3) is not None,
            "right_ear": _get_keypoint_point(keypoints, 4) is not None,
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
