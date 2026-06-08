import math

from stair_monitor.geometry import extract_pose_features
from stair_monitor.settings import (
    CARRY_REGION_MARGIN_RATIO,
    CARRY_ARM_ANGLE_THRESHOLD,
    MIN_CARRY_REGION_MARGIN_PX,
    MIN_WRIST_TOGETHER_X_PX,
    MIN_WRIST_TOGETHER_Y_PX,
    MIN_WRIST_TO_TORSO_X_PX,
    MIN_WRIST_TO_TORSO_Y_PX,
    STRONG_ONE_ARM_ANGLE_THRESHOLD,
    WRIST_TOGETHER_X_RATIO,
    WRIST_TOGETHER_Y_RATIO,
    WRIST_TO_TORSO_X_RATIO,
    WRIST_TO_TORSO_Y_RATIO,
)


# Tai su dung pose feature da tinh san de tranh tinh lap lai trong cung 1 frame.
def _get_features(features, keypoints):
    if features is not None:
        return features
    return extract_pose_features(keypoints, None)


# Khoang cach co ban dung cho shoulder/torso/wrist.
def _point_distance(point_a, point_b):
    if point_a is None or point_b is None:
        return None
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


# Tinh kich thuoc co the dong theo tung nguoi.
# Carry uu tien threshold theo ti le co the, khong dua vao pixel cung.
def _compute_body_scale(pose_features):
    shoulder_width = _point_distance(
        pose_features.get("left_shoulder"),
        pose_features.get("right_shoulder"),
    )
    torso_height = _point_distance(
        pose_features.get("shoulder_center"),
        pose_features.get("hip_center"),
    )

    bbox_height = None
    bbox = pose_features.get("bbox")
    if bbox is not None and len(bbox) >= 4:
        bbox_height = max(0.0, float(bbox[3] - bbox[1]))

    body_scale = torso_height
    if body_scale is None or body_scale <= 0:
        body_scale = bbox_height
    if body_scale is None or body_scale <= 0:
        body_scale = shoulder_width
    if body_scale is None or body_scale <= 0:
        body_scale = 1.0

    return {
        "body_scale": float(body_scale),
        "shoulder_width": shoulder_width,
        "torso_height": torso_height,
        "bbox_height": bbox_height,
    }


# Chieu ngang cho carry uu tien shoulder_width de phu hop voi be rong than tren.
def _horizontal_carry_scale(scale_info):
    shoulder_width = scale_info.get("shoulder_width")
    if shoulder_width is not None and shoulder_width > 0:
        return shoulder_width
    return max(1.0, scale_info["body_scale"] * 0.4)


# Chieu doc cho carry uu tien torso_height de phu hop voi vung truoc nguc/bung.
def _vertical_carry_scale(scale_info):
    torso_height = scale_info.get("torso_height")
    if torso_height is not None and torso_height > 0:
        return torso_height
    return max(1.0, scale_info["body_scale"])


# Kiem tra tay co gap goc giong tu the om/mang vat hay khong.
def is_arm_bent_for_carrying(
    keypoints,
    shoulder_idx,
    elbow_idx,
    wrist_idx,
    angle_threshold=CARRY_ARM_ANGLE_THRESHOLD,
    features=None,
):
    """
    Detect carrying-like arm posture based on elbow angle.
    Returns:
    - carrying: True/False
    - angle: elbow angle or None
    """
    pose_features = _get_features(features, keypoints)
    side = "left" if wrist_idx == 9 else "right"
    angle = pose_features.get(f"{side}_arm_angle")
    return angle is not None and angle < angle_threshold, angle


# Tao torso box dong de kiem tra co tay co nam trong vung truoc nguoi hay khong.
# Cac margin ratio/pixel o day chi phuc vu logic Mang Vac, khong duoc dung sang logic vin tay.
def get_torso_box(keypoints, margin_x=40, margin_y=40, margin_bottom=90, features=None):
    """
    Approximate torso box from shoulders and hips.
    Returns (x1, y1, x2, y2) or None.
    """
    _ = keypoints
    _ = margin_x
    _ = margin_y
    _ = margin_bottom
    pose_features = _get_features(features, keypoints)
    shoulders = [
        point
        for point in (
            pose_features.get("left_shoulder"),
            pose_features.get("right_shoulder"),
        )
        if point is not None
    ]
    hips = [
        point
        for point in (
            pose_features.get("left_hip"),
            pose_features.get("right_hip"),
        )
        if point is not None
    ]
    if not shoulders or not hips:
        return pose_features.get("torso_box")

    scale_info = _compute_body_scale(pose_features)
    horizontal_scale = _horizontal_carry_scale(scale_info)
    vertical_scale = _vertical_carry_scale(scale_info)
    # MIN_*_PX la nguong san de nguoi o xa van khong tao threshold qua nho.
    dynamic_margin_x = max(
        float(MIN_WRIST_TO_TORSO_X_PX),
        horizontal_scale * WRIST_TO_TORSO_X_RATIO,
    )
    dynamic_margin_y = max(
        float(MIN_CARRY_REGION_MARGIN_PX),
        vertical_scale * CARRY_REGION_MARGIN_RATIO,
    )
    dynamic_margin_bottom = max(
        float(MIN_WRIST_TO_TORSO_Y_PX),
        vertical_scale * WRIST_TO_TORSO_Y_RATIO,
    )

    torso_points = shoulders + hips
    return (
        int(min(point[0] for point in torso_points) - dynamic_margin_x),
        int(min(point[1] for point in torso_points) - dynamic_margin_y),
        int(max(point[0] for point in torso_points) + dynamic_margin_x),
        int(max(point[1] for point in hips) + dynamic_margin_bottom),
    )


# Kiem tra co tay co nam trong vung torso hay khong de suy luan dang om vat truoc nguoi.
def is_wrist_in_torso_area(keypoints, wrist_idx, torso_box=None, features=None):
    """
    Check whether wrist lies inside torso/chest/belly area.
    """
    pose_features = _get_features(features, keypoints)
    wrist_key = "left_wrist" if wrist_idx == 9 else "right_wrist"
    wrist_point = pose_features.get(wrist_key)
    if wrist_point is None:
        return False

    if torso_box is None:
        torso_box = pose_features.get("torso_box")
    if torso_box is None:
        return False

    x1, y1, x2, y2 = torso_box
    x, y = wrist_point
    return x1 <= x <= x2 and y1 <= y <= y2


# So sanh hai co tay de xem chung co du gan nhau nhu tu the om vat khong.
def calc_wrist_distance(keypoints, features=None):
    """
    Calculate wrist offsets and Euclidean distance between left and right wrists.
    """
    pose_features = _get_features(features, keypoints)
    left_wrist = pose_features.get("left_wrist")
    right_wrist = pose_features.get("right_wrist")
    if left_wrist is None or right_wrist is None:
        return None, None, None

    wrist_dx = abs(right_wrist[0] - left_wrist[0])
    wrist_dy = abs(right_wrist[1] - left_wrist[1])
    wrist_distance = math.hypot(wrist_dx, wrist_dy)
    return wrist_dx, wrist_dy, wrist_distance


# Suy luan carry tu pose tay/co tay/khuuyu tay.
# Cac threshold ratio trong ham nay chi danh cho Mang Vac, khong duoc dung lai cho handrail/hold.
def detect_carrying_pose(keypoints, holding=False, best_wrist="NONE", features=None):
    """
    Detect carrying from pose.
    """
    _ = holding
    _ = best_wrist
    pose_features = _get_features(features, keypoints)
    scale_info = _compute_body_scale(pose_features)
    horizontal_scale = _horizontal_carry_scale(scale_info)
    vertical_scale = _vertical_carry_scale(scale_info)

    _, left_angle = is_arm_bent_for_carrying(
        keypoints,
        5,
        7,
        9,
        features=pose_features,
    )
    _, right_angle = is_arm_bent_for_carrying(
        keypoints,
        6,
        8,
        10,
        features=pose_features,
    )
    torso_box = get_torso_box(keypoints, features=pose_features)
    left_wrist_in_torso = is_wrist_in_torso_area(
        keypoints,
        9,
        torso_box=torso_box,
        features=pose_features,
    )
    right_wrist_in_torso = is_wrist_in_torso_area(
        keypoints,
        10,
        torso_box=torso_box,
        features=pose_features,
    )
    wrist_dx, wrist_dy, wrist_distance = calc_wrist_distance(
        keypoints,
        features=pose_features,
    )
    # wrist_together dung de kiem tra hai co tay co du gan nhau theo chieu ngang/doc hay khong.
    wrist_dx_threshold = max(
        float(MIN_WRIST_TOGETHER_X_PX),
        horizontal_scale * WRIST_TOGETHER_X_RATIO,
    )
    wrist_dy_threshold = max(
        float(MIN_WRIST_TOGETHER_Y_PX),
        vertical_scale * WRIST_TOGETHER_Y_RATIO,
    )
    left_bent = left_angle is not None and left_angle < CARRY_ARM_ANGLE_THRESHOLD
    right_bent = right_angle is not None and right_angle < CARRY_ARM_ANGLE_THRESHOLD
    wrists_close = (
        wrist_dx is not None
        and wrist_dy is not None
        and wrist_dx <= wrist_dx_threshold
        and wrist_dy <= wrist_dy_threshold
    )
    any_wrist_in_torso = left_wrist_in_torso or right_wrist_in_torso
    both_wrist_in_torso = left_wrist_in_torso and right_wrist_in_torso

    # One-arm strong case cho phep bat duoc tu the om vat bang 1 tay ro rang.
    strong_left_front = (
        left_bent
        and left_wrist_in_torso
        and left_angle is not None
        and left_angle < STRONG_ONE_ARM_ANGLE_THRESHOLD
    )
    strong_right_front = (
        right_bent
        and right_wrist_in_torso
        and right_angle is not None
        and right_angle < STRONG_ONE_ARM_ANGLE_THRESHOLD
    )

    front_carry_two_hand = (
        wrists_close
        and both_wrist_in_torso
        and (left_bent or right_bent)
    )
    front_carry_strong_one_arm = strong_left_front or strong_right_front
    left_carry_raw = front_carry_two_hand or strong_left_front
    right_carry_raw = front_carry_two_hand or strong_right_front
    front_carry_raw = left_carry_raw or right_carry_raw

    left_carry = left_carry_raw
    right_carry = right_carry_raw
    is_carrying = left_carry or right_carry
    carry_type = "FRONT_CARRY" if front_carry_raw else "NONE"

    if left_carry and right_carry:
        carrying_arm = "BOTH"
    elif left_carry:
        carrying_arm = "LEFT_ARM"
    elif right_carry:
        carrying_arm = "RIGHT_ARM"
    else:
        carrying_arm = "NONE"

    return {
        "is_carrying": is_carrying,
        "carrying_arm": carrying_arm,
        "carry_type": carry_type,
        "left_arm_angle": left_angle,
        "right_arm_angle": right_angle,
        "left_wrist_in_torso": left_wrist_in_torso,
        "right_wrist_in_torso": right_wrist_in_torso,
        "body_scale": scale_info["body_scale"],
        "shoulder_width": scale_info.get("shoulder_width"),
        "torso_height": scale_info.get("torso_height"),
        "wrist_dx": wrist_dx,
        "wrist_dx_threshold": wrist_dx_threshold,
        "wrist_dy": wrist_dy,
        "wrist_dy_threshold": wrist_dy_threshold,
        "wrist_distance": wrist_distance,
        "left_bent": left_bent,
        "right_bent": right_bent,
        "wrists_close": wrists_close,
        "any_wrist_in_torso": any_wrist_in_torso,
        "both_wrist_in_torso": both_wrist_in_torso,
        "strong_left_front": strong_left_front,
        "strong_right_front": strong_right_front,
        "front_carry_two_hand": front_carry_two_hand,
        "front_carry_strong_one_arm": front_carry_strong_one_arm,
        "front_carry_two_hand_raw": front_carry_two_hand,
        "front_carry_one_arm_raw": front_carry_strong_one_arm,
        "front_carry": front_carry_raw,
        "front_carry_raw": front_carry_raw,
        "left_carry_raw": left_carry_raw,
        "right_carry_raw": right_carry_raw,
        "left_carry": left_carry,
        "right_carry": right_carry,
    }
