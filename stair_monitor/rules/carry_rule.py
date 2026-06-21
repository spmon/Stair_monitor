from __future__ import annotations

import math

from stair_monitor.common.types import BBox, KeypointsArray, Point, PoseFeatures
from stair_monitor.config.settings import SETTINGS
from stair_monitor.vision.geometry import extract_pose_features


# Tai su dung pose feature da tinh san de tranh tinh lap lai trong cung 1 frame.
def _get_features(
    features: PoseFeatures | None,
    keypoints: KeypointsArray | None,
) -> PoseFeatures:
    """Lay pose feature da extract hoac tu extract moi neu can.

    Args:
        features: Dict feature da co san, co the la None.
        keypoints: Mang keypoint YOLO pose.

    Returns:
        dict: Bo feature da san sang cho carry logic.

    Notes:
        Carry code co the duoc goi tu analyzer sau khi feature da tinh san.
        Ham nay giup giu 1 diem vao chung.
    """
    if features is not None:
        return features
    return extract_pose_features(keypoints, None)


# Khoang cach co ban dung cho shoulder/torso/wrist.
def _point_distance(
    point_a: Point | None,
    point_b: Point | None,
) -> float | None:
    """Tinh khoang cach Euclidean giua 2 diem.

    Args:
        point_a: Diem thu nhat.
        point_b: Diem thu hai.

    Returns:
        float | None: Khoang cach neu du 2 diem, nguoc lai la None.

    Notes:
        Day la helper nho cho body scale va cac threshold dong.
    """
    if point_a is None or point_b is None:
        return None
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


# Tinh kich thuoc co the dong theo tung nguoi.
# Carry uu tien threshold theo ti le co the, khong dua vao pixel cung.
def _compute_body_scale(pose_features: PoseFeatures) -> dict[str, float | None]:
    """Tinh scale dong cua co the de carry khong phu thuoc pixel cung.

    Args:
        pose_features: Dict feature da extract.

    Returns:
        dict: body_scale, shoulder_width, torso_height, bbox_height.

    Notes:
        body_scale uu tien torso_height, roi moi fallback bbox_height va shoulder_width.
    """
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
def _horizontal_carry_scale(scale_info: dict[str, float | None]) -> float:
    """Lay scale ngang cho carry threshold.

    Args:
        scale_info: Dict tra ve tu _compute_body_scale.

    Returns:
        float: Scale ngang de tinh threshold dong.

    Notes:
        Uu tien shoulder_width de theo sat be rong than tren that.
    """
    shoulder_width = scale_info.get("shoulder_width")
    if shoulder_width is not None and shoulder_width > 0:
        return shoulder_width
    return max(1.0, scale_info["body_scale"] * 0.4)


# Chieu doc cho carry uu tien torso_height de phu hop voi vung truoc nguc/bung.
def _vertical_carry_scale(scale_info: dict[str, float | None]) -> float:
    """Lay scale doc cho carry threshold.

    Args:
        scale_info: Dict tra ve tu _compute_body_scale.

    Returns:
        float: Scale doc de tinh threshold dong.

    Notes:
        Torso height giup torso box va wrist check dong theo tung nguoi.
    """
    torso_height = scale_info.get("torso_height")
    if torso_height is not None and torso_height > 0:
        return torso_height
    return max(1.0, scale_info["body_scale"])


# Kiem tra tay co gap goc giong tu the om/mang vat hay khong.
def is_arm_bent_for_carrying(
    keypoints: KeypointsArray | None,
    wrist_idx: int,
    angle_threshold: float = SETTINGS.carry.carry_arm_angle_threshold,
    features: PoseFeatures | None = None,
) -> tuple[bool, float | None]:
    """Kiem tra 1 tay co gap theo tu the Mang Vac hay khong.

    Args:
        keypoints: Mang keypoint YOLO pose.
        wrist_idx: Chi so wrist cua tay dang xet.
        angle_threshold: Nguong goc toi da de xem la tay dang gap.
        features: Dict feature da extract neu caller co san.

    Returns:
        tuple: (carrying_like, elbow_angle)

    Notes:
        Day chi la bang chung raw cho carry. Ket luan cuoi cung con phai qua
        torso-box check va history o carry_analysis.
    """
    pose_features = _get_features(features, keypoints)
    side = "left" if wrist_idx == 9 else "right"
    angle = pose_features.get(f"{side}_arm_angle")
    return angle is not None and angle < angle_threshold, angle


# Tao torso box dong de kiem tra co tay co nam trong vung truoc nguoi hay khong.
# Cac margin ratio/pixel o day chi phuc vu logic Mang Vac, khong duoc dung sang logic vin tay.
def get_torso_box(
    keypoints: KeypointsArray | None,
    margin_x: int = 40,
    margin_y: int = 40,
    margin_bottom: int = 90,
    features: PoseFeatures | None = None,
) -> BBox | None:
    """Tao torso box dong de kiem tra co tay dang om vat truoc nguoi.

    Args:
        keypoints: Mang keypoint YOLO pose.
        margin_x: Margin cu duoc giu de giu API.
        margin_y: Margin cu duoc giu de giu API.
        margin_bottom: Margin cu duoc giu de giu API.
        features: Dict feature da extract neu caller co san.

    Returns:
        tuple | None: (x1, y1, x2, y2) neu du shoulder/hip.

    Notes:
        Torso box nay chi phuc vu carry raw. Khong nen dung lai cho hold.
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
        float(SETTINGS.carry.min_wrist_to_torso_x_px),
        horizontal_scale * SETTINGS.carry.wrist_to_torso_x_ratio,
    )
    dynamic_margin_y = max(
        float(SETTINGS.carry.min_carry_region_margin_px),
        vertical_scale * SETTINGS.carry.carry_region_margin_ratio,
    )
    dynamic_margin_bottom = max(
        float(SETTINGS.carry.min_wrist_to_torso_y_px),
        vertical_scale * SETTINGS.carry.wrist_to_torso_y_ratio,
    )

    torso_points = shoulders + hips
    return (
        int(min(point[0] for point in torso_points) - dynamic_margin_x),
        int(min(point[1] for point in torso_points) - dynamic_margin_y),
        int(max(point[0] for point in torso_points) + dynamic_margin_x),
        int(max(point[1] for point in hips) + dynamic_margin_bottom),
    )


# Kiem tra co tay co nam trong vung torso hay khong de suy luan dang om vat truoc nguoi.
def is_wrist_in_torso_area(
    keypoints: KeypointsArray | None,
    wrist_idx: int,
    torso_box: BBox | None = None,
    features: PoseFeatures | None = None,
) -> bool:
    """Kiem tra wrist co nam trong torso box hay khong.

    Args:
        keypoints: Mang keypoint YOLO pose.
        wrist_idx: Chi so wrist trai/phai can kiem tra.
        torso_box: Torso box da tinh san, co the la None.
        features: Dict feature da extract neu caller co san.

    Returns:
        bool: True neu wrist nam trong torso box.

    Notes:
        Wrist in torso la bang chung raw quan trong cho front carry.
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
def calc_wrist_distance(
    keypoints: KeypointsArray | None,
    features: PoseFeatures | None = None,
) -> tuple[int | None, int | None, float | None]:
    """Tinh do gan nhau giua hai wrist.

    Args:
        keypoints: Mang keypoint YOLO pose.
        features: Dict feature da extract neu caller co san.

    Returns:
        tuple: (wrist_dx, wrist_dy, wrist_distance)

    Notes:
        Carry raw dung ca dx va dy vi tu the om vat can hai co tay vua gan ngang
        vua gan doc.
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
def detect_carrying_pose(
    keypoints: KeypointsArray | None,
    holding: bool = False,
    features: PoseFeatures | None = None,
) -> dict[str, object]:
    """Tao bang chung carry raw cho 1 frame.

    Args:
        keypoints: Mang keypoint YOLO pose.
        holding: Bien cu duoc giu de giu API voi caller hien tai.
        features: Dict feature da extract neu caller co san.

    Returns:
        dict: Nhieu field raw cho carry, torso, wrist va arm posture.

    Notes:
        front_carry_raw la ket qua frame-level. Confirmed/final se duoc carry
        history xu ly o carry_analysis. Carry khong duoc ghi de ket qua hold.
    """
    _ = holding
    pose_features = _get_features(features, keypoints)
    scale_info = _compute_body_scale(pose_features)
    horizontal_scale = _horizontal_carry_scale(scale_info)
    vertical_scale = _vertical_carry_scale(scale_info)

    _, left_angle = is_arm_bent_for_carrying(
        keypoints,
        9,
        features=pose_features,
    )
    _, right_angle = is_arm_bent_for_carrying(
        keypoints,
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
        float(SETTINGS.carry.min_wrist_together_x_px),
        horizontal_scale * SETTINGS.carry.wrist_together_x_ratio,
    )
    wrist_dy_threshold = max(
        float(SETTINGS.carry.min_wrist_together_y_px),
        vertical_scale * SETTINGS.carry.wrist_together_y_ratio,
    )
    left_bent = (
        left_angle is not None
        and left_angle < SETTINGS.carry.carry_arm_angle_threshold
    )
    right_bent = (
        right_angle is not None
        and right_angle < SETTINGS.carry.carry_arm_angle_threshold
    )
    wrists_close = (
        wrist_dx is not None
        and wrist_dy is not None
        and wrist_dx <= wrist_dx_threshold
        and wrist_dy <= wrist_dy_threshold
    )
    any_wrist_in_torso = left_wrist_in_torso or right_wrist_in_torso
    both_wrist_in_torso = left_wrist_in_torso and right_wrist_in_torso

    # One-arm strong case cho phep bat duoc tu the om vat bang 1 tay ro rang.
    # Day la nguon bang chung cho front carry one-arm.
    strong_left_front = (
        left_bent
        and left_wrist_in_torso
        and left_angle is not None
        and left_angle < SETTINGS.carry.strong_one_arm_angle_threshold
    )
    strong_right_front = (
        right_bent
        and right_wrist_in_torso
        and right_angle is not None
        and right_angle < SETTINGS.carry.strong_one_arm_angle_threshold
    )

    # two-hand carry = hai wrist gan nhau + nam trong torso + co it nhat 1 tay gap.
    front_carry_two_hand = (
        wrists_close
        and both_wrist_in_torso
        and (left_bent or right_bent)
    )
    # one-arm carry = 1 tay gap ro rang va nam trong torso.
    front_carry_strong_one_arm = strong_left_front or strong_right_front
    left_carry_raw = front_carry_two_hand or strong_left_front
    right_carry_raw = front_carry_two_hand or strong_right_front
    front_carry_raw = left_carry_raw or right_carry_raw
    left_carry_score = (
        float(SETTINGS.carry.strong_one_arm_angle_threshold - left_angle)
        if strong_left_front and left_angle is not None
        else None
    )
    right_carry_score = (
        float(SETTINGS.carry.strong_one_arm_angle_threshold - right_angle)
        if strong_right_front and right_angle is not None
        else None
    )

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
        "left_carry_score": left_carry_score,
        "right_carry_score": right_carry_score,
        "left_carry": left_carry,
        "right_carry": right_carry,
    }
