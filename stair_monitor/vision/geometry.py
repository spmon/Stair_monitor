import math

from dataclasses import dataclass

import numpy as np

from stair_monitor.common.types import BBox, Point, PoseFeatures
from stair_monitor.config.settings import SETTINGS


# Kiem tra keypoint co du tin cay de dung cho cac logic suy luan hay khong.
def _keypoint_is_visible(keypoints, idx, conf_th):
    """Kiem tra 1 keypoint co du tin cay de duoc su dung hay khong.

    Args:
        keypoints: Mang keypoint YOLO pose cua 1 nguoi.
        idx: Chi so keypoint can kiem tra.
        conf_th: Nguong confidence toi thieu.

    Returns:
        bool: True neu keypoint ton tai va vuot nguong confidence.

    Notes:
        Day la lop loc dau vao co ban de cac logic phia sau khong phai lap lai
        viec check do dai mang va confidence.
    """
    return (
        keypoints is not None
        and len(keypoints) > idx
        and len(keypoints[idx]) > 2
        and keypoints[idx][2] > conf_th
    )


# Lay toa do pixel cua keypoint neu do tin cay vuot nguong.
def _get_keypoint_point(keypoints, idx, conf_th=0.5):
    """Lay toa do pixel cua keypoint neu co the tin duoc.

    Args:
        keypoints: Mang keypoint YOLO pose cua 1 nguoi.
        idx: Chi so keypoint can lay.
        conf_th: Nguong confidence toi thieu.

    Returns:
        tuple | None: (x, y) neu keypoint hop le, nguoc lai la None.

    Notes:
        Ham nay la wrapper cua _keypoint_is_visible de dong nhat cach lay point.
    """
    if not _keypoint_is_visible(keypoints, idx, conf_th):
        return None
    return (int(keypoints[idx][0]), int(keypoints[idx][1]))


def _get_keypoint_confidence(keypoints, idx):
    if keypoints is None or len(keypoints) <= idx or len(keypoints[idx]) <= 2:
        return 0.0
    return float(keypoints[idx][2])


# Midpoint duoc dung de tao cac diem dai dien on dinh hon so voi dung 1 keypoint le.
def _midpoint(point_a, point_b):
    """Tinh midpoint giua 2 diem.

    Args:
        point_a: Diem thu nhat.
        point_b: Diem thu hai.

    Returns:
        tuple | None: Trung diem neu du 2 diem, nguoc lai la None.

    Notes:
        Midpoint giup tao diem dai dien on dinh hon cho feet, hip, shoulder.
    """
    if point_a is None or point_b is None:
        return None
    return (
        int((point_a[0] + point_b[0]) / 2),
        int((point_a[1] + point_b[1]) / 2),
    )


def _average_points(points):
    """Lay trung binh toa do cua cac diem hop le.

    Args:
        points: Danh sach diem co the chua None.

    Returns:
        tuple | None: Diem trung binh cua cac diem hop le.

    Notes:
        Ham nay thuong dung cho head_center khi mot so keypoint mat tam thoi.
    """
    valid_points = [point for point in points if point is not None]
    if not valid_points:
        return None
    return (
        int(sum(point[0] for point in valid_points) / len(valid_points)),
        int(sum(point[1] for point in valid_points) / len(valid_points)),
    )


# Tinh goc tai diem b de suy luan tu the tay trong logic carry.
def calculate_angle(a, b, c):
    """Tinh goc ABC theo do.

    Args:
        a: Diem thu nhat.
        b: Dinh cua goc.
        c: Diem thu ba.

    Returns:
        float: Gia tri goc theo do.

    Notes:
        Ham nay duoc carry logic tai su dung de suy ra tay dang gap hay duoi.
    """
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
    """Tinh signed distance tu diem toi duong thang 2 diem.

    Args:
        pt: Diem can tinh khoang cach.
        line_pts: Hai diem tao thanh line tham chieu.

    Returns:
        float: Signed distance. Am/duong cho biet diem nam ben nao cua line.

    Notes:
        Dau cua khoang cach phu thuoc thu tu 2 diem line trong JSON. Vi vay
        mapping LEFT/RIGHT khong duoc sua tuy tien o file config hoac code.
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
    """Tinh khoang cach tu diem toi doan thang.

    Args:
        p: Diem can tinh khoang cach.
        a: Dau mut thu nhat cua doan.
        b: Dau mut thu hai cua doan.

    Returns:
        tuple:
            - dist: Khoang cach Euclidean toi doan.
            - t: He so hinh chieu truoc khi clamp.
            - closest: Diem gan nhat tren doan.

    Notes:
        t cho biet hinh chieu roi vao trong doan [0, 1] hay nam ngoai doan.
        Dieu nay giup handrail logic tranh bat nham phan keo dai vo han.
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
    """Xay bang chung body-facing tu cac keypoint dau/mat.

    Args:
        keypoints: Mang keypoint YOLO pose.
        conf_th: Nguong confidence toi thieu.

    Returns:
        dict: Bang chung head-facing o muc frame-level.

    Notes:
        Ham nay chi vote FRONT khi thay du du mat/mui/tai. Mat keypoint dau
        khong duoc tu dong xem la bang chung BACK.
    """
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
    """Tong hop bang chung body-facing tu hip, shoulder, ear va head.

    Args:
        keypoints: Mang keypoint YOLO pose.
        conf_th: Nguong confidence toi thieu.

    Returns:
        dict: Nhieu field debug cho body_facing va ly do ket luan.

    Notes:
        Ket qua nay duoc backward logic dung de suy ra Di Lui, khong phai lane.
    """
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
    elif valid_evidence_count < SETTINGS.backward.min_valid_evidence:
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
    """Rut gon ket qua body_facing thanh 1 label duy nhat.

    Args:
        keypoints: Mang keypoint YOLO pose.
        conf_th: Nguong confidence toi thieu.

    Returns:
        str: FRONT_TO_CAMERA, BACK_TO_CAMERA hoac UNKNOWN.

    Notes:
        Ham nay chi la wrapper nhe quanh compute_body_facing_evidence.
    """
    return compute_body_facing_evidence(keypoints, conf_th).get(
        "body_facing", "UNKNOWN"
    )


# Xac dinh tay trai dang nam ben trai hay ben phai anh de debug body orientation.
def estimate_arm_side_order(keypoints, conf_th=0.5):
    """Uoc luong tay trai dang xuat hien ben nao trong anh.

    Args:
        keypoints: Mang keypoint YOLO pose.
        conf_th: Nguong confidence toi thieu.

    Returns:
        str: LEFT_ARM_ON_IMAGE_LEFT, LEFT_ARM_ON_IMAGE_RIGHT hoac UNKNOWN.

    Notes:
        Day la field debug ho tro doc body orientation, khong dung de tinh lane.
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

@dataclass(frozen=True)
class ShoulderHipVirtualFeetEstimate:
    point: Point | None
    source: str
    selected_pair: str
    visible_shoulder_count: int
    visible_hip_count: int
    shoulder_anchor: Point | None
    hip_anchor: Point | None
    unavailable_reason: str

def _build_shoulder_hip_unavailable_estimate(
    visible_shoulder_count: int,
    visible_hip_count: int,
    unavailable_reason: str,
) -> ShoulderHipVirtualFeetEstimate:
    return ShoulderHipVirtualFeetEstimate(
        point=None,
        source=f"FEET_UNAVAILABLE_{unavailable_reason}",
        selected_pair="NONE",
        visible_shoulder_count=visible_shoulder_count,
        visible_hip_count=visible_hip_count,
        shoulder_anchor=None,
        hip_anchor=None,
        unavailable_reason=unavailable_reason,
    )


def _project_virtual_foot_from_shoulder_and_hip(
    shoulder_point: Point | None,
    hip_point: Point | None,
    foot_x: int,
    source: str,
    selected_pair: str,
    visible_shoulder_count: int,
    visible_hip_count: int,
) -> ShoulderHipVirtualFeetEstimate:
    if shoulder_point is None or hip_point is None:
        return _build_shoulder_hip_unavailable_estimate(
            visible_shoulder_count,
            visible_hip_count,
            "NO_VALID_SH_HIP_PAIR",
        )

    back_len_y = abs(hip_point[1] - shoulder_point[1])
    if back_len_y <= 0:
        return _build_shoulder_hip_unavailable_estimate(
            visible_shoulder_count,
            visible_hip_count,
            "NO_VALID_SH_HIP_PAIR",
        )

    scale_used = SETTINGS.virtual_feet.shoulder_hip_scale
    virtual_y = int(hip_point[1] + back_len_y * scale_used)
    return ShoulderHipVirtualFeetEstimate(
        point=(int(foot_x), virtual_y),
        source=source,
        selected_pair=selected_pair,
        visible_shoulder_count=visible_shoulder_count,
        visible_hip_count=visible_hip_count,
        shoulder_anchor=shoulder_point,
        hip_anchor=hip_point,
        unavailable_reason="NONE",
    )


def _select_real_ankle_feet_point(
    left_ankle: Point | None,
    right_ankle: Point | None,
) -> tuple[Point | None, str, bool]:
    if left_ankle is not None and right_ankle is not None:
        return _midpoint(left_ankle, right_ankle), "REAL_BOTH_ANKLES", True
    if left_ankle is not None:
        return left_ankle, "REAL_LEFT_ANKLE", True
    if right_ankle is not None:
        return right_ankle, "REAL_RIGHT_ANKLE", True
    return None, "NO_REAL_FEET", False


def _estimate_feet_from_shoulder_and_hip(
    left_shoulder: Point | None,
    right_shoulder: Point | None,
    left_hip: Point | None,
    right_hip: Point | None,
) -> ShoulderHipVirtualFeetEstimate:
    visible_shoulders = [
        ("L", left_shoulder),
        ("R", right_shoulder),
    ]
    visible_shoulders = [
        (label, point)
        for label, point in visible_shoulders
        if point is not None
    ]
    visible_hips = [
        ("L", left_hip),
        ("R", right_hip),
    ]
    visible_hips = [
        (label, point)
        for label, point in visible_hips
        if point is not None
    ]
    visible_shoulder_count = len(visible_shoulders)
    visible_hip_count = len(visible_hips)

    if visible_shoulder_count == 0 and visible_hip_count == 0:
        return _build_shoulder_hip_unavailable_estimate(
            visible_shoulder_count,
            visible_hip_count,
            "NO_SHOULDER_NO_HIP",
        )
    if visible_shoulder_count == 0:
        return _build_shoulder_hip_unavailable_estimate(
            visible_shoulder_count,
            visible_hip_count,
            "NO_SHOULDER",
        )
    if visible_hip_count == 0:
        return _build_shoulder_hip_unavailable_estimate(
            visible_shoulder_count,
            visible_hip_count,
            "NO_HIP",
        )

    if visible_shoulder_count == 2 and visible_hip_count == 2:
        left_estimate = _project_virtual_foot_from_shoulder_and_hip(
            left_shoulder,
            left_hip,
            left_hip[0],
            "VIRTUAL_FROM_SH_LEFT_HIP_LEFT",
            "SH_L+HIP_L",
            visible_shoulder_count,
            visible_hip_count,
        )
        right_estimate = _project_virtual_foot_from_shoulder_and_hip(
            right_shoulder,
            right_hip,
            right_hip[0],
            "VIRTUAL_FROM_SH_RIGHT_HIP_RIGHT",
            "SH_R+HIP_R",
            visible_shoulder_count,
            visible_hip_count,
        )
        if left_estimate.point is not None and right_estimate.point is not None:
            return ShoulderHipVirtualFeetEstimate(
                point=_midpoint(left_estimate.point, right_estimate.point),
                source="VIRTUAL_FROM_SH_HIP_BOTH_SIDES",
                selected_pair="SH_L+HIP_L|SH_R+HIP_R",
                visible_shoulder_count=visible_shoulder_count,
                visible_hip_count=visible_hip_count,
                shoulder_anchor=_average_points([left_shoulder, right_shoulder]),
                hip_anchor=_average_points([left_hip, right_hip]),
                unavailable_reason="NONE",
            )
        if left_estimate.point is not None:
            return left_estimate
        if right_estimate.point is not None:
            return right_estimate
        return _build_shoulder_hip_unavailable_estimate(
            visible_shoulder_count,
            visible_hip_count,
            "NO_VALID_SH_HIP_PAIR",
        )

    if visible_shoulder_count == 2 and visible_hip_count == 1:
        hip_label, hip_point = visible_hips[0]
        shoulder_point = _average_points([left_shoulder, right_shoulder])
        return _project_virtual_foot_from_shoulder_and_hip(
            shoulder_point,
            hip_point,
            hip_point[0],
            "VIRTUAL_FROM_AVG_SHOULDERS_1HIP",
            f"AVG_SH+HIP_{hip_label}",
            visible_shoulder_count,
            visible_hip_count,
        )

    if visible_shoulder_count == 1 and visible_hip_count == 2:
        shoulder_label, shoulder_point = visible_shoulders[0]
        hip_point = _average_points([left_hip, right_hip])
        return _project_virtual_foot_from_shoulder_and_hip(
            shoulder_point,
            hip_point,
            hip_point[0],
            "VIRTUAL_FROM_1SH_2HIP_AVG_HIP",
            f"SH_{shoulder_label}+AVG_2HIP",
            visible_shoulder_count,
            visible_hip_count,
        )

    shoulder_label, shoulder_point = visible_shoulders[0]
    hip_label, hip_point = visible_hips[0]
    if shoulder_label == hip_label:
        return _project_virtual_foot_from_shoulder_and_hip(
            shoulder_point,
            hip_point,
            hip_point[0],
            (
                "VIRTUAL_FROM_SH_LEFT_HIP_LEFT"
                if shoulder_label == "L"
                else "VIRTUAL_FROM_SH_RIGHT_HIP_RIGHT"
            ),
            f"SH_{shoulder_label}+HIP_{hip_label}",
            visible_shoulder_count,
            visible_hip_count,
        )
    return _project_virtual_foot_from_shoulder_and_hip(
        shoulder_point,
        hip_point,
        int((shoulder_point[0] + hip_point[0]) / 2),
        (
            "VIRTUAL_FROM_SH_LEFT_HIP_RIGHT"
            if shoulder_label == "L"
            else "VIRTUAL_FROM_SH_RIGHT_HIP_LEFT"
        ),
        f"SH_{shoulder_label}+HIP_{hip_label}",
        visible_shoulder_count,
        visible_hip_count,
    )


def _select_feet_point_from_candidates(
    real_feet_point: Point | None,
    real_feet_source: str,
    virtual_feet_from_shoulder_hip: Point | None,
) -> tuple[Point | None, str, bool]:
    # Priority bat buoc:
    # 1. ankle that
    # 2. shoulder + hip
    if real_feet_point is not None:
        return real_feet_point, real_feet_source, True
    if virtual_feet_from_shoulder_hip is not None:
        return virtual_feet_from_shoulder_hip, "VIRTUAL_FROM_SHOULDER_HIP", True
    return None, "FEET_UNAVAILABLE", False


def _compare_feet_points(
    real_feet_point: Point | None,
    candidate_feet_point: Point | None,
) -> tuple[int | None, int | None, float | None, bool]:
    if real_feet_point is None or candidate_feet_point is None:
        return None, None, None, False

    dx = int(candidate_feet_point[0] - real_feet_point[0])
    dy = int(candidate_feet_point[1] - real_feet_point[1])
    return dx, dy, float(math.hypot(dx, dy)), True


# Tao torso box tu shoulder va hip de carry check vung truoc nguc/bung.
def _get_torso_box_from_features(features, margin_x=40, margin_y=40, margin_bottom=90):
    """Dung pose feature de tao torso box xap xi.

    Args:
        features: Dict pose feature da extract.
        margin_x: Margin ngang cu duoc giu de giu API.
        margin_y: Margin doc cu duoc giu de giu API.
        margin_bottom: Margin duoi cu duoc giu de giu API.

    Returns:
        tuple | None: (x1, y1, x2, y2) neu du shoulder va hip.

    Notes:
        Torso box nay chu yeu phuc vu carry logic, khong phai handrail.
    """
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
# p_lane/inside chi dung chan that hoac SH-HIP; khong dung 2 shoulder hay bbox fallback.
# monitor_point_hip va monitor_point_shoulder duoc tach rieng cho direction de tranh tron source.
# motion_point chi con la diem compatibility cho cac logic cu van can p_motion.
def extract_pose_features(
    keypoints,
    bbox,
) -> PoseFeatures:
    """Gom cac pose feature co the tai su dung o nhieu logic.

    Args:
        keypoints: Mang keypoint YOLO pose cua 1 nguoi.
        bbox: Bounding box cua nguoi do trong frame hien tai.

    Returns:
        dict: Bo feature chuan hoa duoc dung boi analyzer, carry va rendering.

    Notes:
        p_lane uu tien feet_point vi lane can vi tri chan that. Direction moi
        dung monitor_point_hip va monitor_point_shoulder rieng. motion_point
        chi giu backward compatibility cho cac caller/logic van can p_motion.
    """
    bbox_tuple: BBox | None = None
    if bbox is not None and len(bbox) >= 4:
        bbox_tuple = (
            int(bbox[0]),
            int(bbox[1]),
            int(bbox[2]),
            int(bbox[3]),
        )

    nose = _get_keypoint_point(keypoints, 0)
    left_eye = _get_keypoint_point(keypoints, 1)
    right_eye = _get_keypoint_point(keypoints, 2)
    left_ear = _get_keypoint_point(keypoints, 3)
    right_ear = _get_keypoint_point(keypoints, 4)
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
    left_ankle_conf = _get_keypoint_confidence(keypoints, 15)
    right_ankle_conf = _get_keypoint_confidence(keypoints, 16)

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
    monitor_point_hip = hip_center
    monitor_point_hip_source = (
        "HIP_CENTER" if monitor_point_hip is not None else "NO_HIP_CENTER"
    )
    monitor_point_shoulder = shoulder_center
    monitor_point_shoulder_source = (
        "SHOULDER_CENTER"
        if monitor_point_shoulder is not None
        else "NO_SHOULDER_CENTER"
    )
    # head_center uu tien tong hop nhieu keypoint dau de giam anh huong keypoint le.
    head_center = _average_points([nose, left_eye, right_eye, left_ear, right_ear])
    body_facing_evidence = compute_body_facing_evidence(keypoints)
    bbox_height = (
        int(bbox_tuple[3] - bbox_tuple[1]) if bbox_tuple is not None else None
    )
    shoulder_hip_scale_used = SETTINGS.virtual_feet.shoulder_hip_scale

    ankle_valid_count = int(left_ankle is not None) + int(right_ankle is not None)
    real_feet_point, real_feet_source, _ = _select_real_ankle_feet_point(
        left_ankle,
        right_ankle,
    )
    sh_hip_estimate = _estimate_feet_from_shoulder_and_hip(
        left_shoulder,
        right_shoulder,
        left_hip,
        right_hip,
    )
    virtual_feet_from_shoulder_hip = sh_hip_estimate.point
    virtual_feet_from_shoulder_hip_source = sh_hip_estimate.source

    # left_ankle/right_ankle la dau vao uu tien nhat cho lane va inside stairs.
    # feet_point dai dien vi tri nguoi so voi vach giua de xet sai lan.
    feet_point, feet_point_source, feet_reliable = _select_feet_point_from_candidates(
        real_feet_point,
        real_feet_source,
        virtual_feet_from_shoulder_hip,
    )
    feet_available = feet_point is not None
    feet_unavailable_reason = (
        "NONE" if feet_available else sh_hip_estimate.unavailable_reason
    )
    inside_feet_point = feet_point
    inside_feet_point_source = feet_point_source
    selected_feet_point = feet_point
    selected_feet_source = feet_point_source
    (
        shoulder_hip_feet_dx,
        shoulder_hip_feet_dy,
        shoulder_hip_feet_distance,
        shoulder_hip_feet_compare_available,
    ) = _compare_feet_points(real_feet_point, virtual_feet_from_shoulder_hip)
    # hip_center/shoulder_center/torso_box giup carry va cac logic cu dung cung 1 bo moc.
    # Direction moi se dung monitor point rieng, con motion_point nay chi giu backward compatibility.
    motion_point = monitor_point_hip or monitor_point_shoulder or bbox_center or bbox_bottom_center
    # bbox_center/bbox_bottom_center chi la fallback khi keypoint bi mat.
    features: PoseFeatures = {
        "bbox": bbox_tuple,
        "bbox_center": bbox_center,
        "bbox_bottom_center": bbox_bottom_center,
        "bbox_height": bbox_height,
        "nose": nose,
        "left_eye": left_eye,
        "right_eye": right_eye,
        "left_ear": left_ear,
        "right_ear": right_ear,
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
        "left_ankle_conf": left_ankle_conf,
        "right_ankle_conf": right_ankle_conf,
        "hip_center": hip_center,
        "head_center": head_center,
        "shoulder_center": shoulder_center,
        "monitor_point_hip": monitor_point_hip,
        "monitor_point_hip_source": monitor_point_hip_source,
        "monitor_point_shoulder": monitor_point_shoulder,
        "monitor_point_shoulder_source": monitor_point_shoulder_source,
        # motion_point la p_motion compatibility cho backward/standing va caller cu.
        "motion_point": motion_point,
        # feet_point la p_lane de analyzer tinh inside/lane.
        "feet_point": feet_point,
        "feet_point_source": feet_point_source,
        "feet_available": feet_available,
        "feet_unavailable_reason": feet_unavailable_reason,
        "real_feet_point": real_feet_point,
        "real_feet_source": real_feet_source,
        "sh_hip_visible_shoulder_count": sh_hip_estimate.visible_shoulder_count,
        "sh_hip_visible_hip_count": sh_hip_estimate.visible_hip_count,
        "sh_hip_selected_pair": sh_hip_estimate.selected_pair,
        "sh_hip_virtual_feet_point": sh_hip_estimate.point,
        "sh_hip_virtual_feet_source": sh_hip_estimate.source,
        "sh_hip_anchor_shoulder_point": sh_hip_estimate.shoulder_anchor,
        "sh_hip_anchor_hip_point": sh_hip_estimate.hip_anchor,
        "virtual_feet_from_shoulder_hip": virtual_feet_from_shoulder_hip,
        "virtual_feet_from_shoulder_hip_source": virtual_feet_from_shoulder_hip_source,
        "selected_feet_point": selected_feet_point,
        "selected_feet_source": selected_feet_source,
        "shoulder_hip_feet_dx": shoulder_hip_feet_dx,
        "shoulder_hip_feet_dy": shoulder_hip_feet_dy,
        "shoulder_hip_feet_distance": shoulder_hip_feet_distance,
        "shoulder_hip_feet_compare_available": shoulder_hip_feet_compare_available,
        "shoulder_hip_scale_used": shoulder_hip_scale_used,
        "inside_feet_point": inside_feet_point,
        "inside_feet_point_source": inside_feet_point_source,
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
        # body_facing va cac vote/reason la bang chung cho logic Di Lui.
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
            "nose": nose is not None,
            "left_eye": left_eye is not None,
            "right_eye": right_eye is not None,
            "left_ear": left_ear is not None,
            "right_ear": right_ear is not None,
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
