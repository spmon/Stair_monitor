import math

from stair_monitor.geometry import calculate_angle
from stair_monitor.settings import (
    CARRY_ARM_ANGLE_THRESHOLD,
    STRONG_ONE_ARM_ANGLE_THRESHOLD,
    WRIST_TOGETHER_X_THRESHOLD,
    WRIST_TOGETHER_Y_THRESHOLD,
)


def is_arm_bent_for_carrying(
    keypoints,
    shoulder_idx,
    elbow_idx,
    wrist_idx,
    angle_threshold=CARRY_ARM_ANGLE_THRESHOLD,
):
    """
    Detect carrying-like arm posture based on elbow angle.
    Returns:
    - carrying: True/False
    - angle: elbow angle or None
    """
    if keypoints is None:
        return False, None

    max_idx = max(shoulder_idx, elbow_idx, wrist_idx)
    if len(keypoints) <= max_idx:
        return False, None

    if (
        len(keypoints[shoulder_idx]) <= 2
        or len(keypoints[elbow_idx]) <= 2
        or len(keypoints[wrist_idx]) <= 2
    ):
        return False, None

    if (
        keypoints[shoulder_idx][2] <= 0.5
        or keypoints[elbow_idx][2] <= 0.5
        or keypoints[wrist_idx][2] <= 0.5
    ):
        return False, None

    pt_shoulder = keypoints[shoulder_idx][:2]
    pt_elbow = keypoints[elbow_idx][:2]
    pt_wrist = keypoints[wrist_idx][:2]
    angle = calculate_angle(pt_shoulder, pt_elbow, pt_wrist)

    return angle < angle_threshold, angle


def get_torso_box(keypoints, margin_x=40, margin_y=40,margin_bottom=90):
    """
    Approximate torso box from shoulders and hips.
    Returns (x1, y1, x2, y2) or None.
    """
    if keypoints is None or len(keypoints) <= 12:
        return None

    torso_indices = [5, 6, 11, 12]
    points = []
    for idx in torso_indices:
        if len(keypoints[idx]) <= 2 or keypoints[idx][2] <= 0.5:
            return None
        points.append((keypoints[idx][0], keypoints[idx][1]))

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    return (
        int(min(xs) - margin_x),
        int(min(keypoints[5][1], keypoints[6][1]) - margin_y),
        int(max(xs) + margin_x),
        int(max(keypoints[11][1], keypoints[12][1]) + margin_y+margin_bottom),
    )


def is_wrist_in_torso_area(keypoints, wrist_idx, torso_box=None):
    """
    Check whether wrist lies inside torso/chest/belly area.
    """
    if keypoints is None or len(keypoints) <= wrist_idx:
        return False
    if len(keypoints[wrist_idx]) <= 2 or keypoints[wrist_idx][2] <= 0.5:
        return False

    if torso_box is None:
        torso_box = get_torso_box(keypoints)
    if torso_box is None:
        return False

    x1, y1, x2, y2 = torso_box
    x, y = keypoints[wrist_idx][:2]
    return x1 <= x <= x2 and y1 <= y <= y2


def calc_wrist_distance(keypoints):
    """
    Calculate wrist offsets and Euclidean distance between left and right wrists.
    """
    if keypoints is None or len(keypoints) <= 10:
        return None, None, None
    if (
        len(keypoints[9]) <= 2
        or len(keypoints[10]) <= 2
        or keypoints[9][2] <= 0.5
        or keypoints[10][2] <= 0.5
    ):
        return None, None, None

    wrist_dx = abs(keypoints[10][0] - keypoints[9][0])
    wrist_dy = abs(keypoints[10][1] - keypoints[9][1])
    wrist_distance = math.hypot(wrist_dx, wrist_dy)

    return wrist_dx, wrist_dy, wrist_distance


def detect_carrying_pose(keypoints, holding=False, best_wrist="NONE"):
    """
    Detect carrying from pose.
    """
    _ = holding
    _ = best_wrist
    _, left_angle = is_arm_bent_for_carrying(keypoints, 5, 7, 9)
    _, right_angle = is_arm_bent_for_carrying(keypoints, 6, 8, 10)
    torso_box = get_torso_box(keypoints)
    left_wrist_in_torso = is_wrist_in_torso_area(keypoints, 9, torso_box)
    right_wrist_in_torso = is_wrist_in_torso_area(keypoints, 10, torso_box)
    wrist_dx, wrist_dy, wrist_distance = calc_wrist_distance(keypoints)
    left_bent = left_angle is not None and left_angle < CARRY_ARM_ANGLE_THRESHOLD
    right_bent = right_angle is not None and right_angle < CARRY_ARM_ANGLE_THRESHOLD
    wrists_close = (
        wrist_dx is not None
        and wrist_dy is not None
        and wrist_dx <= WRIST_TOGETHER_X_THRESHOLD
        and wrist_dy <= WRIST_TOGETHER_Y_THRESHOLD
    )
    any_wrist_in_torso = left_wrist_in_torso or right_wrist_in_torso
    both_wrist_in_torso = left_wrist_in_torso and right_wrist_in_torso

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
    front_carry_raw = front_carry_two_hand or front_carry_strong_one_arm

    left_carry = front_carry_raw
    right_carry = front_carry_raw
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
        "wrist_dx": wrist_dx,
        "wrist_dy": wrist_dy,
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
        "left_carry": left_carry,
        "right_carry": right_carry,
    }
