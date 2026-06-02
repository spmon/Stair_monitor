from stair_monitor.geometry import (
    get_side_name,
    point_to_segment_distance,
    signed_distance_to_line,
)
from stair_monitor.settings import (
    LEFT_HANDRAIL_MAX_DISTANCE,
    RIGHT_HANDRAIL_MAX_DISTANCE,
)

LEFT_HANDRAIL_RULE = "LEFT_HANDRAIL_RULE"
RIGHT_HANDRAIL_RULE = "RIGHT_HANDRAIL_RULE"


def is_front_to_camera(body_facing):
    return body_facing is not None and "FRONT_TO_CAMERA" in str(body_facing)


def is_back_to_camera(body_facing):
    return body_facing is not None and "BACK_TO_CAMERA" in str(body_facing)


def get_best_wrist_for_handrail_by_rule(keypoints, line, rule, segment_max_distance):
    """
    Check both wrists and choose the closest valid wrist to the handrail.

    Rule is fixed by physical handrail side, not motion direction.

    Returns:
    - holding: True/False
    - dist: signed distance to the handrail
    - side: LEFT_SIDE / RIGHT_SIDE / ON_LINE / UNKNOWN
    - wrist_name: LEFT_WRIST / RIGHT_WRIST / NONE
    - wrist_point: chosen wrist point or None
    - hold_status: TRUE / FALSE / UNKNOWN
    - segment_dist: distance to the handrail segment
    - projection_t: unclamped projection coefficient along the segment
    """
    if keypoints is None or len(keypoints) < 11 or len(line) < 2:
        return False, -999, "UNKNOWN", "NONE", None, "UNKNOWN", None, None

    candidates = []

    if len(keypoints[9]) > 2 and keypoints[9][2] > 0.5:
        wrist_point = (int(keypoints[9][0]), int(keypoints[9][1]))
        d = signed_distance_to_line(wrist_point, line)
        segment_dist, projection_t, _ = point_to_segment_distance(
            wrist_point, line[0], line[1]
        )
        valid_d = False
        if rule == LEFT_HANDRAIL_RULE:
            valid_d = -LEFT_HANDRAIL_MAX_DISTANCE <= d <= -10
        elif rule == RIGHT_HANDRAIL_RULE:
            valid_d = 0 <= d <= RIGHT_HANDRAIL_MAX_DISTANCE
        valid_segment = 0.0 <= projection_t <= 1.0 and (
            segment_dist <= segment_max_distance
        )
        candidates.append(
            {
                "name": "LEFT_WRIST",
                "point": wrist_point,
                "dist": d,
                "segment_dist": segment_dist,
                "projection_t": projection_t,
                "side": get_side_name(d),
                "valid_d": valid_d,
                "valid_segment": valid_segment,
                "valid": valid_d and valid_segment,
            }
        )

    if len(keypoints[10]) > 2 and keypoints[10][2] > 0.25:
        wrist_point = (int(keypoints[10][0]), int(keypoints[10][1]))
        d = signed_distance_to_line(wrist_point, line)
        segment_dist, projection_t, _ = point_to_segment_distance(
            wrist_point, line[0], line[1]
        )
        valid_d = False
        if rule == LEFT_HANDRAIL_RULE:
            valid_d = -LEFT_HANDRAIL_MAX_DISTANCE <= d <= -10
        elif rule == RIGHT_HANDRAIL_RULE:
            valid_d = 0 <= d <= RIGHT_HANDRAIL_MAX_DISTANCE
        valid_segment = 0.0 <= projection_t <= 1.0 and (
            segment_dist <= segment_max_distance
        )
        candidates.append(
            {
                "name": "RIGHT_WRIST",
                "point": wrist_point,
                "dist": d,
                "segment_dist": segment_dist,
                "projection_t": projection_t,
                "side": get_side_name(d),
                "valid_d": valid_d,
                "valid_segment": valid_segment,
                "valid": valid_d and valid_segment,
            }
        )

    if not candidates:
        return False, -999, "UNKNOWN", "NONE", None, "UNKNOWN", None, None

    valid_candidates = [candidate for candidate in candidates if candidate["valid"]]
    if valid_candidates:
        best = min(
            valid_candidates,
            key=lambda candidate: (candidate["segment_dist"], abs(candidate["dist"])),
        )
        return (
            True,
            best["dist"],
            best["side"],
            best["name"],
            best["point"],
            "TRUE",
            best["segment_dist"],
            best["projection_t"],
        )

    best = min(
        candidates,
        key=lambda candidate: (candidate["segment_dist"], abs(candidate["dist"])),
    )
    return (
        False,
        best["dist"],
        best["side"],
        best["name"],
        best["point"],
        "FALSE",
        best["segment_dist"],
        best["projection_t"],
    )


class HandrailAnalysisMixin:
    @staticmethod
    def _get_handrail_targets(direction, left_line, right_line):
        if direction == "UP":
            return (
                left_line,
                "LEFT_HANDRAIL",
                LEFT_HANDRAIL_RULE,
                right_line,
                "RIGHT_HANDRAIL",
                RIGHT_HANDRAIL_RULE,
            )
        if direction == "DOWN":
            return (
                right_line,
                "RIGHT_HANDRAIL",
                RIGHT_HANDRAIL_RULE,
                left_line,
                "LEFT_HANDRAIL",
                LEFT_HANDRAIL_RULE,
            )
        return None, "NONE", "NA", None, "NONE", "NA"

    @staticmethod
    def _select_primary_hold_debug(
        hold_raw_status,
        best_wrist_correct,
        best_wrist_correct_point,
        dist_correct,
        wrist_side_correct,
        best_wrist_wrong,
        best_wrist_wrong_point,
        dist_wrong,
        wrist_side_wrong,
    ):
        if hold_raw_status == "CORRECT":
            return (
                best_wrist_correct,
                best_wrist_correct_point,
                dist_correct,
                wrist_side_correct,
            )
        if hold_raw_status == "WRONG_SIDE":
            return (
                best_wrist_wrong,
                best_wrist_wrong_point,
                dist_wrong,
                wrist_side_wrong,
            )

        correct_valid = dist_correct != -999
        wrong_valid = dist_wrong != -999

        if correct_valid and wrong_valid:
            if abs(dist_correct) <= abs(dist_wrong):
                return (
                    best_wrist_correct,
                    best_wrist_correct_point,
                    dist_correct,
                    wrist_side_correct,
                )
            return (
                best_wrist_wrong,
                best_wrist_wrong_point,
                dist_wrong,
                wrist_side_wrong,
            )
        if correct_valid:
            return (
                best_wrist_correct,
                best_wrist_correct_point,
                dist_correct,
                wrist_side_correct,
            )
        if wrong_valid:
            return (
                best_wrist_wrong,
                best_wrist_wrong_point,
                dist_wrong,
                wrist_side_wrong,
            )
        return "NONE", None, -999, "UNKNOWN"
