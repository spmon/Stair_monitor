import cv2

from stair_monitor.settings import (
    DEMO_MODE,
    DRAW_DEBUG_DETAIL,
    DRAW_KEYPOINTS,
    DRAW_SKELETON,
    SHOW_ONLY_VIOLATIONS,
    VIOLATION_COLOR,
)


def draw_scene_guides(frame, config, analyzer):
    if "CENTER_LINE" in config:
        cv2.line(
            frame,
            tuple(config["CENTER_LINE"][0]),
            tuple(config["CENTER_LINE"][1]),
            (0, 255, 255),
            2,
        )
    if len(analyzer.stairs_poly) > 2:
        cv2.polylines(
            frame, [analyzer.stairs_poly.reshape((-1, 1, 2))], True, (255, 0, 0), 2
        )

    if len(analyzer.left_line) >= 2:
        cv2.line(
            frame,
            tuple(analyzer.left_line[0]),
            tuple(analyzer.left_line[1]),
            (0, 165, 255),
            3,
        )
    if len(analyzer.right_line) >= 2:
        cv2.line(
            frame,
            tuple(analyzer.right_line[0]),
            tuple(analyzer.right_line[1]),
            (255, 0, 255),
            3,
        )


def get_feet_point(box, keypoints):
    if keypoints is not None and len(keypoints) > 16:
        conf_l_ankle = keypoints[15][2] if len(keypoints[15]) > 2 else 0
        conf_r_ankle = keypoints[16][2] if len(keypoints[16]) > 2 else 0

        if conf_l_ankle > 0.5 and conf_r_ankle > 0.5:
            return (
                int((keypoints[15][0] + keypoints[16][0]) / 2),
                int((keypoints[15][1] + keypoints[16][1]) / 2),
            )
        if conf_l_ankle > 0.5:
            return (int(keypoints[15][0]), int(keypoints[15][1]))
        if conf_r_ankle > 0.5:
            return (int(keypoints[16][0]), int(keypoints[16][1]))

    return (int((box[0] + box[2]) / 2), int(box[3]))


def get_motion_point(box, keypoints):
    if (
        keypoints is not None
        and len(keypoints) > 12
        and len(keypoints[11]) > 2
        and len(keypoints[12]) > 2
        and keypoints[11][2] > 0.5
        and keypoints[12][2] > 0.5
    ):
        return (
            int((keypoints[11][0] + keypoints[12][0]) / 2),
            int((keypoints[11][1] + keypoints[12][1]) / 2),
        )

    return (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))


def draw_person_overlay(frame, box, keypoints, lane_point, motion_point, analysis):
    display_status = analysis.get("display_status", analysis.get("status", ""))
    if DEMO_MODE and SHOW_ONLY_VIOLATIONS:
        if not display_status:
            return

        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), VIOLATION_COLOR, 2)
        cv2.putText(
            frame,
            display_status,
            (x1, y2 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            VIOLATION_COLOR,
            2,
        )
        return

    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), analysis["color"], 2)

    if display_status:
        cv2.putText(
            frame,
            display_status,
            (x1, y2 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            analysis["color"],
            2,
        )

    if keypoints is not None and DRAW_KEYPOINTS:
        cv2.circle(frame, lane_point, 6, (0, 0, 255), -1)
        cv2.circle(frame, motion_point, 6, (255, 0, 0), -1)

    if keypoints is not None and DRAW_SKELETON:
        if (
            len(keypoints) > 9
            and keypoints[5][2] > 0.5
            and keypoints[7][2] > 0.5
            and keypoints[9][2] > 0.5
        ):
            p_s = (int(keypoints[5][0]), int(keypoints[5][1]))
            p_e = (int(keypoints[7][0]), int(keypoints[7][1]))
            p_w = (int(keypoints[9][0]), int(keypoints[9][1]))
            cv2.line(frame, p_s, p_e, (255, 255, 0), 2)
            cv2.line(frame, p_e, p_w, (255, 255, 0), 2)
            cv2.circle(frame, p_s, 4, (255, 255, 0), -1)
            cv2.circle(frame, p_e, 4, (255, 255, 0), -1)
            cv2.circle(frame, p_w, 4, (255, 255, 0), -1)

        if (
            len(keypoints) > 10
            and keypoints[6][2] > 0.5
            and keypoints[8][2] > 0.5
            and keypoints[10][2] > 0.5
        ):
            p_s = (int(keypoints[6][0]), int(keypoints[6][1]))
            p_e = (int(keypoints[8][0]), int(keypoints[8][1]))
            p_w = (int(keypoints[10][0]), int(keypoints[10][1]))
            cv2.line(frame, p_s, p_e, (0, 255, 255), 2)
            cv2.line(frame, p_e, p_w, (0, 255, 255), 2)
            cv2.circle(frame, p_s, 4, (0, 255, 255), -1)
            cv2.circle(frame, p_e, 4, (0, 255, 255), -1)
            cv2.circle(frame, p_w, 4, (0, 255, 255), -1)

    if DRAW_KEYPOINTS and analysis.get("best_wrist_point") is not None:
        cv2.circle(frame, analysis["best_wrist_point"], 8, (0, 255, 255), -1)

    if DRAW_DEBUG_DETAIL and analysis.get("carry_type", "NONE") != "NONE":
        cv2.putText(
            frame,
            f"CARRY:{analysis['carry_type']}",
            (x1, y2 + 47),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    if DRAW_DEBUG_DETAIL:
        for idx, debug_text in enumerate(build_debug_lines(analysis)):
            cv2.putText(
                frame,
                debug_text,
                (x2 + 5, y1 + 15 + idx * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )


def draw_people_count(frame, current_inside_count):
    text = f"So nguoi trong cau thang: {current_inside_count}"
    cv2.rectangle(frame, (20, 10), (430, 55), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (30, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
    )


def build_debug_lines(analysis):
    return [
        f"BEST_WRIST:{analysis.get('best_wrist', 'NONE')}",
        f"BEST_WRIST_CORRECT:{analysis.get('best_wrist_correct', 'NONE')}",
        f"BEST_WRIST_WRONG:{analysis.get('best_wrist_wrong', 'NONE')}",
        f"D:{int(analysis['dist_wrist'])}"
        if analysis.get("dist_wrist", -999) != -999
        else "D:NA",
        f"D_CORRECT:{int(analysis['dist_correct'])}"
        if analysis.get("dist_correct", -999) != -999
        else "D_CORRECT:NA",
        f"SEG_D_CORRECT:{int(analysis['seg_dist_correct'])}"
        if analysis.get("seg_dist_correct") is not None
        else "SEG_D_CORRECT:NA",
        f"T_CORRECT:{analysis['t_correct']:.2f}"
        if analysis.get("t_correct") is not None
        else "T_CORRECT:NA",
        f"D_WRONG:{int(analysis['dist_wrong'])}"
        if analysis.get("dist_wrong", -999) != -999
        else "D_WRONG:NA",
        f"SEG_D_WRONG:{int(analysis['seg_dist_wrong'])}"
        if analysis.get("seg_dist_wrong") is not None
        else "SEG_D_WRONG:NA",
        f"T_WRONG:{analysis['t_wrong']:.2f}"
        if analysis.get("t_wrong") is not None
        else "T_WRONG:NA",
        f"W_SIDE:{analysis.get('wrist_side', 'UNKNOWN')}",
        f"W_SIDE_CORRECT:{analysis.get('wrist_side_correct', 'UNKNOWN')}",
        f"W_SIDE_WRONG:{analysis.get('wrist_side_wrong', 'UNKNOWN')}",
        f"CORRECT_LINE:{analysis.get('correct_line_name', 'NONE')}",
        f"CORRECT_RULE:{analysis.get('correct_rule', 'NA')}",
        f"WRONG_LINE:{analysis.get('wrong_line_name', 'NONE')}",
        f"WRONG_RULE:{analysis.get('wrong_rule', 'NA')}",
        f"DY:{int(analysis['dy'])}" if analysis.get("dy") is not None else "DY:NA",
        f"LANE_V:{int(analysis['lane_v'])}"
        if analysis.get("lane_v") is not None
        else "LANE_V:NA",
        f"DIR:{analysis.get('direction', 'NA')}",
        f"INSIDE_STAIRS:{analysis.get('inside_stairs', False)}",
        f"LANE_RAW:{analysis.get('lane_raw', False)}",
        f"LANE_HITS:{analysis.get('lane_hits', 0)}",
        f"LANE_CONF:{analysis.get('lane_conf', False)}",
        f"BODY_FACE:{analysis.get('body_facing', 'UNKNOWN')}",
        f"BACKWARD_RAW:{analysis.get('backward_raw', False)}",
        f"BACKWARD_HITS:{analysis.get('backward_hits', 0)}",
        f"BACKWARD_CONF:{analysis.get('backward_confirmed', False)}",
        f"ARM_ORDER:{analysis.get('arm_side_order', 'UNKNOWN')}",
        f"HOLD_CORRECT_RAW:{analysis.get('holding_correct_raw', False)}",
        f"HOLD_WRONG_RAW:{analysis.get('holding_wrong_raw', False)}",
        f"HOLD_RAW:{analysis.get('holding_raw', False)}",
        f"HOLD_RAW_STATUS:{analysis.get('hold_raw_status', 'UNKNOWN')}",
        f"HOLD:{analysis.get('holding', False)}",
        f"HOLD_STATUS:{analysis.get('hold_status', 'UNKNOWN')}",
        f"HOLD_CORRECT_HITS:{analysis.get('hold_correct_hits', 0)}",
        f"HOLD_WRONG_HITS:{analysis.get('hold_wrong_side_hits', 0)}",
        f"HOLD_NONE_HITS:{analysis.get('hold_none_hits', 0)}",
        f"HOLD_UNKNOWN_HITS:{analysis.get('hold_unknown_hits', 0)}",
        f"HOLD_CONF:{analysis.get('hold_confirmed_status', 'UNKNOWN')}",
        f"L_CARRY:{analysis.get('left_carry', False)}",
        f"R_CARRY:{analysis.get('right_carry', False)}",
        f"CARRY:{analysis.get('is_carrying', False)}",
        f"CARRY_TYPE:{analysis.get('carry_type', 'NONE')}",
        f"CARRY_ARM:{analysis.get('carrying_arm', 'NONE')}",
        f"L_ANG:{int(analysis['left_arm_angle'])}"
        if analysis.get("left_arm_angle") is not None
        else "L_ANG:NA",
        f"R_ANG:{int(analysis['right_arm_angle'])}"
        if analysis.get("right_arm_angle") is not None
        else "R_ANG:NA",
        f"L_TORSO:{analysis.get('left_wrist_in_torso', False)}",
        f"R_TORSO:{analysis.get('right_wrist_in_torso', False)}",
        f"W_DX:{int(analysis['wrist_dx'])}"
        if analysis.get("wrist_dx") is not None
        else "W_DX:NA",
        f"W_DY:{int(analysis['wrist_dy'])}"
        if analysis.get("wrist_dy") is not None
        else "W_DY:NA",
        f"W_DIST:{int(analysis['wrist_distance'])}"
        if analysis.get("wrist_distance") is not None
        else "W_DIST:NA",
        f"L_BENT:{analysis.get('left_bent', False)}",
        f"R_BENT:{analysis.get('right_bent', False)}",
        f"W_CLOSE:{analysis.get('wrists_close', False)}",
        f"BOTH_TORSO:{analysis.get('both_wrist_in_torso', False)}",
        f"FRONT:{analysis.get('front_carry', False)}",
        f"FRONT_2H_RAW:{analysis.get('front_carry_two_hand_raw', False)}",
        f"FRONT_2H_HITS:{analysis.get('front_carry_two_hand_hits', 0)}",
        f"FRONT_1H_RAW:{analysis.get('front_carry_one_arm_raw', False)}",
        f"FRONT_1H_HITS:{analysis.get('front_carry_one_arm_hits', 0)}",
        f"FRONT_RAW:{analysis.get('front_carry_raw', False)}",
        f"FRONT_HITS:{analysis.get('front_carry_hits', 0)}",
        f"FRONT_CONF:{analysis.get('front_carry_confirmed', False)}",
        f"LANE_ERR:{analysis.get('wrong_lane', False)}",
        f"P_LANE:{analysis['p_lane'][0]},{analysis['p_lane'][1]}"
        if analysis.get("p_lane") is not None
        else "P_LANE:NA",
        f"MOTION:{analysis['p_motion'][0]},{analysis['p_motion'][1]}"
        if analysis.get("p_motion") is not None
        else "MOTION:NA",
    ]
