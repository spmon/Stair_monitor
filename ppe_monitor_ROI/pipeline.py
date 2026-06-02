import time

import cv2
import numpy as np
from ultralytics import YOLO

from ppe_monitor_ROI.config import (
    DEBUG_MODE,
    DEBUG_RAW_PPE,
    HAT_CLASS_ID,
    HEAD_OVERLAP_THRES,
    PERSON_CONF_THRES,
    PPE_CONF_THRES,
    PPE_IMGSZ,
    POSE_IMGSZ,
    TRACK_TTL,
    TORSO_OVERLAP_THRES,
    VIDEO_INPUT_PATH,
    VIDEO_OUTPUT_PATH,
    VEST_CLASS_ID,
)
from ppe_monitor_ROI.rendering import (
    draw_foot_debug,
    draw_person_debug,
    draw_raw_ppe,
    draw_roi,
    draw_status_panel,
    get_overall_status,
)
from ppe_monitor_ROI.roi import (
    ROI_PTS,
    foot_in_roi,
    get_head_bbox,
    get_torso_bbox,
    overlap_ratio,
)
from ppe_monitor_ROI.tracking import (
    build_person_status,
    get_track_id,
    make_new_track,
    update_stable_state,
)


def parse_ppe_boxes(ppe_results):
    hat_bboxes = []
    vest_bboxes = []

    if ppe_results.boxes is not None:
        for box in ppe_results.boxes:
            class_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            if conf < PPE_CONF_THRES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            if class_id == HAT_CLASS_ID:
                hat_bboxes.append([x1, y1, x2, y2, conf])
            elif class_id == VEST_CLASS_ID:
                vest_bboxes.append([x1, y1, x2, y2, conf])

    return hat_bboxes, vest_bboxes


def match_hat(person_box, head_box, hat_bboxes):
    if head_box is None:
        return False, None

    for hat in hat_bboxes:
        hat_box = hat[:4]
        center_x = (hat_box[0] + hat_box[2]) / 2
        center_y = (hat_box[1] + hat_box[3]) / 2

        if not (
            person_box[0] <= center_x <= person_box[2]
            and person_box[1] <= center_y <= person_box[3]
        ):
            continue

        if overlap_ratio(head_box, hat_box) > HEAD_OVERLAP_THRES:
            return True, hat

    return False, None


def match_vest(person_box, torso_box, vest_bboxes):
    if torso_box is None:
        return False, None

    for vest in vest_bboxes:
        vest_box = vest[:4]
        center_x = (vest_box[0] + vest_box[2]) / 2
        center_y = (vest_box[1] + vest_box[3]) / 2

        if not (
            person_box[0] <= center_x <= person_box[2]
            and person_box[1] <= center_y <= person_box[3]
        ):
            continue

        if overlap_ratio(torso_box, vest_box) > TORSO_OVERLAP_THRES:
            return True, vest

    return False, None


def run_ppe_monitor():
    print("Dang load model...")
    pose_model = YOLO("yolo11m-pose.pt")
    ppe_model = YOLO("runs/detect/ppe-2class-6/weights/best.pt")
    print("PPE classes:", ppe_model.names)

    cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Khong mo duoc video: {VIDEO_INPUT_PATH}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if width <= 0 or height <= 0:
        raise ValueError("Kich thuoc video khong hop le.")
    if fps <= 0:
        fps = 25

    out = cv2.VideoWriter(
        VIDEO_OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not out.isOpened():
        raise RuntimeError(f"Khong tao duoc output video: {VIDEO_OUTPUT_PATH}")

    print("Warming up models...")
    dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
    _ = pose_model(
        dummy_frame,
        imgsz=POSE_IMGSZ,
        conf=PERSON_CONF_THRES,
        verbose=False,
    )[0]
    _ = ppe_model(
        dummy_frame,
        imgsz=PPE_IMGSZ,
        verbose=False,
    )[0]

    frame_idx = 0
    start_time = time.perf_counter()
    next_track_id = 0
    tracks = {}

    print("Bat dau xu ly video...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        infer_frame = frame.copy()
        draw_frame = frame.copy()

        pose_results = pose_model(
            infer_frame,
            imgsz=POSE_IMGSZ,
            conf=PERSON_CONF_THRES,
            verbose=False,
        )[0]
        ppe_results = ppe_model(
            infer_frame,
            imgsz=PPE_IMGSZ,
            verbose=False,
        )[0]

        hat_bboxes, vest_bboxes = parse_ppe_boxes(ppe_results)
        if DEBUG_MODE and DEBUG_RAW_PPE:
            draw_raw_ppe(draw_frame, hat_bboxes, vest_bboxes)

        roi_alert = False
        frame_statuses = []

        if pose_results.keypoints is not None and pose_results.boxes is not None:
            bboxes_array = pose_results.boxes.xyxy.cpu().numpy()
            keypoints_array = pose_results.keypoints.xy.cpu().numpy()
            assigned_track_ids = set()

            for index, keypoints in enumerate(keypoints_array):
                person_box = bboxes_array[index].copy()
                in_roi = foot_in_roi(keypoints)

                if DEBUG_MODE:
                    draw_foot_debug(draw_frame, keypoints)

                if not in_roi:
                    continue

                roi_alert = True

                track_id = get_track_id(person_box, tracks, frame_idx)
                if track_id in assigned_track_ids:
                    track_id = None

                if track_id is None:
                    track_id = next_track_id
                    next_track_id += 1
                    tracks[track_id] = make_new_track(person_box, frame_idx)

                assigned_track_ids.add(track_id)

                track = tracks[track_id]
                track["bbox"] = person_box.copy()
                track["last_seen"] = frame_idx

                head_box = get_head_bbox(keypoints, person_box)
                torso_box = get_torso_bbox(keypoints, person_box)

                raw_has_hat, matched_hat = match_hat(person_box, head_box, hat_bboxes)
                raw_has_vest, matched_vest = match_vest(
                    person_box, torso_box, vest_bboxes
                )

                track["hat_history"].append(1 if raw_has_hat else 0)
                track["vest_history"].append(1 if raw_has_vest else 0)

                if matched_hat is not None:
                    track["matched_hat"] = matched_hat
                if matched_vest is not None:
                    track["matched_vest"] = matched_vest

                update_stable_state(track)

                person_status = build_person_status(track_id, track)
                frame_statuses.append(person_status)

                draw_hat = matched_hat if matched_hat is not None else (
                    track["matched_hat"] if person_status["stable_has_hat"] else None
                )
                draw_vest = matched_vest if matched_vest is not None else (
                    track["matched_vest"] if person_status["stable_has_vest"] else None
                )

                if DEBUG_MODE:
                    draw_person_debug(
                        draw_frame,
                        person_box=person_box,
                        label_text=person_status["label_text"],
                        status_color=person_status["status_color"],
                        head_box=head_box,
                        torso_box=torso_box,
                        matched_hat=draw_hat,
                        matched_vest=draw_vest,
                    )

        draw_roi(draw_frame, ROI_PTS, roi_alert)
        line1, line2, panel_color = get_overall_status(frame_statuses)
        draw_status_panel(draw_frame, line1, line2, panel_color)

        expired_ids = [
            track_id
            for track_id, track in tracks.items()
            if frame_idx - track["last_seen"] > TRACK_TTL
        ]
        for track_id in expired_ids:
            del tracks[track_id]

        out.write(draw_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    total_time = time.perf_counter() - start_time
    print(f"Tong so frame da xu ly: {frame_idx}")
    print(f"Tong thoi gian xu ly: {total_time:.2f} giay")
    print(f"Hoan tat! Video da duoc luu tai: {VIDEO_OUTPUT_PATH}")
