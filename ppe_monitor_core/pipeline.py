import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from ppe_monitor_core.config import (
    DEBUG_MODE,
    DEBUG_RAW_PPE,
    HEAD_OVERLAP_THRES,
    HELMET_CONF_THRES,
    PERSON_CONF_THRES,
    PPE_IMGSZ,
    PPE_MODEL_PATH,
    POSE_IMGSZ,
    POSE_MODEL_PATH,
    TORSO_OVERLAP_THRES,
    VIDEO_INPUT_PATH,
    VIDEO_OUTPUT_PATH,
    VEST_CONF_THRES,
)
from ppe_monitor_core.detection import match_ppe_item, parse_ppe_boxes
from ppe_monitor_core.geometry import get_head_bbox, get_torso_bbox
from ppe_monitor_core.rendering import (
    draw_count_panel,
    draw_person_debug,
    draw_raw_ppe,
)
from ppe_monitor_core.tracking import (
    build_person_status,
    get_track_id,
    make_new_track,
    prune_expired_tracks,
    update_stable_state,
)


def run_ppe_monitor():
    print("Dang load model...")
    pose_model = YOLO(POSE_MODEL_PATH)
    ppe_model = YOLO(PPE_MODEL_PATH)
    print("PPE classes:", ppe_model.names)

    cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Khong mo duoc video: {VIDEO_INPUT_PATH}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if width <= 0 or height <= 0:
        raise ValueError("Kich thuoc video khong hop le.")
    if fps <= 0:
        fps = 25.0

    output_path = Path(VIDEO_OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out = cv2.VideoWriter(
        str(output_path),
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
        conf=min(HELMET_CONF_THRES, VEST_CONF_THRES),
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
            conf=min(HELMET_CONF_THRES, VEST_CONF_THRES),
            verbose=False,
        )[0]

        hat_bboxes, vest_bboxes = parse_ppe_boxes(ppe_results)
        if DEBUG_MODE and DEBUG_RAW_PPE:
            draw_raw_ppe(draw_frame, hat_bboxes, vest_bboxes)

        total_people_count = 0
        checking_count = 0
        full_ppe_count = 0
        no_hat_count = 0
        no_vest_count = 0
        no_hat_no_vest_count = 0

        assigned_track_ids = set()
        used_hat_indices = set()
        used_vest_indices = set()

        if pose_results.keypoints is not None and pose_results.boxes is not None:
            bboxes_array = pose_results.boxes.xyxy.cpu().numpy()
            keypoints_array = pose_results.keypoints.xy.cpu().numpy()
            person_count = min(len(bboxes_array), len(keypoints_array))

            for index in range(person_count):
                person_box = bboxes_array[index].copy()
                keypoints = keypoints_array[index]

                track_id = get_track_id(
                    person_box,
                    tracks,
                    frame_idx,
                    assigned_track_ids,
                )
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

                raw_has_hat, matched_hat = match_ppe_item(
                    person_box,
                    head_box,
                    hat_bboxes,
                    HEAD_OVERLAP_THRES,
                    used_hat_indices,
                    frame=infer_frame,
                    reject_dark_hair=True,
                )
                raw_has_vest, matched_vest = match_ppe_item(
                    person_box,
                    torso_box,
                    vest_bboxes,
                    TORSO_OVERLAP_THRES,
                    used_vest_indices,
                )

                track["hat_history"].append(1 if raw_has_hat else 0)
                track["vest_history"].append(1 if raw_has_vest else 0)
                update_stable_state(track)

                stable_has_hat = track["stable_has_hat"]
                stable_has_vest = track["stable_has_vest"]

                total_people_count += 1

                if stable_has_hat is None or stable_has_vest is None:
                    checking_count += 1
                elif stable_has_hat and stable_has_vest:
                    full_ppe_count += 1
                elif not stable_has_hat and not stable_has_vest:
                    no_hat_no_vest_count += 1
                elif not stable_has_hat:
                    no_hat_count += 1
                elif not stable_has_vest:
                    no_vest_count += 1

                person_status = build_person_status(track_id, track)

                if DEBUG_MODE:
                    draw_person_debug(
                        draw_frame=draw_frame,
                        person_box=person_box,
                        label_text=person_status["label_text"],
                        status_color=person_status["status_color"],
                        head_box=head_box,
                        torso_box=torso_box,
                        matched_hat=matched_hat,
                        matched_vest=matched_vest,
                    )

        draw_count_panel(
            draw_frame,
            total_people_count,
            checking_count,
            full_ppe_count,
            no_hat_count,
            no_vest_count,
            no_hat_no_vest_count,
        )

        prune_expired_tracks(tracks, frame_idx)
        out.write(draw_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    total_time = time.perf_counter() - start_time
    print(f"Tong so frame da xu ly: {frame_idx}")
    print(f"Tong thoi gian xu ly: {total_time:.2f} giay")
    print(f"Hoan tat! Video da duoc luu tai: {VIDEO_OUTPUT_PATH}")
