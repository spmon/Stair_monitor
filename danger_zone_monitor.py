from pathlib import Path
import time

import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from ultralytics import YOLO

VIDEO_INPUT_PATH = "video/raw_video/record_2026-06-06_09-53-43.avi"
VIDEO_OUTPUT_PATH = "video/roi_warning_output3.mp4"

POSE_MODEL_PATH = "yolo11x-pose.pt"

POSE_IMGSZ = 640
PERSON_CONF_THRES = 0.6
DEBUG_MODE = False

roi_coords = [
     [
            748,
            727
        ],
        [
            1568,
            662
        ],
        [
            1558,
            1293
        ],
        [
            609,
            1294
        ]
]

roi_polygon = Polygon(roi_coords)
roi_pts = np.array(roi_coords, np.int32).reshape((-1, 1, 2))


# Kiem tra keypoint co ton tai hop le trong frame hay khong.
def is_point_valid(p):
    return p[0] > 0 and p[1] > 0


# Kiem tra mot diem chan nam trong ROI hoac cham bien ROI.
def point_in_roi(point):
    foot_point = Point(float(point[0]), float(point[1]))
    return roi_polygon.contains(foot_point) or roi_polygon.touches(foot_point)


# Xac dinh co it nhat mot co chan cua nguoi nam trong ROI hay khong.
def foot_in_roi(kpts):
    left_ankle = kpts[15]
    right_ankle = kpts[16]

    for foot in [left_ankle, right_ankle]:
        if not is_point_valid(foot):
            continue

        if point_in_roi(foot):
            return True

    return False


# Ve bbox nguoi va keypoint chan de debug trang thai trong/ngoai ROI.
def draw_person_debug(draw_frame, person_box, keypoints, inside_roi):
    bbox_color = (0, 0, 255) if inside_roi else (180, 180, 180)
    x1, y1, x2, y2 = map(int, person_box)
    cv2.rectangle(draw_frame, (x1, y1), (x2, y2), bbox_color, 2)

    for foot_index in [15, 16]:
        foot = keypoints[foot_index]
        if not is_point_valid(foot):
            continue

        color = (0, 0, 255) if point_in_roi(foot) else (0, 255, 0)
        cv2.circle(draw_frame, (int(foot[0]), int(foot[1])), 6, color, -1)


# Ve vien ROI voi mau phan anh trang thai canh bao hien tai.
def draw_roi(draw_frame, color):
    cv2.polylines(
        draw_frame,
        [roi_pts],
        isClosed=True,
        color=color,
        thickness=4,
    )


# Ve dong canh bao lon khi phat hien nguoi vao khu vuc cam.
def draw_alert_text(draw_frame):
    cv2.putText(
        draw_frame,
        "DANGER: PERSON IN RESTRICTED AREA",
        (40, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        4,
    )


# Ve so nguoi dang nam trong ROI ma khong thay doi tieu chi phat hien hien tai.
def draw_people_count(draw_frame, people_in_roi_count):
    text = f"People in zone: {people_in_roi_count}"
    text_color = (0, 0, 255) if people_in_roi_count > 0 else (0, 255, 0)
    cv2.putText(
        draw_frame,
        text,
        (40, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        text_color,
        3,
    )


# Chay toan bo luong danger zone monitor va tao hieu ung nhay do khi canh bao.
def run_danger_zone_monitor():
    print("Dang load model...")
    pose_model = YOLO(POSE_MODEL_PATH)

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

    print("Warming up model...")
    dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
    _ = pose_model(
        dummy_frame,
        imgsz=POSE_IMGSZ,
        conf=PERSON_CONF_THRES,
        verbose=False,
    )[0]

    frame_idx = 0
    start_time = time.perf_counter()

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

        danger_alert = False
        people_in_roi_count = 0

        if pose_results.keypoints is not None and pose_results.boxes is not None:
            bboxes_array = pose_results.boxes.xyxy.cpu().numpy()
            keypoints_array = pose_results.keypoints.xy.cpu().numpy()
            person_count = min(len(bboxes_array), len(keypoints_array))

            for index in range(person_count):
                person_box = bboxes_array[index]
                keypoints = keypoints_array[index]
                inside_roi = foot_in_roi(keypoints)

                if inside_roi:
                    danger_alert = True
                    people_in_roi_count += 1

                if DEBUG_MODE:
                    draw_person_debug(draw_frame, person_box, keypoints, inside_roi)

        roi_color = (0, 0, 255) if danger_alert else (0, 255, 0)
        draw_roi(draw_frame, roi_color)

        if danger_alert:
            blink_on = (frame_idx // 5) % 2 == 0
            if blink_on:
                overlay = draw_frame.copy()
                overlay[:] = (0, 0, 255)
                draw_frame = cv2.addWeighted(overlay, 0.35, draw_frame, 0.65, 0)

            draw_roi(draw_frame, roi_color)
            draw_alert_text(draw_frame)

        draw_people_count(draw_frame, people_in_roi_count)
        out.write(draw_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    total_time = time.perf_counter() - start_time
    print(f"Tong so frame da xu ly: {frame_idx}")
    print(f"Tong thoi gian xu ly: {total_time:.2f} giay")
    print(f"Hoan tat! Video da duoc luu tai: {VIDEO_OUTPUT_PATH}")


if __name__ == "__main__":
    run_danger_zone_monitor()
