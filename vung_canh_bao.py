import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Point, Polygon

VIDEO_INPUT_PATH = "video/record_2026-05-27_11-01-19.avi"
VIDEO_OUTPUT_PATH = "video/roi_warning_output.mp4"

POSE_MODEL_PATH = "yolo11m-pose.pt"

POSE_IMGSZ = 640
PERSON_CONF_THRES = 0.5

roi_coords = [
    [456, 758],
    [1260, 688],
    [1197, 1291],
    [306, 1291],
]

roi_polygon = Polygon(roi_coords)
roi_pts = np.array(roi_coords, np.int32).reshape((-1, 1, 2))

pose_model = YOLO(POSE_MODEL_PATH)


def is_point_valid(p):
    return p[0] > 0 and p[1] > 0


def person_foot_in_roi(kpts):
    """
    Tra ve True neu co it nhat mot co chan nam trong ROI.
    YOLO pose keypoints:
    15 = left_ankle
    16 = right_ankle
    """
    left_ankle = kpts[15]
    right_ankle = kpts[16]

    for foot in [left_ankle, right_ankle]:
        if not is_point_valid(foot):
            continue

        foot_point = Point(float(foot[0]), float(foot[1]))

        if roi_polygon.contains(foot_point) or roi_polygon.touches(foot_point):
            return True

    return False


cap = cv2.VideoCapture(VIDEO_INPUT_PATH)

if not cap.isOpened():
    raise FileNotFoundError(f"Khong mo duoc video: {VIDEO_INPUT_PATH}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

if width <= 0 or height <= 0:
    raise ValueError("Video loi hoac khong doc duoc kich thuoc.")

if fps <= 0:
    fps = 25

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, fps, (width, height))

print("Bat dau xu ly video...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    infer_frame = frame.copy()
    draw_frame = frame.copy()

    results = pose_model(
        infer_frame,
        imgsz=POSE_IMGSZ,
        conf=PERSON_CONF_THRES,
        verbose=False
    )[0]

    roi_alert = False

    if results.keypoints is not None and results.boxes is not None:
        kpts_array = results.keypoints.xy.cpu().numpy()
        bboxes_array = results.boxes.xyxy.cpu().numpy()

        for i, kpts in enumerate(kpts_array):
            person_box = bboxes_array[i]

            foot_inside = person_foot_in_roi(kpts)

            if foot_inside:
                roi_alert = True

                x1, y1, x2, y2 = map(int, person_box)
                cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    draw_frame,
                    "PERSON IN ROI",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
            else:
                x1, y1, x2, y2 = map(int, person_box)
                cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            # Ve 2 diem chan de debug
            for idx in [15, 16]:
                foot = kpts[idx]
                if is_point_valid(foot):
                    color = (0, 0, 255) if roi_polygon.contains(Point(float(foot[0]), float(foot[1]))) else (0, 255, 0)
                    cv2.circle(draw_frame, (int(foot[0]), int(foot[1])), 6, color, -1)

    # Mau vien ROI
    roi_color = (0, 0, 255) if roi_alert else (0, 255, 0)

    cv2.polylines(
        draw_frame,
        [roi_pts],
        isClosed=True,
        color=roi_color,
        thickness=4,
    )

    if roi_alert:
        cv2.putText(
            draw_frame,
            "WARNING: PERSON ENTERED ROI",
            (50, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255),
            4,
        )

    out.write(draw_frame)

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Hoan tat! Video da luu tai: {VIDEO_OUTPUT_PATH}")