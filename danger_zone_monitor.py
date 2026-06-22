from pathlib import Path
import time
from typing import TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray
from shapely.geometry import Point, Polygon, box as shapely_box
from ultralytics import YOLO
from ultralytics.engine.results import Results

FrameArray: TypeAlias = NDArray[np.uint8]
FloatArray: TypeAlias = NDArray[np.float32] | NDArray[np.float64]
RoiCoords: TypeAlias = list[list[int]]
ColorBGR: TypeAlias = tuple[int, int, int]

VIDEO_INPUT_PATH = "video/raw_video/record_2026-06-06_10-00-21.avi"
VIDEO_OUTPUT_PATH = "video/roi_warning_output3.mp4"

POSE_MODEL_PATH = "yolo11x-pose.pt"

POSE_IMGSZ = 640
PERSON_CONF_THRES = 0.5
KEYPOINT_CONF_THRES = 0.35
VIRTUAL_FOOT_SCALE = 1.0
PERSON_MIN_AREA = 3000
PERSON_MAX_ASPECT_LOW_CONF = 3.5
BOX_ROI_OVERLAP_THRES = 0.05
ALERT_HOLD_FRAMES = 10
BLINK_INTERVAL_FRAMES = 5
DEBUG_MODE = False

roi_coords: RoiCoords = [
    [748, 727],
    [1568, 662],
    [1558, 1293],
    [609, 1294],
]

roi_polygon: Polygon = Polygon(roi_coords)
roi_pts: NDArray[np.int32] = np.array(roi_coords, np.int32).reshape((-1, 1, 2))


def is_point_valid(p: FloatArray) -> bool:
    return bool(p[0] > 0 and p[1] > 0)


def is_keypoint_valid(
    kpts: FloatArray,
    index: int,
    kpt_conf: FloatArray | None = None,
) -> bool:
    point = kpts[index]
    if not is_point_valid(point):
        return False

    if kpt_conf is not None:
        return bool(kpt_conf[index] >= KEYPOINT_CONF_THRES)

    return True


def point_in_roi(point: FloatArray) -> bool:
    roi_point = Point(float(point[0]), float(point[1]))
    return roi_polygon.contains(roi_point) or roi_polygon.touches(roi_point)


def estimate_virtual_feet_from_knees(
    kpts: FloatArray,
    kpt_conf: FloatArray | None = None,
) -> list[tuple[FloatArray, str]]:
    virtual_feet: list[tuple[FloatArray, str]] = []
    pairs = [
        (11, 13, "left_virtual_foot"),
        (12, 14, "right_virtual_foot"),
    ]

    for hip_idx, knee_idx, source in pairs:
        if not is_keypoint_valid(kpts, hip_idx, kpt_conf):
            continue
        if not is_keypoint_valid(kpts, knee_idx, kpt_conf):
            continue

        hip = kpts[hip_idx]
        knee = kpts[knee_idx]
        direction = knee - hip
        virtual_foot = knee + VIRTUAL_FOOT_SCALE * direction
        virtual_feet.append((virtual_foot.astype(np.float32), source))

    return virtual_feet


def box_roi_overlap_ratio(person_box: FloatArray) -> float:
    x1, y1, x2, y2 = map(float, person_box[:4])
    person_poly = shapely_box(x1, y1, x2, y2)
    person_area = person_poly.area
    if person_area <= 0:
        return 0.0

    inter_area = person_poly.intersection(roi_polygon).area
    return float(inter_area / person_area)


def is_valid_person_box(person_box: FloatArray, person_conf: float) -> bool:
    x1, y1, x2, y2 = map(float, person_box[:4])
    width = x2 - x1
    height = y2 - y1

    if width <= 0 or height <= 0:
        return False

    area = width * height
    if area < PERSON_MIN_AREA:
        return False

    aspect = width / max(height, 1.0)
    if person_conf < 0.65 and aspect > PERSON_MAX_ASPECT_LOW_CONF:
        return False

    return True


def person_in_danger_zone(
    kpts: FloatArray,
    person_box: FloatArray,
    kpt_conf: FloatArray | None = None,
) -> tuple[bool, str]:
    candidate_points: list[tuple[FloatArray, str]] = []

    for idx in [15, 16]:
        if is_keypoint_valid(kpts, idx, kpt_conf):
            candidate_points.append((kpts[idx], "real_ankle"))

    for virtual_foot, _source in estimate_virtual_feet_from_knees(kpts, kpt_conf):
        candidate_points.append((virtual_foot, "virtual_foot"))

    for idx in [13, 14]:
        if is_keypoint_valid(kpts, idx, kpt_conf):
            candidate_points.append((kpts[idx], "knee"))

    for idx in range(17):
        if is_keypoint_valid(kpts, idx, kpt_conf):
            candidate_points.append((kpts[idx], "any_keypoint"))

    x1, y1, x2, y2 = map(float, person_box[:4])
    width = x2 - x1
    height = y2 - y1
    bbox_points = [
        np.array([x1 + 0.50 * width, y2], dtype=np.float32),
        np.array([x1 + 0.25 * width, y2], dtype=np.float32),
        np.array([x1 + 0.75 * width, y2], dtype=np.float32),
        np.array([x1 + 0.50 * width, y1 + 0.50 * height], dtype=np.float32),
        np.array([x1 + 0.50 * width, y1 + 0.85 * height], dtype=np.float32),
    ]

    for point in bbox_points:
        candidate_points.append((point, "bbox_point"))

    for point, source in candidate_points:
        if point_in_roi(point):
            return True, source

    overlap = box_roi_overlap_ratio(person_box)
    if overlap >= BOX_ROI_OVERLAP_THRES:
        return True, f"bbox_overlap:{overlap:.2f}"

    return False, "none"


def is_person_class(
    class_id: float,
    model_names: dict[int, str],
) -> bool:
    class_name = model_names.get(int(class_id), "")
    return class_name == "person" or int(class_id) == 0


def draw_roi(draw_frame: FrameArray, color: ColorBGR) -> None:
    cv2.polylines(
        draw_frame,
        [roi_pts],
        isClosed=True,
        color=color,
        thickness=4,
    )


def draw_alert_text(draw_frame: FrameArray) -> None:
    text = "DANGER: PERSON IN RESTRICTED AREA"
    x, y = 40, 80
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 4
    pad_x = 18
    pad_y = 14

    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )

    cv2.rectangle(
        draw_frame,
        (x - pad_x, y - text_height - pad_y),
        (x + text_width + pad_x, y + baseline + pad_y),
        (0, 0, 180),
        -1,
    )

    cv2.putText(
        draw_frame,
        text,
        (x, y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_people_count(
    draw_frame: FrameArray,
    people_in_roi_count: int,
) -> None:
    text = f"People in zone: {people_in_roi_count}"
    x, y = 40, 140
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 3
    pad_x = 14
    pad_y = 12

    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )

    cv2.rectangle(
        draw_frame,
        (x - pad_x, y - text_height - pad_y),
        (x + text_width + pad_x, y + baseline + pad_y),
        (0, 0, 180),
        -1,
    )

    cv2.putText(
        draw_frame,
        text,
        (x, y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_person_debug(
    draw_frame: FrameArray,
    person_box: FloatArray,
    keypoints: FloatArray,
    kpt_conf: FloatArray | None,
    inside_roi: bool,
    roi_source: str,
) -> None:
    bbox_color: ColorBGR = (0, 0, 255) if inside_roi else (180, 180, 180)
    x1, y1, x2, y2 = map(int, person_box[:4])
    cv2.rectangle(draw_frame, (x1, y1), (x2, y2), bbox_color, 2)

    for idx in range(17):
        if not is_keypoint_valid(keypoints, idx, kpt_conf):
            continue

        point = keypoints[idx]
        is_roi_point = point_in_roi(point)
        point_color: ColorBGR = (0, 0, 255) if is_roi_point else (0, 255, 255)
        radius = 5 if idx in {13, 14, 15, 16} else 3
        cv2.circle(
            draw_frame,
            (int(point[0]), int(point[1])),
            radius,
            point_color,
            -1,
        )

    for virtual_foot, _source in estimate_virtual_feet_from_knees(keypoints, kpt_conf):
        point_color = (0, 0, 255) if point_in_roi(virtual_foot) else (255, 255, 0)
        cv2.circle(
            draw_frame,
            (int(virtual_foot[0]), int(virtual_foot[1])),
            6,
            point_color,
            2,
        )

    width = float(person_box[2] - person_box[0])
    height = float(person_box[3] - person_box[1])
    bbox_points = [
        np.array([person_box[0] + 0.50 * width, person_box[3]], dtype=np.float32),
        np.array([person_box[0] + 0.25 * width, person_box[3]], dtype=np.float32),
        np.array([person_box[0] + 0.75 * width, person_box[3]], dtype=np.float32),
        np.array(
            [person_box[0] + 0.50 * width, person_box[1] + 0.50 * height],
            dtype=np.float32,
        ),
        np.array(
            [person_box[0] + 0.50 * width, person_box[1] + 0.85 * height],
            dtype=np.float32,
        ),
    ]

    for point in bbox_points:
        point_color = (0, 0, 255) if point_in_roi(point) else (255, 0, 0)
        cv2.circle(
            draw_frame,
            (int(point[0]), int(point[1])),
            4,
            point_color,
            -1,
        )

    debug_text = f"IN ROI by {roi_source}" if inside_roi else "OUTSIDE ROI"
    cv2.putText(
        draw_frame,
        debug_text,
        (x1, max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        bbox_color,
        2,
        cv2.LINE_AA,
    )


def run_danger_zone_monitor() -> None:
    print("Dang load model...")
    pose_model: YOLO = YOLO(POSE_MODEL_PATH)
    model_names: dict[int, str] = {
        int(key): str(value)
        for key, value in pose_model.names.items()
    }

    cap: cv2.VideoCapture = cv2.VideoCapture(VIDEO_INPUT_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Khong mo duoc video: {VIDEO_INPUT_PATH}")

    width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps: float = float(cap.get(cv2.CAP_PROP_FPS))

    if width <= 0 or height <= 0:
        raise ValueError("Kich thuoc video khong hop le.")
    if fps <= 0:
        fps = 25.0

    output_path = Path(VIDEO_OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out: cv2.VideoWriter = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not out.isOpened():
        raise RuntimeError(f"Khong tao duoc output video: {VIDEO_OUTPUT_PATH}")

    print("Warming up model...")
    dummy_frame: FrameArray = np.zeros((height, width, 3), dtype=np.uint8)
    _ = pose_model(
        dummy_frame,
        imgsz=POSE_IMGSZ,
        conf=PERSON_CONF_THRES,
        classes=[0],
        verbose=False,
    )[0]

    frame_idx: int = 0
    last_alert_frame: int = -999999
    start_time: float = time.perf_counter()

    print("Bat dau xu ly video...")
    while cap.isOpened():
        ret: bool
        frame: FrameArray
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        infer_frame: FrameArray = frame.copy()
        draw_frame: FrameArray = frame.copy()

        pose_results: Results = pose_model(
            infer_frame,
            imgsz=POSE_IMGSZ,
            conf=PERSON_CONF_THRES,
            classes=[0],
            verbose=False,
        )[0]

        raw_danger_alert = False
        people_in_roi_count = 0

        if pose_results.keypoints is not None and pose_results.boxes is not None:
            bboxes_array: FloatArray = pose_results.boxes.xyxy.cpu().numpy()
            person_conf_array: FloatArray = pose_results.boxes.conf.cpu().numpy()
            class_array: FloatArray = pose_results.boxes.cls.cpu().numpy()
            keypoints_array: FloatArray = pose_results.keypoints.xy.cpu().numpy()
            keypoint_conf_array: FloatArray | None = None
            if pose_results.keypoints.conf is not None:
                keypoint_conf_array = pose_results.keypoints.conf.cpu().numpy()

            person_count: int = min(
                len(bboxes_array),
                len(person_conf_array),
                len(class_array),
                len(keypoints_array),
            )

            for index in range(person_count):
                person_box: FloatArray = bboxes_array[index]
                person_conf = float(person_conf_array[index])
                person_cls = float(class_array[index])
                keypoints: FloatArray = keypoints_array[index]
                keypoint_conf: FloatArray | None = None
                if keypoint_conf_array is not None and index < len(keypoint_conf_array):
                    keypoint_conf = keypoint_conf_array[index]

                if not is_person_class(person_cls, model_names):
                    continue
                if not is_valid_person_box(person_box, person_conf):
                    continue

                inside_roi, roi_source = person_in_danger_zone(
                    keypoints,
                    person_box,
                    keypoint_conf,
                )

                if inside_roi:
                    raw_danger_alert = True
                    people_in_roi_count += 1
                    last_alert_frame = frame_idx

                if DEBUG_MODE:
                    draw_person_debug(
                        draw_frame,
                        person_box,
                        keypoints,
                        keypoint_conf,
                        inside_roi,
                        roi_source,
                    )

        danger_alert = raw_danger_alert or (
            frame_idx - last_alert_frame <= ALERT_HOLD_FRAMES
        )

        roi_color: ColorBGR = (0, 0, 255) if danger_alert else (0, 255, 0)
        draw_roi(draw_frame, roi_color)

        if danger_alert:
            blink_on: bool = (frame_idx // BLINK_INTERVAL_FRAMES) % 2 == 0
            if blink_on:
                overlay: FrameArray = draw_frame.copy()
                overlay[:] = (0, 0, 255)
                draw_frame = cv2.addWeighted(overlay, 0.35, draw_frame, 0.65, 0)

            draw_roi(draw_frame, roi_color)
            draw_alert_text(draw_frame)

        if people_in_roi_count > 0:
            draw_people_count(draw_frame, people_in_roi_count)

        out.write(draw_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    total_time: float = time.perf_counter() - start_time
    print(f"Tong so frame da xu ly: {frame_idx}")
    print(f"Tong thoi gian xu ly: {total_time:.2f} giay")
    print(f"Hoan tat! Video da duoc luu tai: {VIDEO_OUTPUT_PATH}")


if __name__ == "__main__":
    run_danger_zone_monitor()
