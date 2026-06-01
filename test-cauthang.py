import cv2
from ultralytics import YOLO
import time
from stair_monitor.analyzer import BehaviorAnalyzer
from stair_monitor.rendering import (
    draw_people_count,
    draw_person_overlay,
    draw_scene_guides,
    get_feet_point,
    get_motion_point,
)
from stair_monitor.settings import (
    DEMO_MODE,
    DRAW_DEBUG,
    SAVE_OUTPUT_VIDEO,
    SHOW_PEOPLE_COUNT,
    VIDEO_INPUT_PATH,
    VIDEO_OUTPUT_PATH,
    load_camera_config,
)

CONFIG = load_camera_config()


def process_video():
    model = YOLO("yolo11x-pose.pt")
    analyzer = BehaviorAnalyzer(CONFIG)
    cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
    out = None
    if SAVE_OUTPUT_VIDEO:
        out = cv2.VideoWriter(
            VIDEO_OUTPUT_PATH,
            cv2.VideoWriter_fourcc(*"mp4v"),
            25,
            (int(cap.get(3)), int(cap.get(4))),
        )
    start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        inside_track_ids = set()
        yolo_frame = frame.copy()
        debug_frame = frame.copy()

        results = model.track(
            yolo_frame,
            conf=0.5,
            persist=True,
            classes=[0],
            tracker="bytetrack.yaml",
            verbose=False,
        )

        if DRAW_DEBUG and not DEMO_MODE:
            draw_scene_guides(debug_frame, CONFIG, analyzer)

        if results[0].boxes.id is not None:
            for box, tid, kpt in zip(
                results[0].boxes.xyxy.cpu().numpy(),
                results[0].boxes.id.int().tolist(),
                results[0].keypoints.data.cpu().numpy(),
            ):
                p_lane = get_feet_point(box, kpt)
                p_motion = get_motion_point(box, kpt)
                if tid is not None and analyzer.is_inside_stairs(p_lane):
                    inside_track_ids.add(int(tid))
                ana = analyzer.analyze(tid, p_lane, p_motion, kpt)
                draw_person_overlay(debug_frame, box, kpt, p_lane, p_motion, ana)

        current_inside_count = len(inside_track_ids)
        if SHOW_PEOPLE_COUNT:
            draw_people_count(debug_frame, current_inside_count)

        if SAVE_OUTPUT_VIDEO and out is not None:
            out.write(debug_frame)
    end_time = time.time()
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    if not DEMO_MODE:
        print(end_time - start_time)

if __name__ == "__main__":
    process_video()
