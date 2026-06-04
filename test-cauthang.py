import cv2
from ultralytics import YOLO
import time
from stair_monitor.analyzer import BehaviorAnalyzer
from stair_monitor.rendering import (
    draw_person_overlay,
    draw_scene_guides,
    draw_violation_summary,
    get_feet_point,
    get_motion_point,
)
from stair_monitor.settings import (
    DEMO_MODE,
    DRAW_DEBUG,
    SAVE_OUTPUT_VIDEO,
    SHOW_PEOPLE_COUNT,
    VIOLATION_COUNT_LABELS,
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
    total_violation_track_ids = set()
    total_violation_by_type = {
        label: set() for label in VIOLATION_COUNT_LABELS
    }
    start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_inside_track_ids = set()
        current_violation_track_ids = set()
        current_violation_counts = {
            label: 0 for label in VIOLATION_COUNT_LABELS
        }
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
                ana = analyzer.analyze(tid, p_lane, p_motion, kpt)
                track_id = ana.get("track_id", tid)
                if ana.get("inside_stairs") and track_id is not None:
                    current_inside_track_ids.add(int(track_id))

                warnings = ana.get("warnings", [])
                real_warnings = [
                    warning
                    for warning in warnings
                    if warning in VIOLATION_COUNT_LABELS
                ]
                if real_warnings and track_id is not None:
                    current_violation_track_ids.add(int(track_id))
                    total_violation_track_ids.add(int(track_id))
                    for warning in real_warnings:
                        current_violation_counts[warning] += 1
                        total_violation_by_type[warning].add(int(track_id))

                draw_person_overlay(debug_frame, box, kpt, p_lane, p_motion, ana)

        current_inside_count = len(current_inside_track_ids)
        current_violation_people_count = len(current_violation_track_ids)
        total_violation_people_count = len(total_violation_track_ids)
        total_violation_counts = {
            label: len(track_ids)
            for label, track_ids in total_violation_by_type.items()
        }
        if SHOW_PEOPLE_COUNT:
            draw_violation_summary(
                debug_frame,
                current_inside_count,
                current_violation_people_count,
                current_violation_counts,
                total_violation_people_count,
                total_violation_counts,
            )

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
