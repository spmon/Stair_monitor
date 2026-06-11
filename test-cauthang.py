import os
import time

import cv2
from ultralytics import YOLO

from stair_monitor.analyzer import BehaviorAnalyzer
from stair_monitor.geometry import extract_pose_features
from stair_monitor.rendering import (
    VietnameseTextDrawer,
    draw_person_overlay,
    draw_scene_guides,
    draw_violation_summary,
)
from stair_monitor.settings import (
    DEMO_ALERT_HOLD_SECONDS,
    DEMO_MODE,
    DRAW_DEBUG,
    ENABLE_PERF_LOG,
    PERF_LOG_INTERVAL,
    SAVE_MODEL_INPUT_DEBUG,
    SAVE_MODEL_INPUT_DEBUG_EVERY,
    SAVE_OUTPUT_VIDEO,
    SHOW_PEOPLE_COUNT,
    VIOLATION_COUNT_LABELS,
    VIDEO_INPUT_PATH,
    VIDEO_OUTPUT_PATH,
    load_camera_config,
)

CONFIG = load_camera_config()
PERF_FIELDS = (
    "model",
    "analyze",
    "direction",
    "lane",
    "hold",
    "carry",
    "standing",
    "backward",
    "overlay",
    "summary_panel",
    "video_write_show",
    "total",
)


# Tao bo dem perf cho ban Windows/demo hien tai.
def _new_perf_totals():
    return {field: 0.0 for field in PERF_FIELDS}


# Cong don perf tung nguoi vao tong perf theo frame.
def _accumulate_analysis_perf(perf_totals, analysis_perf):
    if not analysis_perf:
        return

    for field in ("analyze", "direction", "lane", "hold", "carry", "standing", "backward"):
        perf_totals[field] += analysis_perf.get(field, 0.0)


# In toc do va do tre trung binh theo tung block lon.
def _log_perf(perf_totals, frame_count, elapsed_s):
    if frame_count <= 0 or elapsed_s <= 0:
        return

    fps = frame_count / elapsed_s
    avg_ms = {
        field: perf_totals[field] / frame_count
        for field in PERF_FIELDS
    }
    print(
        "[PERF] "
        f"fps={fps:.1f} "
        f"model={avg_ms['model']:.1f}ms "
        f"analyze={avg_ms['analyze']:.1f}ms "
        f"direction={avg_ms['direction']:.1f}ms "
        f"lane={avg_ms['lane']:.1f}ms "
        f"hold={avg_ms['hold']:.1f}ms "
        f"carry={avg_ms['carry']:.1f}ms "
        f"standing={avg_ms['standing']:.1f}ms "
        f"backward={avg_ms['backward']:.1f}ms "
        f"overlay={avg_ms['overlay']:.1f}ms "
        f"summary={avg_ms['summary_panel']:.1f}ms "
        f"video={avg_ms['video_write_show']:.1f}ms "
        f"total={avg_ms['total']:.1f}ms"
    )


def _build_demo_alert_display_counts(
    current_time,
    active_violations,
    demo_alert_last_seen,
):
    for label in active_violations:
        demo_alert_last_seen[label] = current_time

    display_counts = {label: 0 for label in VIOLATION_COUNT_LABELS}
    for label in VIOLATION_COUNT_LABELS:
        last_seen = demo_alert_last_seen.get(label)
        if (
            last_seen is not None
            and current_time - last_seen <= DEMO_ALERT_HOLD_SECONDS
        ):
            display_counts[label] = 1

    for label in list(demo_alert_last_seen.keys()):
        if current_time - demo_alert_last_seen[label] > DEMO_ALERT_HOLD_SECONDS:
            del demo_alert_last_seen[label]

    return display_counts


def _ensure_debug_snapshot_dirs():
    if not SAVE_MODEL_INPUT_DEBUG:
        return None, None

    model_input_dir = os.path.join("debug_model_input")
    overlay_dir = os.path.join("debug_overlay")
    os.makedirs(model_input_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    return model_input_dir, overlay_dir


def _save_debug_snapshots(frame_index, model_input_frame, overlay_frame, model_input_dir, overlay_dir):
    if not SAVE_MODEL_INPUT_DEBUG or model_input_dir is None or overlay_dir is None:
        return

    interval = max(1, int(SAVE_MODEL_INPUT_DEBUG_EVERY or 1))
    if frame_index % interval != 0:
        return

    filename = f"frame_{frame_index:04d}.jpg"
    cv2.imwrite(os.path.join(model_input_dir, filename), model_input_frame)
    cv2.imwrite(os.path.join(overlay_dir, filename), overlay_frame)


# Main loop cua demo stair_monitor tren Windows.
# Flow: mo video -> doc frame -> chay model/tracker -> xu ly tung person -> ve overlay -> ghi output.
def process_video():
    # Mo model pose va khoi tao analyzer cho logic stair_monitor.
    model = YOLO("yolo11x-pose.pt")
    analyzer = BehaviorAnalyzer(CONFIG)
    # Mo video input/offline demo.
    cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
    out = None
    if SAVE_OUTPUT_VIDEO:
        # Mo writer output neu can luu lai video demo da ve overlay.
        out = cv2.VideoWriter(
            VIDEO_OUTPUT_PATH,
            cv2.VideoWriter_fourcc(*"mp4v"),
            15,
            (int(cap.get(3)), int(cap.get(4))),
        )

    total_violation_track_ids = set()
    total_violation_by_type = {label: set() for label in VIOLATION_COUNT_LABELS}
    total_start_time = time.perf_counter()
    perf_totals = _new_perf_totals() if ENABLE_PERF_LOG else None
    perf_window_start = time.perf_counter() if ENABLE_PERF_LOG else None
    perf_window_frames = 0
    demo_alert_last_seen = {}
    debug_frame_index = 0
    model_input_dir, overlay_dir = _ensure_debug_snapshot_dirs()

    # Vong lap doc tung frame video cho den khi het video hoac doc loi.
    while cap.isOpened():
        frame_start = time.perf_counter() if ENABLE_PERF_LOG else None
        # Doc frame hien tai.
        ret, frame = cap.read()
        if not ret:
            break
        analyzer.begin_frame()
        debug_frame_index += 1

        # frame_raw la anh goc sach tu video.
        # model_frame la ban copy sach chi dung cho inference.
        # overlay_frame la ban copy rieng chi dung de ve demo/debug.
        frame_raw = frame.copy()
        model_frame = frame_raw.copy()
        overlay_frame = frame_raw.copy()

        # current_* chi dem trong frame hien tai.
        # total_* la tong hop tu dau video den hien tai, dung set track_id de 1 nguoi khong bi tinh lap lai.
        current_inside_track_ids = set()
        current_violation_track_ids = set()
        current_violation_counts = {label: 0 for label in VIOLATION_COUNT_LABELS}
        active_violations = set()
        active_track_ids = set()
        model_input_snapshot = model_frame.copy() if SAVE_MODEL_INPUT_DEBUG else None

        # Chay model/tracker de lay bbox, track_id va keypoint cho tung nguoi.
        model_start = time.perf_counter() if ENABLE_PERF_LOG else None
        results = model.track(
            model_frame,
            conf=0.7,
            persist=True,
            classes=[0],
            tracker="bytetrack.yaml",
            verbose=False,
            iou=0.9
        )
        if perf_totals is not None:
            perf_totals["model"] += (time.perf_counter() - model_start) * 1000.0

        # Gom toan bo thao tac ve overlay vao 1 context de giu tieng Viet co dau.
        with VietnameseTextDrawer(overlay_frame) as text_drawer:
            overlay_start = time.perf_counter() if ENABLE_PERF_LOG else None
            if DRAW_DEBUG and not DEMO_MODE:
                draw_scene_guides(overlay_frame, CONFIG, analyzer)

            if results[0].boxes.id is not None:
                # Xu ly tung nguoi trong frame hien tai.
                for box, tid, kpt in zip(
                    results[0].boxes.xyxy.cpu().numpy(),
                    results[0].boxes.id.int().tolist(),
                    results[0].keypoints.data.cpu().numpy(),
                ):
                    active_track_ids.add(int(tid))
                    features = extract_pose_features(kpt, box)
                    # p_lane dung cho sai lan; p_motion dung cho direction/backward/standing.
                    p_lane = features.get("feet_point")
                    p_motion = features.get("motion_point")
                    # Goi analyzer de tong hop toan bo logic cho 1 person/1 frame.
                    ana = analyzer.analyze(
                        tid,
                        p_lane,
                        p_motion,
                        kpt,
                        box=box,
                        features=features,
                    )
                    if perf_totals is not None:
                        _accumulate_analysis_perf(perf_totals, ana.get("perf"))

                    track_id = ana.get("track_id", tid)
                    if ana.get("inside_stairs") and track_id is not None:
                        current_inside_track_ids.add(int(track_id))

                    warnings = ana.get("warnings", [])
                    # Chi dem cac loi thuc su vao thong ke.
                    # Mot nguoi co nhieu loi trong cung frame van duoc tinh vao tung loai loi tuong ung.
                    real_warnings = [
                        warning
                        for warning in warnings
                        if warning in VIOLATION_COUNT_LABELS
                    ]
                    for warning in real_warnings:
                        active_violations.add(warning)

                    if real_warnings and track_id is not None:
                        current_violation_track_ids.add(int(track_id))
                        total_violation_track_ids.add(int(track_id))
                        for warning in real_warnings:
                            current_violation_counts[warning] += 1
                            total_violation_by_type[warning].add(int(track_id))

                    # Ve ket qua len frame sau khi da co full analysis.
                    draw_person_overlay(
                        overlay_frame,
                        box,
                        kpt,
                        p_lane,
                        p_motion,
                        ana,
                        text_drawer=text_drawer,
                    )

            analyzer.cleanup_inactive_tracks(active_track_ids)

            if perf_totals is not None:
                perf_totals["overlay"] += (
                    time.perf_counter() - overlay_start
                ) * 1000.0

            # Chot thong ke cua frame hien tai va thong ke tong tu dau video.
            current_inside_count = len(current_inside_track_ids)
            current_violation_people_count = len(current_violation_track_ids)
            total_violation_people_count = len(total_violation_track_ids)
            total_violation_counts = {
                label: len(track_ids)
                for label, track_ids in total_violation_by_type.items()
            }
            demo_violation_counts = current_violation_counts
            if DEMO_MODE and SHOW_PEOPLE_COUNT:
                demo_violation_counts = _build_demo_alert_display_counts(
                    time.time(),
                    active_violations,
                    demo_alert_last_seen,
                )

            if SHOW_PEOPLE_COUNT:
                # Ve panel tong hop len video.
                summary_start = time.perf_counter() if ENABLE_PERF_LOG else None
                draw_violation_summary(
                    overlay_frame,
                    current_inside_count,
                    current_violation_people_count,
                    current_violation_counts,
                    total_violation_people_count,
                    total_violation_counts,
                    demo_violation_counts=demo_violation_counts,
                    text_drawer=text_drawer,
                )
                if perf_totals is not None:
                    perf_totals["summary_panel"] += (
                        time.perf_counter() - summary_start
                    ) * 1000.0

        _save_debug_snapshots(
            debug_frame_index,
            model_input_snapshot if model_input_snapshot is not None else model_frame,
            overlay_frame,
            model_input_dir,
            overlay_dir,
        )

        # Ghi frame output sau khi da ve xong overlay.
        if SAVE_OUTPUT_VIDEO and out is not None:
            write_start = time.perf_counter() if ENABLE_PERF_LOG else None
            out.write(overlay_frame)
            if perf_totals is not None:
                perf_totals["video_write_show"] += (
                    time.perf_counter() - write_start
                ) * 1000.0

        if perf_totals is not None:
            perf_totals["total"] += (time.perf_counter() - frame_start) * 1000.0

        # Log perf theo cua so N frame de de theo doi FPS.
        if ENABLE_PERF_LOG:
            perf_window_frames += 1
            if perf_window_frames >= max(1, PERF_LOG_INTERVAL):
                elapsed_s = time.perf_counter() - perf_window_start
                _log_perf(perf_totals, perf_window_frames, elapsed_s)
                perf_totals = _new_perf_totals()
                perf_window_start = time.perf_counter()
                perf_window_frames = 0

    # Cleanup resource sau khi het video.
    total_end_time = time.perf_counter()
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

    if not DEMO_MODE:
        print(total_end_time - total_start_time)


if __name__ == "__main__":
    process_video()
