import time

import cv2
from ultralytics import YOLO

from stair_monitor.config.settings import SETTINGS, load_camera_config
from stair_monitor.core.analyzer import BehaviorAnalyzer
from stair_monitor.debug.performance import (
    accumulate_analysis_perf,
    log_perf,
    new_perf_totals,
)
from stair_monitor.debug.snapshots import (
    ensure_debug_snapshot_dirs,
    save_debug_snapshots,
)
from stair_monitor.output.rendering import (
    VietnameseTextDrawer,
    draw_demo_violation_alerts,
    draw_person_overlay,
    draw_scene_guides,
)
from stair_monitor.state.person_identity import PersonIdentityManager
from stair_monitor.vision.geometry import extract_pose_features


CONFIG = load_camera_config()


# Main loop cua demo stair_monitor tren Windows.
# Flow: mo video -> doc frame -> chay model/tracker -> xu ly tung person -> ve overlay -> ghi output.
def process_video():
    # Mo model pose va khoi tao analyzer cho logic stair_monitor.
    model = YOLO("yolo11x-pose.pt")
    analyzer = BehaviorAnalyzer(CONFIG)
    identity_manager = PersonIdentityManager(CONFIG)

    # Mo video input/offline demo.
    cap = cv2.VideoCapture(SETTINGS.video.input_path)
    out = None
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    output_fps = source_fps if source_fps > 0.0 else 15.0
    demo_alert_hold_frames = max(
        1,
        int(round(output_fps * SETTINGS.demo.demo_alert_hold_seconds)),
    )
    active_alert_until_frame: dict[str, int] = {}

    if SETTINGS.video.save_output_video:
        # Mo writer output neu can luu lai video demo da ve overlay.
        out = cv2.VideoWriter(
            SETTINGS.video.output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            output_fps,
            (int(cap.get(3)), int(cap.get(4))),
        )

    total_start_time = time.perf_counter()
    perf_totals = new_perf_totals() if SETTINGS.performance.enable_perf_log else None
    perf_window_start = (
        time.perf_counter() if SETTINGS.performance.enable_perf_log else None
    )
    perf_window_frames = 0

    debug_frame_index = 0
    model_input_dir, overlay_dir = ensure_debug_snapshot_dirs()

    # Vong lap doc tung frame video cho den khi het video hoac doc loi.
    while cap.isOpened():
        frame_start = (
            time.perf_counter() if SETTINGS.performance.enable_perf_log else None
        )

        # Doc frame hien tai.
        ret, frame = cap.read()
        if not ret:
            break

        analyzer.begin_frame()
        identity_manager.begin_frame(analyzer.frame_index)
        debug_frame_index += 1

        # frame_raw la anh goc sach tu video.
        # model_frame la ban copy sach chi dung cho inference.
        # overlay_frame la ban copy rieng chi dung de ve demo/debug.
        frame_raw = frame.copy()
        model_frame = frame_raw.copy()
        overlay_frame = frame_raw.copy()

        model_input_snapshot = (
            model_frame.copy() if SETTINGS.video.save_model_input_debug else None
        )

        # Chay model/tracker de lay bbox, track_id va keypoint cho tung nguoi.
        model_start = (
            time.perf_counter() if SETTINGS.performance.enable_perf_log else None
        )

        results = model.track(
            model_frame,
            conf=0.7,
            persist=True,
            classes=[0],
            tracker="bytetrack.yaml",
            verbose=False,
            iou=0.9,
        )

        if perf_totals is not None and model_start is not None:
            perf_totals["model"] += (time.perf_counter() - model_start) * 1000.0

        # Gom toan bo thao tac ve overlay vao 1 context de giu tieng Viet co dau.
        with VietnameseTextDrawer(overlay_frame) as text_drawer:
            overlay_start = (
                time.perf_counter()
                if SETTINGS.performance.enable_perf_log
                else None
            )

            if SETTINGS.demo.draw_debug and (
                not SETTINGS.demo.demo_mode or SETTINGS.demo.handrail_debug_only
            ):
                draw_scene_guides(overlay_frame, CONFIG, analyzer)

            if (
                results
                and results[0].boxes is not None
                and results[0].boxes.id is not None
                and results[0].keypoints is not None
            ):
                # Xu ly tung nguoi trong frame hien tai.
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().tolist()
                keypoints = results[0].keypoints.data.cpu().numpy()

                for box, tid, kpt in zip(boxes, track_ids, keypoints):
                    yolo_track_id = int(tid)

                    features = extract_pose_features(kpt, box)
                    identity_result = identity_manager.update_detection(
                        yolo_track_id=yolo_track_id,
                        bbox=box,
                        features=features,
                        frame_index=analyzer.frame_index,
                    )

                    # p_lane dung cho sai lan; p_motion giu compatibility cho backward/standing.
                    p_lane = features.get("feet_point")
                    p_motion = features.get("motion_point")

                    merge_from_subject_id = identity_result.merged_from_analysis_subject_id
                    if merge_from_subject_id:
                        analyzer.merge_behavior_history(
                            merge_from_subject_id,
                            identity_result.analysis_subject_id,
                        )

                    if identity_result.analysis_subject_id:
                        analysis = analyzer.analyze(
                            identity_result.analysis_subject_id,
                            p_lane,
                            p_motion,
                            kpt,
                            box=box,
                            features=features,
                        )
                        if identity_result.person_uid is not None:
                            identity_manager.update_from_analysis(
                                identity_result.person_uid,
                                analysis,
                            )
                        identity_manager.augment_analysis_result(
                            analysis,
                            identity_result,
                        )
                        for warning in analysis.get("warnings", []):
                            if warning not in SETTINGS.violation.count_labels:
                                continue
                            active_alert_until_frame[warning] = max(
                                active_alert_until_frame.get(warning, -1),
                                analyzer.frame_index + demo_alert_hold_frames - 1,
                            )

                        if perf_totals is not None:
                            accumulate_analysis_perf(
                                perf_totals,
                                analysis.get("perf"),
                            )
                    else:
                        analysis = identity_manager.build_overlay_result(
                            identity_result
                        )

                    # Ve ket qua len frame sau khi da co full analysis.
                    draw_person_overlay(
                        overlay_frame,
                        box,
                        kpt,
                        p_lane,
                        p_motion,
                        analysis,
                        features=features,
                        text_drawer=text_drawer,
                    )

            active_alert_until_frame = {
                warning: until_frame
                for warning, until_frame in active_alert_until_frame.items()
                if until_frame >= analyzer.frame_index
            }
            if SETTINGS.demo.demo_mode and not SETTINGS.demo.handrail_debug_only:
                draw_demo_violation_alerts(
                    overlay_frame,
                    active_alert_until_frame,
                )

            identity_manager.end_frame()
            analyzer.cleanup_inactive_tracks(
                identity_manager.get_retained_analysis_subject_ids()
            )

            if perf_totals is not None and overlay_start is not None:
                perf_totals["overlay"] += (
                    time.perf_counter() - overlay_start
                ) * 1000.0

        save_debug_snapshots(
            debug_frame_index,
            model_input_snapshot if model_input_snapshot is not None else model_frame,
            overlay_frame,
            model_input_dir,
            overlay_dir,
        )

        # Ghi frame output sau khi da ve xong overlay.
        if SETTINGS.video.save_output_video and out is not None:
            write_start = (
                time.perf_counter()
                if SETTINGS.performance.enable_perf_log
                else None
            )

            out.write(overlay_frame)

            if perf_totals is not None and write_start is not None:
                perf_totals["video_write_show"] += (
                    time.perf_counter() - write_start
                ) * 1000.0

        if perf_totals is not None and frame_start is not None:
            perf_totals["total"] += (time.perf_counter() - frame_start) * 1000.0

        # Log perf theo cua so N frame de de theo doi FPS.
        if SETTINGS.performance.enable_perf_log:
            perf_window_frames += 1

            if perf_window_frames >= max(1, SETTINGS.performance.perf_log_interval):
                elapsed_s = time.perf_counter() - perf_window_start
                log_perf(perf_totals, perf_window_frames, elapsed_s)

                perf_totals = new_perf_totals()
                perf_window_start = time.perf_counter()
                perf_window_frames = 0

    # Cleanup resource sau khi het video.
    total_end_time = time.perf_counter()

    cap.release()

    if out is not None:
        out.release()

    cv2.destroyAllWindows()

    if not SETTINGS.demo.demo_mode:
        print(total_end_time - total_start_time)


if __name__ == "__main__":
    process_video()
