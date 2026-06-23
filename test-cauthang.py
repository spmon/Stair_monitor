import os
import time
from collections.abc import Mapping, Sequence
from datetime import datetime

import cv2
from loguru import logger
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
from stair_monitor.input.latest_frame_capture import LatestFrameCapture
from stair_monitor.output.logging_setup import setup_app_logging
from stair_monitor.output.rendering import (
    VietnameseTextDrawer,
    draw_demo_violation_alerts,
    draw_person_overlay,
    draw_scene_guides,
)
from stair_monitor.output.alert_event_logger import AlertEventLogger, ViolationEvent
from stair_monitor.state.person_identity import PersonIdentityManager
from stair_monitor.vision.geometry import extract_pose_features


RTSP_TUNNEL_URL = (
    "rtsp://admin:vna%40123456@localhost:9999/"
    "cam/realmonitor?channel=1&subtype=0"
)


def open_video_capture(input_path: str) -> cv2.VideoCapture:
    """Open video file or RTSP tunnel input for the Windows demo."""
    is_rtsp_input = input_path.lower().startswith("rtsp://")
    if is_rtsp_input:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(input_path, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        if is_rtsp_input:
            raise RuntimeError(
                "Khong the mo RTSP tunnel. Kiem tra SSH tunnel localhost:9999."
            )
        raise RuntimeError(f"Khong the mo video input: {input_path}")

    return cap


def _read_text_field(raw_value: object, default: str) -> str:
    if raw_value is None:
        return default
    value_text = str(raw_value).strip()
    return value_text if value_text else default


def _read_warning_tuple(raw_value: object) -> tuple[str, ...]:
    if not isinstance(raw_value, Sequence) or isinstance(raw_value, (str, bytes)):
        return ()

    warnings: list[str] = []
    for warning in raw_value:
        warning_text = str(warning).strip()
        if warning_text:
            warnings.append(warning_text)
    return tuple(warnings)


def build_violation_event(
    analysis: Mapping[str, object],
    frame_index: int,
    person_uid: str | None,
    analysis_subject_id: str | None,
    track_id: int | None,
) -> ViolationEvent | None:
    """Build a typed violation event from existing analyzer output."""
    warnings = _read_warning_tuple(analysis.get("warnings"))
    if not warnings:
        return None

    display_status = _read_text_field(analysis.get("display_status"), "")
    status = display_status or _read_text_field(analysis.get("status"), "")
    direction = _read_text_field(analysis.get("direction"), "UNKNOWN")

    return ViolationEvent(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        frame_index=frame_index,
        person_uid=_read_text_field(person_uid, ""),
        analysis_subject_id=_read_text_field(analysis_subject_id, ""),
        track_id=track_id,
        direction=direction,
        warnings=warnings,
        status=status,
    )


def should_draw_person_overlay() -> bool:
    """Return whether person-level overlay should be rendered on the current frame."""
    return not SETTINGS.demo.alert_only_display_active


# Main loop cua demo stair_monitor tren Windows.
# Flow: mo video -> doc frame -> chay model/tracker -> xu ly tung person -> ve overlay -> ghi output.
def process_video() -> None:
    """Chay toan bo pipeline demo offline tren Windows.

    INPUT:
        - `CONFIG`: camera JSON da load san, chua ROI/line/step.
        - `SETTINGS.video.input_path`: file video, RTSP URL hoac sentinel `RTSP_TUNNEL`.

    OUTPUT:
        - Ve overlay len frame.
        - Co the ghi output video va snapshot debug.

    WHY:
        - File nay chi dong vai tro dieu phoi. Business logic that nam o
          `extract_pose_features()`, `PersonIdentityManager` va `BehaviorAnalyzer`.
    """
    setup_app_logging(SETTINGS.video.alert_log_dir)

    # File nay la entry point chinh cua ban Windows/demo.
    # FLOW: video/frame -> YOLO pose -> features -> identity -> analyzer -> overlay -> output video.
    # WHY: Neu doi sang RTSP thi phan "mo source" thay doi, con pipeline tu frame tro di van giong nhau.
    config = load_camera_config()

    # Mo model pose va khoi tao analyzer cho logic stair_monitor.
    model = YOLO("yolo11x-pose.pt")
    analyzer = BehaviorAnalyzer(config)
    identity_manager = PersonIdentityManager(config)

    # Mo video input/offline demo hoac RTSP tunnel tren Windows.
    input_path = str(SETTINGS.video.input_path).strip()
    if input_path == "RTSP_TUNNEL":
        input_path = RTSP_TUNNEL_URL

    is_rtsp_input = input_path.lower().startswith("rtsp://")
    cap = None if is_rtsp_input else open_video_capture(input_path)
    latest_frame_capture = LatestFrameCapture(input_path) if is_rtsp_input else None
    if latest_frame_capture is not None:
        latest_frame_capture.start()
    out: cv2.VideoWriter | None = None
    source_fps = (
        latest_frame_capture.get_source_fps()
        if latest_frame_capture is not None
        else float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    )
    output_fps = source_fps if source_fps > 0.0 else 15.0
    demo_alert_hold_frames = max(
        1,
        int(round(output_fps * SETTINGS.demo.demo_alert_hold_seconds)),
    )
    active_alert_until_frame: dict[str, int] = {}
    failed_read_count = 0
    max_failed_read_count = 5 if is_rtsp_input else 1
    last_processed_sequence_id = -1
    alert_logger: AlertEventLogger | None = None
    if SETTINGS.video.alert_log_enabled:
        alert_log_cooldown_frames = max(
            1,
            int(round(output_fps * SETTINGS.video.alert_log_cooldown_seconds)),
        )
        alert_logger = AlertEventLogger(
            SETTINGS.video.alert_log_dir,
            alert_log_cooldown_frames,
        )

    logger.info("Dang dung input source: {}", input_path)
    if is_rtsp_input:
        logger.info("RTSP dang mo bang TCP qua OpenCV FFMPEG.")
    if alert_logger is not None:
        logger.info("Alert CSV: {}", alert_logger.file_path)

    total_start_time = time.perf_counter()
    perf_totals = new_perf_totals() if SETTINGS.performance.enable_perf_log else None
    perf_window_start = (
        time.perf_counter() if SETTINGS.performance.enable_perf_log else None
    )
    perf_window_frames = 0

    debug_frame_index = 0
    model_input_dir, overlay_dir = ensure_debug_snapshot_dirs()

    # Vong lap doc tung frame video cho den khi het video hoac doc loi.
    while True:
        frame_start = (
            time.perf_counter() if SETTINGS.performance.enable_perf_log else None
        )

        # Doc frame hien tai.
        if latest_frame_capture is not None:
            latest_frame = latest_frame_capture.read_latest()
            if latest_frame is None:
                time.sleep(0.005)
                continue
            if latest_frame.sequence_id == last_processed_sequence_id:
                time.sleep(0.005)
                continue

            last_processed_sequence_id = latest_frame.sequence_id
            frame = latest_frame.frame
        else:
            if cap is None:
                break
            ret, frame = cap.read()
            if not ret:
                failed_read_count += 1
                logger.warning(
                    "Khong doc duoc frame tu input ({}/{})",
                    failed_read_count,
                    max_failed_read_count,
                )
                if failed_read_count >= max_failed_read_count:
                    break
                continue
            failed_read_count = 0

        if SETTINGS.video.save_output_video and out is None:
            # Lazy init writer sau frame dau tien de tranh RTSP tra ve width/height = 0 luc moi mo.
            frame_h, frame_w = frame.shape[:2]
            out = cv2.VideoWriter(
                SETTINGS.video.output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                output_fps,
                (frame_w, frame_h),
            )

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

        # FLOW: Day la dau ra raw cua model, gom bbox + yolo_track_id + keypoints.
        # WHY: Analyzer khong doc truc tiep frame; moi logic sau deu bat dau tu ket qua pose nay.
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
                not SETTINGS.demo.demo_mode or SETTINGS.demo.focus_debug_only
            ):
                draw_scene_guides(overlay_frame, config, analyzer)

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

                    # FLOW: Chuyen keypoint raw thanh features de cac module sau tai su dung cung 1 ngon ngu du lieu.
                    features = extract_pose_features(kpt, box)

                    # FLOW: Bridge tu yolo_track_id tam thoi sang analysis_subject_id/person_uid on dinh hon.
                    # WHY: Tracker YOLO co the doi id khi che khuat; history warning khong nen reset theo moi lan doi id.
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
                        # INPUT: p_lane uu tien chan de xet inside/lane.
                        # INPUT: p_motion uu tien motion point de xet direction/backward/standing.
                        # OUTPUT: analyzer tra ve status, warnings va day du field debug cho rendering.
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

                    # FLOW: Rendering chi ve lai ket qua da tinh xong, khong tu quyet dinh warning.
                    # Ve ket qua len frame sau khi da co full analysis.
                    if should_draw_person_overlay():
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

                    if alert_logger is not None:
                        violation_event = build_violation_event(
                            analysis,
                            analyzer.frame_index,
                            identity_result.person_uid,
                            identity_result.analysis_subject_id,
                            yolo_track_id,
                        )
                        if violation_event is not None:
                            alert_logger.log_if_needed(
                                violation_event,
                                analyzer.frame_index,
                            )

            active_alert_until_frame = {
                warning: until_frame
                for warning, until_frame in active_alert_until_frame.items()
                if until_frame >= analyzer.frame_index
            }
            if SETTINGS.demo.demo_mode and not SETTINGS.demo.focus_debug_only:
                # WARNING: Badge lon nay chi tong hop warning dang active trong demo mode sach.
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

        # DEBUG: Snapshot giup so sanh model input sach voi overlay cuoi cung ma khong doi logic runtime.
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

        if SETTINGS.video.show_live_window:
            display_scale = SETTINGS.video.live_window_scale
            if display_scale > 0.0 and display_scale != 1.0:
                display_frame = cv2.resize(
                    overlay_frame,
                    (0, 0),
                    fx=display_scale,
                    fy=display_scale,
                )
            else:
                display_frame = overlay_frame
            cv2.imshow("Stair Monitor Demo Live", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

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

    if latest_frame_capture is not None:
        latest_frame_capture.stop()
        latest_frame_capture.release()
    elif cap is not None:
        cap.release()

    if out is not None:
        out.release()
    if alert_logger is not None:
        alert_logger.close()

    cv2.destroyAllWindows()

    if not SETTINGS.demo.demo_mode:
        logger.info("Tong thoi gian xu ly: {:.3f}s", total_end_time - total_start_time)


if __name__ == "__main__":
    setup_app_logging()
    process_video()
