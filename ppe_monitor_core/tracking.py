from collections import deque

from ppe_monitor_core.config import (
    HELMET_SAFE_RATIO_THRES,
    HELMET_VIOLATION_RATIO_THRES,
    MIN_HISTORY_TO_DECIDE,
    TEMPORAL_WINDOW,
    TRACK_IOU_THRES,
    TRACK_TTL,
    VEST_SAFE_RATIO_THRES,
    VEST_VIOLATION_RATIO_THRES,
)


def bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0

    return inter_area / union


def get_track_id(person_box, tracks, frame_idx, assigned_track_ids):
    best_id = None
    best_iou = 0.0

    for track_id, track in tracks.items():
        if track_id in assigned_track_ids:
            continue
        if frame_idx - track["last_seen"] > TRACK_TTL:
            continue

        current_iou = bbox_iou(person_box, track["bbox"])
        if current_iou > best_iou:
            best_iou = current_iou
            best_id = track_id

    if best_id is not None and best_iou >= TRACK_IOU_THRES:
        return best_id

    return None


def make_new_track(person_box, frame_idx):
    return {
        "bbox": person_box.copy(),
        "last_seen": frame_idx,
        "hat_history": deque(maxlen=TEMPORAL_WINDOW),
        "vest_history": deque(maxlen=TEMPORAL_WINDOW),
        "stable_has_hat": None,
        "stable_has_vest": None,
    }


def update_stable_state(track):
    if len(track["hat_history"]) >= MIN_HISTORY_TO_DECIDE:
        hat_ratio = sum(track["hat_history"]) / len(track["hat_history"])
        if hat_ratio >= HELMET_SAFE_RATIO_THRES:
            track["stable_has_hat"] = True
        elif hat_ratio <= HELMET_VIOLATION_RATIO_THRES:
            track["stable_has_hat"] = False

    if len(track["vest_history"]) >= MIN_HISTORY_TO_DECIDE:
        vest_ratio = sum(track["vest_history"]) / len(track["vest_history"])
        if vest_ratio >= VEST_SAFE_RATIO_THRES:
            track["stable_has_vest"] = True
        elif vest_ratio <= VEST_VIOLATION_RATIO_THRES:
            track["stable_has_vest"] = False


def build_person_status(track_id, track):
    history_len = len(track["hat_history"])
    hat_ratio = sum(track["hat_history"]) / history_len if history_len > 0 else 0.0
    vest_ratio = (
        sum(track["vest_history"]) / len(track["vest_history"])
        if len(track["vest_history"]) > 0
        else 0.0
    )
    stable_has_hat = track["stable_has_hat"]
    stable_has_vest = track["stable_has_vest"]

    if (
        history_len < MIN_HISTORY_TO_DECIDE
        or stable_has_hat is None
        or stable_has_vest is None
    ):
        label = "CHECKING PPE..."
        status_color = (0, 255, 255)
        severity = 1
    elif stable_has_hat and stable_has_vest:
        label = "SAFE: Full PPE"
        status_color = (0, 255, 0)
        severity = 0
    elif not stable_has_hat and not stable_has_vest:
        label = "WARNING: No Hat + No Vest"
        status_color = (0, 0, 255)
        severity = 4
    elif not stable_has_hat:
        label = "WARNING: No Hat"
        status_color = (0, 0, 255)
        severity = 3
    else:
        label = "WARNING: No Vest"
        status_color = (0, 0, 255)
        severity = 2

    label_text = (
        f"{label} ID:{track_id} H:{hat_ratio:.2f} V:{vest_ratio:.2f} N:{history_len}"
    )

    return {
        "track_id": track_id,
        "label": label,
        "label_text": label_text,
        "severity": severity,
        "status_color": status_color,
        "stable_has_hat": stable_has_hat,
        "stable_has_vest": stable_has_vest,
        "hat_ratio": hat_ratio,
        "vest_ratio": vest_ratio,
        "history_len": history_len,
    }


def prune_expired_tracks(tracks, frame_idx):
    expired_track_ids = [
        track_id
        for track_id, track in tracks.items()
        if frame_idx - track["last_seen"] > TRACK_TTL
    ]
    for track_id in expired_track_ids:
        del tracks[track_id]
