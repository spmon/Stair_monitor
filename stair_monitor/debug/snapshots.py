from __future__ import annotations

import os

import cv2

from stair_monitor.config.settings import SETTINGS


def ensure_debug_snapshot_dirs() -> tuple[str | None, str | None]:
    if not SETTINGS.video.save_model_input_debug:
        return None, None

    model_input_dir = os.path.join("debug_model_input")
    overlay_dir = os.path.join("debug_overlay")
    os.makedirs(model_input_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    return model_input_dir, overlay_dir


def save_debug_snapshots(
    frame_index: int,
    model_input_frame,
    overlay_frame,
    model_input_dir: str | None,
    overlay_dir: str | None,
) -> None:
    if (
        not SETTINGS.video.save_model_input_debug
        or model_input_dir is None
        or overlay_dir is None
    ):
        return

    interval = max(1, int(SETTINGS.video.save_model_input_debug_every or 1))
    if frame_index % interval != 0:
        return

    filename = f"frame_{frame_index:04d}.jpg"
    cv2.imwrite(os.path.join(model_input_dir, filename), model_input_frame)
    cv2.imwrite(os.path.join(overlay_dir, filename), overlay_frame)
