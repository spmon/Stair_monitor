"""Vision helpers for stair_monitor."""

from stair_monitor.vision.geometry import extract_pose_features
from stair_monitor.vision.step_lines import (
    StepBand,
    StepIndexResult,
    StepLine,
    build_step_bands,
    get_step_index_for_point,
    get_step_index_reason_for_point,
    normalize_step_lines,
)

__all__ = [
    "extract_pose_features",
    "StepBand",
    "StepIndexResult",
    "StepLine",
    "build_step_bands",
    "get_step_index_for_point",
    "get_step_index_reason_for_point",
    "normalize_step_lines",
]
