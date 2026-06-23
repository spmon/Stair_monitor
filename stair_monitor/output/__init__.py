"""Output helpers for stair_monitor."""

from stair_monitor.output.alert_event_logger import AlertEventLogger, ViolationEvent
from stair_monitor.output.logging_setup import setup_app_logging
from stair_monitor.output.rendering import (
    VietnameseTextDrawer,
    draw_demo_violation_alerts,
    draw_person_overlay,
    draw_scene_guides,
)

__all__ = [
    "AlertEventLogger",
    "ViolationEvent",
    "setup_app_logging",
    "VietnameseTextDrawer",
    "draw_demo_violation_alerts",
    "draw_person_overlay",
    "draw_scene_guides",
]
