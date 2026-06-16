"""Windows/demo stair monitoring package."""

from stair_monitor.config.settings import SETTINGS, load_camera_config
from stair_monitor.core.analyzer import BehaviorAnalyzer

__all__ = [
    "BehaviorAnalyzer",
    "SETTINGS",
    "load_camera_config",
]
