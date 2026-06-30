from __future__ import annotations

import cv2
import numpy as np

from stair_monitor.common.types import CameraConfigDict
from stair_monitor.config.settings import DemoSettings
from stair_monitor.core.analyzer import BehaviorAnalyzer
from stair_monitor.output.rendering import (
    VietnameseTextDrawer,
    draw_demo_violation_alerts,
    draw_person_overlay,
    draw_scene_guides,
)
from stair_monitor.runtime.runtime_types import Frame, PersonRuntimeResult


class AlertOnlyDisplaySink:
    """Render the current demo frame while preserving alert-only behavior."""

    def __init__(
        self,
        enabled: bool,
        window_name: str,
        scale: float,
        analyzer: BehaviorAnalyzer,
        config: CameraConfigDict,
        demo_settings: DemoSettings,
    ) -> None:
        self.enabled = enabled
        self.window_name = window_name
        self.scale = scale
        self._analyzer = analyzer
        self._config = config
        self._demo_settings = demo_settings

    def render(
        self,
        frame: Frame,
        active_alert_until_frame: dict[str, int],
        person_results: tuple[PersonRuntimeResult, ...] = (),
    ) -> bool:
        with VietnameseTextDrawer(frame) as text_drawer:
            if self._demo_settings.draw_debug and (
                not self._demo_settings.demo_mode or self._demo_settings.focus_debug_only
            ):
                draw_scene_guides(frame, self._config, self._analyzer)

            if not self._demo_settings.alert_only_display_active:
                for person_result in person_results:
                    draw_person_overlay(
                        frame,
                        np.asarray(person_result.bbox, dtype=np.float32),
                        person_result.keypoints,
                        person_result.p_lane,
                        person_result.p_motion,
                        person_result.analysis,
                        features=person_result.features,
                        text_drawer=text_drawer,
                    )

            if self._demo_settings.demo_mode and not self._demo_settings.focus_debug_only:
                draw_demo_violation_alerts(
                    frame,
                    active_alert_until_frame,
                )

        if not self.enabled:
            return True

        display_scale = self.scale
        if display_scale > 0.0 and display_scale != 1.0:
            display_frame = cv2.resize(
                frame,
                (0, 0),
                fx=display_scale,
                fy=display_scale,
            )
        else:
            display_frame = frame

        cv2.imshow(self.window_name, display_frame)
        return (cv2.waitKey(1) & 0xFF) != ord("q")

    def close(self) -> None:
        if self.enabled:
            cv2.destroyAllWindows()
