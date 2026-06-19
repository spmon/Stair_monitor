from __future__ import annotations

import time
from typing import cast

import numpy as np

from stair_monitor.common.types import (
    AnalysisSubjectID,
    AnalysisResult,
    BBoxArray,
    KeypointsArray,
    Point,
    PoseFeatures,
)
from stair_monitor.config.settings import SETTINGS
from stair_monitor.core.person_context import PersonContext
from stair_monitor.output.result_builder import ResultBuilderMixin
from stair_monitor.rules.backward_rule import evaluate_backward
from stair_monitor.rules.carry_analysis import CarryAnalysisMixin, evaluate_carry
from stair_monitor.rules.direction_rule import apply_direction_history, update_direction
from stair_monitor.rules.handrail_rule import (
    HandrailAnalysisMixin,
    evaluate_handrail,
)
from stair_monitor.rules.inside_rule import evaluate_inside_stairs
from stair_monitor.rules.lane_rule import evaluate_lane_violation, get_camera_angle_profile
from stair_monitor.rules.standing_rule import evaluate_standing_still
from stair_monitor.state.behavior_history import BehaviorHistoryMixin
from stair_monitor.state.track_state import AnalyzerState, FootStepState
from stair_monitor.vision.geometry import extract_pose_features
from stair_monitor.vision.step_lines import (
    LOW_ANKLE_CONF_REASON,
    NEAR_STEP_BOUNDARY_REASON,
    NO_STEP_BANDS_REASON,
    OUTSIDE_ALL_BANDS_REASON,
    POINT_INVALID_REASON,
    POINT_NEAR_STEP_BOUNDARY_REASON,
    StepIndexResult,
    build_step_bands,
    get_step_index_for_point,
    normalize_step_lines,
)


class BehaviorAnalyzer(
    BehaviorHistoryMixin,
    CarryAnalysisMixin,
    HandrailAnalysisMixin,
    ResultBuilderMixin,
):
    """Trung tam dieu phoi cac rule cho stair_monitor Windows/demo."""

    def __init__(self, config):
        self.center_line = config.get("CENTER_LINE", [[0, 0], [0, 0]])

        self.left_line = np.array(
            config.get("HANDRAIL_LEFT_LINE", config.get("HANDRAIL_LEFT_POLY", [])),
            np.int32,
        )
        self.right_line = np.array(
            config.get("HANDRAIL_RIGHT_LINE", config.get("HANDRAIL_RIGHT_POLY", [])),
            np.int32,
        )

        step_bottom = config.get("STEP_BOTTOM", [[0, 0], [0, 0]])
        step_top = config.get("STEP_TOP", [[0, 0], [0, 0]])
        bottom_left = [
            step_bottom[0][0]-30,
            step_bottom[0][1],
        ]
        bottom_right = [
            step_bottom[1][0]+30,
            step_bottom[1][1],
        ]
        top_right = [
            step_top[1][0]+30,
            step_top[1][1],
        ]
        top_left = [
            step_top[0][0]-30,
            step_top[0][1],
        ]
        self.stairs_poly = np.array(
            [bottom_left, bottom_right, top_right, top_left],
            np.int32,
        )
        self.step_lines = normalize_step_lines(config.get("STEP_LINES"))
        self.step_bands = build_step_bands(self.step_lines)

        self.state = AnalyzerState()
        self.hip_motion_history = self.state.hip_motion_history
        self.shoulder_motion_history = self.state.shoulder_motion_history
        self.inside_last_state = self.state.inside_last_state
        self.lane_history = self.state.lane_history
        self.lane_last_state = self.state.lane_last_state
        self.lane_last_seen = self.state.lane_last_seen
        self.hold_status_history = self.state.hold_status_history
        self.left_handrail_hit_history = self.state.left_handrail_hit_history
        self.right_handrail_hit_history = self.state.right_handrail_hit_history
        self.last_valid_direction = self.state.last_valid_direction
        self.last_valid_direction_frame = self.state.last_valid_direction_frame
        self.hand_claim_state = self.state.hand_claim_state
        self.front_carry_history = self.state.front_carry_history
        self.front_carry_one_arm_history = self.state.front_carry_one_arm_history
        self.backward_history = self.state.backward_history
        self.standing_history = self.state.standing_history
        self.two_step_skip_history = self.state.two_step_skip_history
        self.left_foot_step_states = self.state.left_foot_step_states
        self.right_foot_step_states = self.state.right_foot_step_states
        self.standing_motion_history = self.state.standing_motion_history

    @property
    def frame_index(self) -> int:
        return self.state.frame_index

    @frame_index.setter
    def frame_index(self, value: int) -> None:
        self.state.frame_index = value

    def begin_frame(self):
        self.frame_index += 1

    @staticmethod
    def _record_perf(perf, key, start_time):
        if perf is None or start_time is None:
            return
        perf[key] = perf.get(key, 0.0) + (time.perf_counter() - start_time) * 1000.0

    def _update_two_step_skip_history(
        self,
        track_id: AnalysisSubjectID,
        candidate: bool,
    ) -> bool:
        history_len = max(1, int(SETTINGS.two_step_skip.confirm_frames or 1))
        history = self.two_step_skip_history.setdefault(track_id, [])
        history.append(candidate)
        if len(history) > history_len:
            del history[:-history_len]
        return len(history) >= history_len and all(history[-history_len:])

    @staticmethod
    def _build_foot_state_label(
        is_planted: bool,
        reason: str,
    ) -> str:
        if is_planted:
            return "PLANTED"
        if reason == "FOOT_MOVING_NO_CHECK":
            return "MOVING"
        if reason == LOW_ANKLE_CONF_REASON:
            return "LOW_CONF"
        if reason in ("ANKLE_MISSING", POINT_INVALID_REASON):
            return "MISSING"
        if reason in (
            "STEP_INDEX_UNKNOWN",
            POINT_NEAR_STEP_BOUNDARY_REASON,
            NO_STEP_BANDS_REASON,
            OUTSIDE_ALL_BANDS_REASON,
            NEAR_STEP_BOUNDARY_REASON,
        ):
            return "UNKNOWN"
        return "WAIT"

    @staticmethod
    def _select_candidate_step_from_history(
        step_history: list[int],
        previous_planted_step: int | None,
    ) -> tuple[int | None, int]:
        if not step_history:
            return None, 0

        step_counts: dict[int, int] = {}
        for step_index in step_history:
            step_counts[step_index] = step_counts.get(step_index, 0) + 1

        best_count = max(step_counts.values())
        best_candidates = [
            step_index
            for step_index, step_count in step_counts.items()
            if step_count == best_count
        ]
        if previous_planted_step in best_candidates:
            return previous_planted_step, best_count

        for step_index in reversed(step_history):
            if step_index in best_candidates:
                return step_index, best_count

        return step_history[-1], best_count

    def _get_foot_step_state_map(
        self,
        foot_side: str,
    ) -> dict[AnalysisSubjectID, FootStepState]:
        if foot_side == "LEFT":
            return self.left_foot_step_states
        return self.right_foot_step_states

    @staticmethod
    def _get_two_step_hold_last_valid_frames() -> int:
        return max(
            0,
            int(SETTINGS.two_step_skip.two_step_hold_last_valid_step_frames),
        )

    def _resolve_two_step_filter_direction(
        self,
        track_id: AnalysisSubjectID,
        current_direction: str,
        direction_source: str,
    ) -> tuple[str, str, bool]:
        if current_direction in ("UP", "DOWN"):
            return current_direction, "CURRENT_DIRECTION", True

        if current_direction == "IDLE" and direction_source != "HIP_SHOULDER_CONFLICT":
            last_valid_direction = self.last_valid_direction.get(track_id)
            last_valid_direction_frame = self.last_valid_direction_frame.get(track_id)
            if (
                last_valid_direction in ("UP", "DOWN")
                and last_valid_direction_frame is not None
                and self.frame_index - last_valid_direction_frame
                <= self._get_two_step_hold_last_valid_frames()
            ):
                return last_valid_direction, "RECENT_LAST_VALID_DIRECTION", True

        last_valid_direction = self.last_valid_direction.get(track_id)
        last_valid_direction_frame = self.last_valid_direction_frame.get(track_id)
        if (
            last_valid_direction in ("UP", "DOWN")
            and last_valid_direction_frame is not None
            and self.frame_index - last_valid_direction_frame
            <= self._get_two_step_hold_last_valid_frames()
        ):
            return last_valid_direction, "RECENT_LAST_VALID_DIRECTION", True
        return "UNKNOWN", "DIRECTION_NOT_RELIABLE", False

    @staticmethod
    def _is_reverse_step_for_direction(
        direction_key: str,
        raw_step: int,
        last_valid_step: int,
    ) -> bool:
        step_index_increases_when_up = bool(
            SETTINGS.two_step_skip.two_step_step_index_increases_when_up
        )
        if direction_key == "UP":
            if step_index_increases_when_up:
                return raw_step < last_valid_step
            return raw_step > last_valid_step
        if direction_key == "DOWN":
            if step_index_increases_when_up:
                return raw_step > last_valid_step
            return raw_step < last_valid_step
        return False

    @staticmethod
    def _get_reverse_step_reject_reason(direction_key: str) -> str:
        if direction_key == "UP":
            return "REJECT_STEP_DECREASE_DURING_UP"
        if direction_key == "DOWN":
            return "REJECT_STEP_INCREASE_DURING_DOWN"
        return "REJECT_STEP_DIRECTION_UNKNOWN"

    def _log_step_filter_reject(
        self,
        track_id: AnalysisSubjectID,
        foot_side: str,
        direction_key: str,
        raw_step: int,
        last_valid_step: int,
        filtered_step: int | None,
        reason: str,
    ) -> None:
        filtered_step_text = str(filtered_step) if filtered_step is not None else "NA"
        foot_label = "L" if foot_side == "LEFT" else "R"
        print(
            "STEP_FILTER_REJECT "
            f"track_id={track_id} "
            f"foot={foot_label} "
            f"direction={direction_key} "
            f"raw_step={raw_step} "
            f"last_valid_step={last_valid_step} "
            f"filtered_step={filtered_step_text} "
            f"reason={reason}"
        )

    def _filter_step_by_direction(
        self,
        track_id: AnalysisSubjectID,
        foot_side: str,
        raw_step: int | None,
        direction_key: str,
        direction_reliable: bool,
    ) -> dict[str, object]:
        foot_state = self._get_foot_step_state_map(foot_side).setdefault(
            track_id,
            FootStepState(),
        )
        hold_last_valid_frames = self._get_two_step_hold_last_valid_frames()
        allow_same_step = bool(SETTINGS.two_step_skip.two_step_allow_same_step)
        use_monotonic_filter = bool(
            SETTINGS.two_step_skip.two_step_use_monotonic_filter
        )

        last_valid_step = foot_state.last_valid_step_index
        last_valid_step_frame = foot_state.last_valid_step_frame
        has_recent_last_valid_step = (
            last_valid_step is not None
            and last_valid_step_frame is not None
            and self.frame_index - last_valid_step_frame <= hold_last_valid_frames
        )

        filtered_step = None
        filter_reason = "NOT_EVALUATED"
        reverse_reject = False
        update_last_valid_from_raw = False

        if raw_step is None:
            if has_recent_last_valid_step:
                filtered_step = last_valid_step
                filter_reason = "HOLD_LAST_VALID_STEP_RAW_STEP_NA"
            else:
                filter_reason = "RAW_STEP_NA"
        elif last_valid_step is None:
            filtered_step = raw_step
            filter_reason = "ACCEPT_FIRST_STEP"
            update_last_valid_from_raw = True
        elif not direction_reliable:
            filtered_step = raw_step
            filter_reason = "ACCEPT_DIRECTION_UNKNOWN"
        elif not use_monotonic_filter:
            filtered_step = raw_step
            filter_reason = "ACCEPT_MONOTONIC_FILTER_DISABLED"
            update_last_valid_from_raw = True
        elif raw_step == last_valid_step:
            if allow_same_step:
                filtered_step = raw_step
                filter_reason = "ACCEPT_SAME_STEP"
                update_last_valid_from_raw = True
            elif has_recent_last_valid_step:
                filtered_step = last_valid_step
                filter_reason = "HOLD_LAST_VALID_STEP_SAME_STEP_BLOCKED"
            else:
                filter_reason = "SAME_STEP_BLOCKED"
        elif self._is_reverse_step_for_direction(
            direction_key,
            raw_step,
            last_valid_step,
        ):
            reverse_reject = True
            filter_reason = self._get_reverse_step_reject_reason(direction_key)
            if has_recent_last_valid_step:
                filtered_step = last_valid_step
            self._log_step_filter_reject(
                track_id,
                foot_side,
                direction_key,
                raw_step,
                last_valid_step,
                filtered_step,
                filter_reason,
            )
        else:
            filtered_step = raw_step
            if direction_key == "UP":
                filter_reason = "ACCEPT_UP_NON_DECREASING"
            elif direction_key == "DOWN":
                filter_reason = "ACCEPT_DOWN_NON_INCREASING"
            else:
                filter_reason = "ACCEPT_DIRECTION_UNKNOWN"
            update_last_valid_from_raw = True

        if update_last_valid_from_raw and raw_step is not None:
            foot_state.last_valid_step_index = raw_step
            foot_state.last_valid_step_frame = self.frame_index

        foot_state.raw_step_index = raw_step
        foot_state.filtered_step_index = filtered_step
        foot_state.filter_reason = filter_reason

        return {
            "raw_step": raw_step,
            "filtered_step": filtered_step,
            "last_valid_step": foot_state.last_valid_step_index,
            "filter_reason": filter_reason,
            "reverse_reject": reverse_reject,
        }

    def _update_foot_step_state(
        self,
        track_id: AnalysisSubjectID,
        foot_side: str,
        ankle_point: Point | None,
        ankle_conf: float,
        current_step_index: int | None,
        current_step_reason: str,
    ) -> dict[str, object]:
        state_map = self._get_foot_step_state_map(foot_side)
        foot_state = state_map.setdefault(track_id, FootStepState())

        history_window = max(1, int(SETTINGS.two_step_skip.foot_step_history_window))
        confirm_frames = max(
            1,
            int(SETTINGS.two_step_skip.foot_planted_confirm_frames),
        )
        speed_threshold = float(
            SETTINGS.two_step_skip.foot_planted_max_speed_px_per_frame
        )
        use_speed_check = bool(SETTINGS.two_step_skip.foot_planted_use_speed_check)

        previous_is_planted = foot_state.is_planted
        previous_planted_step = foot_state.last_planted_step_index
        ankle_speed = None

        if (
            ankle_point is not None
            and foot_state.last_ankle_point is not None
            and foot_state.last_seen_frame is not None
            and self.frame_index - foot_state.last_seen_frame <= 1
        ):
            ankle_speed = float(
                np.hypot(
                    float(ankle_point[0] - foot_state.last_ankle_point[0]),
                    float(ankle_point[1] - foot_state.last_ankle_point[1]),
                )
            )

        foot_state.ankle_speed_px = ankle_speed

        if (
            foot_state.last_seen_frame is not None
            and self.frame_index - foot_state.last_seen_frame > history_window
        ):
            foot_state.step_history.clear()
            foot_state.candidate_step_index = None
            foot_state.candidate_count = 0

        if ankle_point is not None:
            foot_state.last_ankle_point = ankle_point
            foot_state.last_seen_frame = self.frame_index

        ankle_available = ankle_point is not None and ankle_conf >= float(
            SETTINGS.two_step_skip.ankle_conf_threshold
        )
        landed_step_index = None

        if not ankle_available:
            foot_state.is_planted = False
            if ankle_point is None:
                foot_state.reason = POINT_INVALID_REASON
            else:
                foot_state.reason = LOW_ANKLE_CONF_REASON
            foot_state.candidate_step_index = None
            foot_state.candidate_count = 0
            foot_state.step_history.clear()
        elif current_step_index is None:
            foot_state.is_planted = False
            foot_state.reason = current_step_reason or "STEP_INDEX_UNKNOWN"
            foot_state.candidate_step_index = None
            foot_state.candidate_count = 0
            foot_state.step_history.clear()
        elif use_speed_check and ankle_speed is not None and ankle_speed > speed_threshold:
            foot_state.is_planted = False
            foot_state.reason = "FOOT_MOVING_NO_CHECK"
            foot_state.candidate_step_index = current_step_index
            foot_state.candidate_count = 1
            foot_state.step_history.clear()
        else:
            foot_state.step_history.append(current_step_index)
            if len(foot_state.step_history) > history_window:
                del foot_state.step_history[:-history_window]

            candidate_step_index, candidate_count = (
                self._select_candidate_step_from_history(
                    foot_state.step_history,
                    foot_state.last_planted_step_index,
                )
            )
            foot_state.candidate_step_index = candidate_step_index
            foot_state.candidate_count = candidate_count

            if use_speed_check and ankle_speed is None and previous_planted_step is None:
                foot_state.is_planted = False
                foot_state.reason = "WAIT_FOOT_PLANTED"
            elif foot_state.last_planted_step_index is None:
                if (
                    candidate_step_index is not None
                    and candidate_count >= confirm_frames
                ):
                    foot_state.last_planted_step_index = candidate_step_index
                    foot_state.is_planted = True
                    foot_state.reason = "FOOT_LANDED"
                    landed_step_index = candidate_step_index
                else:
                    foot_state.is_planted = False
                    foot_state.reason = "WAIT_FOOT_PLANTED"
            elif current_step_index == foot_state.last_planted_step_index:
                foot_state.is_planted = True
                foot_state.reason = "PLANTED_STABLE"
            else:
                switch_confirm_frames = max(
                    confirm_frames,
                    len(foot_state.step_history) // 2 + 1,
                )
                if (
                    candidate_step_index is not None
                    and candidate_step_index != foot_state.last_planted_step_index
                    and candidate_count >= switch_confirm_frames
                ):
                    foot_state.last_planted_step_index = candidate_step_index
                    foot_state.is_planted = True
                    foot_state.reason = "FOOT_LANDED"
                    landed_step_index = candidate_step_index
                else:
                    foot_state.is_planted = False
                    foot_state.reason = "WAIT_FOOT_PLANTED"

        landed = (
            landed_step_index is not None
            and (
                not previous_is_planted
                or previous_planted_step != foot_state.last_planted_step_index
            )
        )
        state_label = self._build_foot_state_label(
            foot_state.is_planted,
            foot_state.reason,
        )

        return {
            "current_step_index": current_step_index,
            "planted_step_index": foot_state.last_planted_step_index,
            "is_planted": foot_state.is_planted,
            "ankle_speed_px": foot_state.ankle_speed_px,
            "reason": foot_state.reason,
            "state_label": state_label,
            "landed": landed,
            "landed_step_index": landed_step_index,
            "candidate_step_index": foot_state.candidate_step_index,
            "candidate_count": foot_state.candidate_count,
        }

    @staticmethod
    def _override_step_result_for_low_conf(
        step_index_result: StepIndexResult,
    ) -> StepIndexResult:
        return StepIndexResult(
            step_index=None,
            reason=LOW_ANKLE_CONF_REASON,
            nearest_band_id=step_index_result.nearest_band_id,
            nearest_signed_distance=step_index_result.nearest_signed_distance,
            is_inside=step_index_result.is_inside,
            is_near_boundary=step_index_result.is_near_boundary,
        )

    @staticmethod
    def _offset_ankle_step_point(
        ankle_point: Point | None,
        offset_x: int,
        offset_y: int,
    ) -> Point | None:
        if ankle_point is None:
            return None
        return (
            int(ankle_point[0] + offset_x),
            int(ankle_point[1] + offset_y),
        )

    @staticmethod
    def _get_step_offset_pair(direction_key: str) -> tuple[int, int]:
        if direction_key == "UP":
            return (
                int(SETTINGS.two_step_skip.up_ankle_step_offset_x_px),
                int(SETTINGS.two_step_skip.up_ankle_step_offset_y_px),
            )
        if direction_key == "DOWN":
            return (
                int(SETTINGS.two_step_skip.down_ankle_step_offset_x_px),
                int(SETTINGS.two_step_skip.down_ankle_step_offset_y_px),
            )
        return (
            int(SETTINGS.two_step_skip.unknown_ankle_step_offset_x_px),
            int(SETTINGS.two_step_skip.unknown_ankle_step_offset_y_px),
        )

    def _resolve_ankle_step_offset(
        self,
        track_id: AnalysisSubjectID,
        current_direction: str,
        direction_source: str,
    ) -> tuple[int, int, str]:
        if not SETTINGS.two_step_skip.use_directional_ankle_offset:
            return 0, 0, "DISABLED"

        if (
            current_direction in ("UNKNOWN", "ANALYZING", "IDLE")
            or direction_source == "HIP_SHOULDER_CONFLICT"
        ):
            filter_direction, filter_direction_source, direction_reliable = (
                self._resolve_two_step_filter_direction(
                    track_id,
                    current_direction,
                    direction_source,
                )
            )
            if direction_reliable and filter_direction == "UP":
                offset_x, offset_y = self._get_step_offset_pair("UP")
                return offset_x, offset_y, f"{filter_direction_source}_UP"
            if direction_reliable and filter_direction == "DOWN":
                offset_x, offset_y = self._get_step_offset_pair("DOWN")
                return offset_x, offset_y, f"{filter_direction_source}_DOWN"
            offset_x, offset_y = self._get_step_offset_pair("UNKNOWN")
            return offset_x, offset_y, "UNKNOWN"

        if current_direction == "UP":
            offset_x, offset_y = self._get_step_offset_pair("UP")
            return offset_x, offset_y, "UP"
        if current_direction == "DOWN":
            offset_x, offset_y = self._get_step_offset_pair("DOWN")
            return offset_x, offset_y, "DOWN"
        if current_direction == "IDLE":
            offset_x, offset_y = self._get_step_offset_pair("UNKNOWN")
            return offset_x, offset_y, "IDLE"

        offset_x, offset_y = self._get_step_offset_pair("UNKNOWN")
        return offset_x, offset_y, "UNKNOWN"

    def _evaluate_two_step_skip(
        self,
        track_id: AnalysisSubjectID,
        features: PoseFeatures,
        inside_stairs: bool,
        current_direction: str,
        direction_source: str,
    ) -> dict[str, object]:
        left_ankle_point = features.get("left_ankle")
        right_ankle_point = features.get("right_ankle")
        left_ankle_conf = float(features.get("left_ankle_conf", 0.0) or 0.0)
        right_ankle_conf = float(features.get("right_ankle_conf", 0.0) or 0.0)
        ankle_step_offset_x, ankle_step_offset_y, ankle_step_offset_direction = (
            self._resolve_ankle_step_offset(
                track_id,
                current_direction,
                direction_source,
            )
        )
        left_ankle_step_point = self._offset_ankle_step_point(
            left_ankle_point,
            ankle_step_offset_x,
            ankle_step_offset_y,
        )
        right_ankle_step_point = self._offset_ankle_step_point(
            right_ankle_point,
            ankle_step_offset_x,
            ankle_step_offset_y,
        )
        filter_direction, _filter_direction_source, direction_reliable = (
            self._resolve_two_step_filter_direction(
                track_id,
                current_direction,
                direction_source,
            )
        )

        two_step_skip_state: dict[str, object] = {
            "left_ankle_point": left_ankle_point,
            "right_ankle_point": right_ankle_point,
            "left_ankle_raw_point": left_ankle_point,
            "right_ankle_raw_point": right_ankle_point,
            "left_ankle_step_point": left_ankle_step_point,
            "right_ankle_step_point": right_ankle_step_point,
            "left_ankle_conf": left_ankle_conf,
            "right_ankle_conf": right_ankle_conf,
            "left_current_ankle_step": None,
            "right_current_ankle_step": None,
            "left_raw_ankle_step": None,
            "right_raw_ankle_step": None,
            "left_raw_step": None,
            "right_raw_step": None,
            "left_filtered_step": None,
            "right_filtered_step": None,
            "left_last_valid_step": None,
            "right_last_valid_step": None,
            "left_step_filter_reason": "NOT_EVALUATED",
            "right_step_filter_reason": "NOT_EVALUATED",
            "left_adjusted_step": None,
            "right_adjusted_step": None,
            "ankle_step_offset_x": ankle_step_offset_x,
            "ankle_step_offset_y": ankle_step_offset_y,
            "ankle_step_offset_direction": ankle_step_offset_direction,
            "left_step_reason": "NOT_EVALUATED",
            "right_step_reason": "NOT_EVALUATED",
            "left_step_nearest_band_id": None,
            "right_step_nearest_band_id": None,
            "left_step_nearest_distance": None,
            "right_step_nearest_distance": None,
            "left_step_is_inside": False,
            "right_step_is_inside": False,
            "left_step_is_near_boundary": False,
            "right_step_is_near_boundary": False,
            "left_step_index": None,
            "right_step_index": None,
            "foot_gap": None,
            "step_gap": None,
            "left_planted_step": None,
            "right_planted_step": None,
            "planted_step_gap": None,
            "left_is_planted": False,
            "right_is_planted": False,
            "left_ankle_speed": None,
            "right_ankle_speed": None,
            "left_foot_step_reason": "NOT_EVALUATED",
            "right_foot_step_reason": "NOT_EVALUATED",
            "left_foot_state_label": "WAIT",
            "right_foot_state_label": "WAIT",
            "left_foot_landed": False,
            "right_foot_landed": False,
            "two_step_skip_check_available": False,
            "two_step_skip_reason": "NOT_EVALUATED",
            "two_step_skip_confirmed": False,
        }

        candidate_two_step_skip = False
        reason = "NOT_EVALUATED"
        ankle_conf_threshold = float(SETTINGS.two_step_skip.ankle_conf_threshold)
        left_raw_step_result = get_step_index_for_point(left_ankle_point, self.step_bands)
        right_raw_step_result = get_step_index_for_point(
            right_ankle_point,
            self.step_bands,
        )
        left_step_result = get_step_index_for_point(
            left_ankle_step_point,
            self.step_bands,
        )
        right_step_result = get_step_index_for_point(
            right_ankle_step_point,
            self.step_bands,
        )
        if left_ankle_point is not None and left_ankle_conf < ankle_conf_threshold:
            left_step_result = self._override_step_result_for_low_conf(left_step_result)
        if right_ankle_point is not None and right_ankle_conf < ankle_conf_threshold:
            right_step_result = self._override_step_result_for_low_conf(
                right_step_result
            )

        two_step_skip_state.update(
            {
                "left_current_ankle_step": left_raw_step_result.step_index,
                "right_current_ankle_step": right_raw_step_result.step_index,
                "left_raw_ankle_step": left_raw_step_result.step_index,
                "right_raw_ankle_step": right_raw_step_result.step_index,
                "left_raw_step": left_step_result.step_index,
                "right_raw_step": right_step_result.step_index,
                "left_adjusted_step": left_step_result.step_index,
                "right_adjusted_step": right_step_result.step_index,
                "left_step_reason": left_step_result.reason,
                "right_step_reason": right_step_result.reason,
                "left_step_nearest_band_id": left_step_result.nearest_band_id,
                "right_step_nearest_band_id": right_step_result.nearest_band_id,
                "left_step_nearest_distance": left_step_result.nearest_signed_distance,
                "right_step_nearest_distance": right_step_result.nearest_signed_distance,
                "left_step_is_inside": left_step_result.is_inside,
                "right_step_is_inside": right_step_result.is_inside,
                "left_step_is_near_boundary": left_step_result.is_near_boundary,
                "right_step_is_near_boundary": right_step_result.is_near_boundary,
                "left_step_index": left_step_result.step_index,
                "right_step_index": right_step_result.step_index,
            }
        )

        if not SETTINGS.two_step_skip.enabled:
            reason = "TWO_STEP_SKIP_DISABLED"
        elif not inside_stairs:
            reason = "OUTSIDE_STAIRS"
        else:
            left_filter_info = self._filter_step_by_direction(
                track_id,
                "LEFT",
                left_step_result.step_index,
                filter_direction,
                direction_reliable,
            )
            right_filter_info = self._filter_step_by_direction(
                track_id,
                "RIGHT",
                right_step_result.step_index,
                filter_direction,
                direction_reliable,
            )
            left_filtered_step = left_filter_info["filtered_step"]
            right_filtered_step = right_filter_info["filtered_step"]

            left_foot_state = self._update_foot_step_state(
                track_id,
                "LEFT",
                left_ankle_point,
                left_ankle_conf,
                left_filtered_step,
                str(left_filter_info["filter_reason"]),
            )
            right_foot_state = self._update_foot_step_state(
                track_id,
                "RIGHT",
                right_ankle_point,
                right_ankle_conf,
                right_filtered_step,
                str(right_filter_info["filter_reason"]),
            )

            filtered_step_gap = None
            if (
                direction_reliable
                and isinstance(left_filtered_step, int)
                and isinstance(right_filtered_step, int)
            ):
                filtered_step_gap = abs(left_filtered_step - right_filtered_step)

            planted_step_gap = None
            left_planted_step = left_foot_state["planted_step_index"]
            right_planted_step = right_foot_state["planted_step_index"]
            if isinstance(left_planted_step, int) and isinstance(right_planted_step, int):
                planted_step_gap = abs(left_planted_step - right_planted_step)

            two_step_skip_state.update(
                {
                    "left_filtered_step": left_filtered_step,
                    "right_filtered_step": right_filtered_step,
                    "left_last_valid_step": left_filter_info["last_valid_step"],
                    "right_last_valid_step": right_filter_info["last_valid_step"],
                    "left_step_filter_reason": str(left_filter_info["filter_reason"]),
                    "right_step_filter_reason": str(
                        right_filter_info["filter_reason"]
                    ),
                    "foot_gap": filtered_step_gap,
                    "step_gap": filtered_step_gap,
                    "left_planted_step": left_planted_step,
                    "right_planted_step": right_planted_step,
                    "planted_step_gap": planted_step_gap,
                    "left_is_planted": bool(left_foot_state["is_planted"]),
                    "right_is_planted": bool(right_foot_state["is_planted"]),
                    "left_ankle_speed": left_foot_state["ankle_speed_px"],
                    "right_ankle_speed": right_foot_state["ankle_speed_px"],
                    "left_foot_step_reason": str(left_foot_state["reason"]),
                    "right_foot_step_reason": str(right_foot_state["reason"]),
                    "left_foot_state_label": str(left_foot_state["state_label"]),
                    "right_foot_state_label": str(right_foot_state["state_label"]),
                    "left_foot_landed": bool(left_foot_state["landed"]),
                    "right_foot_landed": bool(right_foot_state["landed"]),
                }
            )

            reverse_reject_in_frame = bool(left_filter_info["reverse_reject"]) or bool(
                right_filter_info["reverse_reject"]
            )
            if not direction_reliable:
                reason = "DIRECTION_NOT_RELIABLE_FOR_TWO_STEP"
            elif filtered_step_gap is None:
                if left_filtered_step is None and right_filtered_step is None:
                    reason = "FILTERED_STEP_NA"
                elif left_filtered_step is None:
                    reason = "LEFT_FILTERED_STEP_NA"
                else:
                    reason = "RIGHT_FILTERED_STEP_NA"
            else:
                two_step_skip_state["two_step_skip_check_available"] = True
                if filtered_step_gap >= int(SETTINGS.two_step_skip.min_step_gap):
                    if reverse_reject_in_frame:
                        if int(SETTINGS.two_step_skip.confirm_frames) > 1:
                            candidate_two_step_skip = True
                            reason = "WAIT_CONFIRM_AFTER_REVERSE_REJECT"
                        else:
                            reason = "REVERSE_STEP_REJECT_SUPPRESSED"
                    else:
                        candidate_two_step_skip = True
                        reason = "WAIT_CONFIRM_FRAMES"
                else:
                    reason = "FILTERED_GAP_OK"

        two_step_skip_confirmed = (
            candidate_two_step_skip
            and self._update_two_step_skip_history(track_id, candidate_two_step_skip)
        )
        if not candidate_two_step_skip:
            self._update_two_step_skip_history(track_id, False)
        if two_step_skip_confirmed:
            reason = "TWO_STEP_SKIP_CONFIRMED"

        two_step_skip_state["two_step_skip_reason"] = reason
        two_step_skip_state["two_step_skip_confirmed"] = two_step_skip_confirmed
        return two_step_skip_state

    def analyze(
        self,
        track_id: AnalysisSubjectID,
        p_lane: Point | None,
        p_motion: Point | None,
        keypoints: KeypointsArray,
        box: BBoxArray | None = None,
        features: PoseFeatures | None = None,
    ) -> AnalysisResult:
        perf = {} if SETTINGS.performance.enable_perf_log else None
        analyze_start = time.perf_counter() if perf is not None else None
        if self.frame_index < 0:
            self.frame_index = 0

        features = cast(
            PoseFeatures,
            features or extract_pose_features(keypoints, box),
        )
        if p_lane is None:
            p_lane = features.get("feet_point")
        if p_motion is None:
            p_motion = features.get("motion_point")

        person = PersonContext(
            track_id=track_id,
            keypoints=keypoints,
            box=box,
            features=features,
            p_lane=p_lane,
            p_motion=p_motion,
        )

        use_current_camera_angle = SETTINGS.camera.use_current_camera_angle
        camera_angle_profile = get_camera_angle_profile()
        body_facing = features.get("body_facing", "UNKNOWN")
        body_facing_confidence = float(
            features.get("body_facing_confidence", 0.0) or 0.0
        )
        body_facing_evidence_count = int(
            features.get("body_facing_evidence_count", 0) or 0
        )
        body_facing_front_votes = int(
            features.get("body_facing_front_votes", 0) or 0
        )
        body_facing_back_votes = int(
            features.get("body_facing_back_votes", 0) or 0
        )
        body_facing_reason = features.get("body_facing_reason", "UNKNOWN")
        hip_pair_valid = bool(features.get("hip_pair_valid", False))
        shoulder_pair_valid = bool(features.get("shoulder_pair_valid", False))
        ear_pair_valid = bool(features.get("ear_pair_valid", False))
        head_valid = bool(features.get("head_valid", False))
        arm_side_order = features.get("arm_side_order", "UNKNOWN")

        analysis_context: dict[str, object] = {
            "track_id": track_id,
            "p_lane": person.p_lane,
            "p_motion": person.p_motion,
            "use_current_camera_angle": use_current_camera_angle,
            "camera_angle_profile": camera_angle_profile,
            "lane_mapping_source": camera_angle_profile,
            "handrail_mapping_source": camera_angle_profile,
            "backward_mapping_source": camera_angle_profile,
            "direction": "ANALYZING",
            "final_direction": "ANALYZING",
            "hip_direction": "UNKNOWN",
            "shoulder_direction": "UNKNOWN",
            "direction_source": "NO_VALID_MONITOR_DIRECTION",
            "direction_reason": "ANALYZING",
            "dy": None,
            "direction_dy": None,
            "v": None,
            "lane_direction": "ANALYZING",
            "monitor_point_hip": features.get("monitor_point_hip"),
            "monitor_point_hip_source": features.get(
                "monitor_point_hip_source",
                "NO_HIP_CENTER",
            ),
            "monitor_point_shoulder": features.get("monitor_point_shoulder"),
            "monitor_point_shoulder_source": features.get(
                "monitor_point_shoulder_source",
                "NO_SHOULDER_CENTER",
            ),
            "feet_point_source": features.get("feet_point_source", "FEET_UNAVAILABLE"),
            "inside_feet_point": features.get("inside_feet_point"),
            "inside_feet_point_source": features.get(
                "inside_feet_point_source",
                "FEET_UNAVAILABLE",
            ),
            "feet_available": bool(features.get("feet_available", False)),
            "feet_unavailable_reason": str(
                features.get("feet_unavailable_reason", "NO_SHOULDER_NO_HIP")
            ),
            "sh_hip_visible_shoulder_count": int(
                features.get("sh_hip_visible_shoulder_count", 0) or 0
            ),
            "sh_hip_visible_hip_count": int(
                features.get("sh_hip_visible_hip_count", 0) or 0
            ),
            "sh_hip_selected_pair": str(features.get("sh_hip_selected_pair", "NONE")),
            "sh_hip_virtual_feet_point": features.get("sh_hip_virtual_feet_point"),
            "sh_hip_virtual_feet_source": str(
                features.get("sh_hip_virtual_feet_source", "FEET_UNAVAILABLE")
            ),
            "ankle_valid_count": int(features.get("ankle_valid_count", 0) or 0),
            "feet_reliable": bool(features.get("feet_reliable", False)),
            "bbox_height": features.get("bbox_height"),
            "left_ankle_point": features.get("left_ankle"),
            "right_ankle_point": features.get("right_ankle"),
            "left_ankle_raw_point": features.get("left_ankle"),
            "right_ankle_raw_point": features.get("right_ankle"),
            "left_ankle_step_point": features.get("left_ankle"),
            "right_ankle_step_point": features.get("right_ankle"),
            "left_ankle_conf": float(features.get("left_ankle_conf", 0.0) or 0.0),
            "right_ankle_conf": float(
                features.get("right_ankle_conf", 0.0) or 0.0
            ),
            "left_current_ankle_step": None,
            "right_current_ankle_step": None,
            "left_raw_ankle_step": None,
            "right_raw_ankle_step": None,
            "left_adjusted_step": None,
            "right_adjusted_step": None,
            "ankle_step_offset_x": 0,
            "ankle_step_offset_y": 0,
            "ankle_step_offset_direction": "UNKNOWN",
            "left_step_reason": "NOT_EVALUATED",
            "right_step_reason": "NOT_EVALUATED",
            "left_step_nearest_band_id": None,
            "right_step_nearest_band_id": None,
            "left_step_nearest_distance": None,
            "right_step_nearest_distance": None,
            "left_step_is_inside": False,
            "right_step_is_inside": False,
            "left_step_is_near_boundary": False,
            "right_step_is_near_boundary": False,
            "left_step_index": None,
            "right_step_index": None,
            "foot_gap": None,
            "step_gap": None,
            "left_planted_step": None,
            "right_planted_step": None,
            "planted_step_gap": None,
            "left_is_planted": False,
            "right_is_planted": False,
            "left_ankle_speed": None,
            "right_ankle_speed": None,
            "left_foot_step_reason": "NOT_EVALUATED",
            "right_foot_step_reason": "NOT_EVALUATED",
            "left_foot_state_label": "WAIT",
            "right_foot_state_label": "WAIT",
            "left_foot_landed": False,
            "right_foot_landed": False,
            "two_step_skip_check_available": False,
            "two_step_skip_reason": "NOT_EVALUATED",
            "two_step_skip_confirmed": False,
            "body_facing": body_facing,
            "body_facing_confidence": body_facing_confidence,
            "body_facing_evidence_count": body_facing_evidence_count,
            "body_facing_front_votes": body_facing_front_votes,
            "body_facing_back_votes": body_facing_back_votes,
            "body_facing_reason": body_facing_reason,
            "hip_pair_valid": hip_pair_valid,
            "shoulder_pair_valid": shoulder_pair_valid,
            "ear_pair_valid": ear_pair_valid,
            "head_valid": head_valid,
            "arm_side_order": arm_side_order,
            "left_wrist_valid": False,
            "right_wrist_valid": False,
            "left_wrist_hit": False,
            "right_wrist_hit": False,
            "left_wrist_hit_count": 0,
            "right_wrist_hit_count": 0,
            "left_wrist_miss_count": 0,
            "right_wrist_miss_count": 0,
            "left_wrist_confirm_required": self._get_left_handrail_confirm_required(),
            "right_wrist_confirm_required": self._get_right_handrail_confirm_required(),
            "left_wrist_distance_to_left_rail": None,
            "left_wrist_distance_to_right_rail": None,
            "right_wrist_distance_to_left_rail": None,
            "right_wrist_distance_to_right_rail": None,
            "left_wrist_nearest_distance": None,
            "right_wrist_nearest_distance": None,
            "left_wrist_nearest_rail": "NONE",
            "right_wrist_nearest_rail": "NONE",
            "left_wrist_nearest_point": None,
            "right_wrist_nearest_point": None,
            "left_holding": False,
            "right_holding": False,
            "handrail_status": "NOT_EVALUATED",
            "handrail_reason": "NOT_EVALUATED",
            "handrail_debug_reason": "NOT_EVALUATED",
            "warnings": [],
        }

        def build_result(
            status: str,
            color,
            display_status: str = "",
            carry_info_override=None,
            **overrides,
        ) -> AnalysisResult:
            self._record_perf(perf, "analyze", analyze_start)
            return self._build_result_from_context(
                analysis_context,
                status=status,
                display_status=display_status,
                color=color,
                perf=perf,
                carry_info=carry_info_override,
                **overrides,
            )

        def update_hold_context(hold_direction, skip_handrail=False):
            hold_start = time.perf_counter() if perf is not None else None
            hold_state, carry_pose, hand_claim_state = evaluate_handrail(
                self,
                track_id,
                hold_direction,
                person.keypoints,
                person.features,
            )
            self._record_perf(perf, "hold", hold_start)

            analysis_context.update(
                {
                    "left_hand_claim": hand_claim_state["left"]["claim"] or "NONE",
                    "right_hand_claim": hand_claim_state["right"]["claim"] or "NONE",
                    "left_hold_claim_hits": hand_claim_state["left"]["hold_hits"],
                    "right_hold_claim_hits": hand_claim_state["right"]["hold_hits"],
                    "left_carry_claim_hits": hand_claim_state["left"]["carry_hits"],
                    "right_carry_claim_hits": hand_claim_state["right"]["carry_hits"],
                    "left_hold_raw_before_claim": hold_state[
                        "left_hold_raw_before_claim"
                    ],
                    "right_hold_raw_before_claim": hold_state[
                        "right_hold_raw_before_claim"
                    ],
                    "left_hold_raw_after_claim": hold_state["left_hold_raw_after_claim"],
                    "right_hold_raw_after_claim": hold_state[
                        "right_hold_raw_after_claim"
                    ],
                    "left_carry_raw_before_claim": carry_pose[
                        "left_carry_raw_before_claim"
                    ],
                    "right_carry_raw_before_claim": carry_pose[
                        "right_carry_raw_before_claim"
                    ],
                    "left_carry_raw_after_claim": carry_pose[
                        "left_carry_raw_after_claim"
                    ],
                    "right_carry_raw_after_claim": carry_pose[
                        "right_carry_raw_after_claim"
                    ],
                }
            )

            (
                holding_raw,
                hold_raw_status,
                holding_correct_raw,
                holding_wrong_raw,
                left_wrist_valid,
                right_wrist_valid,
                _left_holding_raw_effective,
                _right_holding_raw_effective,
                _handrail_status_raw,
                _handrail_reason_raw,
            ) = self._copy_hold_state_fields(hold_state)
            handrail_hit_debug = self._update_handrail_hit_history(
                track_id,
                bool(hold_state["left_wrist_hit"]),
                bool(hold_state["right_wrist_hit"]),
            )

            analysis_context.update(
                {
                    "holding_raw": holding_raw,
                    "hold_raw_status": hold_raw_status,
                    "holding_correct_raw": holding_correct_raw,
                    "holding_wrong_raw": holding_wrong_raw,
                    "left_wrist_valid": left_wrist_valid,
                    "right_wrist_valid": right_wrist_valid,
                    "left_wrist_hit": hold_state["left_wrist_hit"],
                    "right_wrist_hit": hold_state["right_wrist_hit"],
                    "left_wrist_hit_count": handrail_hit_debug[
                        "left_wrist_hit_count"
                    ],
                    "right_wrist_hit_count": handrail_hit_debug[
                        "right_wrist_hit_count"
                    ],
                    "left_wrist_miss_count": handrail_hit_debug[
                        "left_wrist_miss_count"
                    ],
                    "right_wrist_miss_count": handrail_hit_debug[
                        "right_wrist_miss_count"
                    ],
                    "left_wrist_confirm_required": handrail_hit_debug[
                        "left_wrist_confirm_required"
                    ],
                    "right_wrist_confirm_required": handrail_hit_debug[
                        "right_wrist_confirm_required"
                    ],
                    "left_wrist_distance_to_left_rail": hold_state[
                        "left_wrist_distance_to_left_rail"
                    ],
                    "left_wrist_distance_to_right_rail": hold_state[
                        "left_wrist_distance_to_right_rail"
                    ],
                    "right_wrist_distance_to_left_rail": hold_state[
                        "right_wrist_distance_to_left_rail"
                    ],
                    "right_wrist_distance_to_right_rail": hold_state[
                        "right_wrist_distance_to_right_rail"
                    ],
                    "left_wrist_nearest_distance": hold_state[
                        "left_wrist_nearest_distance"
                    ],
                    "right_wrist_nearest_distance": hold_state[
                        "right_wrist_nearest_distance"
                    ],
                    "left_wrist_nearest_rail": hold_state["left_wrist_nearest_rail"],
                    "right_wrist_nearest_rail": hold_state[
                        "right_wrist_nearest_rail"
                    ],
                    "left_wrist_nearest_point": hold_state[
                        "left_wrist_nearest_point"
                    ],
                    "right_wrist_nearest_point": hold_state[
                        "right_wrist_nearest_point"
                    ],
                }
            )

            if skip_handrail:
                (
                    hold_correct_hits,
                    hold_wrong_side_hits,
                    hold_none_hits,
                    hold_unknown_hits,
                    hold_not_hold_evidence_hits,
                    hold_final_status,
                    holding,
                ) = self._get_hold_status_history_state(track_id)
            else:
                (
                    hold_correct_hits,
                    hold_wrong_side_hits,
                    hold_none_hits,
                    hold_unknown_hits,
                    hold_not_hold_evidence_hits,
                    hold_final_status,
                    holding,
                ) = self._update_hold_status_history(track_id, hold_raw_status)

            left_holding_confirmed = False
            right_holding_confirmed = False
            if not skip_handrail:
                left_holding_confirmed = hold_final_status == "WRONG_SIDE"
                right_holding_confirmed = hold_final_status == "CORRECT"

            if skip_handrail:
                handrail_decision = {
                    "handrail_status": "SKIP_BACKWARD",
                    "handrail_reason": "BACKWARD_SKIP_HANDRAIL",
                    "handrail_debug_reason": "BACKWARD_SKIP_HANDRAIL",
                }
            else:
                handrail_decision = self._build_confirmed_handrail_decision(
                    left_wrist_valid=left_wrist_valid,
                    right_wrist_valid=right_wrist_valid,
                    left_hit=bool(hold_state["left_wrist_hit"]),
                    right_hit=bool(hold_state["right_wrist_hit"]),
                    left_hit_count=int(handrail_hit_debug["left_wrist_hit_count"]),
                    right_hit_count=int(handrail_hit_debug["right_wrist_hit_count"]),
                    left_confirm_required=int(
                        handrail_hit_debug["left_wrist_confirm_required"]
                    ),
                    right_confirm_required=int(
                        handrail_hit_debug["right_wrist_confirm_required"]
                    ),
                    left_holding=left_holding_confirmed,
                    right_holding=right_holding_confirmed,
                    left_hand_claim=str(analysis_context["left_hand_claim"]),
                    right_hand_claim=str(analysis_context["right_hand_claim"]),
                )

            analysis_context.update(
                {
                    "hold_correct_hits": hold_correct_hits,
                    "hold_wrong_side_hits": hold_wrong_side_hits,
                    "hold_none_hits": hold_none_hits,
                    "hold_unknown_hits": hold_unknown_hits,
                    "hold_not_hold_evidence_hits": hold_not_hold_evidence_hits,
                    "hold_final_status": hold_final_status,
                    "holding": holding,
                    "left_holding": left_holding_confirmed,
                    "right_holding": right_holding_confirmed,
                    "handrail_status": handrail_decision["handrail_status"],
                    "handrail_reason": handrail_decision["handrail_reason"],
                    "handrail_debug_reason": handrail_decision[
                        "handrail_debug_reason"
                    ],
                }
            )
            return carry_pose

        apply_direction_history(self, track_id, person.features)

        standing_start = time.perf_counter() if perf is not None else None
        analysis_context.update(
            evaluate_inside_stairs(
                self,
                track_id,
                person.features,
                person.p_lane,
            )
        )
        analysis_context.update(
            evaluate_standing_still(
                self,
                track_id,
                person.p_motion,
                bool(analysis_context["inside_stairs"]),
            )
        )
        direction_start = time.perf_counter() if perf is not None else None
        analysis_context.update(update_direction(self, track_id, person.features))
        self._record_perf(perf, "direction", direction_start)
        analysis_context.update(
            self._evaluate_two_step_skip(
                track_id,
                person.features,
                bool(analysis_context["inside_stairs"]),
                str(analysis_context["direction"]),
                str(analysis_context["direction_source"]),
            )
        )
        self._record_perf(perf, "standing", standing_start)

        if analysis_context["direction"] == "ANALYZING":
            if not analysis_context["inside_stairs"]:
                self._reset_behavior_histories(track_id)
                analysis_context["hold_raw_status"] = "OUTSIDE"
                analysis_context["hold_final_status"] = "OUTSIDE"
                return build_result(
                    status="ANALYZING...",
                    display_status="" if SETTINGS.demo.demo_mode else "ANALYZING...",
                    color=SETTINGS.violation.outside_color,
                )

            hold_direction = self.last_valid_direction.get(track_id)
            update_hold_context(hold_direction)
            warnings = self._collect_warnings(
                False,
                analysis_context["hold_final_status"],
                False,
                analysis_context["standing_still_confirmed"],
                two_step_skip_confirmed=bool(
                    analysis_context["two_step_skip_confirmed"]
                ),
            )
            analysis_context["warnings"] = warnings
            if warnings:
                status, display_status, color = self._summarize_result(
                    analysis_context["direction"],
                    warnings,
                    analysis_context["hold_final_status"],
                    SETTINGS.violation.unknown_color
                    if SETTINGS.demo.demo_mode
                    else SETTINGS.violation.analyzing_color,
                    "ANALYZING...",
                )
                return build_result(
                    status=status,
                    display_status=display_status,
                    color=color,
                )

            return build_result(
                status="ANALYZING...",
                display_status="" if SETTINGS.demo.demo_mode else "ANALYZING...",
                color=(
                    SETTINGS.violation.unknown_color
                    if SETTINGS.demo.demo_mode
                    else SETTINGS.violation.analyzing_color
                ),
            )

        if not analysis_context["inside_stairs"]:
            self._reset_behavior_histories(track_id)
            analysis_context["warnings"] = []
            return build_result(
                status=f"{analysis_context['direction']} | Ngoai Vung",
                display_status=(
                    ""
                    if SETTINGS.demo.demo_mode
                    else f"{analysis_context['direction']} | Ngoai Vung"
                ),
                color=SETTINGS.violation.outside_color,
                lane_raw=False,
                lane_hits=0,
                lane_conf=False,
                inside_stairs=False,
                hold_status="OUTSIDE",
                hold_raw_status="OUTSIDE",
                hold_confirmed_status="OUTSIDE",
                hold_correct_hits=0,
                hold_wrong_side_hits=0,
                hold_none_hits=0,
                hold_unknown_hits=0,
                holding_correct_raw=False,
                holding_wrong_raw=False,
                left_wrist_valid=False,
                right_wrist_valid=False,
                left_wrist_hit=False,
                right_wrist_hit=False,
                left_wrist_distance_to_left_rail=None,
                left_wrist_distance_to_right_rail=None,
                right_wrist_distance_to_left_rail=None,
                right_wrist_distance_to_right_rail=None,
                left_wrist_nearest_distance=None,
                right_wrist_nearest_distance=None,
                left_wrist_nearest_rail="NONE",
                right_wrist_nearest_rail="NONE",
                left_wrist_nearest_point=None,
                right_wrist_nearest_point=None,
                left_holding=False,
                right_holding=False,
                handrail_status="OUTSIDE",
                handrail_reason="OUTSIDE_STAIRS",
                handrail_debug_reason="OUTSIDE_STAIRS",
            )

        if analysis_context["direction"] in ("UP", "DOWN"):
            self.last_valid_direction[track_id] = analysis_context["direction"]
            self.last_valid_direction_frame[track_id] = self.frame_index

        backward_start = time.perf_counter() if perf is not None else None
        analysis_context.update(
            evaluate_backward(
                self,
                track_id,
                analysis_context["direction"],
                body_facing,
                body_facing_confidence,
                body_facing_evidence_count,
                hip_pair_valid,
                shoulder_pair_valid,
                ear_pair_valid,
                head_valid,
            )
        )
        self._record_perf(perf, "backward", backward_start)

        lane_start = time.perf_counter() if perf is not None else None
        analysis_context.update(
            evaluate_lane_violation(
                self,
                track_id,
                person.features,
                analysis_context["direction"],
            )
        )
        self._record_perf(perf, "lane", lane_start)

        hold_direction = analysis_context["direction"]
        if hold_direction not in ("UP", "DOWN"):
            hold_direction = self.last_valid_direction.get(track_id)
        carry_pose = update_hold_context(
            hold_direction,
            skip_handrail=bool(analysis_context["backward_confirmed"]),
        )

        if (
            analysis_context["direction"] == "IDLE"
            and not analysis_context["standing_still_confirmed"]
        ):
            return build_result(
                status="IDLE",
                display_status="",
                color=(
                    SETTINGS.violation.safe_color
                    if SETTINGS.demo.demo_mode
                    else (200, 200, 200)
                ),
            )

        if analysis_context["direction"] == "IDLE":
            warnings = self._collect_warnings(
                analysis_context["wrong_lane"],
                analysis_context["hold_final_status"],
                analysis_context["backward_confirmed"],
                analysis_context["standing_still_confirmed"],
                two_step_skip_confirmed=bool(
                    analysis_context["two_step_skip_confirmed"]
                ),
            )
            analysis_context["warnings"] = warnings
            status, display_status, color = self._summarize_result(
                analysis_context["direction"],
                warnings,
                analysis_context["hold_final_status"],
                (
                    SETTINGS.violation.safe_color
                    if SETTINGS.demo.demo_mode
                    else (200, 200, 200)
                ),
                analysis_context["direction"],
            )
            return build_result(
                status=status,
                display_status=display_status,
                color=color,
            )

        carry_start = time.perf_counter() if perf is not None else None
        carry_info = evaluate_carry(
            self,
            track_id,
            person.keypoints,
            analysis_context["holding_raw"],
            features=person.features,
            carry_pose=carry_pose,
            handrail_state={
                "left_wrist_hit": analysis_context["left_wrist_hit"],
                "right_wrist_hit": analysis_context["right_wrist_hit"],
                "left_holding": analysis_context["left_holding"],
                "right_holding": analysis_context["right_holding"],
            },
        )
        self._record_perf(perf, "carry", carry_start)
        warnings = self._collect_warnings(
            analysis_context["wrong_lane"],
            analysis_context["hold_final_status"],
            analysis_context["backward_confirmed"],
            analysis_context["standing_still_confirmed"],
            is_carrying=carry_info["is_carrying"],
            two_step_skip_confirmed=bool(
                analysis_context["two_step_skip_confirmed"]
            ),
        )
        analysis_context["warnings"] = warnings
        status, display_status, color = self._summarize_result(
            analysis_context["direction"],
            warnings,
            analysis_context["hold_final_status"],
            SETTINGS.violation.safe_color,
            f"{analysis_context['direction']} | An Toan",
        )
        return build_result(
            status=status,
            display_status=display_status,
            color=color,
            carry_info_override=carry_info,
        )

    def merge_behavior_history(
        self,
        from_subject_id: AnalysisSubjectID,
        to_subject_id: AnalysisSubjectID,
    ) -> None:
        if from_subject_id == to_subject_id:
            return

        list_history_maps = [
            self.hip_motion_history,
            self.shoulder_motion_history,
            self.lane_history,
            self.hold_status_history,
            self.left_handrail_hit_history,
            self.right_handrail_hit_history,
            self.front_carry_history,
            self.front_carry_one_arm_history,
            self.backward_history,
            self.standing_history,
            self.two_step_skip_history,
            self.standing_motion_history,
        ]
        for history_map in list_history_maps:
            if from_subject_id not in history_map:
                continue
            from_value = history_map.pop(from_subject_id)
            to_value = history_map.get(to_subject_id)
            if isinstance(to_value, list) and isinstance(from_value, list):
                history_map[to_subject_id] = to_value + from_value
            else:
                history_map[to_subject_id] = from_value

        latest_value_maps = [
            self.inside_last_state,
            self.lane_last_state,
            self.lane_last_seen,
            self.last_valid_direction,
            self.last_valid_direction_frame,
            self.hand_claim_state,
            self.left_foot_step_states,
            self.right_foot_step_states,
        ]
        for history_map in latest_value_maps:
            if from_subject_id in history_map:
                history_map[to_subject_id] = history_map.pop(from_subject_id)

    def cleanup_inactive_tracks(self, active_track_ids):
        active_track_ids = set(active_track_ids)
        history_maps = [
            self.hip_motion_history,
            self.shoulder_motion_history,
            self.lane_history,
            self.lane_last_state,
            self.lane_last_seen,
            self.hold_status_history,
            self.left_handrail_hit_history,
            self.right_handrail_hit_history,
            self.last_valid_direction,
            self.last_valid_direction_frame,
            self.hand_claim_state,
            self.front_carry_history,
            self.front_carry_one_arm_history,
            self.backward_history,
            self.standing_history,
            self.two_step_skip_history,
            self.left_foot_step_states,
            self.right_foot_step_states,
            self.standing_motion_history,
        ]

        for history_map in history_maps:
            inactive_ids = [
                current_track_id
                for current_track_id in history_map.keys()
                if current_track_id not in active_track_ids
            ]
            for current_track_id in inactive_ids:
                del history_map[current_track_id]
