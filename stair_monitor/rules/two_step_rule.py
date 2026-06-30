"""Two-step behavior rule for stair_monitor Windows/demo."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from stair_monitor.common.types import AnalysisSubjectID, Point, PoseFeatures
from stair_monitor.config.settings import SETTINGS
from stair_monitor.state.track_state import AnalyzerState, FootStepState
from stair_monitor.vision.step_lines import (
    LOW_ANKLE_CONF_REASON,
    NEAR_STEP_BOUNDARY_REASON,
    NO_STEP_BANDS_REASON,
    OUTSIDE_ALL_BANDS_REASON,
    POINT_INVALID_REASON,
    POINT_NEAR_STEP_BOUNDARY_REASON,
    StepBand,
    StepIndexResult,
    get_step_index_for_point,
)


@dataclass(slots=True)
class TwoStepFilterInfo:
    raw_step: int | None
    filtered_step: int | None
    last_valid_step: int | None
    filter_reason: str
    reverse_reject: bool


@dataclass(slots=True)
class FootPlantState:
    current_step_index: int | None
    planted_step_index: int | None
    is_planted: bool
    ankle_speed_px: float | None
    reason: str
    state_label: str
    landed: bool
    landed_step_index: int | None
    candidate_step_index: int | None
    candidate_count: int


@dataclass(slots=True)
class TwoStepAnalysisResult:
    confirmed: bool
    status: str
    reason: str
    left_step_index: int | None = None
    right_step_index: int | None = None
    step_gap: int | None = None
    debug_lines: tuple[str, ...] = ()
    context_fields: dict[str, object] = field(default_factory=dict)

    def to_analysis_context(self) -> dict[str, object]:
        return dict(self.context_fields)


@dataclass(frozen=True, slots=True)
class TwoStepInput:
    subject_id: AnalysisSubjectID
    direction: str
    direction_source: str
    features: PoseFeatures
    inside_stairs: bool
    frame_index: int


@dataclass(frozen=True, slots=True)
class TwoStepFrameState:
    direction_label: str
    left_ankle_point: Point | None
    right_ankle_point: Point | None
    left_ankle_conf: float
    right_ankle_conf: float
    left_ankle_step_point: Point | None
    right_ankle_step_point: Point | None
    ankle_step_offset_x: int
    ankle_step_offset_y: int
    ankle_step_offset_direction: str
    filter_direction: str
    direction_reliable: bool


def build_two_step_input(
    track_id: AnalysisSubjectID,
    direction: str | None,
    features: PoseFeatures,
    inside_stairs: bool,
    frame_index: int,
    direction_source: str,
) -> TwoStepInput:
    return TwoStepInput(
        subject_id=track_id,
        direction=direction or "UNKNOWN",
        direction_source=direction_source,
        features=features,
        inside_stairs=inside_stairs,
        frame_index=frame_index,
    )


class TwoStepAnalyzer:
    """Phan tich loi Buoc 2 Bac cho ban Windows/demo.

    Rule nay nhan direction + pose features tu `BehaviorAnalyzer`.
    Neu huong di khong phai `UP` thi tra ve `SKIPPED` de tranh false positive.
    """

    def __init__(
        self,
        step_bands: list[StepBand],
        state: AnalyzerState,
    ) -> None:
        self.step_bands = step_bands
        self.state = state
        self.current_frame_index = state.frame_index
        self.last_valid_direction = state.last_valid_direction
        self.last_valid_direction_frame = state.last_valid_direction_frame
        self.two_step_skip_history = state.two_step_skip_history
        self.left_foot_step_states = state.left_foot_step_states
        self.right_foot_step_states = state.right_foot_step_states

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
                and self.current_frame_index - last_valid_direction_frame
                <= self._get_two_step_hold_last_valid_frames()
            ):
                return last_valid_direction, "RECENT_LAST_VALID_DIRECTION", True

        last_valid_direction = self.last_valid_direction.get(track_id)
        last_valid_direction_frame = self.last_valid_direction_frame.get(track_id)
        if (
            last_valid_direction in ("UP", "DOWN")
            and last_valid_direction_frame is not None
            and self.current_frame_index - last_valid_direction_frame
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
    ) -> TwoStepFilterInfo:
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
            and self.current_frame_index - last_valid_step_frame
            <= hold_last_valid_frames
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
            foot_state.last_valid_step_frame = self.current_frame_index

        foot_state.raw_step_index = raw_step
        foot_state.filtered_step_index = filtered_step
        foot_state.filter_reason = filter_reason

        return TwoStepFilterInfo(
            raw_step=raw_step,
            filtered_step=filtered_step,
            last_valid_step=foot_state.last_valid_step_index,
            filter_reason=filter_reason,
            reverse_reject=reverse_reject,
        )

    def _update_foot_step_state(
        self,
        track_id: AnalysisSubjectID,
        foot_side: str,
        ankle_point: Point | None,
        ankle_conf: float,
        current_step_index: int | None,
        current_step_reason: str,
    ) -> FootPlantState:
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
            and self.current_frame_index - foot_state.last_seen_frame <= 1
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
            and self.current_frame_index - foot_state.last_seen_frame > history_window
        ):
            foot_state.step_history.clear()
            foot_state.candidate_step_index = None
            foot_state.candidate_count = 0

        if ankle_point is not None:
            foot_state.last_ankle_point = ankle_point
            foot_state.last_seen_frame = self.current_frame_index

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
        elif (
            use_speed_check
            and ankle_speed is not None
            and ankle_speed > speed_threshold
        ):
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

        return FootPlantState(
            current_step_index=current_step_index,
            planted_step_index=foot_state.last_planted_step_index,
            is_planted=foot_state.is_planted,
            ankle_speed_px=foot_state.ankle_speed_px,
            reason=foot_state.reason,
            state_label=state_label,
            landed=landed,
            landed_step_index=landed_step_index,
            candidate_step_index=foot_state.candidate_step_index,
            candidate_count=foot_state.candidate_count,
        )

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

    @staticmethod
    def _build_default_context(
        features: PoseFeatures,
        left_ankle_step_point: Point | None,
        right_ankle_step_point: Point | None,
        ankle_step_offset_x: int,
        ankle_step_offset_y: int,
        ankle_step_offset_direction: str,
    ) -> dict[str, object]:
        left_ankle_point = features.get("left_ankle")
        right_ankle_point = features.get("right_ankle")
        return {
            "left_ankle_point": left_ankle_point,
            "right_ankle_point": right_ankle_point,
            "left_ankle_raw_point": left_ankle_point,
            "right_ankle_raw_point": right_ankle_point,
            "left_ankle_step_point": left_ankle_step_point,
            "right_ankle_step_point": right_ankle_step_point,
            "left_ankle_conf": float(features.get("left_ankle_conf", 0.0) or 0.0),
            "right_ankle_conf": float(features.get("right_ankle_conf", 0.0) or 0.0),
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
            "two_step_skip_status": "NOT_EVALUATED",
            "two_step_skip_reason": "NOT_EVALUATED",
            "two_step_skip_confirmed": False,
        }

    def _build_frame_state(
        self,
        rule_input: TwoStepInput,
    ) -> TwoStepFrameState:
        direction_label = rule_input.direction or "UNKNOWN"
        left_ankle_point = rule_input.features.get("left_ankle")
        right_ankle_point = rule_input.features.get("right_ankle")
        left_ankle_conf = float(rule_input.features.get("left_ankle_conf", 0.0) or 0.0)
        right_ankle_conf = float(
            rule_input.features.get("right_ankle_conf", 0.0) or 0.0
        )
        ankle_step_offset_x, ankle_step_offset_y, ankle_step_offset_direction = (
            self._resolve_ankle_step_offset(
                rule_input.subject_id,
                direction_label,
                rule_input.direction_source,
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
                rule_input.subject_id,
                direction_label,
                rule_input.direction_source,
            )
        )
        return TwoStepFrameState(
            direction_label=direction_label,
            left_ankle_point=left_ankle_point,
            right_ankle_point=right_ankle_point,
            left_ankle_conf=left_ankle_conf,
            right_ankle_conf=right_ankle_conf,
            left_ankle_step_point=left_ankle_step_point,
            right_ankle_step_point=right_ankle_step_point,
            ankle_step_offset_x=ankle_step_offset_x,
            ankle_step_offset_y=ankle_step_offset_y,
            ankle_step_offset_direction=ankle_step_offset_direction,
            filter_direction=filter_direction,
            direction_reliable=direction_reliable,
        )

    def analyze_input(
        self,
        rule_input: TwoStepInput,
    ) -> TwoStepAnalysisResult:
        self.current_frame_index = rule_input.frame_index
        frame_state = self._build_frame_state(rule_input)

        context_fields = self._build_default_context(
            rule_input.features,
            frame_state.left_ankle_step_point,
            frame_state.right_ankle_step_point,
            frame_state.ankle_step_offset_x,
            frame_state.ankle_step_offset_y,
            frame_state.ankle_step_offset_direction,
        )

        if not SETTINGS.two_step_skip.enabled:
            self._update_two_step_skip_history(rule_input.subject_id, False)
            context_fields["two_step_skip_status"] = "DISABLED"
            context_fields["two_step_skip_reason"] = "TWO_STEP_SKIP_DISABLED"
            return TwoStepAnalysisResult(
                confirmed=False,
                status="DISABLED",
                reason="TWO_STEP_SKIP_DISABLED",
                debug_lines=("TWO_STEP=DISABLED",),
                context_fields=context_fields,
            )

        if frame_state.direction_label != "UP":
            self._update_two_step_skip_history(rule_input.subject_id, False)
            context_fields["two_step_skip_status"] = "SKIPPED"
            context_fields["two_step_skip_reason"] = "ONLY_CHECK_WHEN_UP"
            return TwoStepAnalysisResult(
                confirmed=False,
                status="SKIPPED",
                reason="ONLY_CHECK_WHEN_UP",
                debug_lines=(
                    "TWO_STEP=SKIPPED",
                    f"DIR={frame_state.direction_label}",
                    "REASON=ONLY_CHECK_WHEN_UP",
                ),
                context_fields=context_fields,
            )

        candidate_two_step_skip = False
        reason = "NOT_EVALUATED"
        status = "NOT_EVALUATED"
        ankle_conf_threshold = float(SETTINGS.two_step_skip.ankle_conf_threshold)
        left_raw_step_result = get_step_index_for_point(
            frame_state.left_ankle_point,
            self.step_bands,
        )
        right_raw_step_result = get_step_index_for_point(
            frame_state.right_ankle_point,
            self.step_bands,
        )
        left_step_result = get_step_index_for_point(
            frame_state.left_ankle_step_point,
            self.step_bands,
        )
        right_step_result = get_step_index_for_point(
            frame_state.right_ankle_step_point,
            self.step_bands,
        )
        if (
            frame_state.left_ankle_point is not None
            and frame_state.left_ankle_conf < ankle_conf_threshold
        ):
            left_step_result = self._override_step_result_for_low_conf(left_step_result)
        if (
            frame_state.right_ankle_point is not None
            and frame_state.right_ankle_conf < ankle_conf_threshold
        ):
            right_step_result = self._override_step_result_for_low_conf(
                right_step_result
            )

        context_fields.update(
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

        if not rule_input.inside_stairs:
            reason = "OUTSIDE_STAIRS"
            status = "OUTSIDE_STAIRS"
        else:
            left_filter_info = self._filter_step_by_direction(
                rule_input.subject_id,
                "LEFT",
                left_step_result.step_index,
                frame_state.filter_direction,
                frame_state.direction_reliable,
            )
            right_filter_info = self._filter_step_by_direction(
                rule_input.subject_id,
                "RIGHT",
                right_step_result.step_index,
                frame_state.filter_direction,
                frame_state.direction_reliable,
            )
            left_filtered_step = left_filter_info.filtered_step
            right_filtered_step = right_filter_info.filtered_step

            left_foot_state = self._update_foot_step_state(
                rule_input.subject_id,
                "LEFT",
                frame_state.left_ankle_point,
                frame_state.left_ankle_conf,
                left_filtered_step,
                left_filter_info.filter_reason,
            )
            right_foot_state = self._update_foot_step_state(
                rule_input.subject_id,
                "RIGHT",
                frame_state.right_ankle_point,
                frame_state.right_ankle_conf,
                right_filtered_step,
                right_filter_info.filter_reason,
            )

            filtered_step_gap = None
            if (
                frame_state.direction_reliable
                and isinstance(left_filtered_step, int)
                and isinstance(right_filtered_step, int)
            ):
                filtered_step_gap = abs(left_filtered_step - right_filtered_step)

            planted_step_gap = None
            left_planted_step = left_foot_state.planted_step_index
            right_planted_step = right_foot_state.planted_step_index
            if isinstance(left_planted_step, int) and isinstance(right_planted_step, int):
                planted_step_gap = abs(left_planted_step - right_planted_step)

            context_fields.update(
                {
                    "left_filtered_step": left_filtered_step,
                    "right_filtered_step": right_filtered_step,
                    "left_last_valid_step": left_filter_info.last_valid_step,
                    "right_last_valid_step": right_filter_info.last_valid_step,
                    "left_step_filter_reason": left_filter_info.filter_reason,
                    "right_step_filter_reason": right_filter_info.filter_reason,
                    "foot_gap": filtered_step_gap,
                    "step_gap": filtered_step_gap,
                    "left_planted_step": left_planted_step,
                    "right_planted_step": right_planted_step,
                    "planted_step_gap": planted_step_gap,
                    "left_is_planted": left_foot_state.is_planted,
                    "right_is_planted": right_foot_state.is_planted,
                    "left_ankle_speed": left_foot_state.ankle_speed_px,
                    "right_ankle_speed": right_foot_state.ankle_speed_px,
                    "left_foot_step_reason": left_foot_state.reason,
                    "right_foot_step_reason": right_foot_state.reason,
                    "left_foot_state_label": left_foot_state.state_label,
                    "right_foot_state_label": right_foot_state.state_label,
                    "left_foot_landed": left_foot_state.landed,
                    "right_foot_landed": right_foot_state.landed,
                }
            )

            reverse_reject_in_frame = (
                left_filter_info.reverse_reject or right_filter_info.reverse_reject
            )
            if not frame_state.direction_reliable:
                reason = "DIRECTION_NOT_RELIABLE_FOR_TWO_STEP"
                status = "DIRECTION_NOT_RELIABLE"
            elif filtered_step_gap is None:
                status = "INSUFFICIENT_STEP_DATA"
                if left_filtered_step is None and right_filtered_step is None:
                    reason = "FILTERED_STEP_NA"
                elif left_filtered_step is None:
                    reason = "LEFT_FILTERED_STEP_NA"
                else:
                    reason = "RIGHT_FILTERED_STEP_NA"
            else:
                context_fields["two_step_skip_check_available"] = True
                if filtered_step_gap >= int(SETTINGS.two_step_skip.min_step_gap):
                    if reverse_reject_in_frame:
                        if int(SETTINGS.two_step_skip.confirm_frames) > 1:
                            candidate_two_step_skip = True
                            reason = "WAIT_CONFIRM_AFTER_REVERSE_REJECT"
                            status = "WAIT_CONFIRM"
                        else:
                            reason = "REVERSE_STEP_REJECT_SUPPRESSED"
                            status = "SUPPRESSED_REVERSE_REJECT"
                    else:
                        candidate_two_step_skip = True
                        reason = "WAIT_CONFIRM_FRAMES"
                        status = "WAIT_CONFIRM"
                else:
                    reason = "FILTERED_GAP_OK"
                    status = "CLEAR"

        confirmed = candidate_two_step_skip and self._update_two_step_skip_history(
            rule_input.subject_id,
            candidate_two_step_skip,
        )
        if not candidate_two_step_skip:
            self._update_two_step_skip_history(rule_input.subject_id, False)
        if confirmed:
            reason = "TWO_STEP_SKIP_CONFIRMED"
            status = "CONFIRMED"

        context_fields["two_step_skip_status"] = status
        context_fields["two_step_skip_reason"] = reason
        context_fields["two_step_skip_confirmed"] = confirmed

        return TwoStepAnalysisResult(
            confirmed=confirmed,
            status=status,
            reason=reason,
            left_step_index=left_step_result.step_index,
            right_step_index=right_step_result.step_index,
            step_gap=context_fields["step_gap"]
            if isinstance(context_fields["step_gap"], int)
            else None,
            context_fields=context_fields,
        )

    def analyze(
        self,
        track_id: AnalysisSubjectID,
        direction: str | None,
        features: PoseFeatures,
        inside_stairs: bool,
        frame_index: int,
        direction_source: str,
    ) -> TwoStepAnalysisResult:
        rule_input = build_two_step_input(
            track_id=track_id,
            direction=direction,
            features=features,
            inside_stairs=inside_stairs,
            frame_index=frame_index,
            direction_source=direction_source,
        )
        return self.analyze_input(rule_input)
