from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Protocol, TypedDict

from stair_monitor.common.types import (
    AnalysisSubjectID,
    LinePoints,
    Numeric,
    Point,
    PoseFeatures,
)
from stair_monitor.config.settings import SETTINGS

if TYPE_CHECKING:
    from stair_monitor.core.analyzer import BehaviorAnalyzer


class LaneAnalysisUpdate(TypedDict):
    v: Numeric | None
    p_lane: Point | None
    p_lane_source: str
    lane_source: str
    lane_mapping_source: str
    lane_direction: str
    foot_lane_side: Numeric | None
    lane_side_value: Numeric | None
    lane_side_label: str
    correct_lane_side: str
    wrong_lane_raw: bool | None
    wrong_lane: bool
    lane_wrong_hits: int
    lane_status: str
    lane_reason: str
    lane_missing_feet_grace_left: int


class LaneLastStateRecord(TypedDict):
    wrong_lane_raw: bool | None
    wrong_lane: bool
    lane_wrong_hits: int
    lane_status: str
    lane_direction: str
    lane_v: Numeric | None
    foot_lane_side: Numeric | None
    lane_side_value: Numeric | None
    lane_side_label: str
    correct_lane_side: str
    lane_mapping_source: str


@dataclass(frozen=True, slots=True)
class LaneInput:
    subject_id: AnalysisSubjectID
    frame_index: int
    direction: str
    center_line: LinePoints
    feet_point: Point | None
    feet_point_source: str


@dataclass(frozen=True, slots=True)
class LaneSelection:
    point: Point | None
    point_source: str
    selection_reason: str


@dataclass(frozen=True, slots=True)
class LaneResult:
    v: Numeric | None
    p_lane: Point | None
    p_lane_source: str
    lane_source: str
    lane_mapping_source: str
    lane_direction: str
    foot_lane_side: Numeric | None
    lane_side_value: Numeric | None
    lane_side_label: str
    correct_lane_side: str
    wrong_lane_raw: bool | None
    wrong_lane: bool
    lane_wrong_hits: int
    lane_status: str
    lane_reason: str
    lane_missing_feet_grace_left: int

    def to_analysis_update(self) -> LaneAnalysisUpdate:
        return {
            "v": self.v,
            "p_lane": self.p_lane,
            "p_lane_source": self.p_lane_source,
            "lane_source": self.lane_source,
            "lane_mapping_source": self.lane_mapping_source,
            "lane_direction": self.lane_direction,
            "foot_lane_side": self.foot_lane_side,
            "lane_side_value": self.lane_side_value,
            "lane_side_label": self.lane_side_label,
            "correct_lane_side": self.correct_lane_side,
            "wrong_lane_raw": self.wrong_lane_raw,
            "wrong_lane": self.wrong_lane,
            "lane_wrong_hits": self.lane_wrong_hits,
            "lane_status": self.lane_status,
            "lane_reason": self.lane_reason,
            "lane_missing_feet_grace_left": self.lane_missing_feet_grace_left,
        }

    def to_last_state_record(self) -> LaneLastStateRecord:
        return {
            "wrong_lane_raw": self.wrong_lane_raw,
            "wrong_lane": self.wrong_lane,
            "lane_wrong_hits": self.lane_wrong_hits,
            "lane_status": self.lane_status,
            "lane_direction": self.lane_direction,
            "lane_v": self.v,
            "foot_lane_side": self.foot_lane_side,
            "lane_side_value": self.lane_side_value,
            "lane_side_label": self.lane_side_label,
            "correct_lane_side": self.correct_lane_side,
            "lane_mapping_source": self.lane_mapping_source,
        }


class LaneHistoryAdapter(Protocol):
    center_line: LinePoints
    frame_index: int
    lane_last_seen: dict[AnalysisSubjectID, int]
    lane_last_state: dict[AnalysisSubjectID, dict[str, object]]

    def _update_lane_history(
        self,
        track_id: AnalysisSubjectID,
        wrong_lane_raw: bool | None,
    ) -> tuple[int, bool]:
        ...

    def _get_lane_history_state(
        self,
        track_id: AnalysisSubjectID,
    ) -> tuple[int, bool]:
        ...


def compute_lane_side(
    point: Point | None,
    line_points: LinePoints | None,
) -> Numeric | None:
    if point is None or line_points is None or len(line_points) < 2:
        return None

    a, b = line_points[0], line_points[1]
    return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (
        point[0] - a[0]
    )


def is_left_lane_side(side_value: Numeric | None) -> bool:
    if side_value is None or side_value == 0:
        return False
    return side_value * SETTINGS.camera.lane_left_side_sign > 0


def is_right_lane_side(side_value: Numeric | None) -> bool:
    if side_value is None or side_value == 0:
        return False
    return side_value * SETTINGS.camera.lane_left_side_sign < 0


def is_wrong_lane_side(
    direction: str,
    side_value: Numeric | None,
    sign_normal: bool,
) -> bool | None:
    if side_value is None or direction not in ("UP", "DOWN"):
        return None

    if not SETTINGS.camera.use_current_camera_angle:
        if direction == "DOWN":
            return not is_left_lane_side(side_value)
        return not is_right_lane_side(side_value)

    if sign_normal:
        return (direction == "UP" and side_value < 0) or (
            direction == "DOWN" and side_value > 0
        )
    return (direction == "UP" and side_value > 0) or (
        direction == "DOWN" and side_value < 0
    )


def get_lane_side_label(side_value: Numeric | None) -> str:
    if side_value is None:
        return "UNKNOWN"
    if is_left_lane_side(side_value):
        return "LEFT"
    if is_right_lane_side(side_value):
        return "RIGHT"
    return "ON_LINE"


def get_correct_lane_side(direction: str, sign_normal: bool) -> str:
    if direction not in ("UP", "DOWN"):
        return "UNKNOWN"

    if not SETTINGS.camera.use_current_camera_angle:
        return "LEFT" if direction == "DOWN" else "RIGHT"

    if sign_normal:
        return "RIGHT" if direction == "UP" else "LEFT"
    return "LEFT" if direction == "UP" else "RIGHT"


def get_camera_angle_profile() -> str:
    return (
        "CURRENT_CAMERA"
        if SETTINGS.camera.use_current_camera_angle
        else "BOTTOM_STAIR_CAMERA"
    )


def build_lane_input(
    subject_id: AnalysisSubjectID,
    frame_index: int,
    features: PoseFeatures,
    center_line: LinePoints,
) -> LaneInput:
    return LaneInput(
        subject_id=subject_id,
        frame_index=frame_index,
        direction="ANALYZING",
        center_line=center_line,
        feet_point=features.get("feet_point"),
        feet_point_source=features.get("feet_point_source", "FEET_UNAVAILABLE"),
    )


def _with_direction(rule_input: LaneInput, direction: str) -> LaneInput:
    return LaneInput(
        subject_id=rule_input.subject_id,
        frame_index=rule_input.frame_index,
        direction=direction,
        center_line=rule_input.center_line,
        feet_point=rule_input.feet_point,
        feet_point_source=rule_input.feet_point_source,
    )


def select_p_lane_for_lane(rule_input: LaneInput) -> LaneSelection:
    if rule_input.feet_point is None:
        return LaneSelection(None, "NO_FOOT", "NO_VALID_FOOT_FOR_LANE")

    if rule_input.feet_point_source == "REAL_BOTH_ANKLES":
        return LaneSelection(rule_input.feet_point, "FEET_MIDPOINT", "FEET_VISIBLE")
    if rule_input.feet_point_source == "REAL_LEFT_ANKLE":
        return LaneSelection(rule_input.feet_point, "LEFT_FOOT_ONLY", "FEET_VISIBLE")
    if rule_input.feet_point_source == "REAL_RIGHT_ANKLE":
        return LaneSelection(rule_input.feet_point, "RIGHT_FOOT_ONLY", "FEET_VISIBLE")
    return LaneSelection(
        rule_input.feet_point,
        rule_input.feet_point_source,
        "VIRTUAL_FEET",
    )


def update_lane_history(
    history_adapter: LaneHistoryAdapter,
    track_id: AnalysisSubjectID,
    wrong_lane_raw: bool | None,
) -> tuple[int, bool]:
    return history_adapter._update_lane_history(track_id, wrong_lane_raw)


def _new_default_lane_result(
    direction: str,
    p_lane_source: str = "NONE",
) -> LaneResult:
    return LaneResult(
        v=None,
        p_lane=None,
        p_lane_source=p_lane_source,
        lane_source="NO_LANE",
        lane_mapping_source=get_camera_angle_profile(),
        lane_direction=direction,
        foot_lane_side=None,
        lane_side_value=None,
        lane_side_label="UNKNOWN",
        correct_lane_side="UNKNOWN",
        wrong_lane_raw=False,
        wrong_lane=False,
        lane_wrong_hits=0,
        lane_status="UNKNOWN",
        lane_reason="NA",
        lane_missing_feet_grace_left=0,
    )


def _read_bool(
    values: Mapping[str, object],
    key: str,
    default: bool,
) -> bool:
    value = values.get(key, default)
    return value if isinstance(value, bool) else default


def _read_optional_bool(
    values: Mapping[str, object],
    key: str,
) -> bool | None:
    value = values.get(key)
    if value is None or isinstance(value, bool):
        return value
    return None


def _read_int(
    values: Mapping[str, object],
    key: str,
    default: int,
) -> int:
    value = values.get(key, default)
    return value if isinstance(value, int) else default


def _read_str(
    values: Mapping[str, object],
    key: str,
    default: str,
) -> str:
    value = values.get(key, default)
    return value if isinstance(value, str) else default


def _read_numeric(
    values: Mapping[str, object],
    key: str,
) -> Numeric | None:
    value = values.get(key)
    if value is None or isinstance(value, (int, float)):
        return value
    return None


def _evaluate_lane_from_selected_point(
    rule_input: LaneInput,
    history_adapter: LaneHistoryAdapter,
    selection: LaneSelection,
) -> LaneResult:
    result = _new_default_lane_result(rule_input.direction, selection.point_source)
    lane_side_value: Numeric | None = None
    lane_side_label = "UNKNOWN"
    correct_lane_side = "UNKNOWN"
    foot_lane_side: Numeric | None = None
    lane_reason = "LANE_BY_FEET"
    lane_status = "UNKNOWN"
    wrong_lane_raw: bool | None = False
    wrong_lane = False
    lane_wrong_hits = 0
    lane_source = "FOOT_LANE"
    v: Numeric | None = None

    if len(rule_input.center_line) >= 2 and selection.point is not None:
        foot_lane_side = compute_lane_side(selection.point, rule_input.center_line)
        v = foot_lane_side
        lane_side_value = foot_lane_side
        lane_side_label = get_lane_side_label(foot_lane_side)
        correct_lane_side = get_correct_lane_side(
            rule_input.direction,
            SETTINGS.lane.sign_normal,
        )

    if rule_input.direction not in ("UP", "DOWN"):
        lane_wrong_hits, wrong_lane = history_adapter._get_lane_history_state(
            rule_input.subject_id
        )
        lane_status = "UNKNOWN"
    elif len(rule_input.center_line) >= 2:
        wrong_lane_raw = is_wrong_lane_side(
            rule_input.direction,
            foot_lane_side,
            SETTINGS.lane.sign_normal,
        )
        lane_wrong_hits, wrong_lane = update_lane_history(
            history_adapter,
            rule_input.subject_id,
            wrong_lane_raw,
        )
        lane_status = "EVALUATED"
    else:
        lane_status = "UNKNOWN"
        lane_reason = "CENTER_LINE_MISSING"

    return LaneResult(
        v=v,
        p_lane=selection.point,
        p_lane_source=selection.point_source,
        lane_source=lane_source,
        lane_mapping_source=result.lane_mapping_source,
        lane_direction=rule_input.direction,
        foot_lane_side=foot_lane_side,
        lane_side_value=lane_side_value,
        lane_side_label=lane_side_label,
        correct_lane_side=correct_lane_side,
        wrong_lane_raw=wrong_lane_raw,
        wrong_lane=wrong_lane,
        lane_wrong_hits=lane_wrong_hits,
        lane_status=lane_status,
        lane_reason=lane_reason,
        lane_missing_feet_grace_left=0,
    )


def _evaluate_lane_keep_last(
    rule_input: LaneInput,
    history_adapter: LaneHistoryAdapter,
) -> LaneResult:
    result = _new_default_lane_result(rule_input.direction, "NO_FOOT")
    last_seen = history_adapter.lane_last_seen.get(rule_input.subject_id)
    if (
        last_seen is None
        or rule_input.frame_index - last_seen > SETTINGS.lane.missing_feet_grace_frames
    ):
        return LaneResult(
            v=None,
            p_lane=None,
            p_lane_source="NO_FOOT",
            lane_source="NO_LANE",
            lane_mapping_source=result.lane_mapping_source,
            lane_direction=rule_input.direction,
            foot_lane_side=None,
            lane_side_value=None,
            lane_side_label="UNKNOWN",
            correct_lane_side="UNKNOWN",
            wrong_lane_raw=None,
            wrong_lane=False,
            lane_wrong_hits=0,
            lane_status="UNKNOWN",
            lane_reason="MISSING_FEET_NO_DISPLAY",
            lane_missing_feet_grace_left=0,
        )

    lane_missing_feet_grace_left = max(
        0,
        SETTINGS.lane.missing_feet_grace_frames - (rule_input.frame_index - last_seen) + 1,
    )
    last_lane_state = history_adapter.lane_last_state.get(rule_input.subject_id, {})
    return LaneResult(
        v=_read_numeric(last_lane_state, "lane_v"),
        p_lane=None,
        p_lane_source="NO_FOOT",
        lane_source="LANE_KEEP_LAST",
        lane_mapping_source=_read_str(
            last_lane_state,
            "lane_mapping_source",
            result.lane_mapping_source,
        ),
        lane_direction=_read_str(
            last_lane_state,
            "lane_direction",
            rule_input.direction,
        ),
        foot_lane_side=_read_numeric(last_lane_state, "foot_lane_side"),
        lane_side_value=_read_numeric(last_lane_state, "lane_side_value"),
        lane_side_label=_read_str(last_lane_state, "lane_side_label", "UNKNOWN"),
        correct_lane_side=_read_str(
            last_lane_state,
            "correct_lane_side",
            "UNKNOWN",
        ),
        wrong_lane_raw=_read_optional_bool(last_lane_state, "wrong_lane_raw"),
        wrong_lane=_read_bool(last_lane_state, "wrong_lane", False),
        lane_wrong_hits=_read_int(last_lane_state, "lane_wrong_hits", 0),
        lane_status=_read_str(last_lane_state, "lane_status", "UNKNOWN"),
        lane_reason="MISSING_FEET_KEEP_LAST",
        lane_missing_feet_grace_left=lane_missing_feet_grace_left,
    )


def evaluate_lane_violation_typed(
    rule_input: LaneInput,
    history_adapter: LaneHistoryAdapter,
) -> LaneResult:
    selection = select_p_lane_for_lane(rule_input)
    if selection.point is None:
        return _evaluate_lane_keep_last(rule_input, history_adapter)
    return _evaluate_lane_from_selected_point(rule_input, history_adapter, selection)


def evaluate_lane_violation(
    analyzer: BehaviorAnalyzer,
    track_id: AnalysisSubjectID,
    features: PoseFeatures,
    direction: str,
) -> LaneAnalysisUpdate:
    rule_input = _with_direction(
        build_lane_input(
            subject_id=track_id,
            frame_index=analyzer.frame_index,
            features=features,
            center_line=analyzer.center_line,
        ),
        direction,
    )
    result = evaluate_lane_violation_typed(rule_input, analyzer)
    if result.lane_source == "FOOT_LANE":
        analyzer.lane_last_seen[track_id] = analyzer.frame_index
        analyzer.lane_last_state[track_id] = result.to_last_state_record()
    return result.to_analysis_update()
