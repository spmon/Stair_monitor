from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot, inf

import numpy as np

from stair_monitor.common.types import (
    AnalysisSubjectID,
    AnalysisResult,
    BBox,
    BBoxArray,
    CameraConfigDict,
    CountEventReasonLabel,
    IdentityEntryReasonLabel,
    IdentityStatusLabel,
    PersonSessionLifecycleLabel,
    PersonSessionStatusLabel,
    PersonUID,
    Point,
    PoseFeatures,
    ReLinkStateLabel,
)
from stair_monitor.config.settings import SETTINGS
from stair_monitor.rules.inside_rule import is_inside_stairs
from stair_monitor.rules.lane_rule import compute_lane_side, get_lane_side_label

TRUSTED_FEET_SOURCES = frozenset(
    {
        "REAL_BOTH_ANKLES",
        "REAL_LEFT_ANKLE",
        "REAL_RIGHT_ANKLE",
        "VIRTUAL_FROM_SHOULDER_HIP",
    }
)
OCCLUDED_ENTRY_MIN_MOTION_DELTA_PX = 4.0


@dataclass(slots=True)
class IdentityUpdateResult:
    has_active_person_id: bool
    person_uid: PersonUID | None
    person_uid_label: str
    analysis_subject_id: str
    analysis_subject_label: str
    merged_from_analysis_subject_id: str
    yolo_track_id: int | None
    previous_yolo_track_id: int | None
    identity_status: IdentityStatusLabel
    session_status: PersonSessionStatusLabel
    session_lifecycle: PersonSessionLifecycleLabel
    identity_debug: str
    identity_feet_source: str
    identity_feet_reason: str
    identity_gate_reason: str
    identity_inside_test: str
    identity_entry_reason: IdentityEntryReasonLabel
    identity_outside_proof: bool
    identity_enter_confirm_hits: int
    identity_enter_confirm_target: int
    relink_score: float | None
    relink_frame_gap: int
    relink_score_gap: float | None
    relink_best_candidate: str
    relink_second_candidate: str
    relink_state: ReLinkStateLabel
    has_counted_enter: bool
    has_counted_exit: bool
    count_event_reason: CountEventReasonLabel
    total_confirmed_entered_count: int
    total_occluded_entered_count: int
    occluded_entry_candidate_age_frames: int
    occluded_entry_age_target: int
    occluded_entry_inside_frames: int
    occluded_entry_inside_target: int
    occluded_entry_motion_frames: int
    occluded_entry_motion_target: int
    occluded_entry_block_reason: str
    occluded_entry_nearest_ghost_score: float | None
    occluded_entry_nearest_active_iou: float | None
    occluded_entry_nearest_active_distance: float | None


@dataclass(slots=True)
class CandidateSession:
    current_yolo_track_id: int
    first_seen_frame: int
    last_seen_frame: int
    last_bbox: BBox | None
    last_bbox_center: Point | None
    last_anchor_point: Point | None
    previous_anchor_point: Point | None
    last_path_position: float | None
    last_trusted_feet_point: Point | None
    last_trusted_feet_source: str
    has_trusted_feet_outside: bool = False
    entry_confirm_hits: int = 0
    occluded_inside_frames: int = 0
    occluded_motion_frames: int = 0
    last_gate_reason: str = "WAIT_TRUSTED_FEET_ENTER"
    relink_confirm_hits: int = 0
    pending_relink_person_uid: PersonUID | None = None
    pending_relink_best_score: float | None = None
    pending_relink_second_score: float | None = None
    pending_relink_score_gap: float | None = None
    pending_relink_frame_gap: int = 0
    pending_relink_best_label: str = ""
    pending_relink_second_label: str = ""
    pending_relink_state: ReLinkStateLabel = "NONE"
    last_relink_log_signature: str = ""


@dataclass(slots=True)
class PersonSession:
    person_uid: PersonUID
    current_yolo_track_id: int | None
    previous_yolo_track_id: int | None
    first_seen_frame: int
    last_seen_frame: int
    last_bbox: BBox | None
    last_bbox_center: Point | None
    last_trusted_feet_point: Point | None
    last_trusted_feet_source: str
    last_motion_point_hip: Point | None
    last_motion_point_shoulder: Point | None
    last_anchor_point: Point | None
    previous_anchor_point: Point | None
    last_path_position: float | None
    last_direction: str
    last_lane_side: str | None
    last_track_zone_state: str
    last_warnings: list[str] = field(default_factory=list)
    lost_frame_count: int = 0
    exit_confirm_hits: int = 0
    status: PersonSessionStatusLabel = "ACTIVE"
    lifecycle_state: PersonSessionLifecycleLabel = "ACTIVE_INSIDE"
    last_identity_status: IdentityStatusLabel = "NEW"
    last_identity_debug: str = "NEW"
    last_gate_reason: str = "ENTERED_BY_FEET"
    entry_reason: IdentityEntryReasonLabel = "NONE"
    last_relink_score: float | None = None
    last_relink_frame_gap: int = 0
    has_counted_enter: bool = False
    has_counted_exit: bool = False
    last_count_event_reason: CountEventReasonLabel = "NO_COUNT_EVENT"


@dataclass(slots=True)
class ReLinkCandidateScore:
    session: PersonSession
    score: float
    frame_gap: int
    score_reason: str
    reject_reason: str | None


@dataclass(slots=True)
class ReLinkDecision:
    state: ReLinkStateLabel
    best_candidate: ReLinkCandidateScore | None
    second_candidate: ReLinkCandidateScore | None
    score_gap: float | None
    confirm_hits: int = 0


@dataclass(slots=True)
class PendingReLink:
    yolo_track_id: int
    target_person_uid: PersonUID | None
    state: ReLinkStateLabel


@dataclass(slots=True)
class OccludedEntryStatus:
    candidate_age_frames: int
    age_target: int
    inside_frames: int
    inside_target: int
    motion_frames: int
    motion_target: int
    block_reason: str = "NONE"
    nearest_ghost_score: float | None = None
    nearest_active_iou: float | None = None
    nearest_active_distance: float | None = None


class PersonIdentityManager:
    """Quan ly stable person_id duoc gate boi trusted feet cho Windows/demo."""

    def __init__(self, config: CameraConfigDict):
        self.next_person_uid: PersonUID = 1
        self.person_sessions: dict[PersonUID, PersonSession] = {}
        self.candidate_sessions: dict[int, CandidateSession] = {}
        self.pending_relink_by_yolo_id: dict[int, PendingReLink] = {}
        self.yolo_to_person: dict[int, PersonUID] = {}
        self.new_person_guard_log_by_yolo_id: dict[int, str] = {}
        self.current_frame_index = -1
        self.seen_person_uids_this_frame: set[PersonUID] = set()
        self.seen_yolo_ids_this_frame: set[int] = set()
        self.total_entered_count = 0
        self.total_confirmed_entered_count = 0
        self.total_occluded_entered_count = 0
        self.total_exited_count = 0

        self.center_line: list[Point] = []
        raw_center_line = config.get("CENTER_LINE", [])
        if len(raw_center_line) >= 2:
            self.center_line = [
                (int(raw_center_line[0][0]), int(raw_center_line[0][1])),
                (int(raw_center_line[1][0]), int(raw_center_line[1][1])),
            ]

        self.stairs_poly = self._build_stairs_poly(config)

    @staticmethod
    def _build_stairs_poly(config: CameraConfigDict) -> np.ndarray | None:
        step_bottom = config.get("STEP_BOTTOM", [[0, 0], [0, 0]])
        step_top = config.get("STEP_TOP", [[0, 0], [0, 0]])
        if len(step_bottom) < 2 or len(step_top) < 2:
            return None

        bottom_left = [int(step_bottom[0][0]), int(step_bottom[0][1])]
        bottom_right = [int(step_bottom[1][0]), int(step_bottom[1][1])]
        top_right = [int(step_top[1][0]), int(step_top[1][1])]
        top_left = [int(step_top[0][0]), int(step_top[0][1])]
        return np.array(
            [bottom_left, bottom_right, top_right, top_left],
            np.int32,
        )

    @staticmethod
    def _format_person_uid(person_uid: PersonUID | None) -> str:
        if person_uid is None:
            return ""
        return f"P{person_uid:04d}"

    @staticmethod
    def _format_candidate_analysis_subject_id(yolo_track_id: int | None) -> str:
        if yolo_track_id is None:
            return ""
        return f"CANDIDATE_YOLO_{yolo_track_id}"

    @staticmethod
    def _format_candidate_analysis_label(yolo_track_id: int | None) -> str:
        if yolo_track_id is None:
            return "CANDIDATE NA"
        return f"CANDIDATE {yolo_track_id}"

    @staticmethod
    def _format_session_lifecycle_display(session_lifecycle: str) -> str:
        if session_lifecycle.startswith("CANDIDATE_"):
            return "CANDIDATE " + session_lifecycle.removeprefix("CANDIDATE_")
        return session_lifecycle.replace("_", " ")

    @staticmethod
    def _normalize_bbox(
        bbox: BBoxArray | BBox | None,
        features: PoseFeatures,
    ) -> BBox | None:
        feature_bbox = features.get("bbox")
        if feature_bbox is not None:
            return feature_bbox
        if bbox is None or len(bbox) < 4:
            return None
        return (
            int(bbox[0]),
            int(bbox[1]),
            int(bbox[2]),
            int(bbox[3]),
        )

    @staticmethod
    def _get_bbox_center(bbox: BBox | None) -> Point | None:
        if bbox is None:
            return None
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @staticmethod
    def _get_bbox_iou(
        bbox_a: BBox | None,
        bbox_b: BBox | None,
    ) -> float | None:
        if bbox_a is None or bbox_b is None:
            return None

        inter_x1 = max(bbox_a[0], bbox_b[0])
        inter_y1 = max(bbox_a[1], bbox_b[1])
        inter_x2 = min(bbox_a[2], bbox_b[2])
        inter_y2 = min(bbox_a[3], bbox_b[3])
        inter_width = max(0, inter_x2 - inter_x1)
        inter_height = max(0, inter_y2 - inter_y1)
        inter_area = inter_width * inter_height
        if inter_area <= 0:
            return 0.0

        area_a = max(0, bbox_a[2] - bbox_a[0]) * max(0, bbox_a[3] - bbox_a[1])
        area_b = max(0, bbox_b[2] - bbox_b[0]) * max(0, bbox_b[3] - bbox_b[1])
        union_area = area_a + area_b - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    @staticmethod
    def _get_bbox_size_ratio_diff(
        previous_bbox: BBox | None,
        current_bbox: BBox | None,
    ) -> float | None:
        if previous_bbox is None or current_bbox is None:
            return None

        previous_width = max(1, previous_bbox[2] - previous_bbox[0])
        previous_height = max(1, previous_bbox[3] - previous_bbox[1])
        current_width = max(1, current_bbox[2] - current_bbox[0])
        current_height = max(1, current_bbox[3] - current_bbox[1])

        width_diff = abs(current_width - previous_width) / previous_width
        height_diff = abs(current_height - previous_height) / previous_height
        return max(width_diff, height_diff)

    @staticmethod
    def _select_anchor_point(
        features: PoseFeatures,
        bbox: BBox | None,
    ) -> Point | None:
        return (
            features.get("feet_point")
            or features.get("monitor_point_hip")
            or features.get("monitor_point_shoulder")
            or features.get("bbox_center")
            or features.get("bbox_bottom_center")
            or PersonIdentityManager._get_bbox_center(bbox)
        )

    @staticmethod
    def _map_dy_to_direction(dy: int) -> str:
        if SETTINGS.camera.use_current_camera_angle:
            if SETTINGS.direction.sign_normal:
                if dy < -SETTINGS.direction.pixel_threshold:
                    return "UP"
                if dy > SETTINGS.direction.pixel_threshold:
                    return "DOWN"
                return "IDLE"
            if dy < -SETTINGS.direction.pixel_threshold:
                return "DOWN"
            if dy > SETTINGS.direction.pixel_threshold:
                return "UP"
            return "IDLE"

        if dy > SETTINGS.direction.pixel_threshold:
            return "DOWN"
        if dy < -SETTINGS.direction.pixel_threshold:
            return "UP"
        return "IDLE"

    @staticmethod
    def _distance(point_a: Point | None, point_b: Point | None) -> float | None:
        if point_a is None or point_b is None:
            return None
        return float(hypot(point_a[0] - point_b[0], point_a[1] - point_b[1]))

    @staticmethod
    def _is_trusted_feet_source(feet_source: str) -> bool:
        return feet_source in TRUSTED_FEET_SOURCES

    def _get_trusted_feet(
        self,
        features: PoseFeatures,
    ) -> tuple[Point | None, str]:
        feet_point = features.get("feet_point")
        feet_source = str(features.get("feet_point_source", "FEET_UNAVAILABLE"))
        if feet_point is None or not self._is_trusted_feet_source(feet_source):
            return None, "NO_TRUSTED_FEET"
        return feet_point, feet_source

    def _is_trusted_feet_inside(
        self,
        trusted_feet_point: Point | None,
    ) -> bool | None:
        if trusted_feet_point is None:
            return None
        if self.stairs_poly is None or len(self.stairs_poly) < 3:
            return None
        return bool(is_inside_stairs(self.stairs_poly, trusted_feet_point))

    @staticmethod
    def _get_candidate_feet_context(
        features: PoseFeatures,
        trusted_feet_point: Point | None,
    ) -> tuple[str, str]:
        feet_source = str(features.get("feet_point_source", "FEET_UNAVAILABLE"))
        if trusted_feet_point is not None:
            return feet_source, "NONE"

        feet_reason = str(
            features.get("feet_unavailable_reason", "NO_SHOULDER_NO_HIP")
        )
        if feet_reason == "NONE":
            return feet_source, "NO_TRUSTED_FEET"
        return feet_source, feet_reason

    @staticmethod
    def _normalize_candidate_gate_reason(gate_reason: str) -> str:
        if gate_reason == "NEW_PERSON_BLOCKED_PENDING_RELINK":
            return "BLOCKED_BY_PENDING_RELINK"
        if gate_reason == "NEW_PERSON_BLOCKED_GHOST_MATCH":
            return "BLOCKED_BY_GHOST_CANDIDATE"
        return gate_reason

    @staticmethod
    def _update_candidate_proof_state(
        candidate: CandidateSession,
        trusted_feet_point: Point | None,
        trusted_inside: bool | None,
    ) -> None:
        if trusted_feet_point is None:
            candidate.entry_confirm_hits = 0
            return

        if trusted_inside is False:
            candidate.has_trusted_feet_outside = True
            candidate.entry_confirm_hits = 0
            return

        if trusted_inside is True:
            if candidate.has_trusted_feet_outside:
                candidate.entry_confirm_hits += 1
            else:
                candidate.entry_confirm_hits = 0
            return

        candidate.entry_confirm_hits = 0

    @staticmethod
    def _update_candidate_occluded_state(
        candidate: CandidateSession,
        trusted_feet_point: Point | None,
        trusted_inside: bool | None,
    ) -> None:
        if trusted_feet_point is not None and trusted_inside is True:
            candidate.occluded_inside_frames += 1
        else:
            candidate.occluded_inside_frames = 0

        motion_distance = PersonIdentityManager._distance(
            candidate.previous_anchor_point,
            candidate.last_anchor_point,
        )
        if (
            motion_distance is not None
            and motion_distance >= OCCLUDED_ENTRY_MIN_MOTION_DELTA_PX
        ):
            candidate.occluded_motion_frames += 1

    def _get_candidate_age_frames(self, candidate: CandidateSession) -> int:
        return max(1, self.current_frame_index - candidate.first_seen_frame + 1)

    def _build_occluded_entry_status(
        self,
        candidate: CandidateSession,
        *,
        block_reason: str = "NONE",
        nearest_ghost_score: float | None = None,
        nearest_active_iou: float | None = None,
        nearest_active_distance: float | None = None,
    ) -> OccludedEntryStatus:
        return OccludedEntryStatus(
            candidate_age_frames=self._get_candidate_age_frames(candidate),
            age_target=max(1, SETTINGS.identity.occluded_entry_min_age_frames),
            inside_frames=candidate.occluded_inside_frames,
            inside_target=max(1, SETTINGS.identity.occluded_entry_min_inside_frames),
            motion_frames=candidate.occluded_motion_frames,
            motion_target=max(1, SETTINGS.identity.occluded_entry_min_motion_frames),
            block_reason=block_reason,
            nearest_ghost_score=nearest_ghost_score,
            nearest_active_iou=nearest_active_iou,
            nearest_active_distance=nearest_active_distance,
        )

    @staticmethod
    def _get_candidate_session_lifecycle(
        candidate: CandidateSession,
        trusted_feet_point: Point | None,
        trusted_inside: bool | None,
        *,
        allow_new_person: bool,
    ) -> PersonSessionLifecycleLabel:
        if trusted_feet_point is None:
            return "CANDIDATE_NO_FEET"
        if trusted_inside is False:
            return "CANDIDATE_OUTSIDE"
        if trusted_inside is True:
            if not candidate.has_trusted_feet_outside:
                return "OCCLUDED_ENTRY_CANDIDATE"
            if not allow_new_person:
                return "UNASSIGNED_INSIDE_CANDIDATE"
            return "CANDIDATE_WAIT_ENTER"
        return "UNASSIGNED_INSIDE_CANDIDATE"

    @staticmethod
    def _get_candidate_inside_test(
        trusted_feet_point: Point | None,
        trusted_inside: bool | None,
    ) -> str:
        if trusted_feet_point is None:
            return "NO_TRUSTED_FEET"
        if trusted_inside is True:
            return "INSIDE_ROI"
        if trusted_inside is False:
            return "OUTSIDE_ROI"
        return "ROI_UNAVAILABLE"

    def _get_detection_lane_side(self, features: PoseFeatures) -> str | None:
        if len(self.center_line) < 2:
            return None

        feet_point = features.get("feet_point") or features.get("inside_feet_point")
        if feet_point is None:
            return None

        side_value = compute_lane_side(feet_point, self.center_line)
        lane_side_label = get_lane_side_label(side_value)
        if lane_side_label == "ON_LINE":
            return "CENTER"
        if lane_side_label == "UNKNOWN":
            return None
        return lane_side_label

    def _get_path_position(self, point: Point | None) -> float | None:
        if point is None:
            return None

        if len(self.center_line) < 2:
            return float(point[1])

        start_point = self.center_line[0]
        end_point = self.center_line[1]
        line_dx = float(end_point[0] - start_point[0])
        line_dy = float(end_point[1] - start_point[1])
        line_length = hypot(line_dx, line_dy)
        if line_length <= 0.0:
            return float(point[1])

        relative_x = float(point[0] - start_point[0])
        relative_y = float(point[1] - start_point[1])
        return (relative_x * line_dx + relative_y * line_dy) / line_length

    def _get_detection_path_position(
        self,
        features: PoseFeatures,
        bbox: BBox | None,
    ) -> float | None:
        point = (
            features.get("feet_point")
            or features.get("inside_feet_point")
            or self._select_anchor_point(features, bbox)
        )
        return self._get_path_position(point)

    def _approx_direction_from_points(
        self,
        previous_point: Point | None,
        current_point: Point | None,
    ) -> str:
        if previous_point is None or current_point is None:
            return "UNKNOWN"
        dy = int(current_point[1] - previous_point[1])
        return self._map_dy_to_direction(dy)

    def _predict_anchor_point(
        self,
        session: PersonSession,
        frame_gap: int,
    ) -> Point | None:
        if session.last_anchor_point is None:
            return None
        if session.previous_anchor_point is None or frame_gap <= 1:
            return session.last_anchor_point

        dx = session.last_anchor_point[0] - session.previous_anchor_point[0]
        dy = session.last_anchor_point[1] - session.previous_anchor_point[1]
        projection_frames = min(frame_gap, 2)
        return (
            int(session.last_anchor_point[0] + dx * projection_frames),
            int(session.last_anchor_point[1] + dy * projection_frames),
        )

    def _log_event(self, message: str) -> None:
        if SETTINGS.identity.log_events:
            print(message)

    @staticmethod
    def _normalize_relink_log_value(value: float | None) -> str:
        if value is None or value == inf:
            return "NA"
        return f"{value:.2f}"

    def _is_bbox_center_inside_stairs(
        self,
        bbox: BBox | None,
    ) -> bool | None:
        bbox_center = self._get_bbox_center(bbox)
        if bbox_center is None or self.stairs_poly is None or len(self.stairs_poly) < 3:
            return None
        return bool(is_inside_stairs(self.stairs_poly, bbox_center))

    @staticmethod
    def _get_relink_block_reason(decision: ReLinkDecision) -> str:
        if decision.state == "RELINK_WAIT_MORE_FRAMES":
            return "LOST_GHOST_NEARBY"
        if decision.state == "RELINK_REJECTED_AMBIGUOUS":
            return "LOST_GHOST_AMBIGUOUS"
        if decision.state == "RELINK_REJECTED_ORDER_CONFLICT":
            return "LOST_GHOST_ORDER_CONFLICT"
        return "LOST_GHOST_NEARBY"

    @staticmethod
    def _format_occluded_log_metric(value: float | None) -> str:
        if value is None:
            return "NA"
        return f"{value:.2f}"

    def _get_active_person_overlap_status(
        self,
        candidate: CandidateSession,
        bbox: BBox | None,
    ) -> tuple[float | None, float | None, bool]:
        nearest_iou: float | None = None
        nearest_distance: float | None = None
        if not SETTINGS.identity.occluded_entry_block_if_near_active_person:
            return nearest_iou, nearest_distance, False

        candidate_center = (
            candidate.last_anchor_point
            or candidate.last_bbox_center
            or self._get_bbox_center(bbox)
        )
        iou_threshold = max(
            0.0,
            float(SETTINGS.identity.occluded_entry_near_active_iou_threshold),
        )
        distance_threshold = max(
            0.0,
            float(SETTINGS.identity.occluded_entry_near_active_distance_px),
        )
        block_overlap = False

        for session in self.person_sessions.values():
            if session.status != "ACTIVE" or session.lifecycle_state != "ACTIVE_INSIDE":
                continue

            current_iou = self._get_bbox_iou(bbox, session.last_bbox)
            if current_iou is not None and (
                nearest_iou is None or current_iou > nearest_iou
            ):
                nearest_iou = current_iou

            other_center = (
                session.last_anchor_point
                or session.last_bbox_center
                or self._get_bbox_center(session.last_bbox)
            )
            current_distance = self._distance(candidate_center, other_center)
            if current_distance is not None and (
                nearest_distance is None or current_distance < nearest_distance
            ):
                nearest_distance = current_distance

            iou_blocked = current_iou is not None and current_iou >= iou_threshold
            distance_blocked = (
                current_distance is not None
                and current_distance <= distance_threshold
            )
            if iou_blocked or distance_blocked:
                block_overlap = True

        return nearest_iou, nearest_distance, block_overlap

    def _set_session_count_event_reason(
        self,
        session: PersonSession,
        reason: CountEventReasonLabel,
        *,
        log_message: str | None = None,
        log_only_on_change: bool = False,
    ) -> CountEventReasonLabel:
        reason_changed = session.last_count_event_reason != reason
        session.last_count_event_reason = reason
        if log_message is not None and (not log_only_on_change or reason_changed):
            self._log_event(log_message)
        return reason

    def _count_session_enter(
        self,
        session: PersonSession,
    ) -> CountEventReasonLabel:
        if session.has_counted_enter:
            return self._set_session_count_event_reason(
                session,
                "ENTER_ALREADY_COUNTED",
            )

        session.has_counted_enter = True
        self.total_entered_count += 1
        if session.entry_reason == "OCCLUDED_ENTRY":
            self.total_occluded_entered_count += 1
            return self._set_session_count_event_reason(
                session,
                "COUNT_ENTER_OCCLUDED",
                log_message=(
                    "COUNT_ENTER_OCCLUDED "
                    f"person_id={self._format_person_uid(session.person_uid)} "
                    f"total_entered={self.total_entered_count} "
                    f"total_occluded_entered={self.total_occluded_entered_count}"
                ),
            )

        self.total_confirmed_entered_count += 1
        return self._set_session_count_event_reason(
            session,
            "ENTER_COUNTED_BY_FEET",
            log_message=(
                "COUNT_ENTER "
                f"person_id={self._format_person_uid(session.person_uid)} "
                f"total_entered={self.total_entered_count} "
                f"total_confirmed_entered={self.total_confirmed_entered_count}"
            ),
        )

    def _count_session_exit(
        self,
        session: PersonSession,
    ) -> CountEventReasonLabel:
        if session.has_counted_exit:
            return self._set_session_count_event_reason(
                session,
                "EXIT_ALREADY_COUNTED",
            )

        session.has_counted_exit = True
        self.total_exited_count += 1
        return self._set_session_count_event_reason(
            session,
            "EXIT_COUNTED_BY_FEET",
            log_message=(
                "COUNT_EXIT "
                f"person_id={self._format_person_uid(session.person_uid)} "
                f"total_exited={self.total_exited_count}"
            ),
        )

    def _clear_new_person_guard_log(
        self,
        yolo_track_id: int,
    ) -> None:
        self.new_person_guard_log_by_yolo_id.pop(yolo_track_id, None)

    def _log_new_person_guard(
        self,
        yolo_track_id: int,
        reason: str,
        detail: str = "",
    ) -> None:
        signature = f"{reason}|{detail}"
        if self.new_person_guard_log_by_yolo_id.get(yolo_track_id) == signature:
            return

        self.new_person_guard_log_by_yolo_id[yolo_track_id] = signature
        suffix = f" {detail}" if detail else ""
        self._log_event(f"{reason} yolo_id={yolo_track_id}{suffix}")

    def begin_frame(self, frame_index: int) -> None:
        self.current_frame_index = frame_index
        self.seen_person_uids_this_frame = set()
        self.seen_yolo_ids_this_frame = set()

    def _build_identity_result(
        self,
        has_active_person_id: bool,
        person_uid: PersonUID | None,
        yolo_track_id: int | None,
        previous_yolo_track_id: int | None,
        identity_status: IdentityStatusLabel,
        session_status: PersonSessionStatusLabel,
        session_lifecycle: PersonSessionLifecycleLabel,
        identity_debug: str,
        identity_feet_source: str,
        identity_gate_reason: str,
        relink_score: float | None,
        relink_frame_gap: int,
        identity_feet_reason: str = "NONE",
        identity_inside_test: str = "UNKNOWN",
        identity_entry_reason: IdentityEntryReasonLabel = "NONE",
        identity_outside_proof: bool = False,
        identity_enter_confirm_hits: int = 0,
        identity_enter_confirm_target: int = 0,
        relink_score_gap: float | None = None,
        relink_best_candidate: str = "",
        relink_second_candidate: str = "",
        relink_state: ReLinkStateLabel = "NONE",
        has_counted_enter: bool = False,
        has_counted_exit: bool = False,
        count_event_reason: CountEventReasonLabel = "NO_COUNT_EVENT",
        merged_from_analysis_subject_id: str = "",
        occluded_entry_status: OccludedEntryStatus | None = None,
    ) -> IdentityUpdateResult:
        person_uid_label = self._format_person_uid(person_uid)
        candidate_subject_id = self._format_candidate_analysis_subject_id(yolo_track_id)
        analysis_subject_id = person_uid_label or candidate_subject_id
        analysis_subject_label = (
            person_uid_label
            if person_uid_label
            else self._format_candidate_analysis_label(yolo_track_id)
        )
        effective_occluded_status = (
            occluded_entry_status
            if occluded_entry_status is not None
            else OccludedEntryStatus(
                candidate_age_frames=0,
                age_target=max(1, SETTINGS.identity.occluded_entry_min_age_frames),
                inside_frames=0,
                inside_target=max(1, SETTINGS.identity.occluded_entry_min_inside_frames),
                motion_frames=0,
                motion_target=max(1, SETTINGS.identity.occluded_entry_min_motion_frames),
            )
        )
        return IdentityUpdateResult(
            has_active_person_id=has_active_person_id,
            person_uid=person_uid,
            person_uid_label=person_uid_label,
            analysis_subject_id=analysis_subject_id,
            analysis_subject_label=analysis_subject_label,
            merged_from_analysis_subject_id=merged_from_analysis_subject_id,
            yolo_track_id=yolo_track_id,
            previous_yolo_track_id=previous_yolo_track_id,
            identity_status=identity_status,
            session_status=session_status,
            session_lifecycle=session_lifecycle,
            identity_debug=identity_debug,
            identity_feet_source=identity_feet_source,
            identity_feet_reason=identity_feet_reason,
            identity_gate_reason=identity_gate_reason,
            identity_inside_test=identity_inside_test,
            identity_entry_reason=identity_entry_reason,
            identity_outside_proof=identity_outside_proof,
            identity_enter_confirm_hits=identity_enter_confirm_hits,
            identity_enter_confirm_target=identity_enter_confirm_target,
            relink_score=relink_score,
            relink_frame_gap=relink_frame_gap,
            relink_score_gap=relink_score_gap,
            relink_best_candidate=relink_best_candidate,
            relink_second_candidate=relink_second_candidate,
            relink_state=relink_state,
            has_counted_enter=has_counted_enter,
            has_counted_exit=has_counted_exit,
            count_event_reason=count_event_reason,
            total_confirmed_entered_count=self.total_confirmed_entered_count,
            total_occluded_entered_count=self.total_occluded_entered_count,
            occluded_entry_candidate_age_frames=effective_occluded_status.candidate_age_frames,
            occluded_entry_age_target=effective_occluded_status.age_target,
            occluded_entry_inside_frames=effective_occluded_status.inside_frames,
            occluded_entry_inside_target=effective_occluded_status.inside_target,
            occluded_entry_motion_frames=effective_occluded_status.motion_frames,
            occluded_entry_motion_target=effective_occluded_status.motion_target,
            occluded_entry_block_reason=effective_occluded_status.block_reason,
            occluded_entry_nearest_ghost_score=effective_occluded_status.nearest_ghost_score,
            occluded_entry_nearest_active_iou=effective_occluded_status.nearest_active_iou,
            occluded_entry_nearest_active_distance=effective_occluded_status.nearest_active_distance,
        )

    def _update_person_session_detection(
        self,
        session: PersonSession,
        yolo_track_id: int,
        bbox: BBox | None,
        features: PoseFeatures,
        identity_status: IdentityStatusLabel,
        relink_score: float | None,
        relink_frame_gap: int,
        trusted_feet_point: Point | None,
        trusted_feet_source: str,
    ) -> None:
        anchor_point = self._select_anchor_point(features, bbox)
        bbox_center = self._get_bbox_center(bbox)
        path_position = self._get_detection_path_position(features, bbox)

        if session.last_seen_frame != self.current_frame_index:
            session.previous_anchor_point = session.last_anchor_point

        session.current_yolo_track_id = yolo_track_id
        session.last_seen_frame = self.current_frame_index
        session.last_bbox = bbox
        session.last_bbox_center = bbox_center
        session.last_trusted_feet_point = trusted_feet_point
        session.last_trusted_feet_source = trusted_feet_source
        session.last_motion_point_hip = features.get("monitor_point_hip")
        session.last_motion_point_shoulder = features.get("monitor_point_shoulder")
        session.last_anchor_point = anchor_point
        session.last_path_position = path_position
        session.lost_frame_count = 0
        session.status = "ACTIVE"
        session.last_identity_status = identity_status
        session.last_relink_score = relink_score
        session.last_relink_frame_gap = relink_frame_gap

        self.yolo_to_person[yolo_track_id] = session.person_uid
        self.seen_person_uids_this_frame.add(session.person_uid)

    def _create_candidate_result(
        self,
        yolo_track_id: int,
        candidate: CandidateSession,
        session_lifecycle: PersonSessionLifecycleLabel,
        feet_source: str,
        feet_reason: str,
        gate_reason: str,
        inside_test: str,
        occluded_entry_status: OccludedEntryStatus | None = None,
    ) -> IdentityUpdateResult:
        return self._build_identity_result(
            has_active_person_id=False,
            person_uid=None,
            yolo_track_id=yolo_track_id,
            previous_yolo_track_id=None,
            identity_status="CANDIDATE",
            session_status="CANDIDATE",
            session_lifecycle=session_lifecycle,
            identity_debug=f"YOLO {yolo_track_id}",
            identity_feet_source=feet_source,
            identity_feet_reason=feet_reason,
            identity_gate_reason=gate_reason,
            identity_inside_test=inside_test,
            identity_entry_reason="NONE",
            identity_outside_proof=candidate.has_trusted_feet_outside,
            identity_enter_confirm_hits=candidate.entry_confirm_hits,
            identity_enter_confirm_target=(
                SETTINGS.identity.entry_confirm_frames
                if candidate.has_trusted_feet_outside
                else 0
            ),
            relink_score=None,
            relink_frame_gap=0,
            occluded_entry_status=occluded_entry_status,
        )

    def _promote_candidate(
        self,
        candidate: CandidateSession,
        features: PoseFeatures,
        *,
        entry_reason: IdentityEntryReasonLabel,
    ) -> PersonSession:
        person_uid = self.next_person_uid
        self.next_person_uid += 1

        session = PersonSession(
            person_uid=person_uid,
            current_yolo_track_id=candidate.current_yolo_track_id,
            previous_yolo_track_id=None,
            first_seen_frame=candidate.first_seen_frame,
            last_seen_frame=candidate.last_seen_frame,
            last_bbox=candidate.last_bbox,
            last_bbox_center=candidate.last_bbox_center,
            last_trusted_feet_point=candidate.last_trusted_feet_point,
            last_trusted_feet_source=candidate.last_trusted_feet_source,
            last_motion_point_hip=features.get("monitor_point_hip"),
            last_motion_point_shoulder=features.get("monitor_point_shoulder"),
            last_anchor_point=candidate.last_anchor_point,
            previous_anchor_point=candidate.previous_anchor_point,
            last_path_position=candidate.last_path_position,
            last_direction="UNKNOWN",
            last_lane_side=None,
            last_track_zone_state="UNKNOWN",
            status="ACTIVE",
            lifecycle_state="ACTIVE_INSIDE",
            last_identity_status="NEW",
            last_identity_debug=f"ENTER {self._format_person_uid(person_uid)}",
            last_gate_reason=candidate.last_gate_reason,
            entry_reason=entry_reason,
            has_counted_enter=False,
            has_counted_exit=False,
            last_count_event_reason="NO_COUNT_EVENT",
        )
        self.person_sessions[person_uid] = session
        self.yolo_to_person[candidate.current_yolo_track_id] = person_uid
        self._count_session_enter(session)
        if entry_reason == "OCCLUDED_ENTRY":
            self._log_event(
                "PERSON_CREATED_OCCLUDED_ENTRY "
                f"person_id={self._format_person_uid(person_uid)} "
                f"yolo_id={candidate.current_yolo_track_id} "
                f"candidate_age={self._get_candidate_age_frames(candidate)} "
                f"inside_count={candidate.occluded_inside_frames} "
                f"motion_count={candidate.occluded_motion_frames} "
                f"feet_source={candidate.last_trusted_feet_source}"
            )
        else:
            self._log_event(
                "PERSON_ENTERED "
                f"person_id={self._format_person_uid(person_uid)} "
                f"yolo_id={candidate.current_yolo_track_id}"
            )
        self.pending_relink_by_yolo_id.pop(candidate.current_yolo_track_id, None)
        self._clear_new_person_guard_log(candidate.current_yolo_track_id)
        self.candidate_sessions.pop(candidate.current_yolo_track_id, None)
        return session

    def _build_promoted_candidate_result(
        self,
        session: PersonSession,
        yolo_track_id: int,
        trusted_feet_source: str,
        inside_test: str,
        *,
        identity_outside_proof: bool,
        identity_enter_confirm_hits: int,
        identity_enter_confirm_target: int,
        merge_from_candidate_history: bool,
        occluded_entry_status: OccludedEntryStatus | None = None,
    ) -> IdentityUpdateResult:
        return self._build_identity_result(
            has_active_person_id=True,
            person_uid=session.person_uid,
            yolo_track_id=yolo_track_id,
            previous_yolo_track_id=None,
            identity_status="NEW",
            session_status="ACTIVE",
            session_lifecycle="ACTIVE_INSIDE",
            identity_debug=f"{self._format_person_uid(session.person_uid)} / YOLO {yolo_track_id}",
            identity_feet_source=trusted_feet_source,
            identity_gate_reason=session.last_gate_reason,
            relink_score=None,
            relink_frame_gap=0,
            identity_inside_test=inside_test,
            identity_entry_reason=session.entry_reason,
            identity_outside_proof=identity_outside_proof,
            identity_enter_confirm_hits=identity_enter_confirm_hits,
            identity_enter_confirm_target=identity_enter_confirm_target,
            has_counted_enter=session.has_counted_enter,
            has_counted_exit=session.has_counted_exit,
            count_event_reason=session.last_count_event_reason,
            merged_from_analysis_subject_id=(
                self._format_candidate_analysis_subject_id(yolo_track_id)
                if merge_from_candidate_history
                else ""
            ),
            occluded_entry_status=occluded_entry_status,
        )

    def _reset_candidate_relink_state(self, candidate: CandidateSession) -> None:
        candidate.relink_confirm_hits = 0
        candidate.pending_relink_person_uid = None
        candidate.pending_relink_best_score = None
        candidate.pending_relink_second_score = None
        candidate.pending_relink_score_gap = None
        candidate.pending_relink_frame_gap = 0
        candidate.pending_relink_best_label = ""
        candidate.pending_relink_second_label = ""
        candidate.pending_relink_state = "NONE"
        candidate.last_relink_log_signature = ""
        self.pending_relink_by_yolo_id.pop(candidate.current_yolo_track_id, None)

    def _set_pending_relink(
        self,
        yolo_track_id: int,
        target_person_uid: PersonUID | None,
        state: ReLinkStateLabel,
    ) -> None:
        self.pending_relink_by_yolo_id[yolo_track_id] = PendingReLink(
            yolo_track_id=yolo_track_id,
            target_person_uid=target_person_uid,
            state=state,
        )

    def _format_relink_candidate_label(
        self,
        candidate_score: ReLinkCandidateScore | None,
    ) -> str:
        if candidate_score is None:
            return ""
        return self._format_person_uid(candidate_score.session.person_uid)

    def _upsert_candidate_session(
        self,
        yolo_track_id: int,
        bbox: BBox | None,
        features: PoseFeatures,
        trusted_feet_point: Point | None,
        trusted_feet_source: str,
    ) -> CandidateSession:
        candidate = self.candidate_sessions.get(yolo_track_id)
        anchor_point = self._select_anchor_point(features, bbox)
        bbox_center = self._get_bbox_center(bbox)
        path_position = self._get_detection_path_position(features, bbox)

        if candidate is None:
            candidate = CandidateSession(
                current_yolo_track_id=yolo_track_id,
                first_seen_frame=self.current_frame_index,
                last_seen_frame=self.current_frame_index,
                last_bbox=bbox,
                last_bbox_center=bbox_center,
                last_anchor_point=anchor_point,
                previous_anchor_point=None,
                last_path_position=path_position,
                last_trusted_feet_point=trusted_feet_point,
                last_trusted_feet_source=trusted_feet_source,
            )
            self.candidate_sessions[yolo_track_id] = candidate
            self._log_event(f"CANDIDATE_SEEN yolo_id={yolo_track_id}")
            return candidate

        if candidate.last_seen_frame != self.current_frame_index:
            candidate.previous_anchor_point = candidate.last_anchor_point

        candidate.last_seen_frame = self.current_frame_index
        candidate.last_bbox = bbox
        candidate.last_bbox_center = bbox_center
        candidate.last_anchor_point = anchor_point
        candidate.last_path_position = path_position
        candidate.last_trusted_feet_point = trusted_feet_point
        candidate.last_trusted_feet_source = trusted_feet_source
        return candidate

    def _get_order_conflict_reason(
        self,
        session: PersonSession,
        current_path_position: float | None,
    ) -> str | None:
        if current_path_position is None or session.last_path_position is None:
            return None

        min_separation = max(1, SETTINGS.identity.relink_order_min_separation_px)
        for other_session in self.person_sessions.values():
            if other_session.person_uid == session.person_uid:
                continue
            if (
                other_session.status == "COMPLETED"
                or other_session.lifecycle_state == "EXITED"
                or other_session.last_path_position is None
            ):
                continue

            previous_delta = session.last_path_position - other_session.last_path_position
            current_delta = current_path_position - other_session.last_path_position
            if (
                abs(previous_delta) < min_separation
                or abs(current_delta) < min_separation
            ):
                continue
            if previous_delta * current_delta < 0:
                return self._format_person_uid(other_session.person_uid)
        return None

    def _get_same_yolo_lost_matches(
        self,
        yolo_track_id: int,
    ) -> list[PersonSession]:
        same_yolo_matches: list[PersonSession] = []
        same_yolo_relink_window = (
            SETTINGS.identity.max_lost_frames
            + SETTINGS.identity.lost_inside_extra_frames
        )
        for session in self.person_sessions.values():
            if session.person_uid in self.seen_person_uids_this_frame:
                continue
            if session.status != "LOST" or session.lifecycle_state == "EXITED":
                continue
            if session.previous_yolo_track_id != yolo_track_id:
                continue
            frame_gap = self.current_frame_index - session.last_seen_frame
            if frame_gap <= 0 or frame_gap > same_yolo_relink_window:
                continue
            same_yolo_matches.append(session)
        return same_yolo_matches

    def _commit_same_yolo_relink(
        self,
        session: PersonSession,
        yolo_track_id: int,
        bbox: BBox | None,
        features: PoseFeatures,
        trusted_feet_point: Point | None,
        trusted_feet_source: str,
        trusted_inside: bool | None,
    ) -> IdentityUpdateResult:
        frame_gap = self.current_frame_index - session.last_seen_frame
        self._update_person_session_detection(
            session,
            yolo_track_id,
            bbox,
            features,
            identity_status="RELINKED",
            relink_score=0.0,
            relink_frame_gap=frame_gap,
            trusted_feet_point=trusted_feet_point,
            trusted_feet_source=trusted_feet_source,
        )

        if trusted_inside is True:
            session.lifecycle_state = "ACTIVE_INSIDE"
            session.last_gate_reason = "RELINKED_ACTIVE_INSIDE"
            session.last_identity_debug = (
                f"{self._format_person_uid(session.person_uid)} / YOLO {yolo_track_id}"
            )
            has_active_person_id = True
        else:
            session.lifecycle_state = "LOST_INSIDE"
            session.last_gate_reason = "RELINKED_WAIT_TRUSTED_FEET"
            session.last_identity_debug = f"RELINK {self._format_person_uid(session.person_uid)}"
            has_active_person_id = False

        session.exit_confirm_hits = 0
        count_event_reason = self._set_session_count_event_reason(
            session,
            "RELINK_NO_RECOUNT",
            log_message=(
                "RELINK_NO_RECOUNT "
                f"person_id={self._format_person_uid(session.person_uid)} "
                f"old_yolo={yolo_track_id} new_yolo={yolo_track_id}"
            ),
        )
        self.pending_relink_by_yolo_id.pop(yolo_track_id, None)
        self.candidate_sessions.pop(yolo_track_id, None)
        self._clear_new_person_guard_log(yolo_track_id)
        self._log_event(
            "SAME_YOLO_RELINK "
            f"person_id={self._format_person_uid(session.person_uid)} "
            f"yolo_id={yolo_track_id}"
        )
        return self._build_identity_result(
            has_active_person_id=has_active_person_id,
            person_uid=session.person_uid,
            yolo_track_id=yolo_track_id,
            previous_yolo_track_id=yolo_track_id,
            identity_status="RELINKED",
            session_status=session.status,
            session_lifecycle=session.lifecycle_state,
            identity_debug=session.last_identity_debug,
            identity_feet_source=trusted_feet_source,
            identity_gate_reason=session.last_gate_reason,
            identity_entry_reason=session.entry_reason,
            relink_score=0.0,
            relink_frame_gap=frame_gap,
            relink_score_gap=None,
            relink_best_candidate=self._format_person_uid(session.person_uid),
            relink_second_candidate="",
            relink_state="RELINK_CONFIRMED",
            has_counted_enter=session.has_counted_enter,
            has_counted_exit=session.has_counted_exit,
            count_event_reason=count_event_reason,
            merged_from_analysis_subject_id=self._format_candidate_analysis_subject_id(
                yolo_track_id
            ),
        )

    def score_relink_candidate(
        self,
        session: PersonSession,
        bbox: BBox | None,
        features: PoseFeatures,
    ) -> ReLinkCandidateScore:
        if session.status != "LOST" or session.lifecycle_state == "EXITED":
            return ReLinkCandidateScore(
                session=session,
                score=inf,
                frame_gap=0,
                score_reason="SESSION_NOT_RELINKABLE",
                reject_reason="SESSION_NOT_RELINKABLE",
            )

        frame_gap = self.current_frame_index - session.last_seen_frame
        if frame_gap <= 0:
            return ReLinkCandidateScore(
                session=session,
                score=inf,
                frame_gap=frame_gap,
                score_reason="FRAME_GAP_INVALID",
                reject_reason="FRAME_GAP_INVALID",
            )

        if frame_gap > SETTINGS.identity.relink_max_lost_frames:
            return ReLinkCandidateScore(
                session=session,
                score=inf,
                frame_gap=frame_gap,
                score_reason="FRAME_GAP_TOO_LARGE",
                reject_reason="FRAME_GAP_TOO_LARGE",
            )

        current_anchor = self._select_anchor_point(features, bbox)
        predicted_anchor = self._predict_anchor_point(session, frame_gap)
        current_bbox_center = self._get_bbox_center(bbox)
        current_path_position = self._get_detection_path_position(features, bbox)
        anchor_distance = self._distance(predicted_anchor, current_anchor)
        bbox_center_distance = self._distance(
            session.last_bbox_center,
            current_bbox_center,
        )

        if anchor_distance is None and bbox_center_distance is None:
            return ReLinkCandidateScore(
                session=session,
                score=inf,
                frame_gap=frame_gap,
                score_reason="NO_DISTANCE_REFERENCE",
                reject_reason="NO_DISTANCE_REFERENCE",
            )

        distance_reference = (
            anchor_distance if anchor_distance is not None else bbox_center_distance
        )
        if (
            distance_reference is not None
            and distance_reference > SETTINGS.identity.relink_max_distance_px
        ):
            return ReLinkCandidateScore(
                session=session,
                score=inf,
                frame_gap=frame_gap,
                score_reason="DISTANCE_TOO_LARGE",
                reject_reason="DISTANCE_TOO_LARGE",
            )

        bbox_ratio_diff = self._get_bbox_size_ratio_diff(
            session.last_bbox,
            bbox,
        )
        if (
            bbox_ratio_diff is not None
            and bbox_ratio_diff > SETTINGS.identity.relink_max_bbox_size_ratio_diff
        ):
            return ReLinkCandidateScore(
                session=session,
                score=inf,
                frame_gap=frame_gap,
                score_reason="BBOX_RATIO_TOO_LARGE",
                reject_reason="BBOX_RATIO_TOO_LARGE",
            )

        score = 0.0
        reason_parts: list[str] = []
        reject_reason: str | None = None
        max_distance = max(1, SETTINGS.identity.relink_max_distance_px)

        if anchor_distance is not None:
            score += anchor_distance / max_distance
            reason_parts.append(f"d={anchor_distance:.1f}")
        elif bbox_center_distance is not None:
            score += bbox_center_distance / max_distance
            reason_parts.append(f"c={bbox_center_distance:.1f}")

        if bbox_center_distance is not None:
            score += 0.25 * min(2.0, bbox_center_distance / max_distance)
            reason_parts.append(f"center={bbox_center_distance:.1f}")

        score += 0.35 * (
            frame_gap / max(1, SETTINGS.identity.relink_max_lost_frames)
        )
        reason_parts.append(f"gap={frame_gap}")

        if bbox_ratio_diff is not None:
            score += 0.50 * bbox_ratio_diff
            reason_parts.append(f"bbox={bbox_ratio_diff:.2f}")

        approximate_direction = self._approx_direction_from_points(
            session.last_anchor_point,
            current_anchor,
        )
        if (
            session.last_direction in ("UP", "DOWN")
            and approximate_direction in ("UP", "DOWN")
            and approximate_direction != session.last_direction
        ):
            score += 0.35
            reason_parts.append("dir=conflict")

        current_lane_side = self._get_detection_lane_side(features)
        if (
            session.last_lane_side not in (None, "UNKNOWN", "CENTER")
            and current_lane_side not in (None, "UNKNOWN", "CENTER")
            and current_lane_side != session.last_lane_side
        ):
            score += 0.15
            reason_parts.append("lane=shift")

        order_conflict_person = self._get_order_conflict_reason(
            session,
            current_path_position,
        )
        if order_conflict_person:
            reject_reason = "RELINK_REJECTED_ORDER_CONFLICT"
            reason_parts.append(f"order={order_conflict_person}")

        return ReLinkCandidateScore(
            session=session,
            score=score,
            frame_gap=frame_gap,
            score_reason=", ".join(reason_parts),
            reject_reason=reject_reason,
        )

    def _store_candidate_relink_decision(
        self,
        candidate: CandidateSession,
        decision: ReLinkDecision,
    ) -> None:
        best_candidate = decision.best_candidate
        second_candidate = decision.second_candidate
        best_label = self._format_relink_candidate_label(best_candidate)
        second_label = self._format_relink_candidate_label(second_candidate)

        candidate.pending_relink_best_score = (
            best_candidate.score if best_candidate is not None else None
        )
        candidate.pending_relink_second_score = (
            second_candidate.score if second_candidate is not None else None
        )
        candidate.pending_relink_score_gap = decision.score_gap
        candidate.pending_relink_frame_gap = (
            best_candidate.frame_gap if best_candidate is not None else 0
        )
        candidate.pending_relink_best_label = best_label
        candidate.pending_relink_second_label = second_label
        candidate.pending_relink_state = decision.state

        if decision.state == "RELINK_WAIT_MORE_FRAMES" and best_candidate is not None:
            candidate.pending_relink_person_uid = best_candidate.session.person_uid
            candidate.relink_confirm_hits = decision.confirm_hits
            self._set_pending_relink(
                candidate.current_yolo_track_id,
                best_candidate.session.person_uid,
                decision.state,
            )
        elif (
            decision.state in ("RELINK_REJECTED_AMBIGUOUS", "RELINK_REJECTED_ORDER_CONFLICT")
            and best_candidate is not None
        ):
            candidate.pending_relink_person_uid = best_candidate.session.person_uid
            candidate.relink_confirm_hits = 0
            self._set_pending_relink(
                candidate.current_yolo_track_id,
                best_candidate.session.person_uid,
                decision.state,
            )
        else:
            candidate.pending_relink_person_uid = None
            candidate.relink_confirm_hits = 0
            self.pending_relink_by_yolo_id.pop(candidate.current_yolo_track_id, None)

    def _log_relink_decision(
        self,
        yolo_track_id: int,
        candidate: CandidateSession,
        decision: ReLinkDecision,
    ) -> None:
        best_candidate = decision.best_candidate
        if best_candidate is None:
            return

        second_candidate = decision.second_candidate
        best_label = self._format_relink_candidate_label(best_candidate) or "NA"
        second_label = self._format_relink_candidate_label(second_candidate) or "NA"
        best_score_text = self._normalize_relink_log_value(best_candidate.score)
        second_score_text = self._normalize_relink_log_value(
            second_candidate.score if second_candidate is not None else None
        )
        gap_text = self._normalize_relink_log_value(decision.score_gap)
        signature = "|".join(
            (
                decision.state,
                best_label,
                second_label,
                best_score_text,
                second_score_text,
                gap_text,
            )
        )
        if candidate.last_relink_log_signature == signature:
            return

        candidate.last_relink_log_signature = signature
        self._log_event(
            "RELINK_CANDIDATES "
            f"yolo_id={yolo_track_id} best={best_label} score={best_score_text} "
            f"second={second_label} second_score={second_score_text} gap={gap_text}"
        )

        if decision.state == "RELINK_WAIT_MORE_FRAMES":
            self._log_event(f"RELINK_WAIT_MORE_FRAMES yolo_id={yolo_track_id}")
        elif decision.state == "RELINK_REJECTED_AMBIGUOUS":
            self._log_event(f"RELINK_REJECTED_AMBIGUOUS yolo_id={yolo_track_id}")
        elif decision.state == "RELINK_REJECTED_ORDER_CONFLICT":
            self._log_event(
                "RELINK_REJECTED_ORDER_CONFLICT "
                f"person_id={best_label} yolo_id={yolo_track_id}"
            )

    def _evaluate_relink_decision(
        self,
        candidate: CandidateSession,
        bbox: BBox | None,
        features: PoseFeatures,
    ) -> ReLinkDecision | None:
        scored_candidates: list[ReLinkCandidateScore] = []
        for session in self.person_sessions.values():
            if session.person_uid in self.seen_person_uids_this_frame:
                continue

            candidate_score = self.score_relink_candidate(
                session,
                bbox,
                features,
            )
            if candidate_score.score == inf and candidate_score.reject_reason is None:
                continue
            scored_candidates.append(candidate_score)

        if not scored_candidates:
            self._reset_candidate_relink_state(candidate)
            return None

        scored_candidates.sort(key=lambda item: item.score)
        valid_candidates = [
            candidate_score
            for candidate_score in scored_candidates
            if candidate_score.reject_reason is None and candidate_score.score != inf
        ]

        if not valid_candidates:
            self._reset_candidate_relink_state(candidate)
            best_candidate = scored_candidates[0]
            second_candidate = (
                scored_candidates[1] if len(scored_candidates) > 1 else None
            )
            if best_candidate.reject_reason == "RELINK_REJECTED_ORDER_CONFLICT":
                return ReLinkDecision(
                    state="RELINK_REJECTED_ORDER_CONFLICT",
                    best_candidate=best_candidate,
                    second_candidate=second_candidate,
                    score_gap=None,
                )
            return None

        best_candidate = valid_candidates[0]
        second_candidate = valid_candidates[1] if len(valid_candidates) > 1 else None
        if best_candidate.score > SETTINGS.identity.relink_score_threshold:
            self._reset_candidate_relink_state(candidate)
            return None

        score_gap = (
            second_candidate.score - best_candidate.score
            if second_candidate is not None
            else None
        )
        if (
            score_gap is not None
            and score_gap < SETTINGS.identity.relink_ambiguity_margin
        ):
            candidate.pending_relink_person_uid = None
            candidate.relink_confirm_hits = 0
            return ReLinkDecision(
                state="RELINK_REJECTED_AMBIGUOUS",
                best_candidate=best_candidate,
                second_candidate=second_candidate,
                score_gap=score_gap,
            )

        confirm_hits = (
            candidate.relink_confirm_hits + 1
            if candidate.pending_relink_person_uid == best_candidate.session.person_uid
            else 1
        )
        if confirm_hits < SETTINGS.identity.relink_min_confirm_frames:
            return ReLinkDecision(
                state="RELINK_WAIT_MORE_FRAMES",
                best_candidate=best_candidate,
                second_candidate=second_candidate,
                score_gap=score_gap,
                confirm_hits=confirm_hits,
            )

        return ReLinkDecision(
            state="RELINK_CONFIRMED",
            best_candidate=best_candidate,
            second_candidate=second_candidate,
            score_gap=score_gap,
            confirm_hits=confirm_hits,
        )

    def _build_pending_relink_result(
        self,
        yolo_track_id: int,
        feet_source: str,
        feet_reason: str,
        inside_test: str,
        candidate: CandidateSession,
        decision: ReLinkDecision,
    ) -> IdentityUpdateResult:
        best_candidate = decision.best_candidate
        second_candidate = decision.second_candidate
        relink_score = best_candidate.score if best_candidate is not None else None
        relink_frame_gap = best_candidate.frame_gap if best_candidate is not None else 0
        best_label = self._format_relink_candidate_label(best_candidate)
        second_label = self._format_relink_candidate_label(second_candidate)
        is_inside_without_outside_proof = (
            inside_test == "INSIDE_ROI" and not candidate.has_trusted_feet_outside
        )

        if decision.state == "RELINK_WAIT_MORE_FRAMES":
            identity_status: IdentityStatusLabel = "TEMP_REID_CANDIDATE"
            session_lifecycle: PersonSessionLifecycleLabel = "TEMP_REID_CANDIDATE"
            identity_debug = f"TEMP YOLO {yolo_track_id}"
            gate_reason = (
                "OCCLUDED_ENTRY_BLOCKED_GHOST"
                if is_inside_without_outside_proof
                else (
                    "RELINK_WAIT_MORE_FRAMES "
                    f"{decision.confirm_hits}/{SETTINGS.identity.relink_min_confirm_frames}"
                )
            )
        else:
            identity_status = "AMBIGUOUS_REID"
            session_lifecycle = "AMBIGUOUS_REID"
            identity_debug = f"AMBIGUOUS YOLO {yolo_track_id}"
            gate_reason = (
                "OCCLUDED_ENTRY_BLOCKED_GHOST"
                if is_inside_without_outside_proof
                else decision.state
            )

        if is_inside_without_outside_proof:
            occluded_block_reason = self._get_relink_block_reason(decision)
            candidate.last_gate_reason = gate_reason
            return self._build_identity_result(
                has_active_person_id=False,
                person_uid=None,
                yolo_track_id=yolo_track_id,
                previous_yolo_track_id=None,
                identity_status="CANDIDATE",
                session_status="CANDIDATE",
                session_lifecycle="OCCLUDED_ENTRY_CANDIDATE",
                identity_debug=f"YOLO {yolo_track_id}",
                identity_feet_source=feet_source,
                identity_feet_reason=feet_reason,
                identity_gate_reason=gate_reason,
                identity_inside_test=inside_test,
                identity_entry_reason="NONE",
                identity_outside_proof=False,
                identity_enter_confirm_hits=0,
                identity_enter_confirm_target=0,
                relink_score=relink_score,
                relink_frame_gap=relink_frame_gap,
                relink_score_gap=decision.score_gap,
                relink_best_candidate=best_label,
                relink_second_candidate=second_label,
                relink_state=decision.state,
                occluded_entry_status=self._build_occluded_entry_status(
                    candidate,
                    block_reason=occluded_block_reason,
                    nearest_ghost_score=relink_score,
                ),
            )

        return self._build_identity_result(
            has_active_person_id=False,
            person_uid=None,
            yolo_track_id=yolo_track_id,
            previous_yolo_track_id=None,
            identity_status=identity_status,
            session_status="CANDIDATE",
            session_lifecycle=session_lifecycle,
            identity_debug=identity_debug,
            identity_feet_source=feet_source,
            identity_feet_reason=feet_reason,
            identity_gate_reason=gate_reason,
            identity_inside_test=inside_test,
            identity_entry_reason="NONE",
            identity_outside_proof=candidate.has_trusted_feet_outside,
            identity_enter_confirm_hits=candidate.entry_confirm_hits,
            identity_enter_confirm_target=(
                SETTINGS.identity.entry_confirm_frames
                if candidate.has_trusted_feet_outside
                else 0
            ),
            relink_score=relink_score,
            relink_frame_gap=relink_frame_gap,
            relink_score_gap=decision.score_gap,
            relink_best_candidate=best_label,
            relink_second_candidate=second_label,
            relink_state=decision.state,
        )

    def _commit_relink_decision(
        self,
        yolo_track_id: int,
        bbox: BBox | None,
        features: PoseFeatures,
        trusted_feet_point: Point | None,
        trusted_feet_source: str,
        trusted_inside: bool | None,
        decision: ReLinkDecision,
    ) -> IdentityUpdateResult | None:
        best_candidate = decision.best_candidate
        if best_candidate is None:
            return None

        best_session = best_candidate.session
        best_score = best_candidate.score
        best_gap = best_candidate.frame_gap
        second_label = self._format_relink_candidate_label(decision.second_candidate)
        old_yolo_track_id = best_session.previous_yolo_track_id
        self._update_person_session_detection(
            best_session,
            yolo_track_id,
            bbox,
            features,
            identity_status="RELINKED",
            relink_score=best_score,
            relink_frame_gap=best_gap,
            trusted_feet_point=trusted_feet_point,
            trusted_feet_source=trusted_feet_source,
        )

        if trusted_inside:
            best_session.lifecycle_state = "ACTIVE_INSIDE"
            best_session.exit_confirm_hits = 0
            best_session.last_gate_reason = "RELINKED_ACTIVE_INSIDE"
            best_session.last_identity_debug = (
                f"{self._format_person_uid(best_session.person_uid)} / YOLO {yolo_track_id}"
            )
            has_active_person_id = True
        else:
            best_session.lifecycle_state = "LOST_INSIDE"
            best_session.exit_confirm_hits = 0
            best_session.last_gate_reason = "RELINKED_WAIT_TRUSTED_FEET"
            best_session.last_identity_debug = (
                f"RELINK {self._format_person_uid(best_session.person_uid)}"
            )
            has_active_person_id = False
        count_event_reason = self._set_session_count_event_reason(
            best_session,
            "RELINK_NO_RECOUNT",
            log_message=(
                "RELINK_NO_RECOUNT "
                f"person_id={self._format_person_uid(best_session.person_uid)} "
                f"old_yolo={old_yolo_track_id if old_yolo_track_id is not None else 'NA'} "
                f"new_yolo={yolo_track_id}"
            ),
        )
        self.pending_relink_by_yolo_id.pop(yolo_track_id, None)
        self._clear_new_person_guard_log(yolo_track_id)

        self._log_event(
            "RELINK_CONFIRMED "
            f"person_id={self._format_person_uid(best_session.person_uid)} "
            f"yolo_id={yolo_track_id}"
        )
        return self._build_identity_result(
            has_active_person_id=has_active_person_id,
            person_uid=best_session.person_uid,
            yolo_track_id=yolo_track_id,
            previous_yolo_track_id=old_yolo_track_id,
            identity_status="RELINKED",
            session_status=best_session.status,
            session_lifecycle=best_session.lifecycle_state,
            identity_debug=best_session.last_identity_debug,
            identity_feet_source=trusted_feet_source,
            identity_gate_reason=best_session.last_gate_reason,
            identity_entry_reason=best_session.entry_reason,
            relink_score=best_score,
            relink_frame_gap=best_gap,
            relink_score_gap=decision.score_gap,
            relink_best_candidate=self._format_person_uid(best_session.person_uid),
            relink_second_candidate=second_label,
            relink_state="RELINK_CONFIRMED",
            has_counted_enter=best_session.has_counted_enter,
            has_counted_exit=best_session.has_counted_exit,
            count_event_reason=count_event_reason,
            merged_from_analysis_subject_id=self._format_candidate_analysis_subject_id(
                yolo_track_id
            ),
        )

    def _update_active_person_session(
        self,
        session: PersonSession,
        yolo_track_id: int,
        bbox: BBox | None,
        features: PoseFeatures,
        trusted_feet_point: Point | None,
        trusted_feet_source: str,
        trusted_inside: bool | None,
        identity_status: IdentityStatusLabel,
        relink_score: float | None = None,
        relink_frame_gap: int = 0,
    ) -> IdentityUpdateResult:
        self._update_person_session_detection(
            session,
            yolo_track_id,
            bbox,
            features,
            identity_status=identity_status,
            relink_score=relink_score,
            relink_frame_gap=relink_frame_gap,
            trusted_feet_point=trusted_feet_point,
            trusted_feet_source=trusted_feet_source,
        )
        previous_yolo_track_id = session.previous_yolo_track_id

        if trusted_inside is True:
            count_event_reason = (
                self._count_session_enter(session)
                if not session.has_counted_enter
                else self._set_session_count_event_reason(
                    session,
                    "ENTER_ALREADY_COUNTED",
                )
            )
            session.lifecycle_state = "ACTIVE_INSIDE"
            session.exit_confirm_hits = 0
            session.last_gate_reason = (
                "RELINKED_ACTIVE_INSIDE"
                if identity_status == "RELINKED"
                else "ENTERED_BY_FEET"
            )
            session.last_identity_debug = (
                f"{self._format_person_uid(session.person_uid)} / YOLO {yolo_track_id}"
            )
            return self._build_identity_result(
                has_active_person_id=True,
                person_uid=session.person_uid,
                yolo_track_id=yolo_track_id,
                previous_yolo_track_id=previous_yolo_track_id,
                identity_status=identity_status,
                session_status=session.status,
                session_lifecycle=session.lifecycle_state,
                identity_debug=session.last_identity_debug,
                identity_feet_source=trusted_feet_source,
                identity_gate_reason=session.last_gate_reason,
                identity_entry_reason=session.entry_reason,
                relink_score=relink_score,
                relink_frame_gap=relink_frame_gap,
                has_counted_enter=session.has_counted_enter,
                has_counted_exit=session.has_counted_exit,
                count_event_reason=count_event_reason,
            )

        if trusted_inside is False:
            session.exit_confirm_hits += 1
            if session.exit_confirm_hits >= SETTINGS.identity.exit_confirm_frames:
                count_event_reason = self._count_session_exit(session)
                session.lifecycle_state = "EXITED"
                session.status = "COMPLETED"
                session.last_identity_status = "EXITED"
                session.last_gate_reason = "EXIT_CONFIRMED_BY_FEET"
                session.last_identity_debug = self._format_person_uid(session.person_uid)
                self.yolo_to_person.pop(yolo_track_id, None)
                session.current_yolo_track_id = None
                self._log_event(
                    f"PERSON_EXITED person_id={self._format_person_uid(session.person_uid)}"
                )
                self._log_event(
                    f"PERSON_ARCHIVED person_id={self._format_person_uid(session.person_uid)}"
                )
                return self._build_identity_result(
                    has_active_person_id=False,
                    person_uid=session.person_uid,
                    yolo_track_id=yolo_track_id,
                    previous_yolo_track_id=previous_yolo_track_id,
                    identity_status="EXITED",
                    session_status="COMPLETED",
                    session_lifecycle="EXITED",
                    identity_debug=self._format_person_uid(session.person_uid),
                    identity_feet_source=trusted_feet_source,
                    identity_gate_reason=session.last_gate_reason,
                    identity_entry_reason=session.entry_reason,
                    relink_score=relink_score,
                    relink_frame_gap=relink_frame_gap,
                    has_counted_enter=session.has_counted_enter,
                    has_counted_exit=session.has_counted_exit,
                    count_event_reason=count_event_reason,
                )

            count_event_reason = self._set_session_count_event_reason(
                session,
                "NO_COUNT_EVENT",
            )
            session.lifecycle_state = "LOST_INSIDE"
            session.last_gate_reason = (
                f"EXIT_WAIT_CONFIRM {session.exit_confirm_hits}/"
                f"{SETTINGS.identity.exit_confirm_frames}"
            )
            session.last_identity_debug = self._format_person_uid(session.person_uid)
            return self._build_identity_result(
                has_active_person_id=False,
                person_uid=session.person_uid,
                yolo_track_id=yolo_track_id,
                previous_yolo_track_id=previous_yolo_track_id,
                identity_status=identity_status,
                session_status=session.status,
                session_lifecycle="LOST_INSIDE",
                identity_debug=session.last_identity_debug,
                identity_feet_source=trusted_feet_source,
                identity_gate_reason=session.last_gate_reason,
                identity_entry_reason=session.entry_reason,
                relink_score=relink_score,
                relink_frame_gap=relink_frame_gap,
                has_counted_enter=session.has_counted_enter,
                has_counted_exit=session.has_counted_exit,
                count_event_reason=count_event_reason,
            )

        count_event_reason: CountEventReasonLabel
        if self._is_bbox_center_inside_stairs(bbox) is False:
            count_event_reason = self._set_session_count_event_reason(
                session,
                "BBOX_OUTSIDE_IGNORED_NO_EXIT_COUNT",
                log_message=(
                    "NO_EXIT_COUNT_BBOX_ONLY "
                    f"person_id={self._format_person_uid(session.person_uid)}"
                ),
                log_only_on_change=True,
            )
        else:
            count_event_reason = self._set_session_count_event_reason(
                session,
                "NO_COUNT_EVENT",
            )
        session.exit_confirm_hits = 0
        session.lifecycle_state = "LOST_INSIDE"
        session.last_gate_reason = (
            "RELINKED_WAIT_TRUSTED_FEET"
            if identity_status == "RELINKED"
            else "WAIT_TRUSTED_FEET"
        )
        session.last_identity_debug = self._format_person_uid(session.person_uid)
        return self._build_identity_result(
            has_active_person_id=False,
            person_uid=session.person_uid,
            yolo_track_id=yolo_track_id,
            previous_yolo_track_id=previous_yolo_track_id,
            identity_status=identity_status,
            session_status=session.status,
            session_lifecycle="LOST_INSIDE",
            identity_debug=session.last_identity_debug,
            identity_feet_source=trusted_feet_source,
            identity_gate_reason=session.last_gate_reason,
            identity_entry_reason=session.entry_reason,
            relink_score=relink_score,
            relink_frame_gap=relink_frame_gap,
            has_counted_enter=session.has_counted_enter,
            has_counted_exit=session.has_counted_exit,
            count_event_reason=count_event_reason,
        )

    def _format_occluded_entry_log_detail(
        self,
        status: OccludedEntryStatus,
        feet_source: str,
    ) -> str:
        detail_parts = [
            f"frame_idx={self.current_frame_index}",
            f"candidate_age={status.candidate_age_frames}",
            f"inside_count={status.inside_frames}",
            f"motion_count={status.motion_frames}",
            f"feet_source={feet_source}",
            f"nearest_ghost_score={self._format_occluded_log_metric(status.nearest_ghost_score)}",
            f"nearest_active_iou={self._format_occluded_log_metric(status.nearest_active_iou)}",
            f"nearest_active_distance={self._format_occluded_log_metric(status.nearest_active_distance)}",
            f"reason={status.block_reason}",
        ]
        return " ".join(detail_parts)

    def _assess_occluded_entry_candidate(
        self,
        candidate: CandidateSession,
        bbox: BBox | None,
        trusted_feet_point: Point | None,
    ) -> OccludedEntryStatus:
        nearest_active_iou, nearest_active_distance, block_active_overlap = (
            self._get_active_person_overlap_status(candidate, bbox)
        )
        status = self._build_occluded_entry_status(
            candidate,
            nearest_active_iou=nearest_active_iou,
            nearest_active_distance=nearest_active_distance,
        )

        if SETTINGS.identity.occluded_entry_require_trusted_feet and (
            trusted_feet_point is None
            or not self._is_trusted_feet_source(candidate.last_trusted_feet_source)
        ):
            status.block_reason = "NO_TRUSTED_FEET"
            return status

        if block_active_overlap:
            status.block_reason = "ACTIVE_OVERLAP"
            return status

        if status.candidate_age_frames < status.age_target:
            status.block_reason = "TOO_YOUNG"
            return status

        if status.inside_frames < status.inside_target:
            status.block_reason = "INSIDE_NOT_STABLE"
            return status

        if status.motion_frames < status.motion_target:
            status.block_reason = "NOT_ENOUGH_MOTION"
            return status

        return status

    def _update_candidate_session(
        self,
        yolo_track_id: int,
        bbox: BBox | None,
        features: PoseFeatures,
        trusted_feet_point: Point | None,
        trusted_feet_source: str,
        trusted_inside: bool | None,
        *,
        allow_new_person: bool,
        blocked_new_person_reason: str | None = None,
        blocked_target_person_uid: PersonUID | None = None,
    ) -> IdentityUpdateResult:
        candidate = self._upsert_candidate_session(
            yolo_track_id,
            bbox,
            features,
            trusted_feet_point,
            trusted_feet_source,
        )
        candidate_feet_source, candidate_feet_reason = (
            self._get_candidate_feet_context(features, trusted_feet_point)
        )
        self._update_candidate_proof_state(
            candidate,
            trusted_feet_point,
            trusted_inside,
        )
        self._update_candidate_occluded_state(
            candidate,
            trusted_feet_point,
            trusted_inside,
        )
        candidate_lifecycle = self._get_candidate_session_lifecycle(
            candidate,
            trusted_feet_point,
            trusted_inside,
            allow_new_person=allow_new_person,
        )
        inside_test = self._get_candidate_inside_test(
            trusted_feet_point,
            trusted_inside,
        )
        is_inside_without_outside_proof = (
            trusted_inside is True and not candidate.has_trusted_feet_outside
        )

        relink_decision = self._evaluate_relink_decision(
            candidate,
            bbox,
            features,
        )
        if relink_decision is not None:
            self._store_candidate_relink_decision(candidate, relink_decision)
            self._log_relink_decision(
                yolo_track_id,
                candidate,
                relink_decision,
            )
            if relink_decision.state == "RELINK_CONFIRMED":
                relink_result = self._commit_relink_decision(
                    yolo_track_id,
                    bbox,
                    features,
                    trusted_feet_point,
                    trusted_feet_source,
                    trusted_inside,
                    relink_decision,
                )
                if relink_result is not None:
                    self.candidate_sessions.pop(yolo_track_id, None)
                    return relink_result
            elif (
                not is_inside_without_outside_proof
                or SETTINGS.identity.occluded_entry_block_if_any_lost_inside_ghost
            ):
                if blocked_new_person_reason is None:
                    best_candidate = relink_decision.best_candidate
                    blocked_target = (
                        self._format_person_uid(best_candidate.session.person_uid)
                        if best_candidate is not None
                        else (
                            self._format_person_uid(blocked_target_person_uid)
                            if blocked_target_person_uid is not None
                            else ""
                        )
                    )
                    self._log_new_person_guard(
                        yolo_track_id,
                        "NEW_PERSON_BLOCKED_GHOST_MATCH",
                        f"target={blocked_target or 'NA'}",
                    )
                if is_inside_without_outside_proof:
                    ghost_status = self._build_occluded_entry_status(
                        candidate,
                        block_reason=self._get_relink_block_reason(relink_decision),
                        nearest_ghost_score=(
                            relink_decision.best_candidate.score
                            if relink_decision.best_candidate is not None
                            else None
                        ),
                    )
                    self._log_new_person_guard(
                        yolo_track_id,
                        "OCCLUDED_ENTRY_BLOCKED_GHOST",
                        self._format_occluded_entry_log_detail(
                            ghost_status,
                            candidate_feet_source,
                        ),
                    )
                return self._build_pending_relink_result(
                    yolo_track_id,
                    candidate_feet_source,
                    candidate_feet_reason,
                    inside_test,
                    candidate,
                    relink_decision,
                )
            else:
                self._reset_candidate_relink_state(candidate)

        if trusted_inside is True:
            if not candidate.has_trusted_feet_outside:
                base_occluded_status = self._build_occluded_entry_status(candidate)
                if not allow_new_person:
                    blocked_target = (
                        self._format_person_uid(blocked_target_person_uid)
                        if blocked_target_person_uid is not None
                        else ""
                    )
                    if blocked_new_person_reason == "NEW_PERSON_BLOCKED_PENDING_RELINK":
                        base_occluded_status.block_reason = "PENDING_RELINK"
                        candidate.last_gate_reason = "OCCLUDED_ENTRY_WAIT"
                        self._log_new_person_guard(
                            yolo_track_id,
                            "OCCLUDED_ENTRY_WAIT",
                            self._format_occluded_entry_log_detail(
                                base_occluded_status,
                                candidate_feet_source,
                            ),
                        )
                    else:
                        base_occluded_status.block_reason = "LOST_GHOST_NEARBY"
                        candidate.last_gate_reason = "OCCLUDED_ENTRY_BLOCKED_GHOST"
                        detail = self._format_occluded_entry_log_detail(
                            base_occluded_status,
                            candidate_feet_source,
                        )
                        if blocked_target:
                            detail = f"{detail} target={blocked_target}"
                        self._log_new_person_guard(
                            yolo_track_id,
                            "OCCLUDED_ENTRY_BLOCKED_GHOST",
                            detail,
                        )
                    return self._create_candidate_result(
                        yolo_track_id,
                        candidate,
                        candidate_lifecycle,
                        candidate_feet_source,
                        candidate_feet_reason,
                        candidate.last_gate_reason,
                        inside_test,
                        occluded_entry_status=base_occluded_status,
                    )

                if not SETTINGS.identity.allow_occluded_entry_promotion:
                    base_occluded_status.block_reason = "OCCLUDED_ENTRY_DISABLED"
                    candidate.last_gate_reason = "OCCLUDED_ENTRY_WAIT"
                    return self._create_candidate_result(
                        yolo_track_id,
                        candidate,
                        candidate_lifecycle,
                        candidate_feet_source,
                        candidate_feet_reason,
                        candidate.last_gate_reason,
                        inside_test,
                        occluded_entry_status=base_occluded_status,
                    )

                occluded_status = self._assess_occluded_entry_candidate(
                    candidate,
                    bbox,
                    trusted_feet_point,
                )
                if occluded_status.block_reason == "NONE":
                    candidate.last_gate_reason = "OCCLUDED_ENTRY_PROMOTED"
                    session = self._promote_candidate(
                        candidate,
                        features,
                        entry_reason="OCCLUDED_ENTRY",
                    )
                    self.seen_person_uids_this_frame.add(session.person_uid)
                    return self._build_promoted_candidate_result(
                        session,
                        yolo_track_id,
                        trusted_feet_source,
                        inside_test,
                        identity_outside_proof=False,
                        identity_enter_confirm_hits=0,
                        identity_enter_confirm_target=0,
                        merge_from_candidate_history=False,
                        occluded_entry_status=occluded_status,
                    )

                if occluded_status.block_reason == "ACTIVE_OVERLAP":
                    candidate.last_gate_reason = "OCCLUDED_ENTRY_BLOCKED_ACTIVE_OVERLAP"
                    self._log_new_person_guard(
                        yolo_track_id,
                        "OCCLUDED_ENTRY_BLOCKED_ACTIVE_OVERLAP",
                        self._format_occluded_entry_log_detail(
                            occluded_status,
                            candidate_feet_source,
                        ),
                    )
                elif occluded_status.block_reason == "NO_TRUSTED_FEET":
                    candidate.last_gate_reason = "OCCLUDED_ENTRY_BLOCKED_NO_FEET"
                    self._log_new_person_guard(
                        yolo_track_id,
                        "OCCLUDED_ENTRY_BLOCKED_NO_FEET",
                        self._format_occluded_entry_log_detail(
                            occluded_status,
                            candidate_feet_source,
                        ),
                    )
                elif occluded_status.block_reason == "TOO_YOUNG":
                    candidate.last_gate_reason = "OCCLUDED_ENTRY_WAIT"
                    self._log_new_person_guard(
                        yolo_track_id,
                        "OCCLUDED_ENTRY_BLOCKED_TOO_YOUNG",
                        self._format_occluded_entry_log_detail(
                            occluded_status,
                            candidate_feet_source,
                        ),
                    )
                else:
                    candidate.last_gate_reason = "OCCLUDED_ENTRY_WAIT"
                    self._log_new_person_guard(
                        yolo_track_id,
                        "OCCLUDED_ENTRY_WAIT",
                        self._format_occluded_entry_log_detail(
                            occluded_status,
                            candidate_feet_source,
                        ),
                    )

                return self._create_candidate_result(
                    yolo_track_id,
                    candidate,
                    candidate_lifecycle,
                    candidate_feet_source,
                    candidate_feet_reason,
                    candidate.last_gate_reason,
                    inside_test,
                    occluded_entry_status=occluded_status,
                )

            if not allow_new_person:
                if blocked_new_person_reason is not None:
                    blocked_target = (
                        self._format_person_uid(blocked_target_person_uid)
                        if blocked_target_person_uid is not None
                        else ""
                    )
                    detail = f"target={blocked_target}" if blocked_target else ""
                    self._log_new_person_guard(
                        yolo_track_id,
                        blocked_new_person_reason,
                        detail,
                    )
                    candidate.last_gate_reason = self._normalize_candidate_gate_reason(
                        blocked_new_person_reason
                    )
                return self._create_candidate_result(
                    yolo_track_id,
                    candidate,
                    candidate_lifecycle,
                    candidate_feet_source,
                    candidate_feet_reason,
                    candidate.last_gate_reason,
                    inside_test,
                )

            if candidate.entry_confirm_hits >= SETTINGS.identity.entry_confirm_frames:
                self._log_new_person_guard(
                    yolo_track_id,
                    "NEW_PERSON_ALLOWED_NO_MATCH",
                )
                candidate.last_gate_reason = "OUTSIDE_TO_INSIDE_CONFIRMED"
                session = self._promote_candidate(
                    candidate,
                    features,
                    entry_reason="CONFIRMED_ENTER",
                )
                self.seen_person_uids_this_frame.add(session.person_uid)
                return self._build_promoted_candidate_result(
                    session,
                    yolo_track_id,
                    trusted_feet_source,
                    inside_test,
                    identity_outside_proof=True,
                    identity_enter_confirm_hits=SETTINGS.identity.entry_confirm_frames,
                    identity_enter_confirm_target=SETTINGS.identity.entry_confirm_frames,
                    merge_from_candidate_history=True,
                )

            candidate.last_gate_reason = (
                f"WAIT_ENTER_CONFIRM {candidate.entry_confirm_hits}/"
                f"{SETTINGS.identity.entry_confirm_frames}"
            )
            return self._create_candidate_result(
                yolo_track_id,
                candidate,
                candidate_lifecycle,
                candidate_feet_source,
                candidate_feet_reason,
                candidate.last_gate_reason,
                inside_test,
            )

        candidate.last_gate_reason = (
            "TRUSTED_FEET_OUTSIDE_ROI"
            if trusted_inside is False
            else (
                "NO_FEET"
                if trusted_feet_point is None
                else candidate_feet_reason
                if trusted_feet_point is None
                else "INSIDE_TEST_UNAVAILABLE"
            )
        )
        if trusted_feet_point is None and candidate_feet_reason != "NONE":
            candidate.last_gate_reason = "NO_FEET"
        return self._create_candidate_result(
            yolo_track_id,
            candidate,
            candidate_lifecycle,
            candidate_feet_source,
            candidate_feet_reason,
            candidate.last_gate_reason,
            inside_test,
        )

    def update_detection(
        self,
        yolo_track_id: int,
        bbox: BBoxArray | BBox | None,
        features: PoseFeatures,
        frame_index: int,
    ) -> IdentityUpdateResult:
        if self.current_frame_index != frame_index:
            self.begin_frame(frame_index)
        self.seen_yolo_ids_this_frame.add(yolo_track_id)
        bbox_tuple = self._normalize_bbox(bbox, features)
        trusted_feet_point, trusted_feet_source = self._get_trusted_feet(features)
        trusted_inside = self._is_trusted_feet_inside(trusted_feet_point)

        mapped_person_uid = self.yolo_to_person.get(yolo_track_id)
        if mapped_person_uid is not None:
            session = self.person_sessions.get(mapped_person_uid)
            if session is None or session.status == "COMPLETED":
                self.yolo_to_person.pop(yolo_track_id, None)
            else:
                self._log_new_person_guard(
                    yolo_track_id,
                    "NEW_PERSON_BLOCKED_EXISTING_MAPPING",
                    f"person_id={self._format_person_uid(session.person_uid)}",
                )
                self.candidate_sessions.pop(yolo_track_id, None)
                self.pending_relink_by_yolo_id.pop(yolo_track_id, None)
                return self._update_active_person_session(
                    session,
                    yolo_track_id,
                    bbox_tuple,
                    features,
                    trusted_feet_point,
                    trusted_feet_source,
                    trusted_inside,
                    identity_status="ACTIVE",
                )

        pending_relink = self.pending_relink_by_yolo_id.get(yolo_track_id)
        if pending_relink is not None:
            pending_target = self._format_person_uid(pending_relink.target_person_uid)
            self._log_new_person_guard(
                yolo_track_id,
                "NEW_PERSON_BLOCKED_PENDING_RELINK",
                f"target={pending_target or 'NA'}",
            )
            return self._update_candidate_session(
                yolo_track_id,
                bbox_tuple,
                features,
                trusted_feet_point,
                trusted_feet_source,
                trusted_inside,
                allow_new_person=False,
                blocked_new_person_reason="NEW_PERSON_BLOCKED_PENDING_RELINK",
                blocked_target_person_uid=pending_relink.target_person_uid,
            )

        same_yolo_lost_matches = self._get_same_yolo_lost_matches(yolo_track_id)
        if len(same_yolo_lost_matches) == 1:
            return self._commit_same_yolo_relink(
                same_yolo_lost_matches[0],
                yolo_track_id,
                bbox_tuple,
                features,
                trusted_feet_point,
                trusted_feet_source,
                trusted_inside,
            )
        if same_yolo_lost_matches:
            blocked_target_person_uid = same_yolo_lost_matches[0].person_uid
            self._log_new_person_guard(
                yolo_track_id,
                "NEW_PERSON_BLOCKED_GHOST_MATCH",
                f"target={self._format_person_uid(blocked_target_person_uid)}",
            )
            return self._update_candidate_session(
                yolo_track_id,
                bbox_tuple,
                features,
                trusted_feet_point,
                trusted_feet_source,
                trusted_inside,
                allow_new_person=False,
                blocked_new_person_reason="NEW_PERSON_BLOCKED_GHOST_MATCH",
                blocked_target_person_uid=blocked_target_person_uid,
            )

        return self._update_candidate_session(
            yolo_track_id,
            bbox_tuple,
            features,
            trusted_feet_point,
            trusted_feet_source,
            trusted_inside,
            allow_new_person=True,
        )

    def update_from_analysis(
        self,
        person_uid: PersonUID,
        analysis: AnalysisResult,
    ) -> None:
        session = self.person_sessions.get(person_uid)
        if session is None:
            return

        direction = str(analysis.get("direction", session.last_direction))
        if direction:
            session.last_direction = direction

        lane_side_label = str(analysis.get("lane_side_label", "UNKNOWN"))
        if lane_side_label in ("LEFT", "RIGHT", "CENTER"):
            session.last_lane_side = lane_side_label

        session.last_track_zone_state = str(
            analysis.get("track_zone_state", session.last_track_zone_state)
        )

        warnings_value = analysis.get("warnings")
        if isinstance(warnings_value, list):
            session.last_warnings = [str(warning) for warning in warnings_value]

    def _archive_lost_session(
        self,
        session: PersonSession,
        archive_reason: str,
    ) -> None:
        if session.status == "COMPLETED":
            return

        if session.current_yolo_track_id is not None:
            self.yolo_to_person.pop(session.current_yolo_track_id, None)
            session.current_yolo_track_id = None

        session.status = "COMPLETED"
        self._log_event(
            "PERSON_ARCHIVED "
            f"person_id={self._format_person_uid(session.person_uid)} "
            f"reason={archive_reason}"
        )

    def end_frame(self) -> None:
        for session in self.person_sessions.values():
            if session.status == "COMPLETED":
                continue
            if session.last_seen_frame == self.current_frame_index:
                continue

            if session.status == "ACTIVE":
                old_yolo_track_id = session.current_yolo_track_id
                if old_yolo_track_id is not None:
                    self.yolo_to_person.pop(old_yolo_track_id, None)
                session.previous_yolo_track_id = old_yolo_track_id
                session.current_yolo_track_id = None
                session.status = "LOST"
                session.lifecycle_state = "LOST_INSIDE"
                session.last_identity_status = "LOST"
                session.last_gate_reason = "WAIT_RELINK"
                session.lost_frame_count = (
                    self.current_frame_index - session.last_seen_frame
                )
                self._set_session_count_event_reason(
                    session,
                    "TRACK_LOST_NO_EXIT_COUNT",
                    log_message=(
                        "NO_EXIT_COUNT_TRACK_LOST "
                        f"person_id={self._format_person_uid(session.person_uid)}"
                    ),
                )
                self._log_event(
                    "PERSON_LOST_INSIDE "
                    f"person_id={self._format_person_uid(session.person_uid)} "
                    f"old_yolo_id={old_yolo_track_id if old_yolo_track_id is not None else 'NA'}"
                )
            else:
                session.lost_frame_count = (
                    self.current_frame_index - session.last_seen_frame
                )

            if (
                session.lost_frame_count
                > SETTINGS.identity.max_lost_frames + SETTINGS.identity.lost_inside_extra_frames
            ):
                self._archive_lost_session(session, "LOST_INSIDE_TIMEOUT")

        stale_candidate_ids = [
            yolo_track_id
            for yolo_track_id, candidate in self.candidate_sessions.items()
            if self.current_frame_index - candidate.last_seen_frame
            > SETTINGS.identity.candidate_timeout_frames
        ]
        for yolo_track_id in stale_candidate_ids:
            self.pending_relink_by_yolo_id.pop(yolo_track_id, None)
            self._clear_new_person_guard_log(yolo_track_id)
            del self.candidate_sessions[yolo_track_id]

    def get_retained_person_uids(self) -> set[PersonUID]:
        return {
            person_uid
            for person_uid, session in self.person_sessions.items()
            if session.status != "COMPLETED"
        }

    def get_retained_analysis_subject_ids(self) -> set[AnalysisSubjectID]:
        retained_subject_ids: set[AnalysisSubjectID] = {
            self._format_person_uid(person_uid)
            for person_uid, session in self.person_sessions.items()
            if session.status != "COMPLETED"
        }
        retained_subject_ids.update(
            self._format_candidate_analysis_subject_id(yolo_track_id)
            for yolo_track_id in self.candidate_sessions
        )
        return retained_subject_ids

    def get_active_inside_count(self) -> int:
        return sum(
            1
            for session in self.person_sessions.values()
            if session.status == "ACTIVE"
            and session.lifecycle_state == "ACTIVE_INSIDE"
            and not session.has_counted_exit
        )

    def get_lost_inside_count(self) -> int:
        return sum(
            1
            for session in self.person_sessions.values()
            if session.status in ("ACTIVE", "LOST")
            and session.lifecycle_state == "LOST_INSIDE"
            and not session.has_counted_exit
        )

    def get_active_or_lost_inside_count(self) -> int:
        return sum(
            1
            for session in self.person_sessions.values()
            if session.status in ("ACTIVE", "LOST")
            and session.lifecycle_state in ("ACTIVE_INSIDE", "LOST_INSIDE")
            and not session.has_counted_exit
        )

    def augment_analysis_result(
        self,
        analysis: AnalysisResult,
        identity_result: IdentityUpdateResult,
    ) -> None:
        active_inside_count = self.get_active_inside_count()
        lost_inside_count = self.get_lost_inside_count()
        identity_fields: dict[str, object] = {
            "person_uid": identity_result.person_uid,
            "person_uid_label": identity_result.person_uid_label,
            "current_person_id": identity_result.person_uid_label,
            "analysis_subject_id": identity_result.analysis_subject_id,
            "analysis_subject_label": identity_result.analysis_subject_label,
            "merged_from_analysis_subject_id": identity_result.merged_from_analysis_subject_id,
            "yolo_track_id": identity_result.yolo_track_id,
            "previous_yolo_track_id": identity_result.previous_yolo_track_id,
            "identity_status": identity_result.identity_status,
            "session_status": identity_result.session_status,
            "session_lifecycle": identity_result.session_lifecycle,
            "person_lifecycle_state": identity_result.session_lifecycle,
            "identity_debug": identity_result.identity_debug,
            "identity_feet_source": identity_result.identity_feet_source,
            "identity_feet_reason": identity_result.identity_feet_reason,
            "identity_gate_reason": identity_result.identity_gate_reason,
            "identity_inside_test": identity_result.identity_inside_test,
            "identity_entry_reason": identity_result.identity_entry_reason,
            "identity_outside_proof": identity_result.identity_outside_proof,
            "identity_enter_confirm_hits": identity_result.identity_enter_confirm_hits,
            "identity_enter_confirm_target": identity_result.identity_enter_confirm_target,
            "has_active_person_id": identity_result.has_active_person_id,
            "relink_score": identity_result.relink_score,
            "relink_frame_gap": identity_result.relink_frame_gap,
            "relink_score_gap": identity_result.relink_score_gap,
            "relink_best_candidate": identity_result.relink_best_candidate,
            "relink_second_candidate": identity_result.relink_second_candidate,
            "relink_state": identity_result.relink_state,
            "has_counted_enter": identity_result.has_counted_enter,
            "has_counted_exit": identity_result.has_counted_exit,
            "count_event_reason": identity_result.count_event_reason,
            "total_entered_count": self.total_entered_count,
            "total_confirmed_entered_count": self.total_confirmed_entered_count,
            "total_occluded_entered_count": self.total_occluded_entered_count,
            "total_exited_count": self.total_exited_count,
            "entered_count": self.total_entered_count,
            "exited_count": self.total_exited_count,
            "active_inside_count": active_inside_count,
            "lost_inside_count": lost_inside_count,
            "active_or_lost_inside_count": self.get_active_or_lost_inside_count(),
            "occluded_entry_candidate_age_frames": identity_result.occluded_entry_candidate_age_frames,
            "occluded_entry_age_target": identity_result.occluded_entry_age_target,
            "occluded_entry_inside_frames": identity_result.occluded_entry_inside_frames,
            "occluded_entry_inside_target": identity_result.occluded_entry_inside_target,
            "occluded_entry_motion_frames": identity_result.occluded_entry_motion_frames,
            "occluded_entry_motion_target": identity_result.occluded_entry_motion_target,
            "occluded_entry_block_reason": identity_result.occluded_entry_block_reason,
            "occluded_entry_nearest_ghost_score": identity_result.occluded_entry_nearest_ghost_score,
            "occluded_entry_nearest_active_iou": identity_result.occluded_entry_nearest_active_iou,
            "occluded_entry_nearest_active_distance": identity_result.occluded_entry_nearest_active_distance,
        }
        analysis.update(identity_fields)

        debug_info = analysis.get("debug_info")
        if isinstance(debug_info, dict):
            debug_info.update(identity_fields)

    def build_overlay_result(
        self,
        identity_result: IdentityUpdateResult,
    ) -> AnalysisResult:
        active_inside_count = self.get_active_inside_count()
        lost_inside_count = self.get_lost_inside_count()
        if identity_result.session_lifecycle == "CANDIDATE_OUTSIDE":
            color = SETTINGS.violation.outside_color
        elif identity_result.session_lifecycle == "CANDIDATE_NO_FEET":
            color = (0, 165, 255)
        elif identity_result.session_lifecycle == "OCCLUDED_ENTRY_CANDIDATE":
            color = (0, 185, 255)
        elif identity_result.session_lifecycle == "UNASSIGNED_INSIDE_CANDIDATE":
            color = (0, 185, 255)
        elif identity_result.session_lifecycle == "LOST_INSIDE":
            color = (0, 165, 255)
        elif identity_result.session_lifecycle == "TEMP_REID_CANDIDATE":
            color = (0, 200, 255)
        elif identity_result.session_lifecycle == "AMBIGUOUS_REID":
            color = (0, 140, 255)
        elif identity_result.session_lifecycle == "EXITED":
            color = (255, 255, 0)
        else:
            color = SETTINGS.violation.safe_color
        status = identity_result.session_lifecycle
        display_status = self._format_session_lifecycle_display(status)
        result: AnalysisResult = {
            "status": status,
            "display_status": display_status,
            "color": color,
            "track_id": identity_result.person_uid or -1,
            "person_uid": identity_result.person_uid,
            "person_uid_label": identity_result.person_uid_label,
            "current_person_id": identity_result.person_uid_label,
            "analysis_subject_id": identity_result.analysis_subject_id,
            "analysis_subject_label": identity_result.analysis_subject_label,
            "merged_from_analysis_subject_id": identity_result.merged_from_analysis_subject_id,
            "yolo_track_id": identity_result.yolo_track_id,
            "previous_yolo_track_id": identity_result.previous_yolo_track_id,
            "identity_status": identity_result.identity_status,
            "session_status": identity_result.session_status,
            "session_lifecycle": identity_result.session_lifecycle,
            "person_lifecycle_state": identity_result.session_lifecycle,
            "identity_debug": identity_result.identity_debug,
            "identity_feet_source": identity_result.identity_feet_source,
            "identity_feet_reason": identity_result.identity_feet_reason,
            "identity_gate_reason": identity_result.identity_gate_reason,
            "identity_inside_test": identity_result.identity_inside_test,
            "identity_entry_reason": identity_result.identity_entry_reason,
            "identity_outside_proof": identity_result.identity_outside_proof,
            "identity_enter_confirm_hits": identity_result.identity_enter_confirm_hits,
            "identity_enter_confirm_target": identity_result.identity_enter_confirm_target,
            "has_active_person_id": identity_result.has_active_person_id,
            "relink_score": identity_result.relink_score,
            "relink_frame_gap": identity_result.relink_frame_gap,
            "relink_score_gap": identity_result.relink_score_gap,
            "relink_best_candidate": identity_result.relink_best_candidate,
            "relink_second_candidate": identity_result.relink_second_candidate,
            "relink_state": identity_result.relink_state,
            "has_counted_enter": identity_result.has_counted_enter,
            "has_counted_exit": identity_result.has_counted_exit,
            "count_event_reason": identity_result.count_event_reason,
            "total_entered_count": self.total_entered_count,
            "total_confirmed_entered_count": self.total_confirmed_entered_count,
            "total_occluded_entered_count": self.total_occluded_entered_count,
            "total_exited_count": self.total_exited_count,
            "entered_count": self.total_entered_count,
            "exited_count": self.total_exited_count,
            "active_inside_count": active_inside_count,
            "lost_inside_count": lost_inside_count,
            "active_or_lost_inside_count": self.get_active_or_lost_inside_count(),
            "occluded_entry_candidate_age_frames": identity_result.occluded_entry_candidate_age_frames,
            "occluded_entry_age_target": identity_result.occluded_entry_age_target,
            "occluded_entry_inside_frames": identity_result.occluded_entry_inside_frames,
            "occluded_entry_inside_target": identity_result.occluded_entry_inside_target,
            "occluded_entry_motion_frames": identity_result.occluded_entry_motion_frames,
            "occluded_entry_motion_target": identity_result.occluded_entry_motion_target,
            "occluded_entry_block_reason": identity_result.occluded_entry_block_reason,
            "occluded_entry_nearest_ghost_score": identity_result.occluded_entry_nearest_ghost_score,
            "occluded_entry_nearest_active_iou": identity_result.occluded_entry_nearest_active_iou,
            "occluded_entry_nearest_active_distance": identity_result.occluded_entry_nearest_active_distance,
            "wrong_lane": False,
            "direction": "ANALYZING",
            "final_direction": "ANALYZING",
            "hip_direction": "UNKNOWN",
            "shoulder_direction": "UNKNOWN",
            "direction_source": "NO_VALID_MONITOR_DIRECTION",
            "direction_reason": identity_result.identity_gate_reason,
            "warnings": [],
        }
        if SETTINGS.demo.enable_debug_overlay:
            result["debug_info"] = dict(result)
        else:
            result["debug_info"] = None
        return result
