from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot, inf

import numpy as np

from stair_monitor.common.types import (
    AnalysisResult,
    BBox,
    BBoxArray,
    CameraConfigDict,
    IdentityStatusLabel,
    PersonSessionLifecycleLabel,
    PersonSessionStatusLabel,
    PersonUID,
    Point,
    PoseFeatures,
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
        "VIRTUAL_FROM_TWO_SHOULDERS",
    }
)


@dataclass(slots=True)
class IdentityUpdateResult:
    has_active_person_id: bool
    person_uid: PersonUID | None
    person_uid_label: str
    yolo_track_id: int | None
    previous_yolo_track_id: int | None
    identity_status: IdentityStatusLabel
    session_status: PersonSessionStatusLabel
    session_lifecycle: PersonSessionLifecycleLabel
    identity_debug: str
    identity_feet_source: str
    identity_gate_reason: str
    relink_score: float | None
    relink_frame_gap: int


@dataclass(slots=True)
class CandidateSession:
    current_yolo_track_id: int
    first_seen_frame: int
    last_seen_frame: int
    last_bbox: BBox | None
    last_bbox_center: Point | None
    last_anchor_point: Point | None
    last_trusted_feet_point: Point | None
    last_trusted_feet_source: str
    entry_confirm_hits: int = 0
    last_gate_reason: str = "WAIT_TRUSTED_FEET_ENTER"


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
    last_relink_score: float | None = None
    last_relink_frame_gap: int = 0
    entered_counted: bool = False
    exited_counted: bool = False


class PersonIdentityManager:
    """Quan ly stable person_id duoc gate boi trusted feet cho Windows/demo."""

    def __init__(self, config: CameraConfigDict):
        self.next_person_uid: PersonUID = 1
        self.person_sessions: dict[PersonUID, PersonSession] = {}
        self.candidate_sessions: dict[int, CandidateSession] = {}
        self.yolo_to_person: dict[int, PersonUID] = {}
        self.current_frame_index = -1
        self.seen_person_uids_this_frame: set[PersonUID] = set()
        self.seen_yolo_ids_this_frame: set[int] = set()
        self.entered_count = 0
        self.exited_count = 0

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
        feet_source = str(features.get("feet_point_source", "NO_FEET_POINT"))
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
    ) -> IdentityUpdateResult:
        return IdentityUpdateResult(
            has_active_person_id=has_active_person_id,
            person_uid=person_uid,
            person_uid_label=self._format_person_uid(person_uid),
            yolo_track_id=yolo_track_id,
            previous_yolo_track_id=previous_yolo_track_id,
            identity_status=identity_status,
            session_status=session_status,
            session_lifecycle=session_lifecycle,
            identity_debug=identity_debug,
            identity_feet_source=identity_feet_source,
            identity_gate_reason=identity_gate_reason,
            relink_score=relink_score,
            relink_frame_gap=relink_frame_gap,
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
        trusted_feet_source: str,
        gate_reason: str,
    ) -> IdentityUpdateResult:
        return self._build_identity_result(
            has_active_person_id=False,
            person_uid=None,
            yolo_track_id=yolo_track_id,
            previous_yolo_track_id=None,
            identity_status="CANDIDATE",
            session_status="CANDIDATE",
            session_lifecycle="CANDIDATE_OUTSIDE",
            identity_debug=f"YOLO {yolo_track_id}",
            identity_feet_source=trusted_feet_source,
            identity_gate_reason=gate_reason,
            relink_score=None,
            relink_frame_gap=0,
        )

    def _promote_candidate(
        self,
        candidate: CandidateSession,
        features: PoseFeatures,
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
            previous_anchor_point=None,
            last_direction="UNKNOWN",
            last_lane_side=None,
            last_track_zone_state="UNKNOWN",
            status="ACTIVE",
            lifecycle_state="ACTIVE_INSIDE",
            last_identity_status="NEW",
            last_identity_debug=f"ENTER {self._format_person_uid(person_uid)}",
            last_gate_reason="ENTERED_BY_FEET",
            entered_counted=True,
            exited_counted=False,
        )
        self.person_sessions[person_uid] = session
        self.yolo_to_person[candidate.current_yolo_track_id] = person_uid
        self.entered_count += 1
        self._log_event(
            "PERSON_ENTERED "
            f"person_id={self._format_person_uid(person_uid)} "
            f"yolo_id={candidate.current_yolo_track_id}"
        )
        self.candidate_sessions.pop(candidate.current_yolo_track_id, None)
        return session

    def score_relink_candidate(
        self,
        session: PersonSession,
        bbox: BBox | None,
        features: PoseFeatures,
    ) -> tuple[float, str, int]:
        if session.status != "LOST" or session.lifecycle_state == "EXITED":
            return inf, "SESSION_NOT_RELINKABLE", 0

        frame_gap = self.current_frame_index - session.last_seen_frame
        if frame_gap <= 0:
            return inf, "FRAME_GAP_INVALID", frame_gap

        if (
            frame_gap
            > SETTINGS.identity.max_lost_frames + SETTINGS.identity.lost_inside_extra_frames
        ):
            return inf, "FRAME_GAP_TOO_LARGE", frame_gap

        current_anchor = self._select_anchor_point(features, bbox)
        predicted_anchor = self._predict_anchor_point(session, frame_gap)
        current_bbox_center = self._get_bbox_center(bbox)
        anchor_distance = self._distance(predicted_anchor, current_anchor)
        bbox_center_distance = self._distance(
            session.last_bbox_center,
            current_bbox_center,
        )

        if anchor_distance is None and bbox_center_distance is None:
            return inf, "NO_DISTANCE_REFERENCE", frame_gap

        distance_reference = (
            anchor_distance
            if anchor_distance is not None
            else bbox_center_distance
        )
        if (
            distance_reference is not None
            and distance_reference > SETTINGS.identity.relink_max_distance_px
        ):
            return inf, "DISTANCE_TOO_LARGE", frame_gap

        bbox_ratio_diff = self._get_bbox_size_ratio_diff(
            session.last_bbox,
            bbox,
        )
        if (
            bbox_ratio_diff is not None
            and bbox_ratio_diff > SETTINGS.identity.relink_max_bbox_size_ratio_diff
        ):
            return inf, "BBOX_RATIO_TOO_LARGE", frame_gap

        score = 0.0
        reason_parts: list[str] = []
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
            frame_gap / max(1, SETTINGS.identity.max_lost_frames)
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

        return score, ", ".join(reason_parts), frame_gap

    def _try_relink_lost_session(
        self,
        yolo_track_id: int,
        bbox: BBox | None,
        features: PoseFeatures,
        trusted_feet_point: Point | None,
        trusted_feet_source: str,
        trusted_inside: bool | None,
    ) -> IdentityUpdateResult | None:
        best_session: PersonSession | None = None
        best_score = inf
        best_reason = "NO_GHOST_MATCH"
        best_gap = 0

        for session in self.person_sessions.values():
            if session.person_uid in self.seen_person_uids_this_frame:
                continue
            score, reason, frame_gap = self.score_relink_candidate(
                session,
                bbox,
                features,
            )
            if score < best_score:
                best_session = session
                best_score = score
                best_reason = reason
                best_gap = frame_gap

        if (
            best_session is None
            or best_score > SETTINGS.identity.relink_score_threshold
        ):
            return None

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

        self._log_event(
            "PERSON_RELINKED "
            f"person_id={self._format_person_uid(best_session.person_uid)} "
            f"old_yolo_id={old_yolo_track_id if old_yolo_track_id is not None else 'NA'} "
            f"new_yolo_id={yolo_track_id} gap={best_gap} score={best_score:.2f} "
            f"reason={best_reason}"
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
            relink_score=best_score,
            relink_frame_gap=best_gap,
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
                relink_score=relink_score,
                relink_frame_gap=relink_frame_gap,
            )

        if trusted_inside is False:
            session.exit_confirm_hits += 1
            if session.exit_confirm_hits >= SETTINGS.identity.exit_confirm_frames:
                if not session.exited_counted:
                    session.exited_counted = True
                    self.exited_count += 1
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
                    relink_score=relink_score,
                    relink_frame_gap=relink_frame_gap,
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
                relink_score=relink_score,
                relink_frame_gap=relink_frame_gap,
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
            relink_score=relink_score,
            relink_frame_gap=relink_frame_gap,
        )

    def _update_candidate_session(
        self,
        yolo_track_id: int,
        bbox: BBox | None,
        features: PoseFeatures,
        trusted_feet_point: Point | None,
        trusted_feet_source: str,
        trusted_inside: bool | None,
    ) -> IdentityUpdateResult:
        candidate = self.candidate_sessions.get(yolo_track_id)
        if candidate is None:
            candidate = CandidateSession(
                current_yolo_track_id=yolo_track_id,
                first_seen_frame=self.current_frame_index,
                last_seen_frame=self.current_frame_index,
                last_bbox=bbox,
                last_bbox_center=self._get_bbox_center(bbox),
                last_anchor_point=self._select_anchor_point(features, bbox),
                last_trusted_feet_point=trusted_feet_point,
                last_trusted_feet_source=trusted_feet_source,
            )
            self.candidate_sessions[yolo_track_id] = candidate
            self._log_event(f"CANDIDATE_SEEN yolo_id={yolo_track_id}")
        else:
            candidate.last_seen_frame = self.current_frame_index
            candidate.last_bbox = bbox
            candidate.last_bbox_center = self._get_bbox_center(bbox)
            candidate.last_anchor_point = self._select_anchor_point(features, bbox)
            candidate.last_trusted_feet_point = trusted_feet_point
            candidate.last_trusted_feet_source = trusted_feet_source

        if trusted_inside is True:
            candidate.entry_confirm_hits += 1
            if candidate.entry_confirm_hits >= SETTINGS.identity.entry_confirm_frames:
                session = self._promote_candidate(candidate, features)
                self.seen_person_uids_this_frame.add(session.person_uid)
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
                    identity_gate_reason="ENTERED_BY_FEET",
                    relink_score=None,
                    relink_frame_gap=0,
                )

            candidate.last_gate_reason = (
                f"WAIT_ENTRY_CONFIRM {candidate.entry_confirm_hits}/"
                f"{SETTINGS.identity.entry_confirm_frames}"
            )
            return self._create_candidate_result(
                yolo_track_id,
                trusted_feet_source,
                candidate.last_gate_reason,
            )

        candidate.entry_confirm_hits = 0
        candidate.last_gate_reason = (
            "TRUSTED_FEET_OUTSIDE"
            if trusted_inside is False
            else "WAIT_TRUSTED_FEET_ENTER"
        )
        return self._create_candidate_result(
            yolo_track_id,
            trusted_feet_source,
            candidate.last_gate_reason,
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
                self.candidate_sessions.pop(yolo_track_id, None)
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

        relink_result = self._try_relink_lost_session(
            yolo_track_id,
            bbox_tuple,
            features,
            trusted_feet_point,
            trusted_feet_source,
            trusted_inside,
        )
        if relink_result is not None:
            self.candidate_sessions.pop(yolo_track_id, None)
            return relink_result

        return self._update_candidate_session(
            yolo_track_id,
            bbox_tuple,
            features,
            trusted_feet_point,
            trusted_feet_source,
            trusted_inside,
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
            del self.candidate_sessions[yolo_track_id]

    def get_retained_person_uids(self) -> set[PersonUID]:
        return {
            person_uid
            for person_uid, session in self.person_sessions.items()
            if session.status != "COMPLETED"
        }

    def get_active_or_lost_inside_count(self) -> int:
        return sum(
            1
            for session in self.person_sessions.values()
            if session.status in ("ACTIVE", "LOST")
            and session.lifecycle_state in ("ACTIVE_INSIDE", "LOST_INSIDE")
            and not session.exited_counted
        )

    def augment_analysis_result(
        self,
        analysis: AnalysisResult,
        identity_result: IdentityUpdateResult,
    ) -> None:
        identity_fields: dict[str, object] = {
            "person_uid": identity_result.person_uid,
            "person_uid_label": identity_result.person_uid_label,
            "yolo_track_id": identity_result.yolo_track_id,
            "previous_yolo_track_id": identity_result.previous_yolo_track_id,
            "identity_status": identity_result.identity_status,
            "session_status": identity_result.session_status,
            "session_lifecycle": identity_result.session_lifecycle,
            "identity_debug": identity_result.identity_debug,
            "identity_feet_source": identity_result.identity_feet_source,
            "identity_gate_reason": identity_result.identity_gate_reason,
            "has_active_person_id": identity_result.has_active_person_id,
            "relink_score": identity_result.relink_score,
            "relink_frame_gap": identity_result.relink_frame_gap,
            "entered_count": self.entered_count,
            "exited_count": self.exited_count,
            "active_or_lost_inside_count": self.get_active_or_lost_inside_count(),
        }
        analysis.update(identity_fields)

        debug_info = analysis.get("debug_info")
        if isinstance(debug_info, dict):
            debug_info.update(identity_fields)

    def build_overlay_result(
        self,
        identity_result: IdentityUpdateResult,
    ) -> AnalysisResult:
        color = (
            SETTINGS.violation.outside_color
            if identity_result.session_lifecycle == "CANDIDATE_OUTSIDE"
            else (0, 165, 255)
            if identity_result.session_lifecycle == "LOST_INSIDE"
            else (255, 255, 0)
            if identity_result.session_lifecycle == "EXITED"
            else SETTINGS.violation.safe_color
        )
        status = identity_result.session_lifecycle
        result: AnalysisResult = {
            "status": status,
            "display_status": status,
            "color": color,
            "track_id": identity_result.person_uid or -1,
            "person_uid": identity_result.person_uid,
            "person_uid_label": identity_result.person_uid_label,
            "yolo_track_id": identity_result.yolo_track_id,
            "previous_yolo_track_id": identity_result.previous_yolo_track_id,
            "identity_status": identity_result.identity_status,
            "session_status": identity_result.session_status,
            "session_lifecycle": identity_result.session_lifecycle,
            "identity_debug": identity_result.identity_debug,
            "identity_feet_source": identity_result.identity_feet_source,
            "identity_gate_reason": identity_result.identity_gate_reason,
            "has_active_person_id": identity_result.has_active_person_id,
            "relink_score": identity_result.relink_score,
            "relink_frame_gap": identity_result.relink_frame_gap,
            "entered_count": self.entered_count,
            "exited_count": self.exited_count,
            "active_or_lost_inside_count": self.get_active_or_lost_inside_count(),
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
