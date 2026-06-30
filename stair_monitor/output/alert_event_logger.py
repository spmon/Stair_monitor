from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import TextIO

from loguru import logger

from stair_monitor.runtime.runtime_types import ViolationEvent


class AlertEventLogger:
    def __init__(
        self,
        log_dir: str,
        cooldown_frames: int,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.log_dir = Path(log_dir)
        self.cooldown_frames = max(1, cooldown_frames)
        self.file_path: Path | None = None
        self._last_logged_frame_by_key: dict[tuple[str, str], int] = {}
        self._file: TextIO | None = None
        self._writer: csv._writer | None = None

        if not self.enabled:
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.log_dir / (
            f"violations_{datetime.now():%Y%m%d_%H%M%S}.csv"
        )
        self._file = self.file_path.open(
            "w",
            newline="",
            encoding="utf-8",
        )
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            [
                "timestamp",
                "frame_index",
                "person_uid",
                "analysis_subject_id",
                "track_id",
                "direction",
                "warnings",
                "status",
            ]
        )
        self._file.flush()

    def _build_identity_key(self, event: ViolationEvent) -> str:
        if event.person_uid:
            return event.person_uid
        if event.analysis_subject_id:
            return event.analysis_subject_id
        if event.yolo_track_id is not None:
            return f"track:{event.yolo_track_id}"
        return "UNKNOWN"

    def _filter_warnings_by_cooldown(
        self,
        event: ViolationEvent,
        current_frame_index: int,
    ) -> tuple[str, ...]:
        identity_key = self._build_identity_key(event)
        warnings_to_log: list[str] = []

        for warning in event.warnings:
            cooldown_key = (identity_key, warning)
            last_logged_frame = self._last_logged_frame_by_key.get(cooldown_key)
            if (
                last_logged_frame is not None
                and current_frame_index - last_logged_frame < self.cooldown_frames
            ):
                continue

            self._last_logged_frame_by_key[cooldown_key] = current_frame_index
            warnings_to_log.append(warning)

        return tuple(warnings_to_log)

    def log_if_needed(
        self,
        event: ViolationEvent,
        current_frame_index: int,
    ) -> None:
        if not self.enabled or not event.warnings:
            return

        warnings_to_log = self._filter_warnings_by_cooldown(
            event,
            current_frame_index,
        )
        if not warnings_to_log:
            return

        logged_event = ViolationEvent(
            timestamp=event.timestamp,
            frame_index=event.frame_index,
            yolo_track_id=event.yolo_track_id,
            person_uid=event.person_uid,
            analysis_subject_id=event.analysis_subject_id,
            direction=event.direction,
            warnings=warnings_to_log,
            status=event.status,
        )

        warnings_text = ";".join(logged_event.warnings)
        person_text = (
            logged_event.person_uid
            or logged_event.analysis_subject_id
            or "UNKNOWN"
        )
        track_text = (
            str(logged_event.yolo_track_id)
            if logged_event.yolo_track_id is not None
            else "NA"
        )

        logger.bind(event="violation").warning(
            (
                "ALERT time={} frame={} person={} subject={} track={} dir={} "
                'warnings={} status="{}"'
            ),
            logged_event.timestamp,
            logged_event.frame_index,
            person_text,
            logged_event.analysis_subject_id,
            track_text,
            logged_event.direction,
            warnings_text,
            logged_event.status,
        )

        if self._writer is None or self._file is None:
            return

        self._writer.writerow(
            [
                logged_event.timestamp,
                logged_event.frame_index,
                logged_event.person_uid,
                logged_event.analysis_subject_id,
                (
                    ""
                    if logged_event.yolo_track_id is None
                    else logged_event.yolo_track_id
                ),
                logged_event.direction,
                warnings_text,
                logged_event.status,
            ]
        )
        self._file.flush()

    def log_events(
        self,
        events: tuple[ViolationEvent, ...],
        current_frame_index: int,
    ) -> None:
        for event in events:
            self.log_if_needed(event, current_frame_index)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None
