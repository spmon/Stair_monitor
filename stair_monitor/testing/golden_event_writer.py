from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from stair_monitor.runtime.runtime_types import ViolationEvent


@dataclass(frozen=True, slots=True)
class GoldenBehaviorEvent:
    frame_index: int
    yolo_track_id: int | None
    analysis_subject_id: str
    person_uid: str
    direction: str
    status: str
    warnings: tuple[str, ...]


def _to_golden_behavior_event(event: ViolationEvent) -> GoldenBehaviorEvent:
    return GoldenBehaviorEvent(
        frame_index=event.frame_index,
        yolo_track_id=event.yolo_track_id,
        analysis_subject_id=event.analysis_subject_id,
        person_uid=event.person_uid,
        direction=event.direction,
        status=event.status,
        warnings=event.warnings,
    )


def _serialize_golden_behavior_event(event: GoldenBehaviorEvent) -> str:
    payload = {
        "frame_index": event.frame_index,
        "yolo_track_id": event.yolo_track_id,
        "analysis_subject_id": event.analysis_subject_id,
        "person_uid": event.person_uid,
        "direction": event.direction,
        "status": event.status,
        "warnings": list(event.warnings),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _parse_golden_behavior_event(raw_event: object) -> GoldenBehaviorEvent:
    if not isinstance(raw_event, dict):
        raise ValueError("Golden behavior event line must be a JSON object.")

    frame_index = raw_event.get("frame_index")
    yolo_track_id = raw_event.get("yolo_track_id")
    analysis_subject_id = raw_event.get("analysis_subject_id")
    person_uid = raw_event.get("person_uid")
    direction = raw_event.get("direction")
    status = raw_event.get("status")
    warnings = raw_event.get("warnings")

    if not isinstance(frame_index, int):
        raise ValueError("Golden behavior event frame_index must be an int.")
    if yolo_track_id is not None and not isinstance(yolo_track_id, int):
        raise ValueError("Golden behavior event yolo_track_id must be an int or null.")
    if not isinstance(analysis_subject_id, str):
        raise ValueError("Golden behavior event analysis_subject_id must be a string.")
    if not isinstance(person_uid, str):
        raise ValueError("Golden behavior event person_uid must be a string.")
    if not isinstance(direction, str):
        raise ValueError("Golden behavior event direction must be a string.")
    if not isinstance(status, str):
        raise ValueError("Golden behavior event status must be a string.")
    if not isinstance(warnings, list) or not all(
        isinstance(warning, str) for warning in warnings
    ):
        raise ValueError("Golden behavior event warnings must be a list of strings.")

    return GoldenBehaviorEvent(
        frame_index=frame_index,
        yolo_track_id=yolo_track_id,
        analysis_subject_id=analysis_subject_id,
        person_uid=person_uid,
        direction=direction,
        status=status,
        warnings=tuple(warnings),
    )


class GoldenEventWriter:
    def __init__(self, output_path: str) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO | None = self.output_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        )

    def write_events(self, events: tuple[ViolationEvent, ...]) -> None:
        if self._file is None:
            raise RuntimeError("GoldenEventWriter is already closed.")

        for event in events:
            if not event.warnings:
                continue
            golden_event = _to_golden_behavior_event(event)
            self._file.write(_serialize_golden_behavior_event(golden_event))
            self._file.write("\n")

        self._file.flush()

    def log_events(
        self,
        events: tuple[ViolationEvent, ...],
        current_frame_index: int,
    ) -> None:
        del current_frame_index
        self.write_events(events)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def load_golden_behavior_events(path: str) -> tuple[GoldenBehaviorEvent, ...]:
    input_path = Path(path)
    events: list[GoldenBehaviorEvent] = []

    with input_path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                raw_event = json.loads(line)
                events.append(_parse_golden_behavior_event(raw_event))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid golden behavior event at line {line_number} in {input_path}: {exc}"
                ) from exc

    return tuple(events)
