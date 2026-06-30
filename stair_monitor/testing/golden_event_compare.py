from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from stair_monitor.testing.golden_event_writer import (
    GoldenBehaviorEvent,
    load_golden_behavior_events,
)


@dataclass(frozen=True, slots=True)
class NormalizedGoldenBehaviorEvent:
    frame_index: int
    analysis_subject_id: str
    person_uid: str
    warnings: tuple[str, ...]
    direction: str
    status: str


@dataclass(frozen=True, slots=True)
class GoldenBehaviorComparison:
    baseline_count: int
    current_count: int
    missing_events: tuple[NormalizedGoldenBehaviorEvent, ...]
    new_events: tuple[NormalizedGoldenBehaviorEvent, ...]

    @property
    def is_match(self) -> bool:
        return not self.missing_events and not self.new_events


def normalize_golden_behavior_event(
    event: GoldenBehaviorEvent,
) -> NormalizedGoldenBehaviorEvent:
    return NormalizedGoldenBehaviorEvent(
        frame_index=event.frame_index,
        analysis_subject_id=event.analysis_subject_id,
        person_uid=event.person_uid,
        warnings=event.warnings,
        direction=event.direction,
        status=event.status,
    )


def _build_event_counter(
    events: tuple[GoldenBehaviorEvent, ...],
) -> Counter[NormalizedGoldenBehaviorEvent]:
    return Counter(normalize_golden_behavior_event(event) for event in events)


def _expand_counter_diff(
    event_counter: Counter[NormalizedGoldenBehaviorEvent],
) -> tuple[NormalizedGoldenBehaviorEvent, ...]:
    expanded_events: list[NormalizedGoldenBehaviorEvent] = []
    for event, count in sorted(
        event_counter.items(),
        key=lambda item: (
            item[0].frame_index,
            item[0].person_uid,
            item[0].analysis_subject_id,
            item[0].warnings,
            item[0].direction,
            item[0].status,
        ),
    ):
        expanded_events.extend(event for _ in range(count))
    return tuple(expanded_events)


def compare_golden_events(
    baseline_events: tuple[GoldenBehaviorEvent, ...],
    current_events: tuple[GoldenBehaviorEvent, ...],
) -> GoldenBehaviorComparison:
    baseline_counter = _build_event_counter(baseline_events)
    current_counter = _build_event_counter(current_events)

    missing_counter = baseline_counter - current_counter
    new_counter = current_counter - baseline_counter

    return GoldenBehaviorComparison(
        baseline_count=len(baseline_events),
        current_count=len(current_events),
        missing_events=_expand_counter_diff(missing_counter),
        new_events=_expand_counter_diff(new_counter),
    )


def compare_golden_event_files(
    baseline_path: str,
    current_path: str,
) -> GoldenBehaviorComparison:
    baseline_events = load_golden_behavior_events(baseline_path)
    current_events = load_golden_behavior_events(current_path)
    return compare_golden_events(baseline_events, current_events)
