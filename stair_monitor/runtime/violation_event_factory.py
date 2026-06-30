from __future__ import annotations

from datetime import datetime

from stair_monitor.runtime.runtime_types import (
    FrameAnalysisResult,
    OutputFrameContext,
    ViolationEvent,
)


def build_violation_events(
    frame_analysis: FrameAnalysisResult | OutputFrameContext,
) -> tuple[ViolationEvent, ...]:
    events: list[ViolationEvent] = []

    for person_result in frame_analysis.person_results:
        if not person_result.warnings:
            continue

        events.append(
            ViolationEvent(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                frame_index=person_result.frame_index,
                yolo_track_id=person_result.yolo_track_id,
                analysis_subject_id=person_result.analysis_subject_id,
                person_uid=person_result.person_uid,
                direction=person_result.direction,
                warnings=person_result.warnings,
                status=person_result.status,
            )
        )

    return tuple(events)
