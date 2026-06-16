from __future__ import annotations

from typing import TYPE_CHECKING

from stair_monitor.common.types import Point

if TYPE_CHECKING:
    from stair_monitor.core.analyzer import BehaviorAnalyzer


def update_standing_history(
    analyzer: BehaviorAnalyzer,
    track_id: int,
    p_motion: Point | None,
):
    return analyzer.update_standing_still(track_id, p_motion)


def evaluate_standing_still(
    analyzer: BehaviorAnalyzer,
    track_id: int,
    p_motion: Point | None,
    inside_stairs: bool,
) -> dict[str, object]:
    standing_raw = False
    standing_hits = 0
    standing_still_confirmed = False
    standing_motion_range = None
    standing_len = 0

    if inside_stairs:
        (
            standing_raw,
            standing_hits,
            standing_still_confirmed,
            standing_motion_range,
            standing_len,
        ) = update_standing_history(analyzer, track_id, p_motion)

    return {
        "standing_raw": standing_raw,
        "standing_hits": standing_hits,
        "standing_still_confirmed": standing_still_confirmed,
        "standing_motion_range": standing_motion_range,
        "standing_len": standing_len,
    }
