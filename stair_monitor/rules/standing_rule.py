from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypedDict

from stair_monitor.common.types import AnalysisSubjectID, Point

if TYPE_CHECKING:
    from stair_monitor.core.analyzer import BehaviorAnalyzer


class StandingAnalysisUpdate(TypedDict):
    standing_raw: bool
    standing_hits: int
    standing_still_confirmed: bool
    standing_motion_range: float | None
    standing_len: int


@dataclass(frozen=True, slots=True)
class StandingInput:
    subject_id: AnalysisSubjectID
    p_motion: Point | None
    inside_stairs: bool


@dataclass(frozen=True, slots=True)
class StandingResult:
    standing_raw: bool
    standing_hits: int
    standing_still_confirmed: bool
    standing_motion_range: float | None
    standing_len: int

    def to_analysis_update(self) -> StandingAnalysisUpdate:
        return {
            "standing_raw": self.standing_raw,
            "standing_hits": self.standing_hits,
            "standing_still_confirmed": self.standing_still_confirmed,
            "standing_motion_range": self.standing_motion_range,
            "standing_len": self.standing_len,
        }


class StandingHistoryUpdater(Protocol):
    def update_standing_still(
        self,
        track_id: AnalysisSubjectID,
        p_motion: Point | None,
    ) -> tuple[bool, int, bool, float | None, int]:
        ...


def build_standing_input(
    subject_id: AnalysisSubjectID,
    p_motion: Point | None,
    inside_stairs: bool,
) -> StandingInput:
    return StandingInput(
        subject_id=subject_id,
        p_motion=p_motion,
        inside_stairs=inside_stairs,
    )


def _empty_standing_result() -> StandingResult:
    return StandingResult(
        standing_raw=False,
        standing_hits=0,
        standing_still_confirmed=False,
        standing_motion_range=None,
        standing_len=0,
    )


def update_standing_history(
    history_updater: StandingHistoryUpdater,
    rule_input: StandingInput,
) -> StandingResult:
    (
        standing_raw,
        standing_hits,
        standing_still_confirmed,
        standing_motion_range,
        standing_len,
    ) = history_updater.update_standing_still(
        rule_input.subject_id,
        rule_input.p_motion,
    )
    return StandingResult(
        standing_raw=standing_raw,
        standing_hits=standing_hits,
        standing_still_confirmed=standing_still_confirmed,
        standing_motion_range=standing_motion_range,
        standing_len=standing_len,
    )


def evaluate_standing_still_typed(
    rule_input: StandingInput,
    history_updater: StandingHistoryUpdater,
) -> StandingResult:
    if not rule_input.inside_stairs:
        return _empty_standing_result()
    return update_standing_history(history_updater, rule_input)


def evaluate_standing_still(
    analyzer: BehaviorAnalyzer,
    track_id: AnalysisSubjectID,
    p_motion: Point | None,
    inside_stairs: bool,
) -> StandingAnalysisUpdate:
    rule_input = build_standing_input(track_id, p_motion, inside_stairs)
    result = evaluate_standing_still_typed(rule_input, analyzer)
    return result.to_analysis_update()
