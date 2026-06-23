from __future__ import annotations

from loguru import logger

from stair_monitor.common.types import PerfStats

PERF_FIELDS = (
    "model",
    "analyze",
    "direction",
    "lane",
    "hold",
    "carry",
    "standing",
    "backward",
    "overlay",
    "video_write_show",
    "total",
)


def new_perf_totals() -> PerfStats:
    return {field: 0.0 for field in PERF_FIELDS}


def accumulate_analysis_perf(
    perf_totals: PerfStats,
    analysis_perf: PerfStats | None,
) -> None:
    if not analysis_perf:
        return

    for field in ("analyze", "direction", "lane", "hold", "carry", "standing", "backward"):
        perf_totals[field] += analysis_perf.get(field, 0.0)


def log_perf(
    perf_totals: PerfStats,
    frame_count: int,
    elapsed_s: float,
) -> None:
    if frame_count <= 0 or elapsed_s <= 0:
        return

    fps = frame_count / elapsed_s
    avg_ms = {
        field: perf_totals[field] / frame_count
        for field in PERF_FIELDS
    }
    logger.info(
        (
            "[PERF] fps={:.1f} model={:.1f}ms analyze={:.1f}ms direction={:.1f}ms "
            "lane={:.1f}ms hold={:.1f}ms carry={:.1f}ms standing={:.1f}ms "
            "backward={:.1f}ms overlay={:.1f}ms video={:.1f}ms total={:.1f}ms"
        ),
        fps,
        avg_ms["model"],
        avg_ms["analyze"],
        avg_ms["direction"],
        avg_ms["lane"],
        avg_ms["hold"],
        avg_ms["carry"],
        avg_ms["standing"],
        avg_ms["backward"],
        avg_ms["overlay"],
        avg_ms["video_write_show"],
        avg_ms["total"],
    )
