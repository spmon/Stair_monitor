from __future__ import annotations

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
    print(
        "[PERF] "
        f"fps={fps:.1f} "
        f"model={avg_ms['model']:.1f}ms "
        f"analyze={avg_ms['analyze']:.1f}ms "
        f"direction={avg_ms['direction']:.1f}ms "
        f"lane={avg_ms['lane']:.1f}ms "
        f"hold={avg_ms['hold']:.1f}ms "
        f"carry={avg_ms['carry']:.1f}ms "
        f"standing={avg_ms['standing']:.1f}ms "
        f"backward={avg_ms['backward']:.1f}ms "
        f"overlay={avg_ms['overlay']:.1f}ms "
        f"video={avg_ms['video_write_show']:.1f}ms "
        f"total={avg_ms['total']:.1f}ms"
    )
