from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from stair_monitor.common.types import Point
from stair_monitor.config.settings import SETTINGS

STEP_INDEX_OK_REASON = "OK"
STEP_INDEX_OK_WITH_TOLERANCE_REASON = "OK_WITH_TOLERANCE"
LOW_ANKLE_CONF_REASON = "LOW_ANKLE_CONF"
NO_STEP_BANDS_REASON = "NO_STEP_BANDS"
NEAR_STEP_BOUNDARY_REASON = "NEAR_STEP_BOUNDARY"
OUTSIDE_ALL_BANDS_REASON = "OUTSIDE_ALL_BANDS"
POINT_INVALID_REASON = "POINT_INVALID"

POINT_NEAR_STEP_BOUNDARY_REASON = NEAR_STEP_BOUNDARY_REASON
POINT_OUTSIDE_STEP_BANDS_REASON = OUTSIDE_ALL_BANDS_REASON
STEP_INDEX_AVAILABLE_REASON = STEP_INDEX_OK_REASON


@dataclass(frozen=True, slots=True)
class StepLine:
    id: int
    p1: Point
    p2: Point


@dataclass(frozen=True, slots=True)
class StepBand:
    id: int
    polygon: tuple[Point, Point, Point, Point]


@dataclass(frozen=True, slots=True)
class StepIndexResult:
    step_index: int | None
    reason: str
    nearest_band_id: int | None
    nearest_signed_distance: float | None
    is_inside: bool
    is_near_boundary: bool


def _normalize_step_line_point(raw_point: object) -> Point | None:
    if not isinstance(raw_point, (list, tuple)) or len(raw_point) != 2:
        return None

    x_value, y_value = raw_point
    if not isinstance(x_value, (int, float)) or not isinstance(y_value, (int, float)):
        return None
    return (int(x_value), int(y_value))


def _normalize_step_line_id(raw_id: object, fallback_id: int) -> int:
    if raw_id is None:
        return fallback_id
    if isinstance(raw_id, int):
        return raw_id
    if isinstance(raw_id, float) and raw_id.is_integer():
        return int(raw_id)
    return fallback_id


def normalize_step_lines(raw_step_lines: object) -> list[StepLine]:
    if not isinstance(raw_step_lines, list):
        return []

    normalized_lines: list[StepLine] = []
    for fallback_id, raw_step_line in enumerate(raw_step_lines):
        if not isinstance(raw_step_line, dict):
            continue

        point_1 = _normalize_step_line_point(raw_step_line.get("p1"))
        point_2 = _normalize_step_line_point(raw_step_line.get("p2"))
        if point_1 is None or point_2 is None:
            continue

        normalized_lines.append(
            StepLine(
                id=_normalize_step_line_id(raw_step_line.get("id"), fallback_id),
                p1=point_1,
                p2=point_2,
            )
        )

    normalized_lines.sort(key=lambda step_line: step_line.id)
    return normalized_lines


def build_step_bands(step_lines: list[StepLine]) -> list[StepBand]:
    if len(step_lines) < 2:
        return []

    step_bands: list[StepBand] = []
    for band_index, (line_i, line_j) in enumerate(zip(step_lines, step_lines[1:])):
        step_bands.append(
            StepBand(
                id=band_index,
                polygon=(
                    line_i.p1,
                    line_i.p2,
                    line_j.p2,
                    line_j.p1,
                ),
            )
        )
    return step_bands


def _polygon_to_contour(
    polygon: tuple[Point, Point, Point, Point],
) -> np.ndarray:
    return np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))


def get_step_index_for_point(
    point: Point | None,
    step_bands: list[StepBand],
) -> StepIndexResult:
    if point is None:
        return StepIndexResult(
            step_index=None,
            reason=POINT_INVALID_REASON,
            nearest_band_id=None,
            nearest_signed_distance=None,
            is_inside=False,
            is_near_boundary=False,
        )

    if not step_bands:
        return StepIndexResult(
            step_index=None,
            reason=NO_STEP_BANDS_REASON,
            nearest_band_id=None,
            nearest_signed_distance=None,
            is_inside=False,
            is_near_boundary=False,
        )

    boundary_margin_px = max(0, int(SETTINGS.step_band.boundary_margin_px))
    outside_tolerance_px = max(0, int(SETTINGS.step_band.outside_tolerance_px))
    point_xy = (float(point[0]), float(point[1]))
    nearest_band_id = None
    nearest_signed_distance = None

    for step_band in step_bands:
        contour = _polygon_to_contour(step_band.polygon)
        signed_distance = float(cv2.pointPolygonTest(contour, point_xy, True))
        if (
            nearest_signed_distance is None
            or signed_distance > nearest_signed_distance
        ):
            nearest_band_id = step_band.id
            nearest_signed_distance = signed_distance

    if nearest_band_id is None or nearest_signed_distance is None:
        return StepIndexResult(
            step_index=None,
            reason=NO_STEP_BANDS_REASON,
            nearest_band_id=None,
            nearest_signed_distance=None,
            is_inside=False,
            is_near_boundary=False,
        )

    is_inside = nearest_signed_distance >= 0.0
    is_near_boundary = (
        boundary_margin_px > 0 and abs(nearest_signed_distance) < boundary_margin_px
    )

    if is_near_boundary:
        return StepIndexResult(
            step_index=None,
            reason=NEAR_STEP_BOUNDARY_REASON,
            nearest_band_id=nearest_band_id,
            nearest_signed_distance=nearest_signed_distance,
            is_inside=is_inside,
            is_near_boundary=True,
        )

    if is_inside:
        return StepIndexResult(
            step_index=nearest_band_id,
            reason=STEP_INDEX_OK_REASON,
            nearest_band_id=nearest_band_id,
            nearest_signed_distance=nearest_signed_distance,
            is_inside=True,
            is_near_boundary=False,
        )

    if abs(nearest_signed_distance) <= outside_tolerance_px:
        return StepIndexResult(
            step_index=nearest_band_id,
            reason=STEP_INDEX_OK_WITH_TOLERANCE_REASON,
            nearest_band_id=nearest_band_id,
            nearest_signed_distance=nearest_signed_distance,
            is_inside=False,
            is_near_boundary=False,
        )

    return StepIndexResult(
        step_index=None,
        reason=OUTSIDE_ALL_BANDS_REASON,
        nearest_band_id=nearest_band_id,
        nearest_signed_distance=nearest_signed_distance,
        is_inside=False,
        is_near_boundary=False,
    )


def get_step_index_reason_for_point(
    point: Point | None,
    step_bands: list[StepBand],
) -> tuple[int | None, str]:
    step_index_result = get_step_index_for_point(point, step_bands)
    return step_index_result.step_index, step_index_result.reason
