from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_IMAGE_PATH = Path("snapshot_2026-05-27T01-39-57.jpg")
DEFAULT_CONFIG_PATH = Path("camera_config.json")

SHAPE_STYLES = [
    {
        "name": "ROI",
        "color": (0, 255, 0),
        "closed": True,
        "filled": True,
        "alpha": 0.18,
    },
    {
        "name": "HANDRAIL_LEFT_POLY",
        "color": (0, 165, 255),
        "closed": True,
        "filled": True,
        "alpha": 0.3,
    },
    {
        "name": "HANDRAIL_RIGHT_POLY",
        "color": (255, 0, 255),
        "closed": True,
        "filled": True,
        "alpha": 0.3,
    },
    {
        "name": "STEP_BOTTOM",
        "color": (0, 0, 255),
        "closed": False,
        "filled": False,
        "alpha": 0.0,
    },
    {
        "name": "STEP_TOP",
        "color": (255, 191, 0),
        "closed": False,
        "filled": False,
        "alpha": 0.0,
    },
    {
        "name": "CENTER_LINE",
        "color": (0, 255, 255),
        "closed": False,
        "filled": False,
        "alpha": 0.0,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw points from camera_config.json on top of an image."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE_PATH,
        help=f"Input image path. Default: {DEFAULT_IMAGE_PATH}",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Camera config JSON path. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Default: <image_stem>_annotated<suffix>",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the annotated image in a window after saving.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_output_path(image_path: Path, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path
    return image_path.with_name(f"{image_path.stem}_annotated{image_path.suffix}")


def draw_text(image: np.ndarray, text: str, position: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = position
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def normalize_points(raw_points: list[list[int]]) -> np.ndarray:
    return np.asarray(raw_points, dtype=np.int32)


def draw_point_markers(
    image: np.ndarray,
    name: str,
    points: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    for index, point in enumerate(points):
        x, y = int(point[0]), int(point[1])
        cv2.circle(image, (x, y), 6, color, -1, cv2.LINE_AA)
        draw_text(image, f"[{index}]", (x + 10, y - 10), color)

    if len(points) > 0:
        x, y = int(points[0][0]), int(points[0][1])
        draw_text(image, name, (x + 10, y + 25), color)


def draw_shape(
    image: np.ndarray,
    name: str,
    raw_points: list[list[int]] | None,
    color: tuple[int, int, int],
    *,
    closed: bool,
    filled: bool,
    alpha: float,
) -> None:
    if not raw_points:
        return

    points = normalize_points(raw_points)
    poly_points = points.reshape((-1, 1, 2))

    if filled and len(points) >= 3:
        overlay = image.copy()
        cv2.fillPoly(overlay, [poly_points], color)
        cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, dst=image)

    if closed and len(points) >= 3:
        cv2.polylines(image, [poly_points], True, color, 3, cv2.LINE_AA)
    elif len(points) >= 2:
        cv2.polylines(image, [poly_points], False, color, 3, cv2.LINE_AA)

    draw_point_markers(image, name, points, color)


def annotate_image(image: np.ndarray, config: dict) -> np.ndarray:
    annotated = image.copy()

    for style in SHAPE_STYLES:
        draw_shape(
            annotated,
            name=style["name"],
            raw_points=config.get(style["name"]),
            color=style["color"],
            closed=style["closed"],
            filled=style["filled"],
            alpha=style["alpha"],
        )

    return annotated


def main() -> None:
    args = parse_args()

    image_path = args.image.resolve()
    config_path = args.config.resolve()
    output_path = build_output_path(image_path, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    config = load_config(config_path)
    annotated = annotate_image(image, config)

    success = cv2.imwrite(str(output_path), annotated)
    if not success:
        raise RuntimeError(f"Cannot write output image: {output_path}")

    print(f"Annotated image saved to: {output_path}")

    if args.show:
        cv2.imshow("Annotated Camera Points", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
