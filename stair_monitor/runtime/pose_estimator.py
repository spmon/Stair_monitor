from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from stair_monitor.runtime.runtime_types import Frame, PoseDetection


def _read_track_confidences(
    first_result: object,
    detection_count: int,
) -> tuple[float, ...]:
    boxes = getattr(first_result, "boxes", None)
    if boxes is None:
        return tuple(0.0 for _ in range(detection_count))

    confidence_tensor = getattr(boxes, "conf", None)
    if confidence_tensor is None:
        return tuple(0.0 for _ in range(detection_count))

    confidence_values = confidence_tensor.tolist()
    if not isinstance(confidence_values, list):
        return tuple(0.0 for _ in range(detection_count))

    confidences: list[float] = []
    for raw_confidence in confidence_values[:detection_count]:
        if isinstance(raw_confidence, (int, float)):
            confidences.append(float(raw_confidence))
        else:
            confidences.append(0.0)

    if len(confidences) < detection_count:
        confidences.extend(0.0 for _ in range(detection_count - len(confidences)))

    return tuple(confidences)


class YoloPoseEstimator:
    """YOLO pose estimation boundary for the Windows/demo runtime."""

    def __init__(
        self,
        model_path: str,
        confidence: float,
        iou: float,
        tracker_config: str,
    ) -> None:
        self._model = YOLO(model_path)
        self._confidence = confidence
        self._iou = iou
        self._tracker_config = tracker_config

    def estimate(self, frame: Frame) -> tuple[PoseDetection, ...]:
        results = self._model.track(
            frame,
            conf=self._confidence,
            persist=True,
            classes=[0],
            tracker=self._tracker_config,
            verbose=False,
            iou=self._iou,
        )

        if not results:
            return ()

        first_result = results[0]
        boxes_container = getattr(first_result, "boxes", None)
        keypoints_container = getattr(first_result, "keypoints", None)
        if (
            boxes_container is None
            or getattr(boxes_container, "id", None) is None
            or keypoints_container is None
        ):
            return ()

        boxes = boxes_container.xyxy.cpu().numpy()
        track_ids_raw = boxes_container.id.int().tolist()
        keypoints = keypoints_container.data.cpu().numpy()
        if not isinstance(track_ids_raw, list):
            return ()

        track_ids: list[int] = []
        for raw_track_id in track_ids_raw:
            if isinstance(raw_track_id, (int, float)):
                track_ids.append(int(raw_track_id))
            else:
                return ()

        confidences = _read_track_confidences(first_result, len(track_ids))

        detections: list[PoseDetection] = []
        for box, track_id, keypoint_set, confidence in zip(
            boxes,
            track_ids,
            keypoints,
            confidences,
        ):
            detections.append(
                PoseDetection(
                    yolo_track_id=track_id,
                    bbox=(
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3]),
                    ),
                    keypoints=np.asarray(
                        keypoint_set,
                        dtype=np.float32,
                    ).astype(np.float32, copy=False),
                    confidence=confidence,
                )
            )
        return tuple(detections)
