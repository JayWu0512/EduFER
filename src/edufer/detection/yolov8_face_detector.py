from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from edufer.core.schemas import BoundingBox
from edufer.detection.base import FaceDetector


class YOLOv8FaceDetector(FaceDetector):
    def __init__(
        self,
        model_path: str | Path,
        input_size: int = 640,
        confidence_threshold: float = 0.6,
        iou_threshold: float = 0.45,
    ) -> None:
        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"YOLO face model was not found at {self._model_path}. "
                "Run the download script before starting the app."
            )

        self._net = None
        self._input_size = input_size
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold

    @property
    def name(self) -> str:
        return "YOLOv8n-Face (OpenCV DNN)"

    @property
    def is_ready(self) -> bool:
        return True

    def detect(self, frame: np.ndarray) -> list[BoundingBox]:
        net = self._load_net()
        height, width = frame.shape[:2]
        length = max(height, width)
        scale = length / self._input_size

        canvas = np.zeros((length, length, 3), dtype=np.uint8)
        canvas[0:height, 0:width] = frame

        blob = cv2.dnn.blobFromImage(
            canvas,
            scalefactor=1 / 255,
            size=(self._input_size, self._input_size),
            swapRB=True,
        )
        net.setInput(blob)
        outputs = net.forward()
        predictions = cv2.transpose(outputs[0])

        boxes: list[list[float]] = []
        scores: list[float] = []

        for row in predictions:
            score = float(np.max(row[4:])) if row.shape[0] > 5 else float(row[4])
            if score < self._confidence_threshold:
                continue

            x = float((row[0] - (0.5 * row[2])) * scale)
            y = float((row[1] - (0.5 * row[3])) * scale)
            w = float(row[2] * scale)
            h = float(row[3] * scale)
            boxes.append([x, y, w, h])
            scores.append(score)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            self._confidence_threshold,
            self._iou_threshold,
        )
        detections: list[BoundingBox] = []
        for index in self._normalize_indices(indices):
            x, y, w, h = boxes[index]
            x1 = max(0, int(round(x)))
            y1 = max(0, int(round(y)))
            x2 = min(width, int(round(x + w)))
            y2 = min(height, int(round(y + h)))

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                BoundingBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=scores[index],
                )
            )

        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections

    def _load_net(self):
        if self._net is None:
            self._net = cv2.dnn.readNetFromONNX(str(self._model_path))
        return self._net

    @staticmethod
    def _normalize_indices(indices: np.ndarray | tuple | list) -> list[int]:
        if indices is None or len(indices) == 0:
            return []

        normalized: list[int] = []
        for index in indices:
            if isinstance(index, (list, tuple, np.ndarray)):
                normalized.append(int(index[0]))
            else:
                normalized.append(int(index))
        return normalized
