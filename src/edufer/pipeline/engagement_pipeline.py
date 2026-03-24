from __future__ import annotations

import numpy as np

from edufer.classification.base import EmotionClassifier
from edufer.core.schemas import BoundingBox, EngagementResult
from edufer.detection.base import FaceDetector


class EngagementPipeline:
    def __init__(self, detector: FaceDetector, classifier: EmotionClassifier) -> None:
        self._detector = detector
        self._classifier = classifier

    @property
    def detector(self) -> FaceDetector:
        return self._detector

    @property
    def classifier(self) -> EmotionClassifier:
        return self._classifier

    def analyze_frame(self, frame: np.ndarray) -> EngagementResult:
        height, width = frame.shape[:2]
        notes: list[str] = []

        if not self._detector.is_ready:
            notes.append(self._detector.status_message)
            notes.append(
                "The placeholder classifier is ready, but face detection needs the YOLO ONNX weights."
            )
            return self._build_result(
                width=width,
                height=height,
                notes=notes,
            )

        detections = self._detector.detect(frame)
        if not detections:
            notes.append("No face detected in the current frame.")
            return self._build_result(
                width=width,
                height=height,
                notes=notes,
            )

        primary_face = self._select_primary_face(detections)
        face_crop = frame[primary_face.y1 : primary_face.y2, primary_face.x1 : primary_face.x2]
        prediction = self._classifier.classify(face_crop)
        if prediction.placeholder:
            notes.append(
                "The emotion module is intentionally stubbed and currently always reports 'bored'."
            )

        return self._build_result(
            width=width,
            height=height,
            primary_face=primary_face,
            prediction=prediction,
            face_count=len(detections),
            notes=notes,
        )

    def _build_result(
        self,
        *,
        width: int,
        height: int,
        primary_face: BoundingBox | None = None,
        prediction=None,
        face_count: int = 0,
        notes: list[str] | None = None,
    ) -> EngagementResult:
        return EngagementResult(
            face_detected=primary_face is not None,
            face_count=face_count,
            image_width=width,
            image_height=height,
            detector_name=self._detector.name,
            detector_ready=self._detector.is_ready,
            classifier_name=self._classifier.name,
            classifier_placeholder=self._classifier.is_placeholder,
            primary_face=primary_face,
            prediction=prediction,
            notes=notes or [],
        )

    @staticmethod
    def _select_primary_face(detections: list[BoundingBox]) -> BoundingBox:
        return max(detections, key=lambda box: box.area)
