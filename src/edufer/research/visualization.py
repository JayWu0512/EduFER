from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from edufer.research.models import ModelPrediction
from edufer.research.preprocessing import ProcessedImageArtifacts


@dataclass(slots=True, frozen=True)
class AnnotatedPredictionFrame:
    model_name: str
    image_rgb: np.ndarray


class PredictionFrameBuilder:
    def build(
        self,
        *,
        artifacts: ProcessedImageArtifacts,
        prediction: ModelPrediction,
    ) -> AnnotatedPredictionFrame:
        canvas = artifacts.cropped_bgr.copy()
        banner_color = (34, 139, 34) if prediction.predicted_label == "engaged" else (32, 72, 160)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 56), banner_color, thickness=-1)
        cv2.putText(
            canvas,
            prediction.model_name,
            (12, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Pred: {prediction.predicted_label}",
            (12, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"P(engaged)={prediction.engaged_probability:.2f}",
            (12, canvas.shape[0] - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        return AnnotatedPredictionFrame(
            model_name=prediction.model_name,
            image_rgb=cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB),
        )
