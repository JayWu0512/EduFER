from __future__ import annotations

import numpy as np

from edufer.classification.base import EmotionClassifier
from edufer.core.schemas import EmotionPrediction


class PlaceholderEmotionClassifier(EmotionClassifier):
    _SIGNALS = {
        "bored": (
            "Low engagement signal",
            "The student may need a pacing change, shorter segment, or a quick interactive check-in.",
        ),
        "engaged": (
            "High engagement signal",
            "Continue the current flow and maintain the same level of difficulty.",
        ),
        "confused": (
            "Potential confusion signal",
            "Slow down, add a worked example, or restate the concept in simpler language.",
        ),
        "frustrated": (
            "Potential frustration signal",
            "Offer a break, scaffold the task, or provide targeted guidance before continuing.",
        ),
    }

    def __init__(self, default_label: str = "bored", default_confidence: float = 0.88) -> None:
        self._default_label = default_label.lower()
        self._default_confidence = default_confidence

    @property
    def name(self) -> str:
        return "Placeholder Emotion Classifier"

    @property
    def is_placeholder(self) -> bool:
        return True

    def classify(self, face_image: np.ndarray) -> EmotionPrediction:
        signal, recommendation = self._SIGNALS.get(
            self._default_label,
            self._SIGNALS["bored"],
        )
        return EmotionPrediction(
            label=self._default_label,
            confidence=self._default_confidence,
            educational_signal=signal,
            recommendation=recommendation,
            placeholder=True,
        )
