from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from edufer.core.schemas import EmotionPrediction


class EmotionClassifier(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable classifier name."""

    @property
    def is_placeholder(self) -> bool:
        return False

    @abstractmethod
    def classify(self, face_image: np.ndarray) -> EmotionPrediction:
        """Return a prediction for a cropped face image."""
