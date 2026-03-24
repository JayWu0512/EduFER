from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from edufer.core.schemas import BoundingBox


class FaceDetector(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable detector name."""

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Whether the detector is ready for inference."""

    @property
    def status_message(self) -> str:
        return "ready" if self.is_ready else "not ready"

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[BoundingBox]:
        """Return all detected faces in a BGR frame."""
