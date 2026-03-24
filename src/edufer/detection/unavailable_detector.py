from __future__ import annotations

import numpy as np

from edufer.core.schemas import BoundingBox
from edufer.detection.base import FaceDetector


class UnavailableFaceDetector(FaceDetector):
    def __init__(self, reason: str) -> None:
        self._reason = reason

    @property
    def name(self) -> str:
        return "Unavailable Detector"

    @property
    def is_ready(self) -> bool:
        return False

    @property
    def status_message(self) -> str:
        return self._reason

    def detect(self, frame: np.ndarray) -> list[BoundingBox]:
        return []
