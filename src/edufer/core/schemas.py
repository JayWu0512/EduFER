from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass(slots=True)
class EmotionPrediction:
    label: str
    confidence: float
    educational_signal: str
    recommendation: str
    placeholder: bool = False


@dataclass(slots=True)
class EngagementResult:
    face_detected: bool
    face_count: int
    image_width: int
    image_height: int
    detector_name: str
    detector_ready: bool
    classifier_name: str
    classifier_placeholder: bool
    primary_face: BoundingBox | None = None
    prediction: EmotionPrediction | None = None
    notes: list[str] = field(default_factory=list)
