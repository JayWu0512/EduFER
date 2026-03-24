from __future__ import annotations

from pydantic import BaseModel, Field


class AnalyzeFrameRequest(BaseModel):
    image_data: str = Field(..., description="Base64 data URL captured from the webcam.")


class BoundingBoxResponse(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


class EmotionPredictionResponse(BaseModel):
    label: str
    confidence: float
    educational_signal: str
    recommendation: str
    placeholder: bool


class AnalyzeFrameResponse(BaseModel):
    face_detected: bool
    face_count: int
    image_width: int
    image_height: int
    detector_name: str
    detector_ready: bool
    classifier_name: str
    classifier_placeholder: bool
    primary_face: BoundingBoxResponse | None
    prediction: EmotionPredictionResponse | None
    notes: list[str]


class HealthResponse(BaseModel):
    status: str
    detector_name: str
    detector_ready: bool
    classifier_name: str
    classifier_placeholder: bool
