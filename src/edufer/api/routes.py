from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from edufer.api.schemas import (
    AnalyzeFrameRequest,
    AnalyzeFrameResponse,
    BoundingBoxResponse,
    EmotionPredictionResponse,
    HealthResponse,
)
from edufer.pipeline.engagement_pipeline import EngagementPipeline
from edufer.utils.image import decode_data_url_to_bgr


router = APIRouter()


def _pipeline_from_request(request: Request) -> EngagementPipeline:
    return request.app.state.pipeline


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    pipeline = _pipeline_from_request(request)
    return HealthResponse(
        status="ok",
        detector_name=pipeline.detector.name,
        detector_ready=pipeline.detector.is_ready,
        classifier_name=pipeline.classifier.name,
        classifier_placeholder=pipeline.classifier.is_placeholder,
    )


@router.post("/analyze", response_model=AnalyzeFrameResponse)
def analyze_frame(payload: AnalyzeFrameRequest, request: Request) -> AnalyzeFrameResponse:
    pipeline = _pipeline_from_request(request)
    try:
        frame = decode_data_url_to_bgr(payload.image_data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result = pipeline.analyze_frame(frame)
    primary_face = None
    if result.primary_face:
        primary_face = BoundingBoxResponse(
            x1=result.primary_face.x1,
            y1=result.primary_face.y1,
            x2=result.primary_face.x2,
            y2=result.primary_face.y2,
            confidence=result.primary_face.confidence,
        )

    prediction = None
    if result.prediction:
        prediction = EmotionPredictionResponse(
            label=result.prediction.label,
            confidence=result.prediction.confidence,
            educational_signal=result.prediction.educational_signal,
            recommendation=result.prediction.recommendation,
            placeholder=result.prediction.placeholder,
        )

    return AnalyzeFrameResponse(
        face_detected=result.face_detected,
        face_count=result.face_count,
        image_width=result.image_width,
        image_height=result.image_height,
        detector_name=result.detector_name,
        detector_ready=result.detector_ready,
        classifier_name=result.classifier_name,
        classifier_placeholder=result.classifier_placeholder,
        primary_face=primary_face,
        prediction=prediction,
        notes=result.notes,
    )
