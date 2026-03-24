from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from edufer import __version__
from edufer.api.middleware import MaxRequestSizeMiddleware, SecurityHeadersMiddleware
from edufer.api.routes import router as api_router
from edufer.classification.placeholder_classifier import PlaceholderEmotionClassifier
from edufer.core.settings import DEFAULT_CONFIG_PATH, AppSettings
from edufer.detection.unavailable_detector import UnavailableFaceDetector
from edufer.detection.yolov8_face_detector import YOLOv8FaceDetector
from edufer.pipeline.engagement_pipeline import EngagementPipeline


def _build_detector(settings: AppSettings):
    try:
        return YOLOv8FaceDetector(
            model_path=settings.detection.model_path,
            input_size=settings.detection.input_size,
            confidence_threshold=settings.detection.confidence_threshold,
            iou_threshold=settings.detection.iou_threshold,
        )
    except FileNotFoundError as exc:
        return UnavailableFaceDetector(str(exc))


def create_app(config_path: str | Path = DEFAULT_CONFIG_PATH) -> FastAPI:
    settings = AppSettings.from_file(config_path)
    detector = _build_detector(settings)
    classifier = PlaceholderEmotionClassifier(
        default_label=settings.classification.default_label,
        default_confidence=settings.classification.default_confidence,
    )
    pipeline = EngagementPipeline(detector=detector, classifier=classifier)

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        summary="Educational engagement demo with webcam capture, YOLO face detection, and a pluggable DL module.",
    )
    app.state.settings = settings
    app.state.pipeline = pipeline

    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.security.trusted_hosts,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.security.cors_allowed_origins,
        allow_credentials=settings.security.cors_allow_credentials,
        allow_methods=settings.security.cors_allowed_methods,
        allow_headers=settings.security.cors_allowed_headers,
    )
    app.add_middleware(
        MaxRequestSizeMiddleware,
        max_request_bytes=settings.security.max_request_bytes,
    )
    app.add_middleware(
        SecurityHeadersMiddleware,
        content_security_policy=settings.security.content_security_policy,
        permissions_policy=settings.security.permissions_policy,
    )
    if settings.security.enable_https_redirect:
        app.add_middleware(HTTPSRedirectMiddleware)

    app.mount("/static", StaticFiles(directory=settings.web.static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(settings.web.static_dir / "index.html")

    app.include_router(api_router, prefix="/api")
    return app


app = create_app()
