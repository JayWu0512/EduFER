from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _parse_csv(value: str | None, fallback: list[str]) -> list[str]:
    if value is None:
        return fallback
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    return parsed or fallback


def _parse_bool(value: str | None, fallback: bool) -> bool:
    if value is None:
        return fallback
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class DetectionSettings:
    model_path: Path
    download_url: str
    sha256: str
    input_size: int
    confidence_threshold: float
    iou_threshold: float


@dataclass(slots=True)
class ClassificationSettings:
    default_label: str
    default_confidence: float


@dataclass(slots=True)
class WebSettings:
    static_dir: Path
    capture_interval_ms: int


@dataclass(slots=True)
class SecuritySettings:
    cors_allowed_origins: list[str]
    cors_allowed_methods: list[str]
    cors_allowed_headers: list[str]
    cors_allow_credentials: bool
    trusted_hosts: list[str]
    max_request_bytes: int
    enable_https_redirect: bool
    content_security_policy: str
    permissions_policy: str


@dataclass(slots=True)
class AppSettings:
    app_name: str
    environment: str
    host: str
    port: int
    detection: DetectionSettings
    classification: ClassificationSettings
    web: WebSettings
    security: SecuritySettings

    @classmethod
    def from_file(cls, config_path: str | Path) -> "AppSettings":
        resolved_path = _resolve_path(config_path)
        raw = json.loads(resolved_path.read_text(encoding="utf-8"))
        detection = raw["detection"]
        classification = raw["classification"]
        web = raw["web"]
        security = raw["security"]
        return cls(
            app_name=raw["app_name"],
            environment=raw["environment"],
            host=os.getenv("EDUFER_HOST", raw["host"]),
            port=int(os.getenv("EDUFER_PORT", raw["port"])),
            detection=DetectionSettings(
                model_path=_resolve_path(detection["model_path"]),
                download_url=detection["download_url"],
                sha256=detection["sha256"],
                input_size=int(detection["input_size"]),
                confidence_threshold=float(detection["confidence_threshold"]),
                iou_threshold=float(detection["iou_threshold"]),
            ),
            classification=ClassificationSettings(
                default_label=classification["default_label"],
                default_confidence=float(classification["default_confidence"]),
            ),
            web=WebSettings(
                static_dir=_resolve_path(web["static_dir"]),
                capture_interval_ms=int(web["capture_interval_ms"]),
            ),
            security=SecuritySettings(
                cors_allowed_origins=_parse_csv(
                    os.getenv("EDUFER_ALLOWED_ORIGINS"),
                    list(security["cors_allowed_origins"]),
                ),
                cors_allowed_methods=_parse_csv(
                    os.getenv("EDUFER_ALLOWED_METHODS"),
                    list(security["cors_allowed_methods"]),
                ),
                cors_allowed_headers=_parse_csv(
                    os.getenv("EDUFER_ALLOWED_HEADERS"),
                    list(security["cors_allowed_headers"]),
                ),
                cors_allow_credentials=_parse_bool(
                    os.getenv("EDUFER_ALLOW_CREDENTIALS"),
                    bool(security["cors_allow_credentials"]),
                ),
                trusted_hosts=_parse_csv(
                    os.getenv("EDUFER_TRUSTED_HOSTS"),
                    list(security["trusted_hosts"]),
                ),
                max_request_bytes=int(
                    os.getenv("EDUFER_MAX_REQUEST_BYTES", security["max_request_bytes"])
                ),
                enable_https_redirect=_parse_bool(
                    os.getenv("EDUFER_ENABLE_HTTPS_REDIRECT"),
                    bool(security["enable_https_redirect"]),
                ),
                content_security_policy=os.getenv(
                    "EDUFER_CONTENT_SECURITY_POLICY",
                    security["content_security_policy"],
                ),
                permissions_policy=os.getenv(
                    "EDUFER_PERMISSIONS_POLICY",
                    security["permissions_policy"],
                ),
            ),
        )


DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "app_config.json"
