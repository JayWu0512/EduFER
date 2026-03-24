from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


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
class AppSettings:
    app_name: str
    environment: str
    host: str
    port: int
    detection: DetectionSettings
    classification: ClassificationSettings
    web: WebSettings

    @classmethod
    def from_file(cls, config_path: str | Path) -> "AppSettings":
        resolved_path = _resolve_path(config_path)
        raw = json.loads(resolved_path.read_text(encoding="utf-8"))
        detection = raw["detection"]
        classification = raw["classification"]
        web = raw["web"]
        return cls(
            app_name=raw["app_name"],
            environment=raw["environment"],
            host=raw["host"],
            port=raw["port"],
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
        )


DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "app_config.json"
