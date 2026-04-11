from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from edufer.research.preprocessing import ProcessedImageArtifacts


@dataclass(slots=True, frozen=True)
class ModelArtifact:
    model_name: str
    architecture: str
    class_names: tuple[str, str]
    input_size: int
    threshold: float
    feature_weights: dict[str, float]
    notes: str = ""


@dataclass(slots=True, frozen=True)
class ModelSpec:
    display_name: str
    checkpoint_path: Path
    architecture: str
    positive_label: str = "engaged"
    negative_label: str = "not_engaged"
    class_names: tuple[str, str] = ("not_engaged", "engaged")
    threshold: float = 0.5


@dataclass(slots=True, frozen=True)
class ModelPrediction:
    model_name: str
    predicted_label: str
    engaged_probability: float
    not_engaged_probability: float
    threshold: float
    feature_snapshot: dict[str, float]


class ImageModelRunner(ABC):
    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable model name."""

    @abstractmethod
    def predict(self, artifacts: ProcessedImageArtifacts) -> ModelPrediction:
        """Run one prediction against processed artifacts."""


class PlaceholderCheckpointLoader:
    def load(self, checkpoint_path: Path | str) -> ModelArtifact:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path} is not a supported placeholder checkpoint. "
                "Replace it with JSON placeholder metadata or extend the loader for real torch checkpoints."
            ) from exc

        return ModelArtifact(
            model_name=payload["model_name"],
            architecture=payload["architecture"],
            class_names=tuple(payload.get("class_names", ["not_engaged", "engaged"])),
            input_size=int(payload.get("input_size", 224)),
            threshold=float(payload.get("threshold", 0.5)),
            feature_weights={key: float(value) for key, value in payload["feature_weights"].items()},
            notes=payload.get("notes", ""),
        )


class PlaceholderModelRunner(ImageModelRunner):
    def __init__(self, artifact: ModelArtifact, spec: ModelSpec) -> None:
        self._artifact = artifact
        self._spec = spec

    @property
    def display_name(self) -> str:
        return self._spec.display_name

    @property
    def threshold(self) -> float:
        return self._artifact.threshold

    def predict(self, artifacts: ProcessedImageArtifacts) -> ModelPrediction:
        features = self._extract_features(artifacts.cropped_bgr)
        engaged_probability = self._score(features)
        not_engaged_probability = 1.0 - engaged_probability
        predicted_label = (
            self._spec.positive_label
            if engaged_probability >= self._artifact.threshold
            else self._spec.negative_label
        )
        return ModelPrediction(
            model_name=self._spec.display_name,
            predicted_label=predicted_label,
            engaged_probability=engaged_probability,
            not_engaged_probability=not_engaged_probability,
            threshold=self._artifact.threshold,
            feature_snapshot=features,
        )

    def _score(self, features: dict[str, float]) -> float:
        linear_score = self._artifact.feature_weights.get("bias", 0.0)
        for key, value in features.items():
            linear_score += self._artifact.feature_weights.get(key, 0.0) * value
        return float(1.0 / (1.0 + np.exp(-linear_score)))

    @staticmethod
    def _extract_features(image_bgr: np.ndarray) -> dict[str, float]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        center_region = gray[gray.shape[0] // 4 : gray.shape[0] * 3 // 4, gray.shape[1] // 4 : gray.shape[1] * 3 // 4]

        brightness = float(gray.mean() / 255.0)
        contrast = float(np.clip(gray.std() / 64.0, 0.0, 1.5))
        edge_density = float(edges.mean() / 255.0)
        center_brightness = float(center_region.mean() / 255.0)
        warm_tone = float(image_bgr[:, :, 2].mean() / max(image_bgr[:, :, 0].mean(), 1.0))

        return {
            "brightness": brightness,
            "contrast": contrast,
            "edge_density": edge_density,
            "center_brightness": center_brightness,
            "warm_tone": float(np.clip(warm_tone / 2.0, 0.0, 1.5)),
        }


class TorchVisionModelRunner(ImageModelRunner):
    def __init__(self, spec: ModelSpec) -> None:
        self._spec = spec
        self._model = None
        self._torch = None
        self._positive_index = 1
        self._threshold = spec.threshold

    @property
    def display_name(self) -> str:
        return self._spec.display_name

    def predict(self, artifacts: ProcessedImageArtifacts) -> ModelPrediction:
        if self._model is None or self._torch is None:
            self._load_model()

        torch = self._torch
        assert torch is not None
        assert self._model is not None

        with torch.no_grad():
            tensor = torch.from_numpy(artifacts.normalized_chw).unsqueeze(0)
            logits = self._model(tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

        engaged_probability = float(probabilities[self._positive_index])
        not_engaged_probability = float(1.0 - engaged_probability)
        predicted_label = (
            self._spec.positive_label
            if engaged_probability >= self._threshold
            else self._spec.negative_label
        )
        return ModelPrediction(
            model_name=self._spec.display_name,
            predicted_label=predicted_label,
            engaged_probability=engaged_probability,
            not_engaged_probability=not_engaged_probability,
            threshold=self._threshold,
            feature_snapshot={},
        )

    def _load_model(self) -> None:
        try:
            import torch
            import torchvision.models as tv_models
            from torch import nn
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "torch and torchvision are required to load a real .pt checkpoint. "
                "Install the notebook extras before replacing the placeholder files."
            ) from exc

        checkpoint = torch.load(self._spec.checkpoint_path, map_location="cpu")
        state_dict, class_names, threshold = self._extract_checkpoint_parts(checkpoint)

        self._threshold = threshold
        self._positive_index = class_names.index(self._spec.positive_label)
        model = self._build_model(
            tv_models=tv_models,
            nn=nn,
            num_classes=len(class_names),
        )
        cleaned_state_dict = self._clean_state_dict(state_dict)
        model.load_state_dict(cleaned_state_dict, strict=False)
        model.eval()

        self._model = model
        self._torch = torch

    def _extract_checkpoint_parts(
        self,
        checkpoint: Any,
    ) -> tuple[dict[str, Any], tuple[str, ...], float]:
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            class_names = tuple(checkpoint.get("class_names", self._spec.class_names))
            threshold = float(checkpoint.get("threshold", self._spec.threshold))
            return state_dict, class_names, threshold

        if isinstance(checkpoint, dict):
            tensor_values = [value for value in checkpoint.values() if hasattr(value, "shape")]
            if tensor_values:
                return checkpoint, self._spec.class_names, self._spec.threshold

        raise ValueError(
            f"Unsupported checkpoint format in {self._spec.checkpoint_path}. "
            "Expected either a raw state_dict or a dict with 'state_dict'."
        )

    def _build_model(self, *, tv_models: Any, nn: Any, num_classes: int) -> Any:
        architecture = self._spec.architecture
        if architecture == "resnet18":
            model = tv_models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model
        if architecture == "vgg16":
            model = tv_models.vgg16(weights=None)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
            return model
        if architecture == "vit_b_16":
            model = tv_models.vit_b_16(weights=None)
            model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
            return model
        raise ValueError(f"Unsupported architecture '{architecture}'.")

    @staticmethod
    def _clean_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
        cleaned: dict[str, Any] = {}
        for key, value in state_dict.items():
            normalized_key = key
            for prefix in ("module.", "model."):
                if normalized_key.startswith(prefix):
                    normalized_key = normalized_key[len(prefix) :]
            cleaned[normalized_key] = value
        return cleaned


class ModelRunnerFactory:
    def __init__(self, checkpoint_loader: PlaceholderCheckpointLoader | None = None) -> None:
        self._checkpoint_loader = checkpoint_loader or PlaceholderCheckpointLoader()

    def build(self, spec: ModelSpec) -> ImageModelRunner:
        if self._is_json_placeholder(spec.checkpoint_path):
            artifact = self._checkpoint_loader.load(spec.checkpoint_path)
            if artifact.architecture != spec.architecture:
                raise ValueError(
                    f"Checkpoint architecture '{artifact.architecture}' does not match "
                    f"the configured spec '{spec.architecture}' for {spec.display_name}."
                )
            return PlaceholderModelRunner(artifact=artifact, spec=spec)
        return TorchVisionModelRunner(spec=spec)

    @staticmethod
    def _is_json_placeholder(checkpoint_path: Path | str) -> bool:
        path = Path(checkpoint_path)
        try:
            with path.open("r", encoding="utf-8") as handle:
                first_char = handle.read(1)
        except UnicodeDecodeError:
            return False
        return first_char in {"{", "["}
