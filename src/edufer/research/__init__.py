"""Notebook-oriented helpers for model comparison experiments."""

from edufer.research.datasets import BinaryImageDatasetCatalog, DatasetSample
from edufer.research.evaluation import (
    BinaryClassificationMetrics,
    ModelComparisonRunner,
    ModelEvaluationResult,
)
from edufer.research.models import (
    ImageModelRunner,
    ModelArtifact,
    ModelPrediction,
    ModelRunnerFactory,
    ModelSpec,
    PlaceholderCheckpointLoader,
    PlaceholderModelRunner,
    TorchVisionModelRunner,
)
from edufer.research.preprocessing import ProcessedImageArtifacts, ResNetStylePreprocessor
from edufer.research.visualization import AnnotatedPredictionFrame

__all__ = [
    "AnnotatedPredictionFrame",
    "BinaryClassificationMetrics",
    "BinaryImageDatasetCatalog",
    "DatasetSample",
    "ImageModelRunner",
    "ModelArtifact",
    "ModelComparisonRunner",
    "ModelEvaluationResult",
    "ModelPrediction",
    "ModelRunnerFactory",
    "ModelSpec",
    "PlaceholderCheckpointLoader",
    "PlaceholderModelRunner",
    "ProcessedImageArtifacts",
    "ResNetStylePreprocessor",
    "TorchVisionModelRunner",
]
