from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from edufer.research.datasets import DatasetSample
from edufer.research.models import ModelPrediction, ModelRunnerFactory, ModelSpec
from edufer.research.preprocessing import ProcessedImageArtifacts, ResNetStylePreprocessor
from edufer.research.visualization import AnnotatedPredictionFrame, PredictionFrameBuilder


@dataclass(slots=True, frozen=True)
class BinaryClassificationMetrics:
    accuracy: float
    confusion_matrix: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    thresholds: np.ndarray


@dataclass(slots=True, frozen=True)
class EvaluatedSample:
    sample: DatasetSample
    artifacts: ProcessedImageArtifacts
    prediction: ModelPrediction


@dataclass(slots=True, frozen=True)
class ModelEvaluationResult:
    model_name: str
    samples: tuple[EvaluatedSample, ...]
    metrics: BinaryClassificationMetrics
    annotated_frames: tuple[AnnotatedPredictionFrame, ...]


class ModelComparisonRunner:
    def __init__(
        self,
        *,
        preprocessor: ResNetStylePreprocessor,
        model_runner_factory: ModelRunnerFactory | None = None,
        frame_builder: PredictionFrameBuilder | None = None,
    ) -> None:
        self._preprocessor = preprocessor
        self._model_runner_factory = model_runner_factory or ModelRunnerFactory()
        self._frame_builder = frame_builder or PredictionFrameBuilder()

    def evaluate(
        self,
        *,
        model_specs: list[ModelSpec],
        samples: list[DatasetSample],
    ) -> list[ModelEvaluationResult]:
        processed_cache = {
            sample.image_path: self._preprocessor.process_sample(sample)
            for sample in samples
        }
        results: list[ModelEvaluationResult] = []
        for spec in model_specs:
            runner = self._model_runner_factory.build(spec)
            evaluated_samples: list[EvaluatedSample] = []
            annotated_frames: list[AnnotatedPredictionFrame] = []

            for sample in samples:
                artifacts = processed_cache[sample.image_path]
                prediction = runner.predict(artifacts)
                evaluated_samples.append(
                    EvaluatedSample(
                        sample=sample,
                        artifacts=artifacts,
                        prediction=prediction,
                    )
                )
                annotated_frames.append(
                    self._frame_builder.build(
                        artifacts=artifacts,
                        prediction=prediction,
                    )
                )

            metrics = self._compute_metrics(evaluated_samples)
            results.append(
                ModelEvaluationResult(
                    model_name=spec.display_name,
                    samples=tuple(evaluated_samples),
                    metrics=metrics,
                    annotated_frames=tuple(annotated_frames),
                )
            )
        return results

    @staticmethod
    def _compute_metrics(samples: list[EvaluatedSample]) -> BinaryClassificationMetrics:
        y_true = np.asarray([sample.sample.label_id for sample in samples], dtype=np.int32)
        y_score = np.asarray(
            [sample.prediction.engaged_probability for sample in samples],
            dtype=np.float32,
        )
        decision_threshold = float(samples[0].prediction.threshold) if samples else 0.5
        y_pred = (y_score >= decision_threshold).astype(np.int32)

        true_negative = int(np.sum((y_true == 0) & (y_pred == 0)))
        false_positive = int(np.sum((y_true == 0) & (y_pred == 1)))
        false_negative = int(np.sum((y_true == 1) & (y_pred == 0)))
        true_positive = int(np.sum((y_true == 1) & (y_pred == 1)))

        confusion_matrix = np.asarray(
            [
                [true_negative, false_positive],
                [false_negative, true_positive],
            ],
            dtype=np.int32,
        )
        accuracy = float((true_negative + true_positive) / max(len(samples), 1))

        precision, recall, thresholds = ModelComparisonRunner._precision_recall_curve(
            y_true=y_true,
            y_score=y_score,
        )
        return BinaryClassificationMetrics(
            accuracy=accuracy,
            confusion_matrix=confusion_matrix,
            precision=precision,
            recall=recall,
            thresholds=thresholds,
        )

    @staticmethod
    def _precision_recall_curve(
        *,
        y_true: np.ndarray,
        y_score: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        candidate_thresholds = np.unique(np.concatenate(([0.0], y_score, [1.0])))
        candidate_thresholds = np.sort(candidate_thresholds)[::-1]

        precision_values: list[float] = []
        recall_values: list[float] = []
        for threshold in candidate_thresholds:
            y_pred = (y_score >= threshold).astype(np.int32)
            true_positive = np.sum((y_true == 1) & (y_pred == 1))
            false_positive = np.sum((y_true == 0) & (y_pred == 1))
            false_negative = np.sum((y_true == 1) & (y_pred == 0))

            precision = float(true_positive / max(true_positive + false_positive, 1))
            recall = float(true_positive / max(true_positive + false_negative, 1))
            precision_values.append(precision)
            recall_values.append(recall)

        return (
            np.asarray(precision_values, dtype=np.float32),
            np.asarray(recall_values, dtype=np.float32),
            candidate_thresholds.astype(np.float32),
        )
