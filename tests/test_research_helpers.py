import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from edufer.research.datasets import BinaryImageDatasetCatalog, DatasetSample
from edufer.research.evaluation import ModelComparisonRunner
from edufer.research.models import ModelSpec, PlaceholderCheckpointLoader
from edufer.research.preprocessing import ResNetStylePreprocessor


class ResearchHelpersTests(unittest.TestCase):
    def test_dataset_catalog_discovers_binary_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir)
            (root_dir / "0").mkdir()
            (root_dir / "1").mkdir()
            self._write_image(root_dir / "0" / "sample_0.jpg", 32)
            self._write_image(root_dir / "1" / "sample_1.jpg", 224)

            catalog = BinaryImageDatasetCatalog(root_dir=root_dir, label_mapping={0: "not_engaged", 1: "engaged"})
            samples = catalog.discover()

        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].label_name, "not_engaged")
        self.assertEqual(samples[1].label_name, "engaged")

    def test_preprocessor_matches_resnet_style_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.jpg"
            self._write_image(image_path, 128)
            dataset_sample = DatasetSample(
                image_path=image_path,
                label_id=0,
                label_name="not_engaged",
            )

            preprocessor = ResNetStylePreprocessor()
            artifacts = preprocessor.process_sample(dataset_sample)

        self.assertEqual(artifacts.resized_bgr.shape[:2], (256, 256))
        self.assertEqual(artifacts.cropped_bgr.shape[:2], (224, 224))
        self.assertEqual(artifacts.normalized_chw.shape, (3, 224, 224))

    def test_model_comparison_runner_produces_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir)
            models_dir = root_dir / "models"
            models_dir.mkdir()
            for label_id, brightness in ((0, 40), (1, 215)):
                label_dir = root_dir / "processed" / str(label_id)
                label_dir.mkdir(parents=True, exist_ok=True)
                self._write_image(label_dir / f"sample_{label_id}.jpg", brightness)

            checkpoint_path = models_dir / "resnet18_placeholder.pt"
            checkpoint_path.write_text(
                json.dumps(
                    {
                        "model_name": "ResNet18 Placeholder",
                        "architecture": "resnet18",
                        "class_names": ["not_engaged", "engaged"],
                        "input_size": 224,
                        "threshold": 0.5,
                        "feature_weights": {
                            "bias": -1.0,
                            "brightness": 2.4,
                            "contrast": 0.4,
                            "edge_density": 0.2,
                            "center_brightness": 1.2,
                            "warm_tone": 0.1,
                        },
                    }
                ),
                encoding="utf-8",
            )

            samples = BinaryImageDatasetCatalog(
                root_dir=root_dir / "processed",
                label_mapping={0: "not_engaged", 1: "engaged"},
            ).discover()

            results = ModelComparisonRunner(preprocessor=ResNetStylePreprocessor()).evaluate(
                model_specs=[
                    ModelSpec(
                        display_name="ResNet-18",
                        checkpoint_path=checkpoint_path,
                        architecture="resnet18",
                    )
                ],
                samples=samples,
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metrics.confusion_matrix.shape, (2, 2))
        self.assertGreaterEqual(results[0].metrics.accuracy, 0.0)
        self.assertLessEqual(results[0].metrics.accuracy, 1.0)
        self.assertEqual(len(results[0].annotated_frames), 2)

    def test_placeholder_checkpoint_loader_reads_json_pt_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "vit_b16_placeholder.pt"
            checkpoint_path.write_text(
                json.dumps(
                    {
                        "model_name": "ViT-B/16 Placeholder",
                        "architecture": "vit_b_16",
                        "class_names": ["not_engaged", "engaged"],
                        "input_size": 224,
                        "threshold": 0.55,
                        "feature_weights": {
                            "bias": -0.2,
                            "brightness": 1.4,
                            "contrast": 1.0,
                            "edge_density": 0.9,
                            "center_brightness": 0.6,
                            "warm_tone": 0.3,
                        },
                        "notes": "Replace with a real ViT checkpoint later.",
                    }
                ),
                encoding="utf-8",
            )

            artifact = PlaceholderCheckpointLoader().load(checkpoint_path)

        self.assertEqual(artifact.model_name, "ViT-B/16 Placeholder")
        self.assertEqual(artifact.architecture, "vit_b_16")
        self.assertAlmostEqual(artifact.threshold, 0.55)

    @staticmethod
    def _write_image(path: Path, brightness: int) -> None:
        image = np.full((256, 256, 3), brightness, dtype=np.uint8)
        cv2.imwrite(str(path), image)


if __name__ == "__main__":
    unittest.main()
