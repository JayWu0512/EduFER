from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from edufer.research.datasets import DatasetSample


@dataclass(slots=True)
class ProcessedImageArtifacts:
    sample: DatasetSample
    original_bgr: np.ndarray
    resized_bgr: np.ndarray
    cropped_bgr: np.ndarray
    normalized_chw: np.ndarray

    @property
    def original_rgb(self) -> np.ndarray:
        return cv2.cvtColor(self.original_bgr, cv2.COLOR_BGR2RGB)

    @property
    def resized_rgb(self) -> np.ndarray:
        return cv2.cvtColor(self.resized_bgr, cv2.COLOR_BGR2RGB)

    @property
    def cropped_rgb(self) -> np.ndarray:
        return cv2.cvtColor(self.cropped_bgr, cv2.COLOR_BGR2RGB)

    @property
    def normalized_preview_rgb(self) -> np.ndarray:
        chw = np.transpose(self.normalized_chw, (1, 2, 0))
        chw = np.clip(chw, -2.5, 2.5)
        chw = (chw + 2.5) / 5.0
        preview = (chw * 255.0).astype(np.uint8)
        return preview


class ResNetStylePreprocessor:
    def __init__(
        self,
        *,
        resize_short_side: int = 256,
        crop_size: int = 224,
        normalization_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalization_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self._resize_short_side = resize_short_side
        self._crop_size = crop_size
        self._normalization_mean = np.asarray(normalization_mean, dtype=np.float32)
        self._normalization_std = np.asarray(normalization_std, dtype=np.float32)

    @property
    def resize_short_side(self) -> int:
        return self._resize_short_side

    @property
    def crop_size(self) -> int:
        return self._crop_size

    def process_sample(self, sample: DatasetSample) -> ProcessedImageArtifacts:
        original_bgr = self.load_image(sample.image_path)
        return self.process_image(sample=sample, image_bgr=original_bgr)

    def process_image(
        self,
        *,
        sample: DatasetSample,
        image_bgr: np.ndarray,
    ) -> ProcessedImageArtifacts:
        resized_bgr = self._resize_with_short_side(image_bgr)
        cropped_bgr = self._center_crop(resized_bgr)
        normalized_chw = self._normalize(cropped_bgr)
        return ProcessedImageArtifacts(
            sample=sample,
            original_bgr=image_bgr,
            resized_bgr=resized_bgr,
            cropped_bgr=cropped_bgr,
            normalized_chw=normalized_chw,
        )

    @staticmethod
    def load_image(path: Path | str) -> np.ndarray:
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image at {path}.")
        return image_bgr

    def _resize_with_short_side(self, image_bgr: np.ndarray) -> np.ndarray:
        height, width = image_bgr.shape[:2]
        short_side = min(height, width)
        scale = self._resize_short_side / float(short_side)
        resized_width = int(round(width * scale))
        resized_height = int(round(height * scale))
        return cv2.resize(
            image_bgr,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )

    def _center_crop(self, image_bgr: np.ndarray) -> np.ndarray:
        height, width = image_bgr.shape[:2]
        crop_size = self._crop_size
        offset_x = max((width - crop_size) // 2, 0)
        offset_y = max((height - crop_size) // 2, 0)
        cropped = image_bgr[offset_y : offset_y + crop_size, offset_x : offset_x + crop_size]

        if cropped.shape[0] != crop_size or cropped.shape[1] != crop_size:
            cropped = cv2.resize(cropped, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)

        return cropped

    def _normalize(self, image_bgr: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (image_rgb - self._normalization_mean) / self._normalization_std
        return np.transpose(normalized, (2, 0, 1)).astype(np.float32)
