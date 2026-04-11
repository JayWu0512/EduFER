from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(slots=True, frozen=True)
class DatasetSample:
    image_path: Path
    label_id: int
    label_name: str


class BinaryImageDatasetCatalog:
    def __init__(self, root_dir: Path | str, label_mapping: dict[int, str]) -> None:
        self._root_dir = Path(root_dir)
        self._label_mapping = dict(label_mapping)

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    @property
    def label_mapping(self) -> dict[int, str]:
        return dict(self._label_mapping)

    def discover(self) -> list[DatasetSample]:
        samples: list[DatasetSample] = []
        for label_id, label_name in sorted(self._label_mapping.items()):
            label_dir = self._root_dir / str(label_id)
            if not label_dir.exists():
                continue

            for image_path in sorted(label_dir.iterdir()):
                if image_path.suffix.lower() not in _IMAGE_SUFFIXES:
                    continue
                samples.append(
                    DatasetSample(
                        image_path=image_path,
                        label_id=label_id,
                        label_name=label_name,
                    )
                )
        return samples
