from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = REPO_ROOT / "data" / "models" / "face_detection" / "yolov8n-face-lindevs.onnx"
DEFAULT_URL = "https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.onnx"
DEFAULT_SHA256 = "8d0bfb0c3383c5bd7a78dd24ef79a21e2aa456619b6ab5e53867092d1c7dc414"


def sha256sum(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, output_path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the YOLO face detector used by EduFER.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sha256", default=DEFAULT_SHA256)
    args = parser.parse_args()

    print(f"Downloading model from {args.url}")
    download_file(args.url, args.output)
    digest = sha256sum(args.output)
    print(f"Saved model to {args.output}")
    print(f"SHA-256: {digest}")

    if args.sha256 and digest != args.sha256:
        print("Checksum mismatch. Delete the file and try again.", file=sys.stderr)
        return 1

    print("Checksum verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
