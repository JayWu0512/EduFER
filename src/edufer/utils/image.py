from __future__ import annotations

import base64

import cv2
import numpy as np


def decode_data_url_to_bgr(data_url: str) -> np.ndarray:
    if "," not in data_url:
        raise ValueError("Expected a data URL like 'data:image/jpeg;base64,...'.")

    _, encoded = data_url.split(",", maxsplit=1)
    image_bytes = base64.b64decode(encoded)
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image data from the request payload.")
    return frame
