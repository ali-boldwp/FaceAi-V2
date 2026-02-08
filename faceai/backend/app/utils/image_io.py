import base64
from typing import Tuple

import cv2
import numpy as np


def read_image(image_bytes: bytes) -> Tuple[np.ndarray, int, int]:
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode image")
    height, width = image.shape[:2]
    return image, width, height


def to_base64_png(image_bgr: np.ndarray) -> str:
    success, buffer = cv2.imencode(".png", image_bgr)
    if not success:
        raise ValueError("Unable to encode image")
    encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"
