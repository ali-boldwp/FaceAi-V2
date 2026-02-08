from typing import Dict, List

import cv2
import numpy as np


def draw_landmarks(image_bgr: np.ndarray, points: Dict[str, Dict]) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, min(0.7, width / 1200.0))
    thickness = 1

    for label, data in points.items():
        px = int(data["pixel"]["x"])
        py = int(data["pixel"]["y"])

        cv2.circle(image_bgr, (px, py), 3, (0, 255, 255), -1)
        text = label
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        offset_x = 6
        offset_y = -6
        x1 = px + offset_x
        y1 = py + offset_y - th - baseline
        x2 = x1 + tw + 4
        y2 = py + offset_y + 4

        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))

        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.putText(
            image_bgr,
            text,
            (x1 + 2, y2 - 3),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return image_bgr


def draw_all_landmarks(image_bgr: np.ndarray, landmarks: List) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.35, min(0.55, width / 1400.0))
    thickness = 1

    for index, lm in enumerate(landmarks):
        px = int(lm.x * width)
        py = int(lm.y * height)

        cv2.circle(image_bgr, (px, py), 1, (0, 255, 0), -1)
        text = str(index)
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x1 = px + 2
        y1 = py - 2 - th - baseline
        x2 = x1 + tw + 2
        y2 = py - 2

        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))

        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.putText(
            image_bgr,
            text,
            (x1 + 1, y2 - 2),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return image_bgr
