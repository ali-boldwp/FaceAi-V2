import cv2
import numpy as np

from app.services.facemesh import _is_strict_profile_image, _points_from_map, _subject_mask


class _Lm:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


def test_prn_midpoint_from_mesh_points():
    landmarks = [
        _Lm(0.0, 0.0, 0.0),  # 0
        _Lm(0.2, 0.3, 0.1),  # 1
        _Lm(0.0, 0.0, 0.0),  # 2
        _Lm(0.0, 0.0, 0.0),  # 3
        _Lm(0.6, 0.7, 0.5),  # 4
    ]
    mapping = {"Prn": 4}

    points = _points_from_map(landmarks, mapping, width=100, height=200)

    assert points["Prn"]["index"] is None
    assert points["Prn"]["pixel"]["x"] == 40.0
    assert points["Prn"]["pixel"]["y"] == 100.0
    assert points["Prn"]["normalized"]["x"] == 0.4
    assert points["Prn"]["normalized"]["y"] == 0.5
    assert points["Prn"]["normalized"]["z"] == 0.3


def test_profile_mask_ignores_tinted_background():
    image = np.full((420, 360, 3), (214, 206, 186), dtype=np.uint8)
    head = np.array(
        [
            [80, 50],
            [250, 50],
            [285, 115],
            [350, 182],
            [276, 212],
            [258, 300],
            [180, 350],
            [92, 300],
            [50, 178],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(image, [head], (126, 155, 190))

    mask = _subject_mask(image)
    ys, xs = np.where(mask)

    assert int(xs.min()) > 40
    assert int(xs.max()) < 350
    assert _is_strict_profile_image(image)
