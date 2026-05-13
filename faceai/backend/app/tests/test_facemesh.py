from pathlib import Path

import cv2
import numpy as np

from app.services.facemesh import (
    _estimate_side_anthropometric_points,
    _extract_landmarks,
    _filter_profile_visible_points,
    _is_strict_profile_image,
    _points_from_map,
    _restore_landmarks,
    _subject_mask,
)


class _Lm:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


def test_prn_uses_configured_mesh_point_when_index_is_set():
    landmarks = [
        _Lm(0.0, 0.0, 0.0),  # 0
        _Lm(0.2, 0.3, 0.1),  # 1
        _Lm(0.0, 0.0, 0.0),  # 2
        _Lm(0.0, 0.0, 0.0),  # 3
        _Lm(0.6, 0.7, 0.5),  # 4
    ]
    mapping = {"Prn": 4}

    points = _points_from_map(landmarks, mapping, width=100, height=200)

    assert points["Prn"]["index"] == 4
    assert points["Prn"]["pixel"]["x"] == 60.0
    assert points["Prn"]["pixel"]["y"] == 140.0
    assert points["Prn"]["normalized"]["x"] == 0.6
    assert points["Prn"]["normalized"]["y"] == 0.7
    assert points["Prn"]["normalized"]["z"] == 0.5


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


def test_restores_flipped_padded_landmarks_to_original_coordinates():
    restored = _restore_landmarks(
        [[_Lm(0.75, 0.5, 0.2)]],
        variant_w=200,
        variant_h=200,
        original_w=100,
        original_h=160,
        pad_x=50,
        pad_y=20,
        flipped=True,
    )

    assert restored[0][0].x == 0.0
    assert restored[0][0].y == 0.5
    assert restored[0][0].z == 0.2


def test_extracts_landmarks_from_profile_reference_image():
    image_path = Path("faceai/frontend/public/side-reference.png")
    image = cv2.imread(str(image_path))

    faces, count = _extract_landmarks(image)

    assert len(faces) == 1
    assert count >= 468


def test_profile_filter_keeps_only_visible_side_without_moving_points():
    image = cv2.imread("faceai/frontend/public/side-reference.png")
    points = {
        "Prn": {"pixel": {"x": 1.0, "y": 2.0}},
        "Go_R": {"pixel": {"x": 3.0, "y": 4.0}},
        "Go_L": {"pixel": {"x": 5.0, "y": 6.0}},
    }

    filtered = _filter_profile_visible_points(points, image)

    assert filtered["Prn"] is points["Prn"]
    assert filtered["Go_L"] is points["Go_L"]
    assert "Go_R" not in filtered


def test_side_anthropometric_points_add_profile_only_landmarks():
    image = cv2.imread("faceai/frontend/public/side-reference.png")
    faces, _ = _extract_landmarks(image)
    height, width = image.shape[:2]
    mesh_points = _points_from_map(faces[0], {"Prn": 4, "Go_L": 172}, width, height)

    points = _estimate_side_anthropometric_points(image, mesh_points)

    assert points["Prn"]["index"] == 4
    assert "Sa_L" in points
    assert "Pra_L" in points
    assert "Sba_L" in points
    assert "Go_L" in points


def test_side_anthropometric_points_work_without_mesh_points():
    image = cv2.imread("faceai/frontend/public/side-reference.png")

    points = _estimate_side_anthropometric_points(image, {})

    assert "Prn" in points
    assert "Pg" in points
    assert "Me" in points
    assert any(label.startswith("Go_") for label in points)
