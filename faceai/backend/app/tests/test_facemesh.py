from app.services.facemesh import _inverse_transform_xy, _points_from_map


class _Lm:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


def test_prn_uses_mapped_landmark_when_available():
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


def test_inverse_transform_rot90_cw():
    x, y = _inverse_transform_xy(0.2, 0.8, "rot90_cw")
    assert x == 0.8
    assert y == 0.8
