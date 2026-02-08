from app.services.facemesh import _points_from_map


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
