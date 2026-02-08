from app.services.measurements import compute_measurements, compute_ratios


def test_measurements_distance_and_ratio():
    front_points = {
        "Ch_R": {"pixel": {"x": 0.0, "y": 0.0}},
        "Ch_L": {"pixel": {"x": 3.0, "y": 4.0}},
        "Al_R": {"pixel": {"x": 1.0, "y": 1.0}},
        "Al_L": {"pixel": {"x": 1.0, "y": 5.0}},
        "Ls": {"pixel": {"x": 2.0, "y": 2.0}},
        "Li": {"pixel": {"x": 2.0, "y": 6.0}},
        "Sto": {"pixel": {"x": 2.0, "y": 4.0}},
        "Me": {"pixel": {"x": 2.0, "y": 10.0}},
        "Sn": {"pixel": {"x": 2.0, "y": 4.0}},
        "Zy_R": {"pixel": {"x": 0.0, "y": 0.0}},
        "Zy_L": {"pixel": {"x": 10.0, "y": 0.0}},
    }
    side_points = {
        "Prn": {"pixel": {"x": 0.0, "y": 0.0}},
        "Sn": {"pixel": {"x": 3.0, "y": 4.0}},
    }

    measurements = compute_measurements(front_points, side_points)
    measurement_map = {m.id: m for m in measurements}

    assert measurement_map["ch-ch"].value == 5.0
    assert measurement_map["al-al"].value == 4.0

    ratios = compute_ratios(measurements)
    ratio_map = {r.id: r for r in ratios}

    assert ratio_map["mouth_to_nose_width"].value == 5.0 / 4.0
