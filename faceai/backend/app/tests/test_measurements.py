from app.services.measurements import compute_eye_profile, compute_ideal_face_profile, compute_measurements, compute_ratios


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
    assert measurement_map["sn-prn"].value == 3.0

    ratios = compute_ratios(measurements)
    ratio_map = {r.id: r for r in ratios}

    assert ratio_map["mouth_to_nose_width"].value == 5.0 / 4.0


def test_ideal_face_profile_shape_and_phi_ratios():
    front_points = {
        "G": {"pixel": {"x": 0.0, "y": 10.0}},
        "N": {"pixel": {"x": 0.0, "y": 20.0}},
        "Sn": {"pixel": {"x": 0.0, "y": 45.0}},
        "Me": {"pixel": {"x": 0.0, "y": 80.0}},
        "En_R": {"pixel": {"x": -15.0, "y": 35.0}},
        "En_L": {"pixel": {"x": 15.0, "y": 35.0}},
        "Sto": {"pixel": {"x": 0.0, "y": 55.0}},
        "Tr_R": {"pixel": {"x": 0.0, "y": 0.0}},
    }
    side_points = {}

    measurements = compute_measurements(front_points, side_points)
    profile = compute_ideal_face_profile(front_points, measurements)

    assert profile.phi == 1.618
    assert profile.face_shape is not None
    assert profile.indices.ifv is not None
    assert profile.indices.izg is not None
    assert any(ratio.id == "face_width_over_nose_width" for ratio in profile.ratios)


def test_eye_profile_returns_classification_and_ids():
    front_points = {
        "Ex_R": {"pixel": {"x": 20.0, "y": 20.0}},
        "En_R": {"pixel": {"x": 40.0, "y": 20.0}},
        "Ps_R": {"pixel": {"x": 30.0, "y": 15.0}},
        "Pi_R": {"pixel": {"x": 30.0, "y": 25.0}},
        "Ex_L": {"pixel": {"x": 60.0, "y": 20.0}},
        "En_L": {"pixel": {"x": 80.0, "y": 20.0}},
        "Ps_L": {"pixel": {"x": 70.0, "y": 15.0}},
        "Pi_L": {"pixel": {"x": 70.0, "y": 25.0}},
        "Zy_R": {"pixel": {"x": 0.0, "y": 0.0}},
        "Zy_L": {"pixel": {"x": 120.0, "y": 0.0}},
    }
    measurements = compute_measurements(front_points, side_points={})
    eyes = compute_eye_profile(front_points, measurements, gender="female")

    assert len(eyes.standard_ids) == 10
    assert eyes.metrics[0].value is not None
    assert eyes.classification.form is not None
