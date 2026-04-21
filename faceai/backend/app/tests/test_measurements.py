from app.services.measurements import compute_measurements, compute_ratios


def test_measurements_distance_and_ratio():
    front_points = {
        "Ch_R": {"pixel": {"x": 0.0, "y": 0.0}},
        "Ch_L": {"pixel": {"x": 3.0, "y": 4.0}},
        "Ft_R": {"pixel": {"x": 0.0, "y": 0.0}},
        "Ft_L": {"pixel": {"x": 8.0, "y": 0.0}},
        "Al_R": {"pixel": {"x": 1.0, "y": 1.0}},
        "Al_L": {"pixel": {"x": 1.0, "y": 5.0}},
        "Ex_R": {"pixel": {"x": 0.0, "y": 2.0}},
        "En_R": {"pixel": {"x": 4.0, "y": 2.0}},
        "Ps_R": {"pixel": {"x": 2.0, "y": 1.0}},
        "Pi_R": {"pixel": {"x": 2.0, "y": 3.0}},
        "Ex_L": {"pixel": {"x": 12.0, "y": 2.0}},
        "En_L": {"pixel": {"x": 8.0, "y": 2.0}},
        "Ps_L": {"pixel": {"x": 10.0, "y": 1.0}},
        "Pi_L": {"pixel": {"x": 10.0, "y": 3.0}},
        "Ls": {"pixel": {"x": 2.0, "y": 2.0}},
        "Li": {"pixel": {"x": 2.0, "y": 6.0}},
        "Sto": {"pixel": {"x": 2.0, "y": 4.0}},
        "Sl_R": {"pixel": {"x": 2.0, "y": 7.0}},
        "Tr_R": {"pixel": {"x": 2.0, "y": 0.0}},
        "G": {"pixel": {"x": 2.0, "y": 2.0}},
        "N": {"pixel": {"x": 2.0, "y": 3.0}},
        "Me": {"pixel": {"x": 2.0, "y": 10.0}},
        "Pg": {"pixel": {"x": 2.0, "y": 8.0}},
        "Sn": {"pixel": {"x": 2.0, "y": 4.0}},
        "Prn": {"pixel": {"x": 2.0, "y": 5.0}},
        "Zy_R": {"pixel": {"x": 0.0, "y": 0.0}},
        "Zy_L": {"pixel": {"x": 10.0, "y": 0.0}},
        "Go_R": {"pixel": {"x": 0.0, "y": 9.0}},
        "Go_L": {"pixel": {"x": 4.0, "y": 9.0}},
        "Sa_R": {"pixel": {"x": -1.0, "y": 1.0}},
        "Sba_R": {"pixel": {"x": -1.0, "y": 5.0}},
        "Pra_R": {"pixel": {"x": 0.0, "y": 3.0}},
        "Pa_R": {"pixel": {"x": -2.0, "y": 3.0}},
        "Sa_L": {"pixel": {"x": 13.0, "y": 1.0}},
        "Sba_L": {"pixel": {"x": 13.0, "y": 5.0}},
        "Pra_L": {"pixel": {"x": 12.0, "y": 3.0}},
        "Pa_L": {"pixel": {"x": 14.0, "y": 3.0}},
    }
    side_points = {
        "Prn": {"pixel": {"x": 0.0, "y": 0.0}},
        "Sn": {"pixel": {"x": 3.0, "y": 4.0}},
    }

    measurements = compute_measurements(front_points, side_points)
    measurement_map = {m.id: m for m in measurements}

    assert measurement_map["ch-ch"].value == 5.0
    assert measurement_map["al-al"].value == 4.0
    assert measurement_map["tr-me"].value == 10.0
    assert measurement_map["n-g"].value == 1.0
    assert measurement_map["ft-r-g"].value is not None
    assert measurement_map["n-prn"].value == 2.0
    assert measurement_map["sn-prn"].value == 1.0
    assert measurement_map["li-me"].value == 4.0
    assert measurement_map["pg-me"].value == 2.0
    assert measurement_map["sn-sto"].value == 0.0
    assert measurement_map["sto-sl"].value == 3.0
    assert measurement_map["tlt"].value == 6.0
    assert measurement_map["al-base-asym"].value == 0.0
    assert measurement_map["prn-midline-dev"].value == 0.0
    assert measurement_map["ps-pi-r"].value == 2.0
    assert measurement_map["ps-pi-l"].value == 2.0
    assert measurement_map["avg-eye-width"].value == 4.0
    assert measurement_map["canthal-angle-diff"].value == 0.0
    assert measurement_map["zy-midline-asym"].value == 0.0
    assert measurement_map["zy-level-diff"].value == 0.0
    assert measurement_map["zy-tr-vertical"].value == 0.0
    assert measurement_map["sn-tr-vertical"].value == 4.0
    assert measurement_map["sa-sba-r"].value == 4.0
    assert measurement_map["pra-pa-r"].value == 2.0
    assert all(m.image == "front" for m in measurements)

    ratios = compute_ratios(measurements, gender="male")
    ratio_map = {r.id: r for r in ratios}

    assert ratio_map["ideal_mouth_to_nose_width"].value == 5.0 / 4.0
    assert ratio_map["ideal_mouth_to_nose_width"].ideal_value == 1.618
    assert ratio_map["ideal_mouth_to_nose_width"].deviation_pct is not None
    assert ratio_map["ideal_upper_third_balance"].value == 1.0
    assert ratio_map["eye_width_face_share_r"].value == 4.0 / 10.0
    assert ratio_map["eye_form_index_r"].value == 2.0 / 4.0
    assert ratio_map["intercanthal_eye_ratio"].value == 4.0 / 4.0
    assert ratio_map["eye_aperture_symmetry"].value == 1.0
    assert ratio_map["cheekbone_dominance"].value == 10.0 / 4.0
    assert ratio_map["cheekbone_midline_symmetry"].value == 0.0
    assert ratio_map["ear_shape_index_r"].value == 2.0 / 4.0
    assert ratio_map["ear_length_symmetry"].value == 1.0
    assert ratio_map["forehead_height_share"].value == 2.0 / 10.0
    assert ratio_map["forehead_glabellar_balance"].value == 2.0 / 1.0
    assert ratio_map["forehead_symmetry_balance"].note is not None
    assert ratio_map["mouth_width_face_share"].value == 5.0 / 10.0
    assert ratio_map["lip_fullness"].value == 6.0 / 5.0
    assert ratio_map["lip_dominance"].note is not None
    assert ratio_map["mouth_vertical_balance"].value == 0.0
    assert ratio_map["nose_width_height_index"].value == 4.0 / 1.0
    assert ratio_map["nose_projection_index"].value == 1.0 / 1.0
    assert ratio_map["nose_base_index"].value == 4.0 / 1.0
    assert ratio_map["nose_verticality_index"].value == 2.0 / 1.0
    assert ratio_map["nose_base_asymmetry"].value == 0.0
    assert ratio_map["nose_tip_deviation"].value == 0.0
    assert ratio_map["overall_face_height_to_width"].value == 10.0 / 10.0
    assert ratio_map["overall_face_shape"].note is not None
    assert ratio_map["chin_height_share"].value == 4.0 / 6.0
    assert ratio_map["chin_height_share"].note is not None
