import json
import math
from pathlib import Path
from typing import Dict, List, Optional

from app.models.schemas import (
    EarAnalysisOut,
    EyeAnalysisOut,
    EyeClassificationOut,
    EyeMetricOut,
    EyebrowAnalysisOut,
    CheekAnalysisOut,
    FaceIndicesOut,
    ForeheadAnalysisOut,
    IdealDimensionOut,
    IdealFaceOut,
    IdealRatioOut,
    JawAnalysisOut,
    MeasurementOut,
    MouthAnalysisOut,
    NoseAnalysisOut,
    RatioOut,
)

CATALOG_PATH = Path(__file__).resolve().parent.parent / "utils" / "measurements_catalog.json"
PHI = 1.618

RATIO_DEFS = [
    {"id": "face_height_to_width", "numerator": "sn-gn", "denominator": "zy-zy"},
    {"id": "nose_length_to_width", "numerator": "n-sn", "denominator": "al-al"},
    {"id": "mouth_to_nose_width", "numerator": "ch-ch", "denominator": "al-al"},
    {"id": "upper_to_lower_lip", "numerator": "ls-sto", "denominator": "sto-li"},
]


def _distance(point_a: Dict, point_b: Dict) -> float:
    dx = point_a["pixel"]["x"] - point_b["pixel"]["x"]
    dy = point_a["pixel"]["y"] - point_b["pixel"]["y"]
    return (dx ** 2 + dy ** 2) ** 0.5


def _horizontal_distance(point_a: Dict, point_b: Dict) -> float:
    return abs(point_a["pixel"]["x"] - point_b["pixel"]["x"])


def _load_catalog() -> List[Dict]:
    with CATALOG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compute_measurements(front_points: Dict[str, Dict], side_points: Dict[str, Dict]) -> List[MeasurementOut]:
    catalog = _load_catalog()
    results: List[MeasurementOut] = []

    for entry in catalog:
        measurement_id = entry["id"]
        label = entry["label"]
        image = entry["image"]
        point_a, point_b = entry["points"]

        points = front_points if image == "front" else side_points
        value: Optional[float] = None
        note: Optional[str] = None

        if point_a in points and point_b in points:
            if measurement_id == "sn-prn":
                # Profile nasal protrusion is treated as projection (horizontal component).
                value = _horizontal_distance(points[point_a], points[point_b])
            else:
                value = _distance(points[point_a], points[point_b])
        else:
            note = "Missing required landmarks for this measurement."

        results.append(
            MeasurementOut(
                id=measurement_id,
                label=label,
                image=image,
                points=[point_a, point_b],
                value=value,
                unit="px",
                note=note,
            )
        )

    return results


def compute_ratios(measurements: List[MeasurementOut]) -> List[RatioOut]:
    measurement_map = {m.id: m for m in measurements}
    ratios: List[RatioOut] = []

    for entry in RATIO_DEFS:
        numerator_id = entry["numerator"]
        denominator_id = entry["denominator"]
        numerator = measurement_map.get(numerator_id)
        denominator = measurement_map.get(denominator_id)

        value: Optional[float] = None
        note: Optional[str] = None

        if numerator and denominator and numerator.value is not None and denominator.value is not None:
            if denominator.value == 0:
                note = "Denominator is zero for this ratio."
            else:
                value = numerator.value / denominator.value
        else:
            note = "Missing measurements for ratio."

        ratios.append(
            RatioOut(
                id=entry["id"],
                numerator=numerator_id,
                denominator=denominator_id,
                value=value,
                note=note,
            )
        )

    return ratios


def _point_distance(points: Dict[str, Dict], point_a: str, point_b: str) -> Optional[float]:
    if point_a not in points or point_b not in points:
        return None
    return _distance(points[point_a], points[point_b])


def _measurement_value(measurements: List[MeasurementOut], measurement_id: str) -> Optional[float]:
    for measurement in measurements:
        if measurement.id == measurement_id:
            return measurement.value
    return None


def _avg(values: List[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _ratio_entry(
    ratio_id: str,
    label: str,
    formula: str,
    numerator: Optional[float],
    denominator: Optional[float],
    ideal: float,
) -> IdealRatioOut:
    if numerator is None or denominator is None:
        return IdealRatioOut(
            id=ratio_id,
            label=label,
            formula=formula,
            actual=None,
            ideal=ideal,
            diff_percent=None,
            deviation=None,
            interpretation=None,
            note="Missing landmarks or measurements.",
        )
    if denominator == 0:
        return IdealRatioOut(
            id=ratio_id,
            label=label,
            formula=formula,
            actual=None,
            ideal=ideal,
            diff_percent=None,
            deviation=None,
            interpretation=None,
            note="Denominator is zero.",
        )

    actual = numerator / denominator
    diff_percent = ((actual - ideal) / ideal) * 100.0

    if abs(diff_percent) <= 3.0:
        deviation = "balanced"
        interpretation = "Functional symmetry (close to ideal)."
    elif actual > ideal:
        deviation = "above_phi"
        interpretation = "Expansion, expressiveness, openness."
    else:
        deviation = "below_phi"
        interpretation = "Restraint, control, self-censorship."

    return IdealRatioOut(
        id=ratio_id,
        label=label,
        formula=formula,
        actual=actual,
        ideal=ideal,
        diff_percent=diff_percent,
        deviation=deviation,
        interpretation=interpretation,
        note=None,
    )


def _classify_face_shape(
    ifv: Optional[float],
    izg: Optional[float],
    ifm: Optional[float],
    ifzf: Optional[float],
) -> tuple[Optional[str], Optional[str]]:
    if None in (ifv, izg, ifm, ifzf):
        return None, "Insufficient measurements for 10-shape classification."

    assert ifv is not None
    assert izg is not None
    assert ifm is not None
    assert ifzf is not None

    proof = f"IFV={ifv:.2f}, IZG={izg:.2f}, IFM={ifm:.2f}, IFZF={ifzf:.2f}. "

    if 1.20 <= ifv <= 1.35 and 1.10 <= izg <= 1.25 and 0.90 <= ifm <= 1.10 and 0.95 <= ifzf <= 1.20:
        return "oval", proof + "Balanced proportions, mildly dominant cheekbones (Neutral reference)."
    if ifv <= 1.15 and 1.15 <= izg <= 1.30 and 0.95 <= ifm <= 1.05 and 1.00 <= ifzf <= 1.25:
        return "round", proof + "Short/wide face with fuller cheeks (Width dominant, low verticality)."
    if ifv <= 1.15 and 0.95 <= izg <= 1.05 and 0.95 <= ifm <= 1.05 and 0.90 <= ifzf <= 1.15:
        return "square", proof + "Cheekbones, jaw, and forehead are close in width (Vertical equality, dominant jaw)."
    if 1.35 < ifv <= 1.45 and 0.90 <= izg <= 1.10 and 0.90 <= ifm <= 1.10 and 0.90 <= ifzf <= 1.15:
        return "rectangular", proof + "Long face with similar lateral widths (High verticality, constant widths)."
    if ifv > 1.45 and 1.05 <= izg <= 1.25 and 0.90 <= ifm <= 1.10 and 1.00 <= ifzf <= 1.25:
        return "oblong", proof + "Very long face with moderately wider cheekbones (Extreme verticality)."
    if ifm > 1.10 and ifv >= 1.25 and 1.00 <= izg <= 1.30 and 1.00 <= ifzf <= 1.30:
        return "triangular", proof + "Dominant forehead, narrower jaw (Pear shape: bottom heavy)."
    if ifm < 0.90 and izg >= 1.00 and ifv >= 1.20 and 0.95 <= ifzf <= 1.25:
        return "inverted_triangular", proof + "Dominant jaw relative to forehead (Heart shape: dominant forehead)."
    if 1.15 <= ifv <= 1.35 and izg >= 1.30 and ifzf >= 1.25 and 0.90 <= ifm <= 1.10:
        return "diamond", proof + "Cheekbones much wider than forehead and jaw (Maximum cheekbones)."
    if 1.15 <= ifv <= 1.35 and ifm > 1.10 and 1.15 <= izg <= 1.30 and 1.10 <= ifzf <= 1.30:
        return "heart", proof + "Wide forehead, narrow jaw, prominent cheekbones (Dominant forehead)."
    if ifv <= 1.25 and ifm < 0.95 and 0.95 <= izg <= 1.15 and 0.95 <= ifzf <= 1.20:
        return "trapezoidal", proof + "Slightly wider in the lower third (Increasing width top to bottom)."

    if ifv > 1.35 and 0.90 <= ifm <= 1.10:
        return "oblong", proof + "Fallback: long face with balanced top-bottom ratio."
    if ifm > 1.10:
        return "triangular", proof + "Fallback: forehead wider than jaw."
    if ifm < 0.90:
        return "inverted_triangular", proof + "Fallback: jaw wider than forehead."
    return "oval", proof + "Fallback: intermediate shape."


def compute_ideal_face_profile(
    front_points: Dict[str, Dict],
    measurements: List[MeasurementOut],
) -> IdealFaceOut:
    g_gn = _point_distance(front_points, "G", "Me")
    en_mid_sn = _avg(
        [
            _point_distance(front_points, "En_R", "Sn"),
            _point_distance(front_points, "En_L", "Sn"),
        ]
    )
    sn_gn = _point_distance(front_points, "Sn", "Me")
    face_width = _measurement_value(measurements, "zy-zy")
    nose_width = _measurement_value(measurements, "al-al")
    mouth_width = _measurement_value(measurements, "ch-ch")
    interocular = _measurement_value(measurements, "en-en")
    eye_width = _avg(
        [
            _measurement_value(measurements, "ex-en-r"),
            _measurement_value(measurements, "ex-en-l"),
        ]
    )
    nose_height = _measurement_value(measurements, "n-sn")
    forehead_height = _measurement_value(measurements, "tr-g")

    ideal_ratios = [
        _ratio_entry(
            "g_gn_over_en_sn",
            "Face height / eye-to-mouth distance",
            "G-GN / EN-SN",
            g_gn,
            en_mid_sn,
            PHI,
        ),
        _ratio_entry(
            "en_sn_over_sn_gn",
            "Eye-to-mouth / mouth-to-chin distance",
            "EN-SN / SN-GN",
            en_mid_sn,
            sn_gn,
            PHI,
        ),
        _ratio_entry(
            "face_width_over_nose_width",
            "Face width / nose width",
            "Wf / Wn",
            face_width,
            nose_width,
            PHI,
        ),
        _ratio_entry(
            "mouth_width_over_nose_width",
            "Mouth width / nose width",
            "Wm / Wn",
            mouth_width,
            nose_width,
            PHI,
        ),
        _ratio_entry(
            "interocular_over_eye_width",
            "Interocular distance / eye width",
            "Do / Wo",
            interocular,
            eye_width,
            1.0,
        ),
        _ratio_entry(
            "nose_height_over_forehead_height",
            "Nose height / forehead height",
            "Hn / Hf",
            nose_height,
            forehead_height,
            PHI,
        ),
    ]

    base_eye_width = eye_width
    tr_me = _point_distance(front_points, "Tr_R", "Me")
    eye_to_mouth = _avg(
        [
            _point_distance(front_points, "En_R", "Sto"),
            _point_distance(front_points, "En_L", "Sto"),
        ]
    )

    dimensions: List[IdealDimensionOut] = []
    formulas = [
        ("face_width", "Face width", face_width, 5.0),
        ("face_height", "Face height (Tr-Me)", tr_me, 3.0),
        ("eye_to_mouth", "Eye-to-mouth distance", eye_to_mouth, 2.5),
        ("nose_width", "Nose width", nose_width, 1.0),
        ("interocular", "Interocular distance", interocular, 1.0),
        ("mouth_width", "Mouth width", mouth_width, 1.5),
        ("forehead_height", "Tr-G distance", forehead_height, 1.0),
    ]
    for item_id, label, actual, factor in formulas:
        if base_eye_width is None:
            dimensions.append(
                IdealDimensionOut(
                    id=item_id,
                    label=label,
                    actual=actual,
                    ideal=None,
                    unit="px",
                    note="Wo unavailable.",
                )
            )
        else:
            dimensions.append(
                IdealDimensionOut(
                    id=item_id,
                    label=label,
                    actual=actual,
                    ideal=base_eye_width * factor,
                    unit="px",
                    note=None if actual is not None else "Missing landmarks.",
                )
            )

    h = _point_distance(front_points, "N", "Me")
    w = face_width
    j = _measurement_value(measurements, "go-go")
    f = _measurement_value(measurements, "ft-ft")

    ifv = (h / w) if (h is not None and w not in (None, 0)) else None
    izg = (w / j) if (w is not None and j not in (None, 0)) else None
    ifm = (f / j) if (f is not None and j not in (None, 0)) else None
    ifzf = (w / f) if (w is not None and f not in (None, 0)) else None

    face_shape, face_shape_note = _classify_face_shape(ifv, izg, ifm, ifzf)

    return IdealFaceOut(
        phi=PHI,
        base_eye_width=base_eye_width,
        ratios=ideal_ratios,
        dimensions=dimensions,
        indices=FaceIndicesOut(ifv=ifv, izg=izg, ifm=ifm, ifzf=ifzf),
        face_shape=face_shape,
        face_shape_note=face_shape_note,
    )


def _safe_ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return a / b


def _canthal_angle(points: Dict[str, Dict], en: str, ex: str) -> Optional[float]:
    if en not in points or ex not in points:
        return None
    dx = abs(points[ex]["pixel"]["x"] - points[en]["pixel"]["x"])
    if dx == 0:
        return None
    # y grows downward in image coordinates; positive means outer canthus is higher.
    dy = points[en]["pixel"]["y"] - points[ex]["pixel"]["y"]
    return math.degrees(math.atan2(dy, dx))


def _classify_horizontal_ratio(r: Optional[float], gender: Optional[str]) -> Optional[str]:
    if r is None:
        return None
    male = [0.235, 0.255, 0.285, 0.310]
    female = [0.245, 0.265, 0.295, 0.320]
    if gender == "female":
        t = female
    elif gender == "male":
        t = male
    else:
        t = [(male[i] + female[i]) / 2.0 for i in range(4)]
    if r < t[0]:
        return "very_narrow"
    if r < t[1]:
        return "narrow"
    if r < t[2]:
        return "medium"
    if r < t[3]:
        return "wide"
    return "very_wide"


def _classify_ifo(ifo: Optional[float], gender: Optional[str]) -> Optional[str]:
    if ifo is None:
        return None
    male = [0.26, 0.29, 0.34, 0.38]
    female = [0.28, 0.31, 0.36, 0.40]
    if gender == "female":
        t = female
    elif gender == "male":
        t = male
    else:
        t = [(male[i] + female[i]) / 2.0 for i in range(4)]
    if ifo < t[0]:
        return "very_narrow"
    if ifo < t[1]:
        return "narrow"
    if ifo < t[2]:
        return "almond"
    if ifo < t[3]:
        return "round"
    return "very_round"


def _classify_angle(angle: Optional[float], gender: Optional[str]) -> Optional[str]:
    if angle is None:
        return None
    if gender == "female":
        if angle < -2.0:
            return "downturned"
        if angle < -0.5:
            return "slightly_downturned"
        if angle <= 1.5:
            return "neutral"
        if angle <= 5.0:
            return "upturned"
        return "strongly_upturned"
    if gender == "male":
        if angle < -3.0:
            return "downturned"
        if angle < -1.0:
            return "slightly_downturned"
        if angle <= 1.0:
            return "neutral"
        if angle <= 4.0:
            return "upturned"
        return "strongly_upturned"
    # neutral profile for nonbinary/unknown
    if angle < -2.5:
        return "downturned"
    if angle < -0.75:
        return "slightly_downturned"
    if angle <= 1.25:
        return "neutral"
    if angle <= 4.5:
        return "upturned"
    return "strongly_upturned"


def _classify_spacing(r: Optional[float], gender: Optional[str]) -> Optional[str]:
    if r is None:
        return None
    if gender == "female":
        if r < 0.95:
            return "very_close_set"
        if r < 1.02:
            return "close_set"
        if r <= 1.10:
            return "balanced"
        if r <= 1.20:
            return "wide_set"
        return "very_wide_set"
    if gender == "male":
        if r < 0.90:
            return "very_close_set"
        if r < 0.98:
            return "close_set"
        if r <= 1.05:
            return "balanced"
        if r <= 1.15:
            return "wide_set"
        return "very_wide_set"
    if r < 0.925:
        return "very_close_set"
    if r < 1.00:
        return "close_set"
    if r <= 1.075:
        return "balanced"
    if r <= 1.175:
        return "wide_set"
    return "very_wide_set"


def _classify_symmetry(delta_aperture_percent: Optional[float], delta_angle_deg: Optional[float]) -> Optional[str]:
    if delta_aperture_percent is None and delta_angle_deg is None:
        return None
    aperture_ok = delta_aperture_percent is None or delta_aperture_percent < 3.0
    angle_ok = delta_angle_deg is None or delta_angle_deg < 1.0
    if aperture_ok and angle_ok:
        return "good"
    aperture_mod = delta_aperture_percent is None or delta_aperture_percent <= 7.0
    angle_mod = delta_angle_deg is None or delta_angle_deg <= 2.0
    if aperture_mod and angle_mod:
        return "moderate"
    return "high_asymmetry"


def _zoomorphic_label(
    form: Optional[str],
    size: Optional[str],
    orientation: Optional[str],
    spacing: Optional[str],
    symmetry: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    if None in (form, size, orientation, spacing):
        return None, "Insufficient data for zoomorphic label."

    assert form is not None
    assert size is not None
    assert orientation is not None
    assert spacing is not None

    if form in {"round", "very_round"} and size in {"wide", "very_wide"}:
        return "deer eyes", "Large, open, rounded aperture."
    if form == "almond" and orientation in {"upturned", "strongly_upturned"} and symmetry == "good":
        return "cat eyes", "Almond contour with upward canthal tilt."
    if form == "almond" and orientation == "strongly_upturned":
        return "fox eyes", "Elongated almond shape with strong upward axis."
    if form in {"narrow", "very_narrow"} and orientation in {"neutral", "slightly_downturned"}:
        return "horse eyes", "Longer horizontal opening with reduced vertical aperture."
    if form in {"round", "very_round"} and spacing in {"wide_set", "very_wide_set"}:
        return "owl eyes", "Round, wide-set appearance."
    if form == "almond" and spacing in {"wide_set", "very_wide_set"}:
        return "eagle eyes", "Focused almond shape with wider spacing."
    if orientation in {"downturned", "slightly_downturned"}:
        return "camel eyes", "Softer downturned canthal axis."
    return "dolphin eyes", "Balanced intermediate morphology."


def compute_eye_profile(
    front_points: Dict[str, Dict],
    measurements: List[MeasurementOut],
    gender: Optional[str],
) -> EyeAnalysisOut:
    face_width = _measurement_value(measurements, "zy-zy")
    eye_w_r = _measurement_value(measurements, "ex-en-r")
    eye_w_l = _measurement_value(measurements, "ex-en-l")
    eye_w = _avg([eye_w_r, eye_w_l])
    eye_h_r = _point_distance(front_points, "Ps_R", "Pi_R")
    eye_h_l = _point_distance(front_points, "Ps_L", "Pi_L")
    eye_h = _avg([eye_h_r, eye_h_l])
    intercanthal = _measurement_value(measurements, "en-en")
    canthal_r = _canthal_angle(front_points, "En_R", "Ex_R")
    canthal_l = _canthal_angle(front_points, "En_L", "Ex_L")
    canthal = _avg([canthal_r, canthal_l])

    horizontal_rel = _safe_ratio(eye_w, face_width)
    ifo = _safe_ratio(eye_h, eye_w)
    spacing_rel = _safe_ratio(intercanthal, eye_w)

    delta_aperture_percent = None
    if eye_h_l is not None and eye_h_r is not None and (eye_h_l + eye_h_r) > 0:
        delta_aperture_percent = abs(eye_h_l - eye_h_r) / ((eye_h_l + eye_h_r) / 2.0) * 100.0
    delta_angle_deg = abs(canthal_l - canthal_r) if (canthal_l is not None and canthal_r is not None) else None

    size_class = _classify_horizontal_ratio(horizontal_rel, gender)
    form_class = _classify_ifo(ifo, gender)
    orientation_class = _classify_angle(canthal, gender)
    spacing_class = _classify_spacing(spacing_rel, gender)
    symmetry_class = _classify_symmetry(delta_aperture_percent, delta_angle_deg)
    zoomorphic, zoomorphic_note = _zoomorphic_label(
        form_class, size_class, orientation_class, spacing_class, symmetry_class
    )

    sex_tag = "♂" if gender == "male" else "♀" if gender == "female" else "N"
    ids = [f"EYE-{sex_tag}-{i:02d}" for i in range(1, 11)]

    metrics = [
        EyeMetricOut(
            id=ids[0],
            label="Horizontal aperture relative (Ex-En / face width)",
            value=horizontal_rel,
            unit="ratio",
            classification=size_class,
            note=None if horizontal_rel is not None else "Missing Ex/En or face width.",
        ),
        EyeMetricOut(
            id=ids[1],
            label="Vertical aperture absolute (Ps-Pi)",
            value=eye_h,
            unit="px",
            classification=None,
            note=None if eye_h is not None else "Missing Ps/Pi landmarks.",
        ),
        EyeMetricOut(
            id=ids[2],
            label="Ocular form index IFO ((Ps-Pi)/(Ex-En))",
            value=ifo,
            unit="ratio",
            classification=form_class,
            note=None if ifo is not None else "Missing vertical/horizontal aperture.",
        ),
        EyeMetricOut(
            id=ids[3],
            label="Canthal angle",
            value=canthal,
            unit="deg",
            classification=orientation_class,
            note=None if canthal is not None else "Missing En/Ex landmarks.",
        ),
        EyeMetricOut(
            id=ids[4],
            label="Intercanthal spacing (En-En / Ex-En)",
            value=spacing_rel,
            unit="ratio",
            classification=spacing_class,
            note=None if spacing_rel is not None else "Missing En-En or eye width.",
        ),
        EyeMetricOut(
            id=ids[5],
            label="Upper eyelid hoodedness (IDP)",
            value=None,
            unit="ratio",
            classification=None,
            note="Unavailable: palpebral crease landmark not mapped.",
        ),
        EyeMetricOut(
            id=ids[6],
            label="Lower eyelid iris coverage",
            value=None,
            unit="percent",
            classification=None,
            note="Unavailable: iris contour landmarks not mapped.",
        ),
        EyeMetricOut(
            id=ids[7],
            label="Ocular projection",
            value=None,
            unit="ratio",
            classification=None,
            note="Unavailable: orbital plane / profile landmarks not mapped.",
        ),
        EyeMetricOut(
            id=ids[8],
            label="Bilateral aperture asymmetry",
            value=delta_aperture_percent,
            unit="percent",
            classification=symmetry_class,
            note=None if delta_aperture_percent is not None else "Requires both left/right Ps-Pi.",
        ),
        EyeMetricOut(
            id=ids[9],
            label="Bilateral canthal angle asymmetry",
            value=delta_angle_deg,
            unit="deg",
            classification=symmetry_class,
            note=None if delta_angle_deg is not None else "Requires both left/right canthal angles.",
        ),
    ]

    classification = EyeClassificationOut(
        form=form_class,
        size=size_class,
        orientation=orientation_class,
        spacing=spacing_class,
        symmetry=symmetry_class,
        eyelid=None,
        depth=None,
    )

    output_signature = None
    if form_class and orientation_class and spacing_class:
        output_signature = (
            f"{gender or 'unspecified'} - {form_class}, {orientation_class}, "
            f"{spacing_class}, symmetry {symmetry_class or 'unknown'}"
        )

    return EyeAnalysisOut(
        standard_ids=ids,
        metrics=metrics,
        composite_indices={
            "ifo": ifo,
            "io": canthal,
            "idp": None,
            "horizontal_aperture_rel": horizontal_rel,
            "intercanthal_rel": spacing_rel,
            "delta_aperture_percent": delta_aperture_percent,
            "delta_angle_deg": delta_angle_deg,
        },
        classification=classification,
        zoomorphic_label=zoomorphic,
        zoomorphic_note=zoomorphic_note,
        output_signature=output_signature,
    )


def compute_forehead_profile(
    front_points: Dict[str, Dict],
    measurements: List[MeasurementOut],
    gender: str | None,
) -> ForeheadAnalysisOut:
    # Basic points
    tr_g = _measurement_value(measurements, "tr-g")
    n_me = _point_distance(front_points, "N", "Me")
    ft_ft = _measurement_value(measurements, "ft-ft")
    zy_zy = _measurement_value(measurements, "zy-zy")
    
    # R1 = Tr-G / N-Me
    r1 = _safe_ratio(tr_g, n_me)
    # R2 = FtL-FtR / ZyL-ZyR
    r2 = _safe_ratio(ft_ft, zy_zy)

    # Classifications
    height_class = None
    width_class = None
    
    # Typologies (Male vs Female thresholds simulated, or averaged if None)
    # Using the standard provided rules: R1 < 0.30 (low), 0.30-0.35 (medium), >0.35 (high)
    if r1 is not None:
        if r1 < 0.30:
            height_class = "low"
        elif r1 <= 0.35:
            height_class = "medium"
        else:
            height_class = "high"
            
    # Width (simplified rules)
    if r2 is not None:
        if r2 < 0.8:
            width_class = "narrow"
        elif r2 <= 0.9:
            width_class = "medium"
        else:
            width_class = "wide"

    proof = []
    if r1 is not None:
        proof.append(f"R1(Tr-G/N-Me)={r1:.2f}")
    if r2 is not None:
        proof.append(f"R2(Ft-Ft/Zy-Zy)={r2:.2f}")

    signature = " | ".join(proof) if proof else "Insufficient data"

    return ForeheadAnalysisOut(
        r1=r1,
        r2=r2,
        r3=None,  # Not supported with 2D mediapipe (requires profile plane)
        r4=None,
        f7=None,
        height_classification=height_class,
        width_classification=width_class,
        profile_classification="neutral (estimated)", 
        relief_classification=None,
        symmetry_classification=None,
        output_signature=signature
    )

def compute_nose_profile(
    front_points: Dict[str, Dict],
    measurements: List[MeasurementOut],
    gender: str | None,
) -> NoseAnalysisOut:
    nw = _measurement_value(measurements, "al-al")
    nh = _measurement_value(measurements, "n-sn")
    pt = _measurement_value(measurements, "sn-prn")  # projection
    nl = _point_distance(front_points, "N", "Prn")

    in_index = _safe_ratio(nw, nh) * 100 if _safe_ratio(nw, nh) else None
    ip_index = _safe_ratio(pt, nh)
    ib_index = _safe_ratio(nw, pt)
    il_index = _safe_ratio(nl, nh)

    # Simplified Typology
    width_class = None
    if in_index is not None:
        if in_index < 70:
            width_class = "narrow"
        elif in_index <= 85:
            width_class = "medium"
        else:
            width_class = "wide"

    proj_class = None
    if ip_index is not None:
        if ip_index < 0.55:
            proj_class = "low"
        elif ip_index <= 0.65:
            proj_class = "medium"
        else:
            proj_class = "high"
            
    base_class = None
    if ib_index is not None:
        if ib_index < 1.0:
            base_class = "compact"
        elif ib_index <= 1.2:
            base_class = "medium"
        else:
            base_class = "expansive"

    proof = []
    if in_index is not None: proof.append(f"IN={in_index:.1f}")
    if ip_index is not None: proof.append(f"IP={ip_index:.2f}")
    if ib_index is not None: proof.append(f"IB={ib_index:.2f}")

    return NoseAnalysisOut(
        in_index=in_index,
        ip_index=ip_index,
        ib_index=ib_index,
        il_index=il_index,
        nla_angle=None, # Needs profile lines
        nfa_angle=None,
        width_classification=width_class,
        projection_classification=proj_class,
        base_classification=base_class,
        rotation_classification=None,
        symmetry_classification=None,
        output_signature=" | ".join(proof) if proof else "Insufficient data"
    )

def compute_mouth_profile(
    front_points: Dict[str, Dict],
    measurements: List[MeasurementOut],
    gender: str | None,
) -> MouthAnalysisOut:
    mw = _measurement_value(measurements, "ch-ch")
    zy_w = _measurement_value(measurements, "zy-zy")
    uth = _measurement_value(measurements, "ls-sto")
    lth = _measurement_value(measurements, "sto-li")
    
    tlt = None
    if uth is not None and lth is not None:
        tlt = uth + lth
        
    ulh = _point_distance(front_points, "Sn", "Sto")
    llh = _point_distance(front_points, "Sto", "Sl")

    imw = _safe_ratio(mw, zy_w) * 100 if mw and zy_w else None
    igb = _safe_ratio(tlt, mw) * 100 if tlt and mw else None
    itb = _safe_ratio(uth, lth)
    ivv = _safe_ratio(ulh, llh)

    width_class = None
    if imw is not None:
        if imw < 30: width_class = "narrow"
        elif imw <= 35: width_class = "medium"
        else: width_class = "wide"

    vol_class = None
    if igb is not None:
        if igb < 25: vol_class = "thin"
        elif igb <= 35: vol_class = "medium"
        else: vol_class = "full"

    ratio_class = None
    if itb is not None:
        if itb < 0.7: ratio_class = "thick lower lip"
        elif itb <= 1.2: ratio_class = "balanced"
        else: ratio_class = "thick upper lip"

    proof = []
    if imw is not None: proof.append(f"IMW={imw:.1f}")
    if igb is not None: proof.append(f"IGB={igb:.1f}")
    if itb is not None: proof.append(f"ITB={itb:.2f}")

    return MouthAnalysisOut(
        imw=imw,
        igb=igb,
        itb=itb,
        ivv=ivv,
        width_classification=width_class,
        volume_classification=vol_class,
        ratio_classification=ratio_class,
        symmetry_classification=None,
        output_signature=" | ".join(proof) if proof else "Insufficient data"
    )

def compute_jaw_profile(
    front_points: Dict[str, Dict],
    measurements: List[MeasurementOut],
    gender: str | None,
) -> JawAnalysisOut:
    li_me = _point_distance(front_points, "Li", "Me")
    sn_me = _point_distance(front_points, "Sn", "Me")
    
    # R1: Vertical ratio
    r1 = _safe_ratio(li_me, sn_me)
    # R3: Projection (Pg - VL) - Can't calculate accurately without profile line VL
    r3 = None
    
    cw = _measurement_value(measurements, "ft-ft") # forehead width
    jw = _measurement_value(measurements, "go-go") # mandible width
    mw = _measurement_value(measurements, "ch-ch") # using mouth as proxy for chin width since chin width isn't directly measured usually, or we skip
    # For menton width we need distinct points, we'll skip r5 and i3 if Not Available
    r5 = None
    i3 = None

    # I2: Mandibular robustness (Go-Go / Sn-Me)
    i2 = _safe_ratio(jw, sn_me)

    mandible_type = None
    if i2 is not None:
        if i2 < 0.58: mandible_type = "narrow"
        elif i2 <= 0.66: mandible_type = "medium"
        else: mandible_type = "wide"

    proof = []
    if r1 is not None: proof.append(f"R1={r1:.2f}")
    if i2 is not None: proof.append(f"I2={i2:.2f}")

    return JawAnalysisOut(
        r1=r1,
        r3=r3,
        r5=r5,
        c6_angle=None,
        i1=None,
        i2=i2,
        i3=i3,
        jp1_angle=None,
        jm4_angle=None,
        mandible_type=mandible_type,
        chin_type=None,
        profile_type=None,
        coherence_flag="OK",
        output_signature=" | ".join(proof) if proof else "Insufficient data"
    )

def compute_cheek_profile(
    front_points: Dict[str, Dict],
    measurements: List[MeasurementOut],
    gender: str | None,
) -> CheekAnalysisOut:
    zy_w = _measurement_value(measurements, "zy-zy")
    go_w = _measurement_value(measurements, "go-go")
    
    # RZ1: Relative width ZyL-ZyR / GoL-GoR
    rz1 = _safe_ratio(zy_w, go_w)
    
    # RZ2: Vertical position Zy-Sn / Tr-Sn
    zy_sn = _point_distance(front_points, "Zy_L", "Sn") # approximation using distance
    tr_sn = _point_distance(front_points, "Tr_L", "Sn")
    rz2 = _safe_ratio(zy_sn, tr_sn)
    
    bone_class = None
    if rz1 is not None:
        if rz1 < 1.1: bone_class = "narrow"
        elif rz1 <= 1.25: bone_class = "medium"
        else: bone_class = "wide"

    proof = []
    if rz1 is not None: proof.append(f"RZ1(Zy/Go)={rz1:.2f}")

    return CheekAnalysisOut(
        rz1=rz1,
        rz2=rz2,
        rz3=None,
        ro1=None,
        ro2=None,
        ro3=None,
        bone_classification=bone_class,
        volume_classification=None,
        output_signature=" | ".join(proof) if proof else "Insufficient data"
    )

def compute_eyebrow_profile(
    front_points: Dict[str, Dict],
    measurements: List[MeasurementOut],
    gender: str | None,
) -> EyebrowAnalysisOut:
    # Requires specific eyebrow points (Bh, Br, Bt) which might not be mapped in `measurements_catalog.json`
    # We will output empty for now, or estimate if labels exist
    # Let's assume standard points aren't fully mapped to those exact labels yet
    
    return EyebrowAnalysisOut(
        bed=None,
        bt=None,
        delta_h=None,
        bta_angle=None,
        bl_ratio=None,
        form_classification=None,
        position_classification=None,
        thickness_classification=None,
        length_classification=None,
        tail_classification=None,
        symmetry_classification=None,
        output_signature="Insufficient data"
    )

def compute_ear_profile(
    side_points: Dict[str, Dict],
    measurements: List[MeasurementOut],
    gender: str | None,
) -> EarAnalysisOut:
    # Ears use Sa, Sba, Pra, Pa, In, L
    el = _point_distance(side_points, "Sa_R", "Sba_R")
    ew = _point_distance(side_points, "Pra_R", "Pa_R")
    
    ie = _safe_ratio(ew, el)

    form_class = None
    if ie is not None:
        if ie < 0.5: form_class = "elongated"
        elif ie <= 0.65: form_class = "balanced"
        else: form_class = "wide"

    proof = []
    if el is not None: proof.append(f"EL={el:.1f}px")
    if ie is not None: proof.append(f"IE={ie:.2f}")

    return EarAnalysisOut(
        el=el,
        ew=ew,
        ll=None,
        ed=None,
        ie=ie,
        il_index=None,
        ip=None,
        length_classification=None,
        form_classification=form_class,
        lob_classification=None,
        protrusion_classification=None,
        symmetry_classification=None,
        output_signature=" | ".join(proof) if proof else "Insufficient data"
    )
