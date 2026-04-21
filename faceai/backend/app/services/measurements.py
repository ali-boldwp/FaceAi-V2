import json
import math
from pathlib import Path
from typing import Dict, List, Optional

from app.models.schemas import MeasurementOut, RatioOut

CATALOG_PATH = Path(__file__).resolve().parent.parent / "utils" / "measurements_catalog.json"
SUPPORTED_MEASUREMENT_IMAGE = "front"

RATIO_DEFS = [
    {
        "id": "overall_face_height_to_width",
        "label": "Total face height / face width",
        "numerator": "tr-me",
        "denominator": "zy-zy",
        "classifier": "overall_face_height_to_width",
    },
    {
        "id": "overall_jaw_to_cheek_width",
        "label": "Mandible width / cheekbone width",
        "numerator": "go-go",
        "denominator": "zy-zy",
        "classifier": "overall_jaw_to_cheek_width",
    },
    {
        "id": "overall_forehead_to_cheek_width",
        "label": "Forehead width / cheekbone width",
        "numerator": "ft-ft",
        "denominator": "zy-zy",
        "classifier": "overall_forehead_to_cheek_width",
    },
    {
        "id": "overall_upper_third_share",
        "label": "Upper third / total face height",
        "numerator": "tr-g",
        "denominator": "tr-me",
        "ideal_value": 1.0 / 3.0,
    },
    {
        "id": "overall_middle_third_share",
        "label": "Middle third / total face height",
        "numerator": "g-sn",
        "denominator": "tr-me",
        "ideal_value": 1.0 / 3.0,
    },
    {
        "id": "overall_lower_third_share",
        "label": "Lower third / total face height",
        "numerator": "sn-gn",
        "denominator": "tr-me",
        "ideal_value": 1.0 / 3.0,
    },
    {
        "id": "overall_face_shape",
        "label": "Overall face shape classifier",
        "numerator": "tr-me",
        "denominator": "zy-zy",
        "classifier": "overall_face_shape",
    },
    {
        "id": "forehead_height_share",
        "label": "Forehead height / total face height",
        "numerator": "tr-g",
        "denominator": "tr-me",
        "classifier": "forehead_height_share",
    },
    {
        "id": "forehead_width_ratio",
        "label": "Forehead width / forehead height",
        "numerator": "ft-ft",
        "denominator": "tr-g",
        "classifier": "forehead_width_ratio",
    },
    {
        "id": "forehead_width_to_cheek_width",
        "label": "Forehead width / cheekbone width",
        "numerator": "ft-ft",
        "denominator": "zy-zy",
        "classifier": "forehead_width_to_cheek_width",
    },
    {
        "id": "forehead_glabellar_balance",
        "label": "Forehead height / glabellar segment",
        "numerator": "tr-g",
        "denominator": "n-g",
        "classifier": "forehead_glabellar_balance",
    },
    {
        "id": "forehead_symmetry_balance",
        "label": "FtL-G / FtR-G",
        "numerator": "ft-l-g",
        "denominator": "ft-r-g",
        "ideal_value": 1.0,
        "classifier": "forehead_symmetry_balance",
    },
    {
        "id": "mouth_width_face_share",
        "label": "Mouth width / cheekbone width",
        "numerator": "ch-ch",
        "denominator": "zy-zy",
        "classifier": "mouth_width_face_share",
    },
    {
        "id": "mouth_width_nose_height",
        "label": "Mouth width / nose height",
        "numerator": "ch-ch",
        "denominator": "n-sn",
        "classifier": "mouth_width_nose_height",
    },
    {
        "id": "lip_fullness",
        "label": "Total lip thickness / mouth width",
        "numerator": "tlt",
        "denominator": "ch-ch",
        "classifier": "lip_fullness",
    },
    {
        "id": "lip_dominance",
        "label": "Upper lip / lower lip thickness",
        "numerator": "ls-sto",
        "denominator": "sto-li",
        "classifier": "lip_dominance",
    },
    {
        "id": "mouth_vertical_balance",
        "label": "Upper lip height / lower lip height",
        "numerator": "sn-sto",
        "denominator": "sto-sl",
        "classifier": "mouth_vertical_balance",
    },
    {
        "id": "mouth_corner_asymmetry",
        "label": "Corner delta / mouth width",
        "numerator": "ch-delta-y",
        "denominator": "ch-ch",
        "classifier": "mouth_corner_asymmetry",
    },
    {
        "id": "mouth_center_deviation",
        "label": "Stomion deviation / mouth width",
        "numerator": "sto-center-dev",
        "denominator": "ch-ch",
        "classifier": "mouth_center_deviation",
    },
    {
        "id": "mouth_corner_orientation",
        "label": "Corner level / mouth width",
        "numerator": "ch-avg-vs-sto",
        "denominator": "ch-ch",
        "classifier": "mouth_corner_orientation",
    },
    {
        "id": "eye_width_face_share_r",
        "label": "Right eye width / face width",
        "numerator": "ex-en-r",
        "denominator": "zy-zy",
        "classifier": "eye_width_face_share",
    },
    {
        "id": "eye_width_face_share_l",
        "label": "Left eye width / face width",
        "numerator": "ex-en-l",
        "denominator": "zy-zy",
        "classifier": "eye_width_face_share",
    },
    {
        "id": "eye_form_index_r",
        "label": "Right eye form index",
        "numerator": "ps-pi-r",
        "denominator": "ex-en-r",
        "classifier": "eye_form_index",
    },
    {
        "id": "eye_form_index_l",
        "label": "Left eye form index",
        "numerator": "ps-pi-l",
        "denominator": "ex-en-l",
        "classifier": "eye_form_index",
    },
    {
        "id": "intercanthal_eye_ratio",
        "label": "Intercanthal distance / mean eye width",
        "numerator": "en-en",
        "denominator": "avg-eye-width",
        "classifier": "intercanthal_eye_ratio",
    },
    {
        "id": "eye_aperture_symmetry",
        "label": "Right aperture / left aperture",
        "numerator": "ps-pi-r",
        "denominator": "ps-pi-l",
        "ideal_value": 1.0,
        "classifier": "eye_aperture_symmetry",
    },
    {
        "id": "eye_width_symmetry",
        "label": "Right eye width / left eye width",
        "numerator": "ex-en-r",
        "denominator": "ex-en-l",
        "ideal_value": 1.0,
        "classifier": "eye_width_symmetry",
    },
    {
        "id": "cheekbone_dominance",
        "label": "Cheekbone width / jaw width",
        "numerator": "zy-zy",
        "denominator": "go-go",
        "classifier": "cheekbone_dominance",
    },
    {
        "id": "cheekbone_forehead_balance",
        "label": "Cheekbone width / forehead width",
        "numerator": "zy-zy",
        "denominator": "ft-ft",
        "classifier": "cheekbone_forehead_balance",
    },
    {
        "id": "cheekbone_vertical_position",
        "label": "Cheekbone vertical position",
        "numerator": "zy-tr-vertical",
        "denominator": "sn-tr-vertical",
        "classifier": "cheekbone_vertical_position",
    },
    {
        "id": "cheekbone_midline_symmetry",
        "label": "Cheekbone midline asymmetry / width",
        "numerator": "zy-midline-asym",
        "denominator": "zy-zy",
        "classifier": "cheekbone_midline_symmetry",
    },
    {
        "id": "cheekbone_level_symmetry",
        "label": "Cheekbone level asymmetry / width",
        "numerator": "zy-level-diff",
        "denominator": "zy-zy",
        "classifier": "cheekbone_level_symmetry",
    },
    {
        "id": "ear_shape_index_r",
        "label": "Right ear width / ear length",
        "numerator": "pra-pa-r",
        "denominator": "sa-sba-r",
        "classifier": "ear_shape_index",
    },
    {
        "id": "ear_shape_index_l",
        "label": "Left ear width / ear length",
        "numerator": "pra-pa-l",
        "denominator": "sa-sba-l",
        "classifier": "ear_shape_index",
    },
    {
        "id": "ear_length_symmetry",
        "label": "Right ear length / left ear length",
        "numerator": "sa-sba-r",
        "denominator": "sa-sba-l",
        "ideal_value": 1.0,
        "classifier": "ear_length_symmetry",
    },
    {
        "id": "ear_width_symmetry",
        "label": "Right ear width / left ear width",
        "numerator": "pra-pa-r",
        "denominator": "pra-pa-l",
        "ideal_value": 1.0,
        "classifier": "ear_width_symmetry",
    },
    {
        "id": "ear_length_face_share_r",
        "label": "Right ear length / face height",
        "numerator": "sa-sba-r",
        "denominator": "tr-me",
        "classifier": "ear_length_face_share",
    },
    {
        "id": "ear_length_face_share_l",
        "label": "Left ear length / face height",
        "numerator": "sa-sba-l",
        "denominator": "tr-me",
        "classifier": "ear_length_face_share",
    },
    {
        "id": "nose_width_height_index",
        "label": "Nose width / nose height",
        "numerator": "al-al",
        "denominator": "n-sn",
        "classifier": "nose_width_height_index",
    },
    {
        "id": "nose_projection_index",
        "label": "Tip projection / nose height",
        "numerator": "sn-prn",
        "denominator": "n-sn",
        "classifier": "nose_projection_index",
    },
    {
        "id": "nose_base_index",
        "label": "Nose width / tip projection",
        "numerator": "al-al",
        "denominator": "sn-prn",
        "classifier": "nose_base_index",
    },
    {
        "id": "nose_verticality_index",
        "label": "Nose length / nose height",
        "numerator": "n-prn",
        "denominator": "n-sn",
        "classifier": "nose_verticality_index",
    },
    {
        "id": "nose_base_asymmetry",
        "label": "Alar asymmetry / nose width",
        "numerator": "al-base-asym",
        "denominator": "al-al",
        "classifier": "nose_base_asymmetry",
    },
    {
        "id": "nose_tip_deviation",
        "label": "Prn midline deviation / nose width",
        "numerator": "prn-midline-dev",
        "denominator": "al-al",
        "classifier": "nose_tip_deviation",
    },
    {
        "id": "ideal_face_width_to_nose_width",
        "label": "Face width / nose width",
        "numerator": "zy-zy",
        "denominator": "al-al",
        "ideal_value": 1.618,
    },
    {
        "id": "ideal_mouth_to_nose_width",
        "label": "Mouth width / nose width",
        "numerator": "ch-ch",
        "denominator": "al-al",
        "ideal_value": 1.618,
    },
    {
        "id": "ideal_interocular_to_eye_width",
        "label": "Intercanthal distance / eye width",
        "numerator": "en-en",
        "denominator": "ex-en-r",
        "ideal_value": 1.0,
    },
    {
        "id": "ideal_nose_to_forehead_height",
        "label": "Nose height / forehead height",
        "numerator": "n-sn",
        "denominator": "tr-g",
        "ideal_value": 1.618,
    },
    {
        "id": "ideal_glabella_face_to_eye_subnasale",
        "label": "G-Me / En-Sn",
        "numerator": "g-me",
        "denominator": "en-sn-r",
        "ideal_value": 1.618,
    },
    {
        "id": "ideal_eye_subnasale_to_lower_face",
        "label": "En-Sn / Sn-Me",
        "numerator": "en-sn-r",
        "denominator": "sn-gn",
        "ideal_value": 1.618,
    },
    {
        "id": "ideal_upper_third_balance",
        "label": "Forehead height / midface height",
        "numerator": "tr-g",
        "denominator": "g-sn",
        "ideal_value": 1.0,
    },
    {
        "id": "ideal_lower_third_balance",
        "label": "Midface height / lower face height",
        "numerator": "g-sn",
        "denominator": "sn-gn",
        "ideal_value": 1.0,
    },
    {
        "id": "ideal_total_height_to_forehead",
        "label": "Total face height / forehead height",
        "numerator": "tr-me",
        "denominator": "tr-g",
        "ideal_value": 3.0,
    },
    {
        "id": "ideal_total_height_to_midface",
        "label": "Total face height / midface height",
        "numerator": "tr-me",
        "denominator": "g-sn",
        "ideal_value": 3.0,
    },
    {
        "id": "ideal_total_height_to_lower_face",
        "label": "Total face height / lower face height",
        "numerator": "tr-me",
        "denominator": "sn-gn",
        "ideal_value": 3.0,
    },
    {
        "id": "upper_to_lower_lip",
        "label": "Upper lip / lower lip",
        "numerator": "ls-sto",
        "denominator": "sto-li",
        "ideal_value": 1.0,
    },
    {
        "id": "chin_height_share",
        "label": "Chin height / lower face height",
        "numerator": "li-me",
        "denominator": "sn-gn",
        "classifier": "chin_height_share",
    },
    {
        "id": "chin_vertical_dominance",
        "label": "Pg-Me / lower face height",
        "numerator": "pg-me",
        "denominator": "sn-gn",
        "classifier": "chin_vertical_dominance",
    },
    {
        "id": "chin_width_proxy",
        "label": "Lower face width / chin height",
        "numerator": "go-go",
        "denominator": "li-me",
        "classifier": "chin_width_proxy",
    },
    {
        "id": "chin_symmetry_balance",
        "label": "GoL-Me / GoR-Me",
        "numerator": "go-l-me",
        "denominator": "go-r-me",
        "ideal_value": 1.0,
        "classifier": "chin_symmetry_balance",
    },
]


def _distance(point_a: Dict, point_b: Dict) -> float:
    dx = point_a["pixel"]["x"] - point_b["pixel"]["x"]
    dy = point_a["pixel"]["y"] - point_b["pixel"]["y"]
    return (dx ** 2 + dy ** 2) ** 0.5


def _signed_axis_delta(point_a: Dict, point_b: Dict, axis: str) -> float:
    return float(point_a["pixel"][axis] - point_b["pixel"][axis])


def _midline_x(points: Dict[str, Dict]) -> Optional[float]:
    labels = ("G", "N", "Sn", "Me", "Tr_R")
    xs = [float(points[label]["pixel"]["x"]) for label in labels if label in points]
    if not xs:
        return None
    return sum(xs) / len(xs)


def _load_catalog() -> List[Dict]:
    with CATALOG_PATH.open("r", encoding="utf-8") as handle:
        catalog = json.load(handle)
    return [entry for entry in catalog if entry.get("image") == SUPPORTED_MEASUREMENT_IMAGE]


def _format_ratio_note(value: float, ideal_value: Optional[float]) -> str:
    if ideal_value is None:
        return "Computed from front-image landmarks."

    deviation_pct = abs(value - ideal_value) / ideal_value * 100.0 if ideal_value else None
    if deviation_pct is None:
        return "Computed from front-image landmarks."
    if deviation_pct <= 5:
        status = "Very close to ideal"
    elif deviation_pct <= 12:
        status = "Close to ideal"
    elif deviation_pct <= 20:
        status = "Moderately different from ideal"
    else:
        status = "Clearly different from ideal"
    return f"{status}; ideal target {ideal_value:.3f}."


def _gender_bucket(gender: Optional[str]) -> str:
    if gender == "female":
        return "female"
    return "male"


def _classify_chin_height_share(value: float, gender: Optional[str]) -> str:
    bucket = _gender_bucket(gender)
    if bucket == "female":
        if value < 0.31:
            return "Bărbie joasă; raport Li-Me / Sn-Me sub pragul feminin."
        if value <= 0.37:
            return "Bărbie medie; raport Li-Me / Sn-Me în intervalul feminin normal."
        return "Bărbie înaltă; raport Li-Me / Sn-Me peste pragul feminin."
    if value < 0.29:
        return "Bărbie joasă; raport Li-Me / Sn-Me sub pragul masculin."
    if value <= 0.35:
        return "Bărbie medie; raport Li-Me / Sn-Me în intervalul masculin normal."
    return "Bărbie înaltă; raport Li-Me / Sn-Me peste pragul masculin."


def _classify_chin_vertical_dominance(value: float) -> str:
    if value < 0.18:
        return "Dominanță mentonială verticală redusă."
    if value <= 0.24:
        return "Dominanță mentonială verticală echilibrată."
    return "Dominanță mentonială verticală accentuată."


def _classify_chin_width_proxy(value: float, gender: Optional[str]) -> str:
    bucket = _gender_bucket(gender)
    if bucket == "female":
        if value < 2.8:
            return "Bărbie frontală relativ îngustă."
        if value <= 4.1:
            return "Bărbie frontală cu lățime medie."
        return "Bărbie frontală relativ lată."
    if value < 3.0:
        return "Bărbie frontală relativ îngustă."
    if value <= 4.4:
        return "Bărbie frontală cu lățime medie."
    return "Bărbie frontală relativ lată."


def _classify_chin_symmetry_balance(value: float) -> str:
    deviation_pct = abs(value - 1.0) * 100.0
    if deviation_pct <= 3:
        return "Simetrie mentonială bună între GoL-Me și GoR-Me."
    if deviation_pct <= 6:
        return "Asimetrie mentonială moderată între stânga și dreapta."
    return "Asimetrie mentonială mare între stânga și dreapta."


def _classify_overall_face_height_to_width(value: float, gender: Optional[str]) -> str:
    bucket = _gender_bucket(gender)
    if bucket == "female":
        if value < 1.32:
            return "Față lată/rotundă; raportul H/W este sub pragul feminin."
        if value <= 1.40:
            return "Față echilibrată/ovală; raportul H/W este în intervalul feminin mediu."
        if value <= 1.44:
            return "Față dreptunghiulară/oblongă; verticalitate crescută."
        return "Față alungită; verticalitate facială accentuată."
    if value < 1.30:
        return "Față lată/rotundă; raportul H/W este sub pragul masculin."
    if value <= 1.38:
        return "Față echilibrată/ovală; raportul H/W este în intervalul masculin mediu."
    if value <= 1.42:
        return "Față dreptunghiulară/oblongă; verticalitate crescută."
    return "Față alungită; verticalitate facială accentuată."


def _classify_overall_jaw_to_cheek_width(value: float, gender: Optional[str]) -> str:
    bucket = _gender_bucket(gender)
    if bucket == "female":
        if value < 0.86:
            return "Mandibulă îngustă față de pomeți; pattern de tip heart/diamond."
        if value <= 0.96:
            return "Mandibulă echilibrată față de pomeți."
        return "Mandibulă lată față de pomeți; pattern de tip square/pear."
    if value < 0.88:
        return "Mandibulă îngustă față de pomeți; pattern de tip heart/diamond."
    if value <= 0.98:
        return "Mandibulă echilibrată față de pomeți."
    return "Mandibulă lată față de pomeți; pattern de tip square/pear."


def _classify_overall_forehead_to_cheek_width(value: float) -> str:
    if value < 0.90:
        return "Frunte relativ îngustă față de pomeți."
    if value <= 1.00:
        return "Frunte echilibrată față de pomeți."
    return "Frunte relativ lată față de pomeți."


def _classify_overall_face_shape(
    value: float,
    measurement_map: Dict[str, MeasurementOut],
    gender: Optional[str],
) -> str:
    jaw = measurement_map.get("go-go")
    cheek = measurement_map.get("zy-zy")
    forehead = measurement_map.get("ft-ft")
    if not jaw or not cheek or not forehead:
        return "Clasificarea formei feței necesită lățimile mandibulei, pomeților și frunții."
    if jaw.value is None or cheek.value is None or forehead.value is None or cheek.value == 0:
        return "Clasificarea formei feței nu a putut fi calculată din măsurătorile curente."

    jaw_ratio = jaw.value / cheek.value
    forehead_ratio = forehead.value / cheek.value
    bucket = _gender_bucket(gender)
    round_cutoff = 1.32 if bucket == "female" else 1.30
    oval_cutoff = 1.40 if bucket == "female" else 1.38
    oblong_cutoff = 1.44 if bucket == "female" else 1.42

    if value > oblong_cutoff:
        return "Formă facială probabil alungită (Long / Elongated)."
    if value > oval_cutoff:
        if jaw_ratio >= 0.96:
            return "Formă facială probabil dreptunghiulară / oblongă."
        return "Formă facială probabil ovală alungită."
    if value < round_cutoff:
        if jaw_ratio >= 0.98:
            return "Formă facială probabil pătrată."
        return "Formă facială probabil rotundă."
    if forehead_ratio > 1.0 and jaw_ratio < 0.90:
        return "Formă facială probabil triunghiulară inversată / heart."
    if forehead_ratio < 0.92 and jaw_ratio > 0.98:
        return "Formă facială probabil triangulară / pear."
    if jaw_ratio < 0.90 and forehead_ratio < 0.95:
        return "Formă facială probabil diamond."
    if jaw_ratio >= 0.97:
        return "Formă facială probabil square / hexagonală."
    return "Formă facială probabil ovală."


def _classify_forehead_height_share(value: float, gender: Optional[str]) -> str:
    bucket = _gender_bucket(gender)
    if bucket == "female":
        if value < 0.30:
            return "Frunte joasă; proporția frunții este sub pragul feminin."
        if value <= 0.36:
            return "Frunte medie; proporția frunții este în intervalul feminin normal."
        return "Frunte înaltă; proporția frunții este peste pragul feminin."
    if value < 0.28:
        return "Frunte joasă; proporția frunții este sub pragul masculin."
    if value <= 0.34:
        return "Frunte medie; proporția frunții este în intervalul masculin normal."
    return "Frunte înaltă; proporția frunții este peste pragul masculin."


def _classify_forehead_width_ratio(value: float, gender: Optional[str]) -> str:
    bucket = _gender_bucket(gender)
    if bucket == "female":
        if value < 0.85:
            return "Frunte îngustă în raport cu înălțimea."
        if value <= 1.00:
            return "Frunte cu lățime medie în raport cu înălțimea."
        return "Frunte lată în raport cu înălțimea."
    if value < 0.90:
        return "Frunte îngustă în raport cu înălțimea."
    if value <= 1.05:
        return "Frunte cu lățime medie în raport cu înălțimea."
    return "Frunte lată în raport cu înălțimea."


def _classify_forehead_width_to_cheek_width(value: float) -> str:
    if value < 0.85:
        return "Frunte îngustă față de lățimea zigomatică."
    if value <= 1.00:
        return "Frunte echilibrată față de lățimea zigomatică."
    return "Frunte lată față de lățimea zigomatică."


def _classify_forehead_glabellar_balance(value: float) -> str:
    if value < 1.8:
        return "Segment frontal scurt față de zona glabelară."
    if value <= 2.1:
        return "Raport frontal-glabelar echilibrat."
    return "Segment frontal lung față de zona glabelară."


def _classify_forehead_symmetry_balance(value: float) -> str:
    deviation_pct = abs(value - 1.0) * 100.0
    if deviation_pct <= 3:
        return "Simetrie frontală bună între FtL-G și FtR-G."
    if deviation_pct <= 6:
        return "Asimetrie frontală moderată."
    return "Asimetrie frontală mare."


def _classify_mouth_width_face_share(value: float, gender: Optional[str]) -> str:
    bucket = _gender_bucket(gender)
    if bucket == "female":
        if value < 0.33:
            return "Gură foarte îngustă sau îngustă față de lățimea facială."
        if value <= 0.399:
            return "Gură cu lățime medie față de lățimea facială."
        if value <= 0.429:
            return "Gură largă față de lățimea facială."
        return "Gură foarte largă față de lățimea facială."
    if value < 0.34:
        return "Gură foarte îngustă sau îngustă față de lățimea facială."
    if value <= 0.409:
        return "Gură cu lățime medie față de lățimea facială."
    if value <= 0.439:
        return "Gură largă față de lățimea facială."
    return "Gură foarte largă față de lățimea facială."


def _classify_mouth_width_nose_height(value: float) -> str:
    if value < 1.20:
        return "Raport gură-nas compact."
    if value <= 1.65:
        return "Raport gură-nas echilibrat."
    return "Raport gură-nas expansiv."


def _classify_lip_fullness(value: float, gender: Optional[str]) -> str:
    bucket = _gender_bucket(gender)
    if bucket == "female":
        if value < 0.28:
            return "Buze foarte subțiri sau subțiri."
        if value <= 0.379:
            return "Buze medii."
        if value <= 0.449:
            return "Buze pline."
        return "Buze foarte pline."
    if value < 0.26:
        return "Buze foarte subțiri sau subțiri."
    if value <= 0.359:
        return "Buze medii."
    if value <= 0.419:
        return "Buze pline."
    return "Buze foarte pline."


def _classify_lip_dominance(value: float, gender: Optional[str]) -> str:
    bucket = _gender_bucket(gender)
    if bucket == "female":
        if value < 0.80:
            return "Buza inferioară este dominantă."
        if value <= 0.95:
            return "Buze echilibrate."
        return "Buza superioară este dominantă."
    if value < 0.78:
        return "Buza inferioară este dominantă."
    if value <= 0.92:
        return "Buze echilibrate."
    return "Buza superioară este dominantă."


def _classify_mouth_vertical_balance(value: float, gender: Optional[str]) -> str:
    bucket = _gender_bucket(gender)
    if bucket == "female":
        if value < 0.88:
            return "Etajul vertical inferior al gurii este dominant."
        if value <= 1.10:
            return "Raport vertical al gurii aproape echilibrat."
        return "Etajul vertical superior al gurii este dominant."
    if value < 0.85:
        return "Etajul vertical inferior al gurii este dominant."
    if value <= 1.05:
        return "Raport vertical al gurii aproape echilibrat."
    return "Etajul vertical superior al gurii este dominant."


def _classify_mouth_corner_asymmetry(value: float) -> str:
    if value < 0.015:
        return "Asimetrie mică a comisurilor."
    if value <= 0.03:
        return "Asimetrie moderată a comisurilor."
    return "Asimetrie mare a comisurilor."


def _classify_mouth_center_deviation(value: float) -> str:
    if value < 0.01:
        return "Deviație mică a Stomionului față de centrul gurii."
    if value <= 0.025:
        return "Deviație moderată a Stomionului față de centrul gurii."
    return "Deviație mare a Stomionului față de centrul gurii."


def _classify_mouth_corner_orientation(value: float) -> str:
    if value <= -0.015:
        return "Comisuri ascendente."
    if value >= 0.015:
        return "Comisuri descendente."
    return "Comisuri aproape orizontale."


def _classify_eye_width_face_share(value: float, gender: Optional[str]) -> str:
    bucket = _gender_bucket(gender)
    if bucket == "female":
        if value < 0.245:
            return "Ochi foarte înguști sau înguști față de lățimea feței."
        if value <= 0.295:
            return "Ochi de dimensiune medie față de lățimea feței."
        if value <= 0.320:
            return "Ochi largi."
        return "Ochi foarte largi."
    if value < 0.235:
        return "Ochi foarte înguști sau înguști față de lățimea feței."
    if value <= 0.285:
        return "Ochi de dimensiune medie față de lățimea feței."
    if value <= 0.310:
        return "Ochi largi."
    return "Ochi foarte largi."


def _classify_eye_form_index(value: float, gender: Optional[str]) -> str:
    bucket = _gender_bucket(gender)
    if bucket == "female":
        if value < 0.28:
            return "Ochi foarte îngust sau îngust."
        if value <= 0.31:
            return "Ochi îngust."
        if value <= 0.36:
            return "Ochi migdalat."
        if value <= 0.40:
            return "Ochi rotund."
        return "Ochi foarte rotund."
    if value < 0.26:
        return "Ochi foarte îngust sau îngust."
    if value <= 0.29:
        return "Ochi îngust."
    if value <= 0.34:
        return "Ochi migdalat."
    if value <= 0.38:
        return "Ochi rotund."
    return "Ochi foarte rotund."


def _classify_intercanthal_eye_ratio(value: float, gender: Optional[str]) -> str:
    bucket = _gender_bucket(gender)
    if bucket == "female":
        if value < 0.95:
            return "Ochi foarte apropiați sau apropiați."
        if value <= 1.10:
            return "Distanță intercanthală normală."
        if value <= 1.20:
            return "Ochi depărtați."
        return "Ochi foarte depărtați."
    if value < 0.90:
        return "Ochi foarte apropiați sau apropiați."
    if value <= 1.05:
        return "Distanță intercanthală normală."
    if value <= 1.15:
        return "Ochi depărtați."
    return "Ochi foarte depărtați."


def _classify_eye_symmetry_ratio(value: float, label: str) -> str:
    deviation_pct = abs(value - 1.0) * 100.0
    subject = "deschiderea oculară" if "aperture" in label else "lățimea oculară"
    if deviation_pct <= 3:
        return f"Simetrie bună pentru {subject}."
    if deviation_pct <= 7:
        return f"Asimetrie moderată pentru {subject}."
    return f"Asimetrie mare pentru {subject}."


def _classify_canthal_angle(angle: float, gender: Optional[str]) -> str:
    bucket = _gender_bucket(gender)
    if bucket == "female":
        if angle < -2.0:
            return "Canthus căzut."
        if angle <= 1.5:
            return "Canthus neutru."
        if angle <= 5.0:
            return "Canthus ascendent."
        return "Canthus foarte ascendent."
    if angle < -3.0:
        return "Canthus căzut."
    if angle <= 1.0:
        return "Canthus neutru."
    if angle <= 4.0:
        return "Canthus ascendent."
    return "Canthus foarte ascendent."


def _classify_eye_width_face_share(value: float, gender: Optional[str]) -> str:
    bucket = _gender_bucket(gender)
    if bucket == "female":
        if value < 0.245:
            return "Ochi foarte înguști sau înguști față de lățimea feței."
        if value <= 0.295:
            return "Ochi de dimensiune medie față de lățimea feței."
        if value <= 0.320:
            return "Ochi largi."
        return "Ochi foarte largi."
    if value < 0.235:
        return "Ochi foarte înguști sau înguști față de lățimea feței."
    if value <= 0.285:
        return "Ochi de dimensiune medie față de lățimea feței."
    if value <= 0.310:
        return "Ochi largi."
    return "Ochi foarte largi."


def _classify_eye_form_index(value: float, gender: Optional[str]) -> str:
    bucket = _gender_bucket(gender)
    if bucket == "female":
        if value < 0.28:
            return "Ochi foarte îngust sau îngust."
        if value <= 0.31:
            return "Ochi îngust."
        if value <= 0.36:
            return "Ochi migdalat."
        if value <= 0.40:
            return "Ochi rotund."
        return "Ochi foarte rotund."
    if value < 0.26:
        return "Ochi foarte îngust sau îngust."
    if value <= 0.29:
        return "Ochi îngust."
    if value <= 0.34:
        return "Ochi migdalat."
    if value <= 0.38:
        return "Ochi rotund."
    return "Ochi foarte rotund."


def _classify_intercanthal_eye_ratio(value: float, gender: Optional[str]) -> str:
    bucket = _gender_bucket(gender)
    if bucket == "female":
        if value < 0.95:
            return "Ochi foarte apropiați sau apropiați."
        if value <= 1.10:
            return "Distanță intercanthală normală."
        if value <= 1.20:
            return "Ochi depărtați."
        return "Ochi foarte depărtați."
    if value < 0.90:
        return "Ochi foarte apropiați sau apropiați."
    if value <= 1.05:
        return "Distanță intercanthală normală."
    if value <= 1.15:
        return "Ochi depărtați."
    return "Ochi foarte depărtați."


def _classify_eye_symmetry_ratio(value: float, label: str) -> str:
    deviation_pct = abs(value - 1.0) * 100.0
    subject = "deschiderea oculară" if "aperture" in label else "lățimea oculară"
    if deviation_pct <= 3:
        return f"Simetrie bună pentru {subject}."
    if deviation_pct <= 7:
        return f"Asimetrie moderată pentru {subject}."
    return f"Asimetrie mare pentru {subject}."


def _classify_canthal_angle(angle: float, gender: Optional[str]) -> str:
    bucket = _gender_bucket(gender)
    if bucket == "female":
        if angle < -2.0:
            return "Canthus căzut."
        if angle <= 1.5:
            return "Canthus neutru."
        if angle <= 5.0:
            return "Canthus ascendent."
        return "Canthus foarte ascendent."
    if angle < -3.0:
        return "Canthus căzut."
    if angle <= 1.0:
        return "Canthus neutru."
    if angle <= 4.0:
        return "Canthus ascendent."
    return "Canthus foarte ascendent."


def _classify_nose_width_height_index(value: float, gender: Optional[str]) -> str:
    bucket = _gender_bucket(gender)
    if bucket == "female":
        if value < 0.60:
            return "Nas foarte îngust sau îngust."
        if value <= 0.729:
            return "Nas cu lățime medie."
        if value <= 0.799:
            return "Nas larg."
        return "Nas foarte larg."
    if value < 0.62:
        return "Nas foarte îngust sau îngust."
    if value <= 0.749:
        return "Nas cu lățime medie."
    if value <= 0.819:
        return "Nas larg."
    return "Nas foarte larg."


def _classify_nose_projection_index(value: float, gender: Optional[str]) -> str:
    bucket = _gender_bucket(gender)
    if bucket == "female":
        if value < 0.46:
            return "Proiecție mică a vârfului nazal."
        if value <= 0.54:
            return "Proiecție medie a vârfului nazal."
        return "Proiecție mare a vârfului nazal."
    if value < 0.48:
        return "Proiecție mică a vârfului nazal."
    if value <= 0.56:
        return "Proiecție medie a vârfului nazal."
    return "Proiecție mare a vârfului nazal."


def _classify_nose_base_index(value: float, gender: Optional[str]) -> str:
    bucket = _gender_bucket(gender)
    if bucket == "female":
        if value < 1.50:
            return "Bază nazală compactă."
        if value <= 1.80:
            return "Bază nazală medie."
        return "Bază nazală expansivă."
    if value < 1.55:
        return "Bază nazală compactă."
    if value <= 1.85:
        return "Bază nazală medie."
    return "Bază nazală expansivă."


def _classify_nose_verticality_index(value: float, gender: Optional[str]) -> str:
    bucket = _gender_bucket(gender)
    if bucket == "female":
        if value < 1.06:
            return "Nas relativ scurt."
        if value <= 1.16:
            return "Nas cu lungime medie."
        return "Nas alungit."
    if value < 1.08:
        return "Nas relativ scurt."
    if value <= 1.18:
        return "Nas cu lungime medie."
    return "Nas alungit."


def _classify_nose_base_asymmetry(value: float) -> str:
    if value < 0.02:
        return "Asimetrie mică a bazei nazale."
    if value <= 0.04:
        return "Asimetrie moderată a bazei nazale."
    return "Asimetrie mare a bazei nazale."


def _classify_nose_tip_deviation(value: float) -> str:
    if value < 0.02:
        return "Deviație mică a vârfului nazal."
    if value <= 0.05:
        return "Deviație moderată a vârfului nazal."
    return "Deviație mare a vârfului nazal."


def _classify_cheekbone_dominance(value: float) -> str:
    if value < 0.95:
        return "Pomeți relativ înguști față de mandibulă."
    if value <= 1.05:
        return "Pomeți echilibrați față de mandibulă."
    return "Pomeți dominanți / lați față de mandibulă."


def _classify_cheekbone_forehead_balance(value: float) -> str:
    if value < 1.00:
        return "Pomeții sunt mai înguști decât fruntea."
    if value <= 1.10:
        return "Pomeții și fruntea sunt relativ echilibrate."
    return "Pomeții sunt mai dominanți decât fruntea."


def _classify_cheekbone_vertical_position(value: float) -> str:
    if value < 0.42:
        return "Pomeți poziționați jos."
    if value <= 0.48:
        return "Pomeți cu poziție verticală medie."
    return "Pomeți poziționați sus."


def _classify_cheekbone_symmetry(value: float, label: str) -> str:
    if value < 0.02:
        return f"Simetrie bună pentru {label}."
    if value <= 0.04:
        return f"Asimetrie moderată pentru {label}."
    return f"Asimetrie mare pentru {label}."


def _classify_ear_shape_index(value: float) -> str:
    if value < 0.50:
        return "Ureche alungită / îngustă."
    if value <= 0.55:
        return "Ureche echilibrată ca formă."
    return "Ureche lată."


def _classify_ear_symmetry(value: float, label: str) -> str:
    deviation_pct = abs(value - 1.0) * 100.0
    if deviation_pct < 3:
        return f"Simetrie bună pentru {label}."
    if deviation_pct <= 6:
        return f"Asimetrie moderată pentru {label}."
    return f"Asimetrie mare pentru {label}."


def _classify_ear_length_face_share(value: float) -> str:
    if value < 0.22:
        return "Ureche relativ mică față de înălțimea feței."
    if value <= 0.32:
        return "Ureche cu dimensiune medie față de înălțimea feței."
    return "Ureche relativ mare față de înălțimea feței."


def _classifier_note(
    classifier: str,
    value: float,
    gender: Optional[str],
    measurement_map: Optional[Dict[str, MeasurementOut]] = None,
) -> str:
    if classifier == "chin_height_share":
        return _classify_chin_height_share(value, gender)
    if classifier == "chin_vertical_dominance":
        return _classify_chin_vertical_dominance(value)
    if classifier == "chin_width_proxy":
        return _classify_chin_width_proxy(value, gender)
    if classifier == "chin_symmetry_balance":
        return _classify_chin_symmetry_balance(value)
    if classifier == "overall_face_height_to_width":
        return _classify_overall_face_height_to_width(value, gender)
    if classifier == "overall_jaw_to_cheek_width":
        return _classify_overall_jaw_to_cheek_width(value, gender)
    if classifier == "overall_forehead_to_cheek_width":
        return _classify_overall_forehead_to_cheek_width(value)
    if classifier == "overall_face_shape":
        return _classify_overall_face_shape(value, measurement_map or {}, gender)
    if classifier == "forehead_height_share":
        return _classify_forehead_height_share(value, gender)
    if classifier == "forehead_width_ratio":
        return _classify_forehead_width_ratio(value, gender)
    if classifier == "forehead_width_to_cheek_width":
        return _classify_forehead_width_to_cheek_width(value)
    if classifier == "forehead_glabellar_balance":
        return _classify_forehead_glabellar_balance(value)
    if classifier == "forehead_symmetry_balance":
        return _classify_forehead_symmetry_balance(value)
    if classifier == "mouth_width_face_share":
        return _classify_mouth_width_face_share(value, gender)
    if classifier == "mouth_width_nose_height":
        return _classify_mouth_width_nose_height(value)
    if classifier == "lip_fullness":
        return _classify_lip_fullness(value, gender)
    if classifier == "lip_dominance":
        return _classify_lip_dominance(value, gender)
    if classifier == "mouth_vertical_balance":
        return _classify_mouth_vertical_balance(value, gender)
    if classifier == "mouth_corner_asymmetry":
        return _classify_mouth_corner_asymmetry(value)
    if classifier == "mouth_center_deviation":
        return _classify_mouth_center_deviation(value)
    if classifier == "mouth_corner_orientation":
        return _classify_mouth_corner_orientation(value)
    if classifier == "eye_width_face_share":
        return _classify_eye_width_face_share(value, gender)
    if classifier == "eye_form_index":
        return _classify_eye_form_index(value, gender)
    if classifier == "intercanthal_eye_ratio":
        return _classify_intercanthal_eye_ratio(value, gender)
    if classifier == "eye_aperture_symmetry":
        return _classify_eye_symmetry_ratio(value, "aperture")
    if classifier == "eye_width_symmetry":
        return _classify_eye_symmetry_ratio(value, "width")
    if classifier == "cheekbone_dominance":
        return _classify_cheekbone_dominance(value)
    if classifier == "cheekbone_forehead_balance":
        return _classify_cheekbone_forehead_balance(value)
    if classifier == "cheekbone_vertical_position":
        return _classify_cheekbone_vertical_position(value)
    if classifier == "cheekbone_midline_symmetry":
        return _classify_cheekbone_symmetry(value, "poziția laterală a pomeților")
    if classifier == "cheekbone_level_symmetry":
        return _classify_cheekbone_symmetry(value, "nivelul vertical al pomeților")
    if classifier == "ear_shape_index":
        return _classify_ear_shape_index(value)
    if classifier == "ear_length_symmetry":
        return _classify_ear_symmetry(value, "lungimea urechii")
    if classifier == "ear_width_symmetry":
        return _classify_ear_symmetry(value, "lățimea urechii")
    if classifier == "ear_length_face_share":
        return _classify_ear_length_face_share(value)
    if classifier == "nose_width_height_index":
        return _classify_nose_width_height_index(value, gender)
    if classifier == "nose_projection_index":
        return _classify_nose_projection_index(value, gender)
    if classifier == "nose_base_index":
        return _classify_nose_base_index(value, gender)
    if classifier == "nose_verticality_index":
        return _classify_nose_verticality_index(value, gender)
    if classifier == "nose_base_asymmetry":
        return _classify_nose_base_asymmetry(value)
    if classifier == "nose_tip_deviation":
        return _classify_nose_tip_deviation(value)
    return "Computed from front-image landmarks."


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

    measurement_map = {m.id: m for m in results}
    upper_lip = measurement_map.get("ls-sto")
    lower_lip = measurement_map.get("sto-li")
    if upper_lip and lower_lip and upper_lip.value is not None and lower_lip.value is not None:
        results.append(
            MeasurementOut(
                id="tlt",
                label="Total lip thickness",
                image="front",
                points=["Ls", "Sto", "Li"],
                value=upper_lip.value + lower_lip.value,
                unit="px",
                note="Derived from upper and lower lip thickness.",
            )
        )

    ch_r = front_points.get("Ch_R")
    ch_l = front_points.get("Ch_L")
    sto = front_points.get("Sto")
    if ch_r and ch_l:
        results.append(
            MeasurementOut(
                id="ch-delta-y",
                label="Mouth corner vertical delta",
                image="front",
                points=["Ch_R", "Ch_L"],
                value=abs(_signed_axis_delta(ch_r, ch_l, "y")),
                unit="px",
                note="Absolute vertical difference between mouth corners.",
            )
        )

    if ch_r and ch_l and sto:
        center_x = (float(ch_r["pixel"]["x"]) + float(ch_l["pixel"]["x"])) / 2.0
        avg_corner_y = (float(ch_r["pixel"]["y"]) + float(ch_l["pixel"]["y"])) / 2.0
        results.append(
            MeasurementOut(
                id="sto-center-dev",
                label="Stomion deviation from mouth center",
                image="front",
                points=["Sto", "Ch_R", "Ch_L"],
                value=abs(float(sto["pixel"]["x"]) - center_x),
                unit="px",
                note="Horizontal deviation of Stomion from the mouth midpoint.",
            )
        )
        results.append(
            MeasurementOut(
                id="ch-avg-vs-sto",
                label="Corner level relative to Stomion",
                image="front",
                points=["Ch_R", "Ch_L", "Sto"],
                value=avg_corner_y - float(sto["pixel"]["y"]),
                unit="px",
                note="Signed mean corner height relative to Stomion.",
            )
        )

    al_r = front_points.get("Al_R")
    al_l = front_points.get("Al_L")
    prn = front_points.get("Prn")
    target_midline_x = _midline_x(front_points)
    if al_r and al_l:
        if target_midline_x is None:
            target_midline_x = (float(al_r["pixel"]["x"]) + float(al_l["pixel"]["x"])) / 2.0
        left_dist = abs(float(al_l["pixel"]["x"]) - target_midline_x)
        right_dist = abs(float(al_r["pixel"]["x"]) - target_midline_x)
        results.append(
            MeasurementOut(
                id="al-base-asym",
                label="Alar base asymmetry",
                image="front",
                points=["Al_R", "Al_L"],
                value=abs(left_dist - right_dist),
                unit="px",
                note="Difference between left and right alar spread around the facial midline.",
            )
        )

    if prn:
        if target_midline_x is None and al_r and al_l:
            target_midline_x = (float(al_r["pixel"]["x"]) + float(al_l["pixel"]["x"])) / 2.0
        if target_midline_x is not None:
            results.append(
                MeasurementOut(
                    id="prn-midline-dev",
                    label="Pronasale deviation from midline",
                    image="front",
                    points=["Prn"],
                    value=abs(float(prn["pixel"]["x"]) - target_midline_x),
                    unit="px",
                    note="Horizontal deviation of pronasale from the facial midline.",
                )
            )

    zy_r = front_points.get("Zy_R")
    zy_l = front_points.get("Zy_L")
    tr = front_points.get("Tr_R")
    sn = front_points.get("Sn")
    if zy_r and zy_l:
        if target_midline_x is None:
            target_midline_x = (float(zy_r["pixel"]["x"]) + float(zy_l["pixel"]["x"])) / 2.0
        right_dist = abs(float(zy_r["pixel"]["x"]) - target_midline_x)
        left_dist = abs(float(zy_l["pixel"]["x"]) - target_midline_x)
        results.append(
            MeasurementOut(
                id="zy-midline-asym",
                label="Cheekbone midline asymmetry",
                image="front",
                points=["Zy_R", "Zy_L"],
                value=abs(left_dist - right_dist),
                unit="px",
                note="Difference between left and right zygion spread around the facial midline.",
            )
        )
        results.append(
            MeasurementOut(
                id="zy-level-diff",
                label="Cheekbone level difference",
                image="front",
                points=["Zy_R", "Zy_L"],
                value=abs(float(zy_r["pixel"]["y"]) - float(zy_l["pixel"]["y"])),
                unit="px",
                note="Absolute vertical difference between left and right zygion.",
            )
        )

    if zy_r and zy_l and tr and sn:
        avg_zy_y = (float(zy_r["pixel"]["y"]) + float(zy_l["pixel"]["y"])) / 2.0
        tr_y = float(tr["pixel"]["y"])
        sn_y = float(sn["pixel"]["y"])
        results.append(
            MeasurementOut(
                id="zy-tr-vertical",
                label="Cheekbone vertical distance from trichion",
                image="front",
                points=["Zy_R", "Zy_L", "Tr_R"],
                value=max(0.0, avg_zy_y - tr_y),
                unit="px",
                note="Average zygion vertical position relative to trichion.",
            )
        )
        results.append(
            MeasurementOut(
                id="sn-tr-vertical",
                label="Subnasale vertical distance from trichion",
                image="front",
                points=["Sn", "Tr_R"],
                value=max(0.0, sn_y - tr_y),
                unit="px",
                note="Subnasale vertical position relative to trichion.",
            )
        )

    ex_r = front_points.get("Ex_R")
    en_r = front_points.get("En_R")
    ex_l = front_points.get("Ex_L")
    en_l = front_points.get("En_L")
    ps_r = front_points.get("Ps_R")
    pi_r = front_points.get("Pi_R")
    ps_l = front_points.get("Ps_L")
    pi_l = front_points.get("Pi_L")

    if ex_r and en_r:
        angle_r = math.degrees(
            math.atan2(float(en_r["pixel"]["y"]) - float(ex_r["pixel"]["y"]), float(ex_r["pixel"]["x"]) - float(en_r["pixel"]["x"]))
        )
        results.append(
            MeasurementOut(
                id="canthal-angle-r",
                label="Right canthal angle",
                image="front",
                points=["En_R", "Ex_R"],
                value=angle_r,
                unit="deg",
                note="Right eye canthal orientation angle.",
            )
        )

    if ex_l and en_l:
        angle_l = math.degrees(
            math.atan2(float(en_l["pixel"]["y"]) - float(ex_l["pixel"]["y"]), float(ex_l["pixel"]["x"]) - float(en_l["pixel"]["x"]))
        )
        results.append(
            MeasurementOut(
                id="canthal-angle-l",
                label="Left canthal angle",
                image="front",
                points=["En_L", "Ex_L"],
                value=angle_l,
                unit="deg",
                note="Left eye canthal orientation angle.",
            )
        )

    if ps_r and pi_r:
        results.append(
            MeasurementOut(
                id="ps-pi-r",
                label="Right eye aperture height",
                image="front",
                points=["Ps_R", "Pi_R"],
                value=_distance(ps_r, pi_r),
                unit="px",
                note="Right palpebral aperture height.",
            )
        )

    if ps_l and pi_l:
        results.append(
            MeasurementOut(
                id="ps-pi-l",
                label="Left eye aperture height",
                image="front",
                points=["Ps_L", "Pi_L"],
                value=_distance(ps_l, pi_l),
                unit="px",
                note="Left palpebral aperture height.",
            )
        )

    measurement_map = {m.id: m for m in results}
    right_eye = measurement_map.get("ex-en-r")
    left_eye = measurement_map.get("ex-en-l")
    if right_eye and left_eye and right_eye.value is not None and left_eye.value is not None:
        results.append(
            MeasurementOut(
                id="avg-eye-width",
                label="Average eye width",
                image="front",
                points=["Ex_R", "En_R", "Ex_L", "En_L"],
                value=(right_eye.value + left_eye.value) / 2.0,
                unit="px",
                note="Mean horizontal eye aperture width.",
            )
        )

    measurement_map = {m.id: m for m in results}
    angle_right = measurement_map.get("canthal-angle-r")
    angle_left = measurement_map.get("canthal-angle-l")
    if angle_right and angle_left and angle_right.value is not None and angle_left.value is not None:
        results.append(
            MeasurementOut(
                id="canthal-angle-diff",
                label="Canthal angle difference",
                image="front",
                points=["En_R", "Ex_R", "En_L", "Ex_L"],
                value=abs(angle_right.value - angle_left.value),
                unit="deg",
                note="Absolute left/right canthal angle difference.",
            )
        )

    measurement_map = {m.id: m for m in results}
    for measurement_id, label, classifier in (
        ("canthal-angle-r", "right", "canthal"),
        ("canthal-angle-l", "left", "canthal"),
        ("canthal-angle-diff", "symmetry", "diff"),
    ):
        measurement = measurement_map.get(measurement_id)
        if measurement is None or measurement.value is None:
            continue
        if classifier == "canthal":
            measurement.note = _classify_canthal_angle(measurement.value, None)
        else:
            diff = measurement.value
            if diff < 1.0:
                measurement.note = "Simetrie bună a orientării cantale."
            elif diff <= 2.0:
                measurement.note = "Asimetrie moderată a orientării cantale."
            else:
                measurement.note = "Asimetrie mare a orientării cantale."

    return results


def compute_ratios(measurements: List[MeasurementOut], gender: Optional[str] = None) -> List[RatioOut]:
    measurement_map = {m.id: m for m in measurements}
    ratios: List[RatioOut] = []

    for entry in RATIO_DEFS:
        ratio_id = entry["id"]
        label = entry.get("label")
        numerator_id = entry["numerator"]
        denominator_id = entry["denominator"]
        ideal_value = entry.get("ideal_value")
        classifier = entry.get("classifier")
        numerator = measurement_map.get(numerator_id)
        denominator = measurement_map.get(denominator_id)

        value: Optional[float] = None
        deviation_pct: Optional[float] = None
        note: Optional[str] = None

        if numerator and denominator and numerator.value is not None and denominator.value is not None:
            if denominator.value == 0:
                note = "Denominator is zero for this ratio."
            else:
                value = numerator.value / denominator.value
                if ideal_value is not None and ideal_value != 0:
                    deviation_pct = abs(value - ideal_value) / ideal_value * 100.0
                note = (
                    _format_ratio_note(value, ideal_value)
                    if ideal_value is not None
                    else _classifier_note(classifier, value, gender, measurement_map)
                )
        else:
            note = "Missing measurements for ratio."

        ratios.append(
            RatioOut(
                id=ratio_id,
                label=label,
                numerator=numerator_id,
                denominator=denominator_id,
                value=value,
                ideal_value=ideal_value,
                deviation_pct=deviation_pct,
                note=note,
            )
        )

    return ratios
