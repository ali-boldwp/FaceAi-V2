import json
from pathlib import Path
from typing import Dict, List, Optional

from app.models.schemas import MeasurementOut, RatioOut

CATALOG_PATH = Path(__file__).resolve().parent.parent / "utils" / "measurements_catalog.json"

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
