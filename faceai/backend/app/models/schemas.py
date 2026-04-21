from typing import Dict, List, Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    ok: bool


class Point2D(BaseModel):
    x: float
    y: float


class Point3D(BaseModel):
    x: float
    y: float
    z: float


class LandmarkOut(BaseModel):
    label: str
    index: Optional[int]
    pixel: Optional[Point2D]
    normalized: Optional[Point3D]


class MeasurementOut(BaseModel):
    id: str
    label: str
    image: str
    points: List[str]
    value: Optional[float]
    unit: str
    note: Optional[str]


class RatioOut(BaseModel):
    id: str
    label: Optional[str] = None
    numerator: str
    denominator: str
    value: Optional[float]
    ideal_value: Optional[float] = None
    deviation_pct: Optional[float] = None
    note: Optional[str]


class AnalyzeResponse(BaseModel):
    ok: bool
    all_landmarks_count: int
    gender: Optional[str]
    mandatory_landmarks: List[LandmarkOut]
    measurements: List[MeasurementOut]
    ratios: List[RatioOut]
    annotated_images: Dict[str, str]
    warnings: List[str]
