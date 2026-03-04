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
    numerator: str
    denominator: str
    value: Optional[float]
    note: Optional[str]


class IdealRatioOut(BaseModel):
    id: str
    label: str
    formula: str
    actual: Optional[float]
    ideal: float
    diff_percent: Optional[float]
    deviation: Optional[str]
    interpretation: Optional[str]
    note: Optional[str]


class IdealDimensionOut(BaseModel):
    id: str
    label: str
    actual: Optional[float]
    ideal: Optional[float]
    unit: str
    note: Optional[str]


class FaceIndicesOut(BaseModel):
    ifv: Optional[float]
    izg: Optional[float]
    ifm: Optional[float]
    ifzf: Optional[float]


class IdealFaceOut(BaseModel):
    phi: float
    base_eye_width: Optional[float]
    ratios: List[IdealRatioOut]
    dimensions: List[IdealDimensionOut]
    indices: FaceIndicesOut
    face_shape: Optional[str]
    face_shape_note: Optional[str]


class EyeMetricOut(BaseModel):
    id: str
    label: str
    value: Optional[float]
    unit: str
    classification: Optional[str]
    note: Optional[str]


class EyeClassificationOut(BaseModel):
    form: Optional[str]
    size: Optional[str]
    orientation: Optional[str]
    spacing: Optional[str]
    symmetry: Optional[str]
    eyelid: Optional[str]
    depth: Optional[str]


class EyeAnalysisOut(BaseModel):
    standard_ids: List[str]
    metrics: List[EyeMetricOut]
    composite_indices: Dict[str, Optional[float]]
    classification: EyeClassificationOut
    zoomorphic_label: Optional[str]
    zoomorphic_note: Optional[str]
    output_signature: Optional[str]


class ForeheadAnalysisOut(BaseModel):
    r1: Optional[float]
    r2: Optional[float]
    r3: Optional[float]
    r4: Optional[float]
    f7: Optional[float] # angle
    height_classification: Optional[str]
    width_classification: Optional[str]
    profile_classification: Optional[str]
    relief_classification: Optional[str]
    symmetry_classification: Optional[str]
    output_signature: Optional[str]

class NoseAnalysisOut(BaseModel):
    in_index: Optional[float]
    ip_index: Optional[float]
    ib_index: Optional[float]
    il_index: Optional[float]
    nla_angle: Optional[float]
    nfa_angle: Optional[float]
    width_classification: Optional[str]
    projection_classification: Optional[str]
    base_classification: Optional[str]
    rotation_classification: Optional[str]
    symmetry_classification: Optional[str]
    output_signature: Optional[str]

class MouthAnalysisOut(BaseModel):
    imw: Optional[float]
    igb: Optional[float]
    itb: Optional[float]
    ivv: Optional[float]
    width_classification: Optional[str]
    volume_classification: Optional[str]
    ratio_classification: Optional[str]
    symmetry_classification: Optional[str]
    output_signature: Optional[str]

class JawAnalysisOut(BaseModel):
    r1: Optional[float]
    r3: Optional[float]
    r5: Optional[float]
    c6_angle: Optional[float]
    i1: Optional[float]
    i2: Optional[float]
    i3: Optional[float]
    jp1_angle: Optional[float]
    jm4_angle: Optional[float]
    mandible_type: Optional[str]
    chin_type: Optional[str]
    profile_type: Optional[str]
    coherence_flag: Optional[str]
    output_signature: Optional[str]

class CheekAnalysisOut(BaseModel):
    rz1: Optional[float]
    rz2: Optional[float]
    rz3: Optional[float]
    ro1: Optional[float]
    ro2: Optional[float]
    ro3: Optional[float]
    bone_classification: Optional[str]
    volume_classification: Optional[str]
    output_signature: Optional[str]

class EyebrowAnalysisOut(BaseModel):
    bed: Optional[float]
    bt: Optional[float]
    delta_h: Optional[float]
    bta_angle: Optional[float]
    bl_ratio: Optional[float]
    form_classification: Optional[str]
    position_classification: Optional[str]
    thickness_classification: Optional[str]
    length_classification: Optional[str]
    tail_classification: Optional[str]
    symmetry_classification: Optional[str]
    output_signature: Optional[str]

class EarAnalysisOut(BaseModel):
    el: Optional[float]
    ew: Optional[float]
    ll: Optional[float]
    ed: Optional[float]
    ie: Optional[float]
    il_index: Optional[float]
    ip: Optional[float]
    length_classification: Optional[str]
    form_classification: Optional[str]
    lob_classification: Optional[str]
    protrusion_classification: Optional[str]
    symmetry_classification: Optional[str]
    output_signature: Optional[str]

class AnalyzeResponse(BaseModel):
    ok: bool
    all_landmarks_count: int
    gender: Optional[str]
    mandatory_landmarks: List[LandmarkOut]
    measurements: List[MeasurementOut]
    ratios: List[RatioOut]
    ideal_face: IdealFaceOut
    eyes: EyeAnalysisOut
    forehead: Optional[ForeheadAnalysisOut] = None
    nose: Optional[NoseAnalysisOut] = None
    mouth: Optional[MouthAnalysisOut] = None
    jaw: Optional[JawAnalysisOut] = None
    cheek: Optional[CheekAnalysisOut] = None
    eyebrows: Optional[EyebrowAnalysisOut] = None
    ears: Optional[EarAnalysisOut] = None
    annotated_images: Dict[str, str]
    warnings: List[str]
