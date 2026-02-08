from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from app.models.schemas import AnalyzeResponse, LandmarkOut, MeasurementOut, RatioOut
from app.services.hairline import estimate_trichion
from app.services.measurements import compute_measurements, compute_ratios
from app.services.overlay import draw_landmarks, draw_all_landmarks
from app.utils.image_io import read_image, to_base64_png
from app.utils.landmarks_map import load_landmark_map


@dataclass
class FaceSelection:
    landmarks: List
    bbox: Tuple[float, float, float, float]
    score: float


def _bbox_from_landmarks(landmarks: List) -> Tuple[float, float, float, float]:
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    return min(xs), min(ys), max(xs), max(ys)


def _select_best_face(landmark_lists: List, image_w: int, image_h: int) -> Optional[FaceSelection]:
    if not landmark_lists:
        return None

    image_center = (0.5, 0.5)
    best: Optional[FaceSelection] = None

    for face in landmark_lists:
        bbox = _bbox_from_landmarks(face.landmark)
        min_x, min_y, max_x, max_y = bbox
        area = max(0.0, (max_x - min_x)) * max(0.0, (max_y - min_y))
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        dist = ((center_x - image_center[0]) ** 2 + (center_y - image_center[1]) ** 2) ** 0.5
        score = area - (dist * area * 0.5)
        selection = FaceSelection(face.landmark, bbox, score)
        if best is None or selection.score > best.score:
            best = selection

    return best


def _extract_landmarks(image_bgr: np.ndarray) -> Tuple[List, int]:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if not hasattr(mp, "solutions"):
        raise RuntimeError(
            "MediaPipe 'solutions' module not available. "
            "Pin mediapipe to a version that includes solutions (e.g. 0.10.11) "
            "and reinstall backend dependencies."
        )
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=5,
        refine_landmarks=False,
        min_detection_confidence=0.5,
    ) as face_mesh:
        results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return [], 0

    return results.multi_face_landmarks, len(results.multi_face_landmarks[0].landmark)


def _points_from_map(landmarks: List, mapping: Dict[str, Optional[int]], width: int, height: int) -> Dict[str, Dict]:
    points: Dict[str, Dict] = {}
    for label, index in mapping.items():
        if index is None:
            continue
        if index < 0 or index >= len(landmarks):
            continue
        lm = landmarks[index]
        px = float(lm.x * width)
        py = float(lm.y * height)
        points[label] = {
            "index": index,
            "pixel": {"x": px, "y": py},
            "normalized": {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)},
        }

    if "Prn" in mapping and len(landmarks) > 4:
        lm_a = landmarks[4]
        lm_b = landmarks[1]
        nx = (lm_a.x + lm_b.x) / 2.0
        ny = (lm_a.y + lm_b.y) / 2.0
        nz = (lm_a.z + lm_b.z) / 2.0
        points["Prn"] = {
            "index": None,
            "pixel": {"x": float(nx * width), "y": float(ny * height)},
            "normalized": {"x": float(nx), "y": float(ny), "z": float(nz)},
        }

    if "Sto" not in points and "Ls" in points and "Li" in points:
        px = (points["Ls"]["pixel"]["x"] + points["Li"]["pixel"]["x"]) / 2.0
        py = (points["Ls"]["pixel"]["y"] + points["Li"]["pixel"]["y"]) / 2.0
        nx = (points["Ls"]["normalized"]["x"] + points["Li"]["normalized"]["x"]) / 2.0
        ny = (points["Ls"]["normalized"]["y"] + points["Li"]["normalized"]["y"]) / 2.0
        nz = (points["Ls"]["normalized"]["z"] + points["Li"]["normalized"]["z"]) / 2.0
        points["Sto"] = {
            "index": None,
            "pixel": {"x": px, "y": py},
            "normalized": {"x": nx, "y": ny, "z": nz},
        }

    return points


def _tr_from_normalized(tr_x: float, tr_y: float, width: int, height: int) -> Dict:
    px = float(tr_x * width)
    py = float(tr_y * height)
    return {
        "index": None,
        "pixel": {"x": px, "y": py},
        "normalized": {"x": float(tr_x), "y": float(tr_y), "z": 0.0},
    }


def analyze_images(
    front_bytes: bytes,
    side_bytes: bytes,
    tr_x: float | None = None,
    tr_y: float | None = None,
    gender: str | None = None,
) -> AnalyzeResponse:
    front_image, front_w, front_h = read_image(front_bytes)
    side_image, side_w, side_h = read_image(side_bytes)

    front_faces, front_count = _extract_landmarks(front_image)
    side_faces, side_count = _extract_landmarks(side_image)

    if not front_faces:
        raise ValueError("No face detected in front image")
    side_missing = False
    if not side_faces:
        side_missing = True

    front_selection = _select_best_face(front_faces, front_w, front_h)
    side_selection = _select_best_face(side_faces, side_w, side_h) if not side_missing else None

    if front_selection is None:
        raise ValueError("No face detected in front image")
    if side_selection is None and not side_missing:
        raise ValueError("No face detected in side image")

    mapping = load_landmark_map()

    front_points = _points_from_map(front_selection.landmarks, mapping, front_w, front_h)
    side_points = (
        _points_from_map(side_selection.landmarks, mapping, side_w, side_h)
        if side_selection is not None
        else {}
    )
    tr_method = "none"
    trichion = None
    tr_debug = {}
    if tr_x is not None and tr_y is not None:
        trichion = _tr_from_normalized(tr_x, tr_y, front_w, front_h)
        tr_method = "manual"
    else:
        trichion, tr_debug, tr_method = estimate_trichion(
            front_image, front_points, landmarks=front_selection.landmarks, debug=True
        )
    trichion_available = trichion is not None
    if trichion:
        front_points["Tr_R"] = trichion
        front_points["Tr_L"] = trichion

    mandatory_landmarks: List[LandmarkOut] = []
    for label, index in mapping.items():
        entry = front_points.get(label) or side_points.get(label)
        if entry:
            mandatory_landmarks.append(
                LandmarkOut(
                    label=label,
                    index=entry["index"],
                    pixel=entry["pixel"],
                    normalized=entry["normalized"],
                )
            )
        else:
            mandatory_landmarks.append(LandmarkOut(label=label, index=index, pixel=None, normalized=None))

    measurements: List[MeasurementOut] = compute_measurements(front_points, side_points)
    ratios: List[RatioOut] = compute_ratios(measurements)

    annotated_front = draw_landmarks(front_image.copy(), front_points)
    annotated_side = draw_landmarks(side_image.copy(), side_points) if side_points else side_image.copy()
    annotated_front_all = draw_all_landmarks(front_image.copy(), front_selection.landmarks)
    annotated_side_all = (
        draw_all_landmarks(side_image.copy(), side_selection.landmarks) if side_selection is not None else side_image.copy()
    )

    warnings: List[str] = []
    if len(front_faces) > 1:
        warnings.append("Multiple faces detected in front image; selected the most central/largest face.")
    if len(side_faces) > 1:
        warnings.append("Multiple faces detected in side image; selected the most central/largest face.")
    if side_missing:
        warnings.append("No face detected in side image; side measurements are unavailable.")
    if not trichion_available:
        warnings.append("Trichion (Tr) unavailable; hairline segmentation did not return a result.")
    elif tr_method == "fallback":
        warnings.append("Trichion (Tr) estimated with geometric fallback (no hair detected).")
    elif tr_method == "manual":
        warnings.append("Trichion (Tr) set manually.")

    return AnalyzeResponse(
        ok=True,
        all_landmarks_count=front_count,
        gender=gender,
        mandatory_landmarks=mandatory_landmarks,
        measurements=measurements,
        ratios=ratios,
        annotated_images={
            "front": to_base64_png(annotated_front),
            "side": to_base64_png(annotated_side),
            "front_all": to_base64_png(annotated_front_all),
            "side_all": to_base64_png(annotated_side_all),
            **{key: to_base64_png(img) for key, img in tr_debug.items()},
        },
        warnings=warnings,
    )
