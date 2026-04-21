from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

from app.models.schemas import AnalyzeResponse, LandmarkOut, MeasurementOut, RatioOut
from app.services.hairline import estimate_trichion
from app.services.measurements import compute_measurements, compute_ratios
from app.services.overlay import draw_landmarks, draw_all_landmarks
from app.utils.image_io import read_image, to_base64_png
from app.utils.landmarks_map import load_landmark_map

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
FACE_LANDMARKER_MODEL_PATH = PROJECT_ROOT / "model_cache" / "mediapipe" / "face_landmarker.task"
_FACE_LANDMARKER = None


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
        bbox = _bbox_from_landmarks(face)
        min_x, min_y, max_x, max_y = bbox
        area = max(0.0, (max_x - min_x)) * max(0.0, (max_y - min_y))
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        dist = ((center_x - image_center[0]) ** 2 + (center_y - image_center[1]) ** 2) ** 0.5
        score = area - (dist * area * 0.5)
        selection = FaceSelection(face, bbox, score)
        if best is None or selection.score > best.score:
            best = selection

    return best


def _extract_landmarks(image_bgr: np.ndarray) -> Tuple[List, int]:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if hasattr(mp, "solutions"):
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=5,
            refine_landmarks=False,
            min_detection_confidence=0.5,
        ) as face_mesh:
            results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return [], 0

        faces = [face.landmark for face in results.multi_face_landmarks]
        return faces, len(faces[0])

    landmarker = _get_face_landmarker()
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = landmarker.detect(image)
    if not results.face_landmarks:
        return [], 0

    return results.face_landmarks, len(results.face_landmarks[0])


def _get_face_landmarker():
    global _FACE_LANDMARKER

    if _FACE_LANDMARKER is not None:
        return _FACE_LANDMARKER

    if not FACE_LANDMARKER_MODEL_PATH.exists():
        FACE_LANDMARKER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(FACE_LANDMARKER_MODEL_URL, FACE_LANDMARKER_MODEL_PATH)

    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(FACE_LANDMARKER_MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=5,
        min_face_detection_confidence=0.5,
    )
    _FACE_LANDMARKER = vision.FaceLandmarker.create_from_options(options)
    return _FACE_LANDMARKER


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


def _synthetic_point(px: float, py: float, width: int, height: int) -> Dict:
    px = max(0.0, min(float(width - 1), px))
    py = max(0.0, min(float(height - 1), py))
    return {
        "index": None,
        "pixel": {"x": px, "y": py},
        "normalized": {"x": px / width, "y": py / height, "z": 0.0},
    }


def _skin_mask(image_bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    _, cr, cb = cv2.split(ycrcb)
    skin = (cr >= 130) & (cr <= 180) & (cb >= 70) & (cb <= 140)
    not_background = np.any(image_bgr < 245, axis=2)
    return skin & not_background


def _estimate_sa_from_skin(
    skin: np.ndarray,
    zy_point: Dict,
    ex_point: Dict | None,
    side: str,
    face_width: float,
    width: int,
    height: int,
) -> Dict | None:
    zy_x = float(zy_point["pixel"]["x"])
    zy_y = float(zy_point["pixel"]["y"])
    ex_y = float(ex_point["pixel"]["y"]) if ex_point else zy_y - (height * 0.10)
    target_y = ex_y + ((zy_y - ex_y) * 0.18)
    y_start = max(0, int(target_y - (height * 0.035)))
    y_end = min(height - 1, int(target_y + (height * 0.07)))
    inner_gap = max(4, int(face_width * 0.025))
    outer_width = max(18, int(face_width * 0.32))

    if side == "R":
        x_start = max(0, int(zy_x - outer_width))
        x_end = max(0, int(zy_x - inner_gap))
    else:
        x_start = min(width - 1, int(zy_x + inner_gap))
        x_end = min(width - 1, int(zy_x + outer_width))

    if x_end <= x_start or y_end <= y_start:
        return None

    min_pixels = max(3, int((x_end - x_start) * 0.08))
    best: tuple[float, int, np.ndarray] | None = None
    for y in range(y_start, y_end + 1):
        xs = np.where(skin[y, x_start : x_end + 1])[0]
        if len(xs) < min_pixels:
            continue

        distance_penalty = abs(float(y) - target_y) / max(1.0, float(y_end - y_start))
        score = float(len(xs)) * (1.0 - (distance_penalty * 0.65))
        if best is None or score > best[0]:
            best = (score, y, xs + x_start)

    if best is None:
        return None

    _, y, absolute_xs = best
    px = float(np.percentile(absolute_xs, 8 if side == "R" else 92))
    return _synthetic_point(px, float(y), width, height)


def _nearest_by_x(points: Dict[str, Dict], names: tuple[str, ...], target_x: float) -> Dict | None:
    candidates = [points[name] for name in names if name in points]
    if not candidates:
        return None
    return min(candidates, key=lambda point: abs(float(point["pixel"]["x"]) - target_x))


def _add_front_ear_points(points: Dict[str, Dict], image_bgr: np.ndarray, width: int, height: int) -> None:
    zy_r = points.get("Zy_R")
    zy_l = points.get("Zy_L")
    ft_r = points.get("Ft_R")
    ft_l = points.get("Ft_L")
    if not zy_r or not zy_l:
        return

    face_width = abs(zy_l["pixel"]["x"] - zy_r["pixel"]["x"])
    outward = max(width * 0.045, face_width * 0.18)
    skin = _skin_mask(image_bgr)
    ex_r = _nearest_by_x(points, ("Ex_R", "Ex_L"), float(zy_r["pixel"]["x"]))
    ex_l = _nearest_by_x(points, ("Ex_R", "Ex_L"), float(zy_l["pixel"]["x"]))

    if "Sa_R" not in points:
        top_ref = ft_r or ft_l or zy_r
        y = (top_ref["pixel"]["y"] * 0.48) + (zy_r["pixel"]["y"] * 0.52)
        points["Sa_R"] = (
            _estimate_sa_from_skin(skin, zy_r, ex_r, "R", face_width, width, height)
            or _synthetic_point(zy_r["pixel"]["x"] - outward, y, width, height)
        )

    if "Sa_L" not in points:
        top_ref = ft_l or ft_r or zy_l
        y = (top_ref["pixel"]["y"] * 0.48) + (zy_l["pixel"]["y"] * 0.52)
        points["Sa_L"] = (
            _estimate_sa_from_skin(skin, zy_l, ex_l, "L", face_width, width, height)
            or _synthetic_point(zy_l["pixel"]["x"] + outward, y, width, height)
        )


def _subject_mask(image_bgr: np.ndarray) -> np.ndarray:
    not_background = np.any(image_bgr < 245, axis=2).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(not_background, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask.astype(bool)

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def _estimate_profile_orientation(mask: np.ndarray) -> bool:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return True
    top = int(np.min(ys))
    bottom = int(np.max(ys))
    upper_cutoff = top + int((bottom - top) * 0.78)
    upper_xs = xs[ys <= upper_cutoff]
    if len(upper_xs) == 0:
        upper_xs = xs
    bbox_left = int(np.min(xs))
    bbox_right = int(np.max(xs))
    center_x = (bbox_left + bbox_right) / 2.0
    return float(np.mean(upper_xs)) < center_x


def _anterior_score(x: float, faces_left: bool) -> float:
    return -x if faces_left else x


def _find_edge_points(mask: np.ndarray, faces_left: bool) -> tuple[Dict[int, int], Dict[int, int], tuple[int, int, int, int]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return {}, {}, (0, 0, 0, 0)

    min_x = int(np.min(xs))
    max_x = int(np.max(xs))
    min_y = int(np.min(ys))
    max_y = int(np.max(ys))
    front_edge: Dict[int, int] = {}
    back_edge: Dict[int, int] = {}

    for y in range(min_y, max_y + 1):
        row = np.where(mask[y])[0]
        if len(row) == 0:
            continue
        front_edge[y] = int(row[0] if faces_left else row[-1])
        back_edge[y] = int(row[-1] if faces_left else row[0])

    return front_edge, back_edge, (min_x, min_y, max_x, max_y)


def _is_strict_profile_image(image_bgr: np.ndarray) -> bool:
    mask = _subject_mask(image_bgr)
    faces_left = _estimate_profile_orientation(mask)
    front_edge, _, (min_x, min_y, max_x, max_y) = _find_edge_points(mask, faces_left)
    if not front_edge:
        return False

    box_w = max_x - min_x
    box_h = max_y - min_y
    if box_h <= 0 or box_w <= 0:
        return False

    width_ratio = box_w / box_h
    forehead_y = min_y + int(box_h * 0.22)
    nose_y = min_y + int(box_h * 0.44)
    forehead_x = front_edge.get(forehead_y)
    nose_x = front_edge.get(nose_y)
    if forehead_x is None or nose_x is None:
        return False

    nasal_projection = abs(nose_x - forehead_x)
    projection_ratio = nasal_projection / box_h
    return width_ratio >= 0.78 and projection_ratio >= 0.02


def _pick_row(
    edge: Dict[int, int],
    y_start: int,
    y_end: int,
    scorer,
) -> Optional[tuple[int, int]]:
    candidates = [(y, x) for y, x in edge.items() if y_start <= y <= y_end]
    if not candidates:
        return None
    return max(candidates, key=lambda item: scorer(item[0], item[1]))


def _estimate_side_profile_points(image_bgr: np.ndarray, width: int, height: int) -> Dict[str, Dict]:
    mask = _subject_mask(image_bgr)
    front_edge, back_edge, (min_x, min_y, max_x, max_y) = _find_edge_points(mask, _estimate_profile_orientation(mask))
    if not front_edge:
        return {}

    faces_left = _estimate_profile_orientation(mask)
    box_w = max(1, max_x - min_x)
    box_h = max(1, max_y - min_y)

    def anterior(y: int, x: int) -> float:
        return _anterior_score(float(x), faces_left)

    def anterior_point(y: int, x: int) -> Dict:
        return _synthetic_point(float(x), float(y), width, height)

    prn = _pick_row(
        front_edge,
        min_y + int(box_h * 0.24),
        min_y + int(box_h * 0.52),
        lambda y, x: anterior(y, x),
    )
    if prn is None:
        return {}

    prn_y, prn_x = prn

    n = _pick_row(
        front_edge,
        min_y + int(box_h * 0.16),
        max(min_y + int(box_h * 0.20), prn_y - int(box_h * 0.06)),
        lambda y, x: -anterior(y, x),
    )
    sn = _pick_row(
        front_edge,
        prn_y + int(box_h * 0.03),
        min(max_y, prn_y + int(box_h * 0.17)),
        lambda y, x: -anterior(y, x),
    )
    pg = _pick_row(
        front_edge,
        min_y + int(box_h * 0.64),
        min_y + int(box_h * 0.90),
        lambda y, x: anterior(y, x),
    )

    if sn is None:
        sn = (min(max_y, prn_y + int(box_h * 0.12)), prn_x)
    if n is None:
        n = (max(min_y, prn_y - int(box_h * 0.16)), prn_x)
    if pg is None:
        pg = (min_y + int(box_h * 0.78), front_edge.get(min_y + int(box_h * 0.78), prn_x))

    pg_y, pg_x = pg
    chin_band = []
    tol = max(12, int(box_w * 0.06))
    for y, x in front_edge.items():
        if y < pg_y:
            continue
        if abs(x - pg_x) <= tol:
            chin_band.append((y, x))
    me_y, me_x = max(chin_band, key=lambda item: item[0]) if chin_band else pg

    ear_center_y = int(n[0] + ((sn[0] - n[0]) * 0.55))
    front_at_ear = front_edge.get(ear_center_y, prn_x)
    back_at_ear = back_edge.get(ear_center_y, max_x if faces_left else min_x)
    ear_span = abs(back_at_ear - front_at_ear)
    ear_w = max(18.0, ear_span * 0.18)
    ear_h = max(26.0, box_h * 0.18)
    if faces_left:
        ear_center_x = front_at_ear + (ear_span * 0.80)
        pra_x = ear_center_x - (ear_w * 0.55)
        pa_x = ear_center_x + (ear_w * 0.55)
    else:
        ear_center_x = back_at_ear + ((front_at_ear - back_at_ear) * 0.20)
        pra_x = ear_center_x + (ear_w * 0.55)
        pa_x = ear_center_x - (ear_w * 0.55)

    points = {
        "Prn": anterior_point(prn_y, prn_x),
        "N": anterior_point(n[0], n[1]),
        "Sn": anterior_point(sn[0], sn[1]),
        "Pg": anterior_point(pg_y, pg_x),
        "Me": anterior_point(me_y, me_x),
        "Sa_R": _synthetic_point(ear_center_x, ear_center_y - (ear_h * 0.52), width, height),
        "Sba_R": _synthetic_point(ear_center_x, ear_center_y + (ear_h * 0.52), width, height),
        "Pra_R": _synthetic_point(pra_x, ear_center_y, width, height),
        "Pa_R": _synthetic_point(pa_x, ear_center_y, width, height),
        "T_R": _synthetic_point(pra_x, ear_center_y, width, height),
    }

    return points


def analyze_images(
    front_bytes: bytes,
    side_bytes: bytes,
    tr_x: float | None = None,
    tr_y: float | None = None,
    gender: str | None = None,
) -> AnalyzeResponse:
    front_image, front_w, front_h = read_image(front_bytes)
    side_image, side_w, side_h = read_image(side_bytes)
    side_profile_ok = _is_strict_profile_image(side_image)

    front_faces, front_count = _extract_landmarks(front_image)
    side_faces, side_count = _extract_landmarks(side_image) if side_profile_ok else ([], 0)

    if not front_faces:
        raise ValueError("No face detected in front image")
    side_missing = False
    side_fallback_used = False
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
    _add_front_ear_points(front_points, front_image, front_w, front_h)
    side_points = (
        _points_from_map(side_selection.landmarks, mapping, side_w, side_h)
        if side_selection is not None
        else {}
    )
    if side_missing and side_profile_ok:
        side_points = _estimate_side_profile_points(side_image, side_w, side_h)
        side_fallback_used = bool(side_points)
        side_missing = not side_fallback_used
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
    if not side_profile_ok:
        warnings.append("Side image is not a true 90-degree profile; side measurements are unavailable.")
    elif side_missing:
        warnings.append("No face detected in side image; side measurements are unavailable.")
    elif side_fallback_used:
        warnings.append("Side landmarks estimated using profile fallback.")
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
