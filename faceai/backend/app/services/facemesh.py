from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

from app.models.schemas import AnalyzeResponse, LandmarkOut, MeasurementOut, RatioOut
from app.services.measurements import compute_measurements, compute_ratios
from app.services.overlay import draw_landmarks, draw_all_landmarks
from app.utils.image_io import read_image, to_base64_png
from app.utils.landmarks_map import load_landmark_map

try:
    from app.services.hairline import estimate_trichion

    _HAIRLINE_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # noqa: BLE001
    estimate_trichion = None
    _HAIRLINE_IMPORT_ERROR = exc

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


@dataclass
class LandmarkPoint:
    x: float
    y: float
    z: float


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


def _detect_landmarks_with_face_mesh(rgb: np.ndarray) -> Tuple[List, int]:
    if hasattr(mp, "solutions"):
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.25,
        ) as face_mesh:
            results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return [], 0

        faces = [face.landmark for face in results.multi_face_landmarks]
        return faces, len(faces[0])

    return [], 0


def _detect_landmarks_with_face_landmarker(rgb: np.ndarray) -> Tuple[List, int]:
    landmarker = _get_face_landmarker()
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = landmarker.detect(image)
    if not results.face_landmarks:
        return [], 0

    return results.face_landmarks, len(results.face_landmarks[0])


def _detect_landmarks_rgb(rgb: np.ndarray) -> Tuple[List, int]:
    detectors = (
        _detect_landmarks_with_face_mesh,
        _detect_landmarks_with_face_landmarker,
    )
    for detector in detectors:
        try:
            faces, count = detector(rgb)
        except Exception:  # noqa: BLE001
            continue
        if faces:
            return faces, count

    return [], 0


def _enhance_for_face_detection(image_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = cv2.merge((clahe.apply(lightness), a_channel, b_channel))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def _pad_to_square(image_bgr: np.ndarray) -> tuple[np.ndarray, int, int]:
    height, width = image_bgr.shape[:2]
    size = max(height, width)
    pad_x = (size - width) // 2
    pad_y = (size - height) // 2
    padded = cv2.copyMakeBorder(
        image_bgr,
        pad_y,
        size - height - pad_y,
        pad_x,
        size - width - pad_x,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    return padded, pad_x, pad_y


def _restore_landmarks(
    faces: List,
    variant_w: int,
    variant_h: int,
    original_w: int,
    original_h: int,
    pad_x: int = 0,
    pad_y: int = 0,
    flipped: bool = False,
) -> List[List[LandmarkPoint]]:
    restored_faces: List[List[LandmarkPoint]] = []
    for face in faces:
        restored: List[LandmarkPoint] = []
        for lm in face:
            x_abs = float(lm.x) * variant_w
            if flipped:
                x_abs = variant_w - x_abs
            x = (x_abs - pad_x) / original_w
            y = (float(lm.y) * variant_h - pad_y) / original_h
            restored.append(LandmarkPoint(x=x, y=y, z=float(lm.z)))
        restored_faces.append(restored)
    return restored_faces


def _extract_landmarks(image_bgr: np.ndarray) -> Tuple[List, int]:
    height, width = image_bgr.shape[:2]
    enhanced = _enhance_for_face_detection(image_bgr)
    padded, pad_x, pad_y = _pad_to_square(image_bgr)
    padded_enhanced = _enhance_for_face_detection(padded)
    variants = (
        (image_bgr, 0, 0, False),
        (cv2.flip(image_bgr, 1), 0, 0, True),
        (enhanced, 0, 0, False),
        (cv2.flip(enhanced, 1), 0, 0, True),
        (padded, pad_x, pad_y, False),
        (cv2.flip(padded, 1), pad_x, pad_y, True),
        (padded_enhanced, pad_x, pad_y, False),
        (cv2.flip(padded_enhanced, 1), pad_x, pad_y, True),
    )

    for variant, variant_pad_x, variant_pad_y, flipped in variants:
        rgb = cv2.cvtColor(variant, cv2.COLOR_BGR2RGB)
        faces, count = _detect_landmarks_rgb(rgb)
        if faces:
            variant_h, variant_w = variant.shape[:2]
            return _restore_landmarks(
                faces,
                variant_w,
                variant_h,
                width,
                height,
                pad_x=variant_pad_x,
                pad_y=variant_pad_y,
                flipped=flipped,
            ), count

    return [], 0


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
        min_face_detection_confidence=0.25,
        min_face_presence_confidence=0.25,
        min_tracking_confidence=0.25,
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

    if mapping.get("Prn") is None and "Prn" in mapping and len(landmarks) > 4:
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
    height, width = image_bgr.shape[:2]
    border = max(8, int(min(height, width) * 0.04))
    border_pixels = np.concatenate(
        [
            image_bgr[:border].reshape(-1, 3),
            image_bgr[-border:].reshape(-1, 3),
            image_bgr[:, :border].reshape(-1, 3),
            image_bgr[:, -border:].reshape(-1, 3),
        ]
    )
    border_hsv = cv2.cvtColor(border_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    background_candidates = border_pixels[(border_hsv[:, 2] > 120) & (border_hsv[:, 1] < 80)]
    background_pixels = background_candidates if len(background_candidates) > 100 else border_pixels
    background_bgr = np.median(background_pixels, axis=0).astype(np.uint8)

    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    background_lab = cv2.cvtColor(np.array([[background_bgr]], dtype=np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    color_distance = np.linalg.norm(image_lab - background_lab, axis=2)
    not_background = (color_distance > 18).astype(np.uint8)

    if float(np.mean(not_background)) > 0.85:
        not_background = np.any(image_bgr < 245, axis=2).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(not_background, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask.astype(bool)

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def _profile_projection_scores(mask: np.ndarray) -> tuple[float, float]:
    ys, _ = np.where(mask)
    if len(ys) == 0:
        return 0.0, 0.0

    top = int(np.min(ys))
    bottom = int(np.max(ys))
    box_h = bottom - top
    if box_h <= 0:
        return 0.0, 0.0

    forehead_y = top + int(box_h * 0.22)
    nose_y = top + int(box_h * 0.44)
    forehead_row = np.where(mask[forehead_y])[0]
    nose_row = np.where(mask[nose_y])[0]
    if len(forehead_row) == 0 or len(nose_row) == 0:
        return 0.0, 0.0

    left_projection = max(0.0, float(forehead_row[0] - nose_row[0]))
    right_projection = max(0.0, float(nose_row[-1] - forehead_row[-1]))
    return left_projection, right_projection


def _estimate_profile_orientation(mask: np.ndarray) -> bool:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return True
    left_projection, right_projection = _profile_projection_scores(mask)
    if max(left_projection, right_projection) >= 8 and abs(left_projection - right_projection) >= 4:
        return left_projection > right_projection

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


def _visible_profile_suffix(image_bgr: np.ndarray) -> str:
    mask = _subject_mask(image_bgr)
    faces_left = _estimate_profile_orientation(mask)
    return "_R" if faces_left else "_L"


def _filter_profile_visible_points(points: Dict[str, Dict], image_bgr: np.ndarray) -> Dict[str, Dict]:
    visible_suffix = _visible_profile_suffix(image_bgr)
    hidden_suffix = "_L" if visible_suffix == "_R" else "_R"
    filtered: Dict[str, Dict] = {}
    for label, point in points.items():
        if label.endswith(hidden_suffix):
            continue
        filtered[label] = point
    return filtered


def _pick_profile_row(
    edge: Dict[int, int],
    y_start: int,
    y_end: int,
    scorer,
) -> Optional[tuple[int, int]]:
    candidates = [(y, x) for y, x in edge.items() if y_start <= y <= y_end]
    if not candidates:
        return None
    return max(candidates, key=lambda item: scorer(item[0], item[1]))


def _profile_edge_point(edge: Dict[int, int], target_y: float, fallback_x: float, width: int, height: int) -> Dict:
    if edge:
        nearest_y = min(edge.keys(), key=lambda row_y: abs(row_y - target_y))
        return _synthetic_point(float(edge[nearest_y]), float(nearest_y), width, height)
    return _synthetic_point(fallback_x, target_y, width, height)


def _estimate_side_anthropometric_points(image_bgr: np.ndarray, mesh_points: Dict[str, Dict]) -> Dict[str, Dict]:
    height, width = image_bgr.shape[:2]
    mask = _subject_mask(image_bgr)
    faces_left = _estimate_profile_orientation(mask)
    front_edge, back_edge, (min_x, min_y, max_x, max_y) = _find_edge_points(mask, faces_left)
    if not front_edge:
        return {}

    box_w = max(1, max_x - min_x)
    box_h = max(1, max_y - min_y)
    suffix = "_R" if faces_left else "_L"
    posterior = 1.0 if faces_left else -1.0

    def anterior_score(y: int, x: int) -> float:
        return _anterior_score(float(x), faces_left)

    def point(y: float, x: float) -> Dict:
        return _synthetic_point(float(x), float(y), width, height)

    prn = _pick_profile_row(
        front_edge,
        min_y + int(box_h * 0.30),
        min_y + int(box_h * 0.58),
        lambda y, x: anterior_score(y, x),
    )
    if prn is None:
        return {}
    prn_y, prn_x = prn

    n = _pick_profile_row(
        front_edge,
        min_y + int(box_h * 0.18),
        max(min_y + int(box_h * 0.26), prn_y - int(box_h * 0.08)),
        lambda y, x: -anterior_score(y, x),
    ) or (max(min_y, prn_y - int(box_h * 0.18)), prn_x)
    g = _pick_profile_row(
        front_edge,
        min_y + int(box_h * 0.18),
        max(min_y + int(box_h * 0.32), n[0] + int(box_h * 0.06)),
        lambda y, x: anterior_score(y, x) - abs(y - (n[0] + int(box_h * 0.05))) * 0.35,
    ) or (max(min_y, n[0] - int(box_h * 0.05)), n[1])
    sn = _pick_profile_row(
        front_edge,
        prn_y + int(box_h * 0.02),
        min(max_y, prn_y + int(box_h * 0.16)),
        lambda y, x: -anterior_score(y, x),
    ) or (min(max_y, prn_y + int(box_h * 0.10)), prn_x)
    pg_candidate = _pick_profile_row(
        front_edge,
        min_y + int(box_h * 0.66),
        min_y + int(box_h * 0.91),
        lambda y, x: anterior_score(y, x),
    ) or (min_y + int(box_h * 0.78), front_edge.get(min_y + int(box_h * 0.78), prn_x))

    pg_y, pg_x = pg_candidate
    chin_band = [(y, x) for y, x in front_edge.items() if y >= pg_y and abs(x - pg_x) <= max(12, int(box_w * 0.08))]
    me_y, me_x = max(chin_band, key=lambda item: item[0]) if chin_band else pg_candidate
    target_pg_y = int(me_y - box_h * 0.09)
    jaw_x_at_pg = front_edge.get(target_pg_y, pg_x)
    pg_y = target_pg_y
    pg_x = float(jaw_x_at_pg) * 0.45 + float(me_x) * 0.55

    mouth_y = sn[0] + ((pg_y - sn[0]) * 0.34)
    lip = _pick_profile_row(
        front_edge,
        int(sn[0] + ((pg_y - sn[0]) * 0.12)),
        int(sn[0] + ((pg_y - sn[0]) * 0.54)),
        lambda y, x: anterior_score(y, x) - abs(y - mouth_y) * 0.40,
    ) or (int(mouth_y), front_edge.get(int(mouth_y), sn[1]))
    lip_y, lip_x = lip

    tr = _profile_edge_point(front_edge, min_y + box_h * 0.04, prn_x, width, height)
    ft = _profile_edge_point(front_edge, min_y + box_h * 0.18, prn_x, width, height)
    ex = _profile_edge_point(front_edge, n[0] + box_h * 0.07, prn_x, width, height)
    zy = point(n[0] + box_h * 0.18, prn_x + posterior * box_w * 0.18)
    go = point(pg_y + ((me_y - pg_y) * 0.20), prn_x + posterior * box_w * 0.30)

    ear_y = min_y + (box_h * 0.47)
    ear_center_x = (min_x + (box_w * 0.36)) if not faces_left else (max_x - (box_w * 0.36))
    ear_h = max(26.0, box_h * 0.18)
    ear_w = max(18.0, box_w * 0.075)

    estimated = {
        "Tr_R": tr,
        "Tr_L": tr,
        f"Ft{suffix}": ft,
        "Ft_S": ft,
        "G": point(g[0], g[1]),
        "N": point(n[0], n[1]),
        f"Ex{suffix}": ex,
        f"Zy{suffix}": zy,
        "Prn": point(prn_y, prn_x),
        f"Al{suffix}": point(sn[0] - box_h * 0.025, prn_x + posterior * box_w * 0.055),
        "Sn": point(sn[0], sn[1]),
        "Ls": point(lip_y - box_h * 0.025, lip_x),
        "Sto": point(lip_y, lip_x),
        "Li": point(lip_y + box_h * 0.035, lip_x + posterior * box_w * 0.015),
        "Sl_R": point(pg_y - box_h * 0.055, pg_x + posterior * box_w * 0.030),
        "Pg": point(pg_y, pg_x),
        "Me": point(me_y, me_x),
        f"Go{suffix}": go,
        f"Sa{suffix}": point(ear_y - ear_h * 0.52, ear_center_x),
        f"Sba{suffix}": point(ear_y + ear_h * 0.52, ear_center_x),
        f"Pra{suffix}": point(ear_y, ear_center_x - posterior * ear_w * 0.55),
        f"Pa{suffix}": point(ear_y, ear_center_x + posterior * ear_w * 0.55),
        f"T{suffix}": point(ear_y, ear_center_x - posterior * ear_w * 0.30),
    }

    for label in ("Ps_L", "Pi_L", "Ir_L", "Cph_L", "Ch_L"):
        if label.endswith(suffix) and label in mesh_points:
            estimated[label] = mesh_points[label]

    points = _filter_profile_visible_points(mesh_points, image_bgr)
    for label, value in estimated.items():
        if label.startswith(("Sa", "Sba", "Pra", "Pa", "T", "Go", "Tr")) or label in {"Pg", "Me"}:
            points[label] = value
        else:
            points.setdefault(label, value)

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
    mapped_side_points = (
        _points_from_map(side_selection.landmarks, mapping, side_w, side_h)
        if side_selection is not None
        else {}
    )
    side_points = mapped_side_points.copy()
    side_profile_estimated = False
    if side_profile_ok:
        estimated_side_points = _estimate_side_anthropometric_points(side_image, side_points)
        if estimated_side_points:
            side_points = estimated_side_points
            side_profile_estimated = True
        elif side_points:
            side_points = _filter_profile_visible_points(side_points, side_image)
    if side_missing and side_points:
        side_missing = False
    tr_method = "none"
    trichion = None
    tr_debug = {}
    if tr_x is not None and tr_y is not None:
        trichion = _tr_from_normalized(tr_x, tr_y, front_w, front_h)
        tr_method = "manual"
    else:
        if estimate_trichion is not None:
            trichion, tr_debug, tr_method = estimate_trichion(
                front_image, front_points, landmarks=front_selection.landmarks, debug=True
            )
        else:
            tr_method = "none"
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
        warnings.append("Side image did not pass the strict 90-degree profile check; using detected landmarks anyway.")
    if side_missing:
        warnings.append("No face detected in side image; side measurements are unavailable.")
    elif side_profile_estimated and side_selection is None:
        warnings.append("Side face mesh was not detected; side profile landmarks were estimated from the profile outline.")
    if not trichion_available:
        warnings.append("Trichion (Tr) unavailable; hairline segmentation did not return a result.")
        if _HAIRLINE_IMPORT_ERROR is not None:
            warnings.append("Hairline model dependencies are unavailable, so Tr-based measurements may be null.")
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
