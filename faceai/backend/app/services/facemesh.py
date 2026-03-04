from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from app.models.schemas import AnalyzeResponse, EyeAnalysisOut, IdealFaceOut, LandmarkOut, MeasurementOut, RatioOut
from app.services.hairline import estimate_trichion
from app.services.measurements import (
    compute_cheek_profile,
    compute_ear_profile,
    compute_eye_profile,
    compute_eyebrow_profile,
    compute_forehead_profile,
    compute_ideal_face_profile,
    compute_jaw_profile,
    compute_measurements,
    compute_mouth_profile,
    compute_nose_profile,
    compute_ratios,
)
from app.services.overlay import draw_landmarks, draw_all_landmarks
from app.utils.image_io import read_image, to_base64_png
from app.utils.landmarks_map import load_landmark_map


@dataclass
class FaceSelection:
    landmarks: List
    bbox: Tuple[float, float, float, float]
    score: float


@dataclass
class NormalizedLandmark:
    x: float
    y: float
    z: float


@dataclass
class Keypoint2D:
    x: float
    y: float


def _bbox_from_landmarks(landmarks: List) -> Tuple[float, float, float, float]:
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    return min(xs), min(ys), max(xs), max(ys)


def _face_landmarks(face: object) -> List:
    if hasattr(face, "landmark"):
        return getattr(face, "landmark")
    return face  # already a landmark list


def _select_best_face(landmark_lists: List, image_w: int, image_h: int) -> Optional[FaceSelection]:
    if not landmark_lists:
        return None

    image_center = (0.5, 0.5)
    best: Optional[FaceSelection] = None

    for face in landmark_lists:
        face_landmarks = _face_landmarks(face)
        bbox = _bbox_from_landmarks(face_landmarks)
        min_x, min_y, max_x, max_y = bbox
        area = max(0.0, (max_x - min_x)) * max(0.0, (max_y - min_y))
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        dist = ((center_x - image_center[0]) ** 2 + (center_y - image_center[1]) ** 2) ** 0.5
        score = area - (dist * area * 0.5)
        selection = FaceSelection(face_landmarks, bbox, score)
        if best is None or selection.score > best.score:
            best = selection

    return best


def _run_facemesh(image_bgr: np.ndarray) -> Tuple[List, int]:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if not hasattr(mp, "solutions"):
        raise RuntimeError(
            "MediaPipe 'solutions' module not available. "
            "Pin mediapipe to a version that includes solutions (e.g. 0.10.11) "
            "and reinstall backend dependencies."
        )
    configs = [
        {"refine_landmarks": False, "min_detection_confidence": 0.50},
        {"refine_landmarks": True, "min_detection_confidence": 0.50},
        {"refine_landmarks": True, "min_detection_confidence": 0.35},
        {"refine_landmarks": False, "min_detection_confidence": 0.30},
    ]

    for config in configs:
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=5,
            refine_landmarks=config["refine_landmarks"],
            min_detection_confidence=config["min_detection_confidence"],
        ) as face_mesh:
            results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            return results.multi_face_landmarks, len(results.multi_face_landmarks[0].landmark)

    return [], 0


def _enhance_profile_image(image_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    merged = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _inverse_transform_xy(x: float, y: float, mode: str) -> Tuple[float, float]:
    if mode == "original":
        return x, y
    if mode == "flip_h":
        return 1.0 - x, y
    if mode == "rot90_cw":
        return y, 1.0 - x
    if mode == "rot90_ccw":
        return 1.0 - y, x
    if mode == "rot180":
        return 1.0 - x, 1.0 - y
    return x, y


def _transform_landmarks_to_original(landmark_lists: List, mode: str) -> List[List[NormalizedLandmark]]:
    transformed: List[List[NormalizedLandmark]] = []
    for face in landmark_lists:
        face_landmarks = _face_landmarks(face)
        remapped: List[NormalizedLandmark] = []
        for lm in face_landmarks:
            nx, ny = _inverse_transform_xy(float(lm.x), float(lm.y), mode)
            nx = min(max(nx, 0.0), 1.0)
            ny = min(max(ny, 0.0), 1.0)
            remapped.append(NormalizedLandmark(x=nx, y=ny, z=float(lm.z)))
        transformed.append(remapped)
    return transformed


def _extract_landmarks_with_fallback(image_bgr: np.ndarray, allow_orientation_retry: bool) -> Tuple[List, int, str]:
    faces, count = _run_facemesh(image_bgr)
    if faces:
        return _transform_landmarks_to_original(faces, "original"), count, "original"

    retry_inputs: List[Tuple[str, np.ndarray]] = [("original_enhanced", _enhance_profile_image(image_bgr))]
    if allow_orientation_retry:
        retry_inputs.extend(
            [
                ("rot90_cw", cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)),
                ("rot90_ccw", cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)),
                ("rot180", cv2.rotate(image_bgr, cv2.ROTATE_180)),
                ("flip_h", cv2.flip(image_bgr, 1)),
                ("flip_h_enhanced", _enhance_profile_image(cv2.flip(image_bgr, 1))),
            ]
        )

    for mode, candidate in retry_inputs:
        retry_faces, retry_count = _run_facemesh(candidate)
        if retry_faces:
            if mode == "original_enhanced":
                mapped_mode = "original"
            elif mode == "flip_h_enhanced":
                mapped_mode = "flip_h"
            else:
                mapped_mode = mode
            return _transform_landmarks_to_original(retry_faces, mapped_mode), retry_count, mode

    return [], 0, "original"


def _extract_face_detection_keypoints(image_bgr: np.ndarray) -> Dict[str, Keypoint2D]:
    if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "face_detection"):
        return {}
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.35,
    ) as detector:
        results = detector.process(rgb)

    if not results.detections:
        return {}

    # Pick largest detection.
    best = None
    best_area = -1.0
    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        area = max(0.0, float(bbox.width)) * max(0.0, float(bbox.height))
        if area > best_area:
            best = detection
            best_area = area
    if best is None:
        return {}

    face_keypoint = mp.solutions.face_detection.FaceKeyPoint
    keypoint_defs = {
        "right_ear_tragion": face_keypoint.RIGHT_EAR_TRAGION,
        "left_ear_tragion": face_keypoint.LEFT_EAR_TRAGION,
        "nose_tip": face_keypoint.NOSE_TIP,
    }

    out: Dict[str, Keypoint2D] = {}
    for name, enum_value in keypoint_defs.items():
        point = mp.solutions.face_detection.get_key_point(best, enum_value)
        if point is None:
            continue
        out[name] = Keypoint2D(x=float(point.x), y=float(point.y))
    return out


def _build_point_from_normalized(nx: float, ny: float, side_w: int, side_h: int) -> Dict:
    nx = min(max(nx, 0.0), 1.0)
    ny = min(max(ny, 0.0), 1.0)
    return {
        "index": None,
        "pixel": {"x": float(nx * side_w), "y": float(ny * side_h)},
        "normalized": {"x": float(nx), "y": float(ny), "z": 0.0},
    }


def _add_ear_points_for_side(
    side_points: Dict[str, Dict],
    side_image: np.ndarray,
    side_bbox: Optional[Tuple[float, float, float, float]],
) -> bool:
    keypoints = _extract_face_detection_keypoints(side_image)
    if not keypoints:
        return False

    tragion = keypoints.get("right_ear_tragion") or keypoints.get("left_ear_tragion")
    if tragion is None:
        return False

    nose = keypoints.get("nose_tip")
    if nose is None and "Prn" in side_points:
        nose = Keypoint2D(
            x=float(side_points["Prn"]["normalized"]["x"]),
            y=float(side_points["Prn"]["normalized"]["y"]),
        )
    if nose is None:
        return False

    min_x, min_y, max_x, max_y = side_bbox if side_bbox is not None else (0.2, 0.2, 0.8, 0.8)
    face_scale = max(max_x - min_x, max_y - min_y)
    dx_sign = 1.0 if nose.x >= tragion.x else -1.0

    # Proxy ear landmarks derived from tragion and face scale.
    pra = Keypoint2D(x=tragion.x + 0.020 * face_scale * dx_sign, y=tragion.y)
    pa = Keypoint2D(x=tragion.x - 0.060 * face_scale * dx_sign, y=tragion.y + 0.005 * face_scale)
    sa = Keypoint2D(x=tragion.x - 0.010 * face_scale * dx_sign, y=tragion.y - 0.120 * face_scale)
    sba = Keypoint2D(x=tragion.x - 0.005 * face_scale * dx_sign, y=tragion.y + 0.140 * face_scale)

    side_h, side_w = side_image.shape[:2]
    # Fill right labels because current measurement catalog uses right ear IDs.
    side_points["Pra_R"] = _build_point_from_normalized(pra.x, pra.y, side_w, side_h)
    side_points["Pa_R"] = _build_point_from_normalized(pa.x, pa.y, side_w, side_h)
    side_points["Sa_R"] = _build_point_from_normalized(sa.x, sa.y, side_w, side_h)
    side_points["Sba_R"] = _build_point_from_normalized(sba.x, sba.y, side_w, side_h)

    # Mirror into left labels only when absent, to keep outputs complete.
    for a, b in [("Pra_L", "Pra_R"), ("Pa_L", "Pa_R"), ("Sa_L", "Sa_R"), ("Sba_L", "Sba_R")]:
        if a not in side_points:
            side_points[a] = side_points[b]

    return True


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

    if "Prn" not in points and "Prn" in mapping and len(landmarks) > 4:
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

    # Subnasale fallback:
    # 1) prefer FaceMesh point 2 when available
    # 2) otherwise use midpoint between left/right alare points
    if "Sn" not in points and len(landmarks) > 2:
        lm = landmarks[2]
        points["Sn"] = {
            "index": 2,
            "pixel": {"x": float(lm.x * width), "y": float(lm.y * height)},
            "normalized": {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)},
        }
    if "Sn" not in points and "Al_R" in points and "Al_L" in points:
        px = (points["Al_R"]["pixel"]["x"] + points["Al_L"]["pixel"]["x"]) / 2.0
        py = (points["Al_R"]["pixel"]["y"] + points["Al_L"]["pixel"]["y"]) / 2.0
        nx = (points["Al_R"]["normalized"]["x"] + points["Al_L"]["normalized"]["x"]) / 2.0
        ny = (points["Al_R"]["normalized"]["y"] + points["Al_L"]["normalized"]["y"]) / 2.0
        nz = (points["Al_R"]["normalized"]["z"] + points["Al_L"]["normalized"]["z"]) / 2.0
        points["Sn"] = {
            "index": None,
            "pixel": {"x": px, "y": py},
            "normalized": {"x": nx, "y": ny, "z": nz},
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

    front_faces, front_count, _front_mode = _extract_landmarks_with_fallback(front_image, allow_orientation_retry=False)
    side_faces, side_count, side_mode = _extract_landmarks_with_fallback(side_image, allow_orientation_retry=True)

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
    ear_points_added = False
    if side_selection is not None:
        ear_points_added = _add_ear_points_for_side(side_points, side_image, side_selection.bbox)
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
    ideal_face: IdealFaceOut = compute_ideal_face_profile(front_points, measurements)
    eyes: EyeAnalysisOut = compute_eye_profile(front_points, measurements, gender)
    forehead = compute_forehead_profile(front_points, measurements, gender)
    nose = compute_nose_profile(front_points, measurements, gender)
    mouth = compute_mouth_profile(front_points, measurements, gender)
    jaw = compute_jaw_profile(front_points, measurements, gender)
    cheek = compute_cheek_profile(front_points, measurements, gender)
    eyebrows = compute_eyebrow_profile(front_points, measurements, gender)
    ears = compute_ear_profile(side_points, measurements, gender) if side_points else None

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
    elif side_mode != "original":
        warnings.append(f"Side face detected after orientation fallback ({side_mode}).")
    if side_selection is not None and not ear_points_added:
        warnings.append("Ear landmarks are unavailable on side image (tragion keypoint not detected).")
    elif ear_points_added:
        warnings.append("Ear landmarks are estimated from side tragion keypoint.")
    if not trichion_available:
        warnings.append("Trichion (Tr) unavailable; hairline segmentation did not return a result.")
    elif tr_method == "fallback":
        warnings.append("Trichion (Tr) estimated with geometric fallback (no hair detected).")
    elif tr_method == "manual":
        warnings.append("Trichion (Tr) set manually.")
    if any(metric.value is None for metric in eyes.metrics[:5]):
        warnings.append("Eye analysis is partial; one or more eye landmarks are missing.")
    if any(metric.value is None for metric in eyes.metrics[5:8]):
        warnings.append("Some advanced eye metrics are unavailable (requires crease/iris/orbital landmarks).")

    return AnalyzeResponse(
        ok=True,
        all_landmarks_count=front_count,
        gender=gender,
        mandatory_landmarks=mandatory_landmarks,
        measurements=measurements,
        ratios=ratios,
        ideal_face=ideal_face,
        eyes=eyes,
        forehead=forehead,
        nose=nose,
        mouth=mouth,
        jaw=jaw,
        cheek=cheek,
        eyebrows=eyebrows,
        ears=ears,
        annotated_images={
            "front": to_base64_png(annotated_front),
            "side": to_base64_png(annotated_side),
            "front_all": to_base64_png(annotated_front_all),
            "side_all": to_base64_png(annotated_side_all),
            **{key: to_base64_png(img) for key, img in tr_debug.items()},
        },
        warnings=warnings,
    )
