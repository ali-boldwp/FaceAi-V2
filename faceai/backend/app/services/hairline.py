from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn

MODEL_REPO_ZIP = "https://github.com/yakhyo/face-parsing/archive/refs/heads/main.zip"
MODEL_WEIGHTS_URL = "https://github.com/yakhyo/face-parsing/releases/download/weights/resnet34.pt"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "model_cache" / "face_parsing"
WEIGHTS_PATH = CACHE_DIR / "weights" / "resnet34.pt"
REPO_DIR = CACHE_DIR / "face-parsing-main"

HAIR_CLASS_ID = 1  # hair class (confirmed via debug)

_MODEL: Optional[nn.Module] = None
_DEVICE: Optional[torch.device] = None


def _download(url: str, dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, dest.open("wb") as handle:
        handle.write(response.read())


def _ensure_repo() -> None:
    if REPO_DIR.exists():
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = CACHE_DIR / "face-parsing.zip"
    if not zip_path.exists():
        _download(MODEL_REPO_ZIP, zip_path)

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(CACHE_DIR)


def _ensure_weights() -> None:
    if WEIGHTS_PATH.exists():
        return
    _download(MODEL_WEIGHTS_URL, WEIGHTS_PATH)


def _load_model() -> nn.Module:
    global _MODEL, _DEVICE

    if _MODEL is not None:
        return _MODEL

    _ensure_repo()
    _ensure_weights()

    sys.path.insert(0, str(REPO_DIR))
    try:
        from models.bisenet import BiSeNet  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Unable to import BiSeNet from downloaded face-parsing repo.") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiSeNet(19, "resnet34")  # type: ignore[call-arg]
    state = torch.load(str(WEIGHTS_PATH), map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    _MODEL = model
    _DEVICE = device
    return model


def _preprocess(image_bgr: np.ndarray) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(image_rgb)
    pil = pil.resize((512, 512), Image.BILINEAR)
    array = np.asarray(pil).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    array = (array - mean) / std
    array = np.transpose(array, (2, 0, 1))
    tensor = torch.from_numpy(array).unsqueeze(0)
    return tensor


def _predict_mask(image_bgr: np.ndarray) -> np.ndarray:
    model = _load_model()
    device = _DEVICE or torch.device("cpu")
    tensor = _preprocess(image_bgr).to(device)

    with torch.no_grad():
        outputs = model(tensor)

    # BiSeNet returns a tuple; first element is the main output
    if isinstance(outputs, (list, tuple)):
        outputs = outputs[0]
    parsing = outputs.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    return parsing


def _resize_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    width, height = size
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def _colorize_parsing(parsing: np.ndarray) -> np.ndarray:
    # 19 classes in CelebAMask-HQ for BiSeNet face parsing
    palette = np.array(
        [
            [0, 0, 0],
            [128, 0, 0],
            [255, 0, 0],
            [0, 85, 0],
            [170, 0, 51],
            [255, 85, 0],
            [0, 0, 85],
            [0, 119, 221],
            [85, 85, 0],
            [0, 85, 85],
            [85, 51, 0],
            [52, 86, 128],
            [0, 128, 0],
            [0, 0, 255],
            [51, 170, 221],
            [0, 255, 255],
            [85, 255, 170],
            [170, 255, 85],
            [255, 255, 0],
        ],
        dtype=np.uint8,
    )
    colored = palette[parsing % len(palette)]
    return colored


def _legend_image() -> np.ndarray:
    palette = _colorize_parsing(np.arange(19, dtype=np.uint8)).reshape(19, 1, 3)
    row_h = 26
    width = 260
    height = row_h * 19 + 10
    legend = np.zeros((height, width, 3), dtype=np.uint8)
    legend[:] = (18, 18, 18)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx in range(19):
        y = 5 + idx * row_h
        color = tuple(int(v) for v in palette[idx, 0])
        cv2.rectangle(legend, (8, y + 4), (32, y + 20), color, -1)
        cv2.putText(legend, f"Class {idx}", (42, y + 18), font, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    return legend

def estimate_trichion(
    image_bgr: np.ndarray,
    front_points: Dict[str, Dict],
    landmarks: Optional[list] = None,
    debug: bool = False,
) -> Tuple[Optional[Dict[str, Dict]], Dict[str, np.ndarray], str]:
    try:
        parsing = _predict_mask(image_bgr)
    except Exception:
        parsing = None

    height, width = image_bgr.shape[:2]
    hair_mask = None
    if parsing is not None:
        parsing = _resize_mask(parsing, (width, height))
        hair_mask = parsing == HAIR_CLASS_ID

    mid_x = None
    if "Prn" in front_points:
        mid_x = int(front_points["Prn"]["pixel"]["x"])
    elif "N" in front_points:
        mid_x = int(front_points["N"]["pixel"]["x"])

    if mid_x is None:
        mid_x = width // 2

    search_radius = max(3, int(width * 0.01))
    top_y = None
    if hair_mask is not None:
        for y in range(height):
            x_start = max(0, mid_x - search_radius)
            x_end = min(width - 1, mid_x + search_radius)
            if hair_mask[y, x_start : x_end + 1].any():
                top_y = y
                break

    method = "hair" if top_y is not None else "fallback"

    if top_y is None and landmarks:
        # Use top-most mesh point near the midline as a fallback.
        min_y = None
        for lm in landmarks:
            px = int(lm.x * width)
            if abs(px - mid_x) <= max(5, int(width * 0.03)):
                py = int(lm.y * height)
                if min_y is None or py < min_y:
                    min_y = py
        if min_y is None:
            min_y = min(int(lm.y * height) for lm in landmarks)
        top_y = min_y

    if top_y is None:
        return None, {}, "none"

    px = float(mid_x)
    py = float(top_y)
    trichion = {
        "index": None,
        "pixel": {"x": px, "y": py},
        "normalized": {"x": px / width, "y": py / height, "z": 0.0},
    }

    debug_images: Dict[str, np.ndarray] = {}
    if debug:
        mask_vis = np.zeros_like(image_bgr)
        if hair_mask is not None:
            mask_vis[hair_mask] = (0, 200, 0)
        overlay = cv2.addWeighted(image_bgr, 0.65, mask_vis, 0.35, 0)
        cv2.line(overlay, (mid_x, 0), (mid_x, height - 1), (255, 255, 0), 1)
        cv2.circle(overlay, (mid_x, top_y), 4, (0, 0, 255), -1)
        debug_images["tr_hair_mask"] = mask_vis
        debug_images["tr_overlay"] = overlay
        if parsing is not None:
            parsing_vis = _colorize_parsing(parsing)
            debug_images["tr_parsing"] = parsing_vis
            debug_images["tr_parsing_legend"] = _legend_image()

    return trichion, debug_images, method
