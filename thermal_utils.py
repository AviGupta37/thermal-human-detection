"""Thermal image helpers shared by every script: preprocessing, label I/O,
box maths and the training-free hotspot detector."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import config

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ---- preprocessing -------------------------------------------------------------

def to_gray8(frame: np.ndarray) -> np.ndarray:
    """Any camera frame (8/16-bit, 1 or 3 channel) -> single-channel uint8."""
    if frame.ndim == 3:
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame.dtype != np.uint8:
        # Radiometric 16-bit: stretch the 1st-99th percentile to 0-255 so a
        # single hot exhaust pipe does not crush everything else to black.
        lo, hi = np.percentile(frame, (1, 99))
        frame = np.clip((frame.astype(np.float32) - lo) * (255.0 / max(hi - lo, 1.0)), 0, 255)
        frame = frame.astype(np.uint8)
    return frame


_clahe = None


def preprocess(frame: np.ndarray, mode: str | None = None, invert: bool | None = None) -> np.ndarray:
    """Camera frame -> 3-channel uint8 image exactly as the model sees it."""
    mode = config.PREPROCESS if mode is None else mode
    invert = config.INVERT if invert is None else invert
    gray = to_gray8(frame)
    if invert:
        gray = 255 - gray
    if mode == "clahe":
        global _clahe
        if _clahe is None:
            _clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP,
                                     tileGridSize=(config.CLAHE_GRID, config.CLAHE_GRID))
        gray = _clahe.apply(gray)
    elif mode != "none":
        raise ValueError(f"unknown preprocess mode {mode!r}")
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ---- files and labels --------------------------------------------------------

def list_images(folder: Path) -> list[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)


def label_path_for(image: Path, labels_dir: Path) -> Path:
    return Path(labels_dir) / (Path(image).stem + ".txt")


def read_yolo_labels(path: Path, w: int, h: int) -> list[list[float]]:
    """-> [[cls, x1, y1, x2, y2], ...] in pixels. Missing file -> []."""
    boxes = []
    path = Path(path)
    if not path.exists():
        return boxes
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:5])
        boxes.append([c, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return boxes


def write_yolo_labels(path: Path, boxes: list[list[float]], w: int, h: int) -> None:
    """boxes: [[cls, x1, y1, x2, y2], ...] in pixels. An empty list writes an
    empty file, which YOLO treats as a verified 'no person here' image."""
    lines = []
    for c, x1, y1, x2, y2 in boxes:
        x1, x2 = sorted((max(0.0, min(x1, w)), max(0.0, min(x2, w))))
        y1, y2 = sorted((max(0.0, min(y1, h)), max(0.0, min(y2, h))))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        lines.append(f"{int(c)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                     f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""))


def group_of(stem: str) -> str:
    """'flight03__000120' -> 'flight03' (the clip a frame came from)."""
    return stem.split(config.GROUP_SEP)[0] if config.GROUP_SEP in stem else stem


def trained_imgsz(model, default: int) -> int:
    """Image size an Ultralytics model was trained at (fallback: default)."""
    try:
        return int(model.ckpt["train_args"]["imgsz"])
    except (AttributeError, KeyError, TypeError, ValueError):
        return default


# ---- box maths ---------------------------------------------------------------

def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (N,4) and b (M,4), xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a = a[:, None, :]
    b = b[None, :, :]
    iw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = iw * ih
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Plain NMS, returns kept indices sorted by score."""
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        ious = iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious < iou_thr]
    return np.array(keep, dtype=int)


# ---- training-free hotspot detector ----------------------------------------------

class HotspotDetector:
    """Finds warm, person-sized blobs. No training needed; used as a fallback,
    as a sanity check and to find frames the YOLO model may be missing.

    White top-hat keeps only blobs that are warmer than their surroundings
    and smaller than `person_px`, so sun-warmed ground and large hot surfaces
    are suppressed.
    """

    def __init__(self, person_px: int = 40, min_area: int = 12, max_area: int = 5000,
                 thresh: int = 0, max_aspect: float = 4.0):
        self.person_px = person_px      # rough person height in pixels at your altitude
        self.min_area = min_area
        self.max_area = max_area
        self.thresh = thresh            # 0 = Otsu (automatic)
        self.max_aspect = max_aspect
        k = max(3, int(person_px * 1.5) | 1)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    def __call__(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """-> (boxes xyxy float32 (N,4), scores (N,))"""
        gray = to_gray8(frame)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, self.kernel)
        if self.thresh > 0:
            _, mask = cv2.threshold(tophat, self.thresh, 255, cv2.THRESH_BINARY)
        else:
            t, mask = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if t < 15:  # flat scene: Otsu would just threshold noise
                _, mask = cv2.threshold(tophat, 15, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        boxes, scores = [], []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if not (self.min_area <= area <= self.max_area):
                continue
            if max(w, h) / max(1, min(w, h)) > self.max_aspect:
                continue
            contrast = float(tophat[y:y + h, x:x + w].max())
            boxes.append([x, y, x + w, y + h])
            scores.append(min(1.0, contrast / 80.0))
        if not boxes:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
        return np.array(boxes, np.float32), np.array(scores, np.float32)
