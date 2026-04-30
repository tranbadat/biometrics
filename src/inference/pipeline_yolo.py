"""Pipeline 3-stage: MediaPipe (detect) + YOLOv8-cls (mask) + InsightFace (recognize).

Khác với pipeline.py (dùng MTCNN + CNN), file này tối ưu cho:
  - Tốc độ real-time (MediaPipe BlazeFace ~100 FPS CPU)
  - Robust với loá sáng (BlazeFace pretrained trên data đa dạng)
  - Recognition khi đeo mask (ArcFace buffalo_l robust với occlusion)
"""
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

# ── Stage 1: MediaPipe face detection ──────────────────────────────────────
try:
    import mediapipe as mp
    _mp_face = mp.solutions.face_detection.FaceDetection(
        model_selection=1,        # 1 = full-range (>2m), 0 = short-range (<2m)
        min_detection_confidence=0.5,
    )
    _HAS_MP = True
except Exception:
    _mp_face = None
    _HAS_MP = False

# ── Stage 2: YOLOv8 classification ─────────────────────────────────────────
try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except Exception:
    YOLO = None
    _HAS_YOLO = False

# ── Stage 3: InsightFace ArcFace ───────────────────────────────────────────
try:
    from insightface.app import FaceAnalysis
    _HAS_INSIGHT = True
except Exception:
    FaceAnalysis = None
    _HAS_INSIGHT = False


# Đường dẫn weights
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_YOLO_WEIGHTS = _PROJECT_ROOT / "models" / "mask_yolov8n_cls.pt"

# Lazy singletons
_yolo_model: Optional["YOLO"] = None
_insight_app: Optional["FaceAnalysis"] = None
_known_embeddings: dict[str, np.ndarray] = {}  # name -> embedding


# ── Tiền xử lý chống loá sáng ──────────────────────────────────────────────
def _apply_clahe(bgr: np.ndarray) -> np.ndarray:
    """CLAHE trên kênh L (LAB) — cân bằng sáng cục bộ, chống loá."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# ── Stage 1: detect ────────────────────────────────────────────────────────
def _detect_faces_mp(rgb: np.ndarray) -> list[tuple[int, int, int, int, float]]:
    """Trả về list (x1, y1, x2, y2, score) từ MediaPipe."""
    if not _HAS_MP:
        return []
    h, w = rgb.shape[:2]
    results = _mp_face.process(rgb)
    out: list[tuple[int, int, int, int, float]] = []
    if not results.detections:
        return out
    for det in results.detections:
        bbox = det.location_data.relative_bounding_box
        x1 = max(0, int(bbox.xmin * w))
        y1 = max(0, int(bbox.ymin * h))
        x2 = min(w - 1, int((bbox.xmin + bbox.width) * w))
        y2 = min(h - 1, int((bbox.ymin + bbox.height) * h))
        if x2 <= x1 or y2 <= y1:
            continue
        out.append((x1, y1, x2, y2, float(det.score[0])))
    return out


# ── Stage 2: classify mask ─────────────────────────────────────────────────
def _get_yolo() -> Optional["YOLO"]:
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    if not _HAS_YOLO or not _YOLO_WEIGHTS.exists():
        return None
    _yolo_model = YOLO(str(_YOLO_WEIGHTS))
    return _yolo_model


def _classify_mask(face_bgr: np.ndarray) -> tuple[str, float]:
    """Trả về (label, confidence). Fallback heuristic nếu không có model."""
    model = _get_yolo()
    if model is None:
        # Fallback: trung bình kênh xanh lá
        mean_g = float(face_bgr[:, :, 1].mean()) if face_bgr.ndim == 3 else 0.0
        return ("with_mask" if mean_g < 100 else "without_mask", 0.5)

    results = model.predict(face_bgr, imgsz=224, verbose=False)
    probs = results[0].probs
    idx = int(probs.top1)
    label = model.names[idx]
    return label, float(probs.top1conf)


# ── Stage 3: recognize identity ────────────────────────────────────────────
def _get_insight() -> Optional["FaceAnalysis"]:
    global _insight_app
    if _insight_app is not None:
        return _insight_app
    if not _HAS_INSIGHT:
        return None
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    _insight_app = app
    return app


def _recognize_identity(face_bgr: np.ndarray, threshold: float = 0.35) -> tuple[Optional[str], float]:
    """Tính embedding ArcFace, so cosine với DB."""
    app = _get_insight()
    if app is None or not _known_embeddings:
        return None, 0.0
    faces = app.get(face_bgr)
    if not faces:
        return None, 0.0
    emb = faces[0].normed_embedding  # đã L2-normalize
    best_name, best_sim = None, -1.0
    for name, ref in _known_embeddings.items():
        sim = float(np.dot(emb, ref))
        if sim > best_sim:
            best_name, best_sim = name, sim
    if best_sim < threshold:
        return None, best_sim
    return best_name, best_sim


def enroll_identity(name: str, face_bgr: np.ndarray) -> bool:
    """Đăng ký embedding cho 1 người (gọi ngoài pipeline)."""
    app = _get_insight()
    if app is None:
        return False
    faces = app.get(face_bgr)
    if not faces:
        return False
    _known_embeddings[name] = faces[0].normed_embedding
    return True


# ── Pipeline chính ─────────────────────────────────────────────────────────
def infer_pil(img: Image.Image) -> list[dict]:
    """Detect + classify mask + recognize identity. Trả về list dict cho mỗi mặt."""
    rgb = np.array(img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = _apply_clahe(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    detections = _detect_faces_mp(rgb)
    preds: list[dict] = []
    for x1, y1, x2, y2, score in detections:
        face_bgr = bgr[y1:y2, x1:x2]
        if face_bgr.size == 0:
            continue
        label, conf = _classify_mask(face_bgr)
        identity, id_conf = _recognize_identity(face_bgr)
        preds.append({
            "box": [x1, y1, x2, y2],
            "detection_score": score,
            "label": label,
            "confidence": conf,
            "identity": identity,
            "identity_confidence": id_conf,
        })
    return preds
