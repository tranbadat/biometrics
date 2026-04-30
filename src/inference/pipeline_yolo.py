"""Pipeline 2-stage: InsightFace (detect + recognize) + YOLOv8-cls (mask classify).

Khác với pipeline.py (dùng MTCNN + CNN custom):
  - InsightFace `buffalo_l` lo cả detect (RetinaFace) lẫn embedding (ArcFace)
  - YOLOv8n-cls phân loại mask/no_mask trên crop face
  - Robust với loá sáng (CLAHE preprocess + augmentation HSV khi train)
  - Robust với occlusion mask (ArcFace native)

Note: ban đầu thiết kế 3-stage (MediaPipe + YOLO + ArcFace) nhưng MediaPipe 0.10.x
trên Python 3.14 không export `solutions` API → gộp detect vào InsightFace cho gọn.
"""
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

# ── Stage 1+3: InsightFace (detect + ArcFace embedding) ────────────────────
try:
    from insightface.app import FaceAnalysis
    _HAS_INSIGHT = True
except Exception:
    FaceAnalysis = None
    _HAS_INSIGHT = False

# ── Stage 2: YOLOv8 classification ─────────────────────────────────────────
try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except Exception:
    YOLO = None
    _HAS_YOLO = False


# Đường dẫn weights
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_YOLO_WEIGHTS = _PROJECT_ROOT / "models" / "mask_yolov8n_cls.pt"
_ARCFACE_DB = _PROJECT_ROOT / "models" / "arcface_db.npz"

# Lazy singletons
_yolo_model: Optional["YOLO"] = None
_insight_app: Optional["FaceAnalysis"] = None
_known_embeddings: dict[str, np.ndarray] = {}  # name -> embedding


def _load_db() -> None:
    """Load ArcFace embeddings DB từ disk (gọi khi import)."""
    if not _ARCFACE_DB.exists():
        return
    data = np.load(_ARCFACE_DB, allow_pickle=False)
    for key in data.files:
        _known_embeddings[key] = data[key]


def _save_db() -> None:
    """Persist ArcFace embeddings ra disk."""
    _ARCFACE_DB.parent.mkdir(parents=True, exist_ok=True)
    np.savez(_ARCFACE_DB, **_known_embeddings)


_load_db()


# ── Tiền xử lý chống loá sáng ──────────────────────────────────────────────
def _apply_clahe(bgr: np.ndarray) -> np.ndarray:
    """CLAHE trên kênh L (LAB) — cân bằng sáng cục bộ, chống loá."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


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


def _match_embedding(emb: np.ndarray, threshold: float = 0.35) -> tuple[Optional[str], float]:
    """So cosine embedding với DB. Trả về (name, similarity) hoặc (None, best_sim)."""
    if not _known_embeddings:
        return None, 0.0
    best_name, best_sim = None, -1.0
    for name, ref in _known_embeddings.items():
        sim = float(np.dot(emb, ref))
        if sim > best_sim:
            best_name, best_sim = name, sim
    if best_sim < threshold:
        return None, best_sim
    return best_name, best_sim


def enroll_identity(name: str, face_bgr: np.ndarray, persist: bool = True) -> bool:
    """Đăng ký embedding cho 1 người. Nếu nhiều ảnh cho cùng `name` thì
    embedding được trung bình hoá rồi L2-normalize lại.
    """
    app = _get_insight()
    if app is None:
        return False
    faces = app.get(face_bgr)
    if not faces:
        return False
    new_emb = faces[0].normed_embedding
    if name in _known_embeddings:
        merged = _known_embeddings[name] + new_emb
        norm = np.linalg.norm(merged)
        if norm > 0:
            merged = merged / norm
        _known_embeddings[name] = merged
    else:
        _known_embeddings[name] = new_emb
    if persist:
        _save_db()
    return True


def list_identities() -> list[str]:
    return list(_known_embeddings.keys())


# ── Pipeline chính ─────────────────────────────────────────────────────────
def infer_pil(img: Image.Image) -> list[dict]:
    """InsightFace detect → YOLO mask classify → cosine match identity."""
    rgb = np.array(img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = _apply_clahe(bgr)

    app = _get_insight()
    preds: list[dict] = []
    if app is None:
        return preds

    faces = app.get(bgr)
    h, w = bgr.shape[:2]
    for face in faces:
        x1, y1, x2, y2 = (int(v) for v in face.bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        face_bgr = bgr[y1:y2, x1:x2]
        if face_bgr.size == 0:
            continue

        label, conf = _classify_mask(face_bgr)
        identity, id_conf = _match_embedding(face.normed_embedding)

        preds.append({
            "box": [x1, y1, x2, y2],
            "detection_score": float(face.det_score),
            "label": label,
            "confidence": conf,
            "identity": identity,
            "identity_confidence": id_conf,
        })
    return preds
