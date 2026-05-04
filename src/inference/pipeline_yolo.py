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
    raw = str(model.names[idx]).lower()
    # Chuẩn hoá tên class về snake_case để khớp _MASK_LABELS
    if "without" in raw or "no_mask" in raw or raw == "no":
        label = "without_mask"
    elif "mask" in raw or "with" in raw:
        label = "with_mask"
    else:
        label = raw
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


_MASK_LABELS = ("with_mask", "without_mask")


def _slot_key(name: str, mask_label: str) -> str:
    return f"{name}__{mask_label}"


def _split_key(key: str) -> tuple[str, Optional[str]]:
    if "__" in key:
        base, suffix = key.rsplit("__", 1)
        if suffix in _MASK_LABELS:
            return base, suffix
    return key, None


def _match_embedding(
    emb: np.ndarray,
    mask_label: Optional[str] = None,
    threshold: float = 0.35,
) -> tuple[Optional[str], float]:
    """So cosine với DB, ưu tiên slot cùng trạng thái mask.

    - Nếu có slot khớp `mask_label` → chỉ match trong slot đó (cụm chặt hơn).
    - Nếu không có slot khớp (user chỉ enroll 1 trạng thái) → fallback toàn DB.
    """
    if not _known_embeddings:
        return None, 0.0

    if mask_label in _MASK_LABELS:
        candidates = {k: v for k, v in _known_embeddings.items() if k.endswith(f"__{mask_label}")}
        if not candidates:
            candidates = _known_embeddings
    else:
        candidates = _known_embeddings

    best_key, best_sim = None, -1.0
    for key, ref in candidates.items():
        sim = float(np.dot(emb, ref))
        if sim > best_sim:
            best_key, best_sim = key, sim

    if best_sim < threshold or best_key is None:
        return None, best_sim
    base_name, _ = _split_key(best_key)
    return base_name, best_sim


def enroll_identity(name: str, face_bgr: np.ndarray, persist: bool = True) -> bool:
    """Đăng ký embedding cho 1 người, tách slot theo trạng thái mask.

    Pipeline: detect → crop face → classify mask → lưu vào slot `name__<mask_label>`.
    Nhiều ảnh cùng (name, mask_label) sẽ được trung bình + L2-normalize.
    """
    app = _get_insight()
    if app is None:
        return False
    faces = app.get(face_bgr)
    if not faces:
        return False

    face = faces[0]
    x1, y1, x2, y2 = (int(v) for v in face.bbox)
    h, w = face_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return False

    face_h = y2 - y1
    face_w = x2 - x1
    ex1 = max(0, x1 - int(face_w * 0.10))
    ex2 = min(w - 1, x2 + int(face_w * 0.10))
    ey1 = max(0, y1 - int(face_h * 0.10))
    ey2 = min(h - 1, y2 + int(face_h * 0.25))
    crop = face_bgr[ey1:ey2, ex1:ex2]
    if crop.size == 0:
        crop = face_bgr[y1:y2, x1:x2]
    mask_label, _ = _classify_mask(crop)
    if mask_label not in _MASK_LABELS:
        mask_label = "without_mask"

    key = _slot_key(name, mask_label)
    new_emb = face.normed_embedding
    if key in _known_embeddings:
        merged = _known_embeddings[key] + new_emb
        norm = np.linalg.norm(merged)
        if norm > 0:
            merged = merged / norm
        _known_embeddings[key] = merged
    else:
        _known_embeddings[key] = new_emb

    if persist:
        _save_db()
    return True


def list_identities() -> list[str]:
    """Trả về danh sách tên người (đã gộp 2 slot mask/no_mask)."""
    names = {_split_key(k)[0] for k in _known_embeddings.keys()}
    return sorted(names)


def list_slots() -> dict[str, list[str]]:
    """Debug: trả về dict {name: [trạng thái đã enroll]}."""
    out: dict[str, list[str]] = {}
    for k in _known_embeddings.keys():
        base, suffix = _split_key(k)
        out.setdefault(base, []).append(suffix or "legacy")
    return out


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

        # Mở rộng bbox xuống dưới + ngang để YOLO thấy đủ vùng khẩu trang
        face_h = y2 - y1
        face_w = x2 - x1
        ex1 = max(0, x1 - int(face_w * 0.10))
        ex2 = min(w - 1, x2 + int(face_w * 0.10))
        ey1 = max(0, y1 - int(face_h * 0.10))
        ey2 = min(h - 1, y2 + int(face_h * 0.25))
        face_for_mask = bgr[ey1:ey2, ex1:ex2]
        if face_for_mask.size == 0:
            face_for_mask = face_bgr

        label, conf = _classify_mask(face_for_mask)
        identity, id_conf = _match_embedding(face.normed_embedding, mask_label=label)

        preds.append({
            "box": [x1, y1, x2, y2],
            "detection_score": float(face.det_score),
            "label": label,
            "confidence": conf,
            "identity": identity,
            "identity_confidence": id_conf,
        })
    return preds
