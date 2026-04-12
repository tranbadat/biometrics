from PIL import Image
import numpy as np
import cv2

# Try to import MTCNN from facenet-pytorch; otherwise fall back to deterministic stub.
try:
    from facenet_pytorch import MTCNN
    _mtcnn = MTCNN(keep_all=True, post_process=False)
    _HAS_MTCNN = True
except Exception:
    _mtcnn = None
    _HAS_MTCNN = False

# Preprocessing utilities (OpenCV + FFT)
try:
    from src.preprocessing.image_utils import (
        preprocess_frame, expand_box, crop_periocular, prepare_for_embedding
    )
    from src.preprocessing.fft_utils import enhance_periocular
    _HAS_PREPROCESS = True
except Exception:
    _HAS_PREPROCESS = False

# Try to import MaskClassifier wrapper (optional)
try:
    from src.models.mask_classifier import MaskClassifier
    _HAS_MASK_CLS = True
except Exception:
    MaskClassifier = None
    _HAS_MASK_CLS = False

# instantiate classifier lazily
_MASK_CLF = None
try:
    from src.models.recognizer import Recognizer
    _HAS_RECOG = True
except Exception:
    Recognizer = None
    _HAS_RECOG = False

# instantiate recognizer lazily
_RECOG = None


def _heuristic_mask_label(arr_crop: np.ndarray) -> str:
    if arr_crop.ndim == 3:
        mean_g = float(arr_crop[:, :, 1].mean())
    else:
        mean_g = 0.0
    return "mask" if mean_g < 100 else "no_mask"


def _bgr_from_arr(crop: np.ndarray) -> np.ndarray:
    """Chuyển crop RGB numpy → BGR để dùng với OpenCV."""
    if crop.ndim == 3 and crop.shape[2] == 3:
        return crop[:, :, ::-1]
    return crop


def _classify_and_recognize(crop: np.ndarray, score: float, landmarks=None):
    """Phân loại mask và nhận diện danh tính cho một crop khuôn mặt.

    Pipeline:
      1. Tiền xử lý bằng CLAHE + Gaussian (nếu có preprocessing utils)
      2. Phân loại mask/no_mask bằng MaskClassifier
      3. Nếu có mask: crop vùng periocular + FFT high-pass trước khi recognize
      4. Nhận diện danh tính bằng Recognizer (FaceNet + SVM)
    """
    global _MASK_CLF, _RECOG

    crop_bgr = _bgr_from_arr(crop)

    # ── Bước 1: Tiền xử lý CLAHE ────────────────────────────────────────────
    if _HAS_PREPROCESS:
        from src.preprocessing.image_utils import apply_clahe
        crop_bgr = apply_clahe(crop_bgr)

    pil_crop = Image.fromarray(crop_bgr[:, :, ::-1])  # BGR→RGB→PIL

    # ── Bước 2: Phân loại mask ───────────────────────────────────────────────
    if _HAS_MASK_CLS and _MASK_CLF is None:
        try:
            _MASK_CLF = MaskClassifier()
        except Exception:
            pass

    if _MASK_CLF is not None:
        try:
            label, conf = _MASK_CLF.predict(pil_crop)
        except Exception:
            label = _heuristic_mask_label(crop)
            conf = float(score)
    else:
        label = _heuristic_mask_label(crop)
        conf = float(score)

    # ── Bước 3: Chuẩn bị ảnh cho recognition ────────────────────────────────
    has_mask = (label == 'mask')
    if _HAS_PREPROCESS and has_mask and landmarks is not None:
        # Crop vùng periocular khi có khẩu trang
        periocular_bgr = crop_periocular(crop_bgr, landmarks)
        if periocular_bgr.size > 0:
            # FFT high-pass + CLAHE để tăng rõ nét vùng mắt
            periocular_bgr = enhance_periocular(periocular_bgr)
            pil_for_recog = Image.fromarray(periocular_bgr[:, :, ::-1])
        else:
            pil_for_recog = pil_crop
    elif _HAS_PREPROCESS:
        pil_for_recog = Image.fromarray(
            prepare_for_embedding(crop_bgr, has_mask=False)
            if not has_mask else pil_crop
        )
    else:
        pil_for_recog = pil_crop

    # ── Bước 4: Nhận diện danh tính ─────────────────────────────────────────
    identity, identity_conf = None, 0.0
    if _HAS_RECOG and _RECOG is None:
        try:
            _RECOG = Recognizer()
        except Exception:
            pass

    if _RECOG is not None:
        try:
            identity, identity_conf = _RECOG.match(pil_for_recog)
        except Exception:
            pass

    return label, conf, identity, identity_conf


def infer_pil(img: Image.Image):
    """Nhận diện khuôn mặt + phân loại mask + nhận diện danh tính từ PIL image.

    Pipeline:
      1. Tiền xử lý frame (Gaussian blur, resize nếu cần)
      2. MTCNN detect → bounding boxes + landmarks
      3. Với mỗi mặt: mở rộng box → classify mask → recognize identity

    Returns:
        list of dict: {box, label, confidence, identity, identity_confidence}
    """
    # Bước 1: tiền xử lý frame
    arr = np.array(img)           # RGB
    h, w = arr.shape[0], arr.shape[1]

    if _HAS_PREPROCESS:
        arr_bgr = arr[:, :, ::-1].copy()
        arr_bgr = preprocess_frame(arr_bgr)
        img_for_detect = Image.fromarray(arr_bgr[:, :, ::-1])
    else:
        arr_bgr = arr[:, :, ::-1].copy()
        img_for_detect = img

    if _HAS_MTCNN:
        # Detect với landmarks để dùng cho periocular crop
        boxes, probs, landmarks_list = _mtcnn.detect(img_for_detect, landmarks=True)
        preds = []

        if boxes is not None:
            for box, score, lm in zip(boxes, probs, landmarks_list):
                if box is None:
                    continue

                x1, y1, x2, y2 = map(int, box)

                # Mở rộng box xuống dưới để bao trọn khẩu trang
                if _HAS_PREPROCESS:
                    x1, y1, x2, y2 = expand_box(
                        (x1, y1, x2, y2), h, w, expand_ratio=0.18
                    )

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                crop_rgb = arr[y1:y2, x1:x2]   # RGB crop
                score    = float(score) if score is not None else 0.0

                # Điều chỉnh landmarks về tọa độ tương đối trong crop
                lm_rel = lm - np.array([x1, y1]) if lm is not None else None

                label, conf, identity, identity_conf = _classify_and_recognize(
                    crop_rgb, score, landmarks=lm_rel
                )
                preds.append({
                    'box':                [x1, y1, x2, y2],
                    'label':              label,
                    'confidence':         conf,
                    'identity':           identity,
                    'identity_confidence': float(identity_conf),
                })
        return preds

    # Fallback: heuristic central box (khi không có MTCNN)
    x1 = int(w * 0.25)
    y1 = int(h * 0.25)
    x2 = int(w * 0.75)
    y2 = int(h * 0.75)
    crop = arr[y1:y2, x1:x2]
    label = _heuristic_mask_label(crop)
    return [{
        'box':        [x1, y1, x2, y2],
        'label':      label,
        'confidence': 0.85,
    }]
