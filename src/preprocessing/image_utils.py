"""OpenCV-based image processing utilities.

Các hàm xử lý ảnh dùng trong pipeline và chuẩn bị dataset:
  - apply_clahe         : cân bằng histogram thích nghi (vùng mắt tối)
  - apply_gaussian_blur : khử noise trước khi detect
  - apply_canny         : trích cạnh viền mắt/lông mày
  - is_sharp_enough     : lọc ảnh mờ khỏi dataset
  - expand_box          : mở rộng bounding box để bao trọn khẩu trang
  - crop_periocular     : crop vùng mắt + trán từ facial landmarks
  - preprocess_frame    : tiền xử lý frame đầu vào trước khi detect
"""

import numpy as np
import cv2
from PIL import Image


# ---------------------------------------------------------------------------
# Tiền xử lý frame trước khi detect
# ---------------------------------------------------------------------------

def preprocess_frame(img_bgr: np.ndarray, max_side: int = 1280) -> np.ndarray:
    """Resize nếu ảnh quá lớn, sau đó khử noise nhẹ bằng Gaussian blur.

    Vấn đề: ảnh độ phân giải cao làm MTCNN chậm; noise từ webcam rẻ
    khiến MTCNN sinh false positive.

    Args:
        img_bgr: ảnh BGR uint8
        max_side: cạnh dài tối đa (pixel)

    Returns:
        ảnh BGR uint8 đã xử lý
    """
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                             interpolation=cv2.INTER_AREA)

    # Gaussian blur nhẹ — kernel 3×3, sigma nhỏ: khử noise mà không mờ cạnh
    return cv2.GaussianBlur(img_bgr, (3, 3), sigmaX=0.8)


# ---------------------------------------------------------------------------
# Cân bằng histogram thích nghi
# ---------------------------------------------------------------------------

def apply_clahe(img_bgr: np.ndarray,
                clip_limit: float = 2.0,
                tile_size: int = 8) -> np.ndarray:
    """CLAHE trên kênh L (LAB color space) để cân bằng sáng cục bộ.

    Vấn đề: vùng mắt tối khi ngược sáng → FaceNet tạo embedding kém chất lượng.
    CLAHE cân bằng histogram theo từng ô nhỏ (tile_size × tile_size), không làm
    cháy vùng đã sáng như global histogram equalization.

    Args:
        img_bgr: ảnh BGR uint8
        clip_limit: giới hạn khuếch đại tương phản (2.0 là an toàn)
        tile_size: kích thước ô histogram cục bộ

    Returns:
        ảnh BGR uint8 đã cân bằng sáng
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                             tileGridSize=(tile_size, tile_size))
    l_enhanced = clahe.apply(l_ch)

    lab_enhanced = cv2.merge([l_enhanced, a_ch, b_ch])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Phát hiện cạnh
# ---------------------------------------------------------------------------

def apply_canny(img_gray: np.ndarray,
                low: int = 40,
                high: int = 120) -> np.ndarray:
    """Phát hiện cạnh Canny để trích đặc trưng viền mắt/lông mày.

    Vấn đề: FaceNet cần đặc trưng hình dạng mắt, không chỉ màu sắc.
    Canny dùng gradient Sobel → non-maximum suppression → hysteresis thresholding
    → cho ra đường viền sắc nét của mắt, lông mày.

    Kết quả được blend vào ảnh gốc trước khi đưa vào FaceNet (tỉ lệ nhỏ ~15%).

    Args:
        img_gray: ảnh grayscale uint8
        low: ngưỡng hysteresis thấp
        high: ngưỡng hysteresis cao

    Returns:
        edge map uint8, giá trị 0 (nền) hoặc 255 (cạnh)
    """
    return cv2.Canny(img_gray, threshold1=low, threshold2=high)


def blend_edges(img_bgr: np.ndarray, edge_weight: float = 0.15) -> np.ndarray:
    """Ghép edge map Canny vào ảnh gốc làm kênh bổ sung.

    Args:
        img_bgr: ảnh BGR uint8
        edge_weight: tỉ lệ contribution của edge (0.0–1.0)

    Returns:
        ảnh BGR uint8 đã blend edges
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = apply_canny(gray)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(img_bgr, 1.0 - edge_weight, edges_bgr, edge_weight, 0)


# ---------------------------------------------------------------------------
# Đánh giá chất lượng ảnh
# ---------------------------------------------------------------------------

def is_sharp_enough(img_gray: np.ndarray, threshold: float = 80.0) -> bool:
    """Kiểm tra ảnh có đủ sắc nét để đưa vào training không.

    Vấn đề: dataset từ nhiều nguồn có nhiều ảnh blur do chuyển động hoặc
    lấy nét kém → embedding không ổn định → nhiễu training.

    Dùng Laplacian variance: ảnh sắc nét có gradient lớn → variance cao.

    Args:
        img_gray: ảnh grayscale uint8
        threshold: ngưỡng variance (80 phù hợp với ảnh khuôn mặt 160×160)

    Returns:
        True nếu đủ sắc nét
    """
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    return float(laplacian.var()) >= threshold


# ---------------------------------------------------------------------------
# Bounding box
# ---------------------------------------------------------------------------

def expand_box(box, img_h: int, img_w: int, expand_ratio: float = 0.18):
    """Mở rộng bounding box xuống dưới để bao trọn khẩu trang.

    Vấn đề: MTCNN trả về box khuôn mặt, nhưng khi đeo khẩu trang cằm thường
    bị cắt ra ngoài → mask classifier nhầm mask thành no_mask.

    Args:
        box: (x1, y1, x2, y2)
        img_h, img_w: kích thước ảnh gốc
        expand_ratio: tỉ lệ mở rộng so với chiều cao box

    Returns:
        (x1, y1, x2, y2_expanded)
    """
    x1, y1, x2, y2 = map(int, box)
    face_h = y2 - y1
    y2_new = min(img_h - 1, y2 + int(face_h * expand_ratio))
    return x1, y1, x2, y2_new


# ---------------------------------------------------------------------------
# Crop vùng periocular (mắt + trán)
# ---------------------------------------------------------------------------

def crop_periocular(face_img_bgr: np.ndarray, landmarks, expand: float = 1.3):
    """Crop vùng mắt + trán từ facial landmarks của MTCNN.

    Vấn đề: khi có khẩu trang, chỉ còn vùng mắt dùng được cho recognition.
    MTCNN trả về 5 landmarks: [left_eye, right_eye, nose, mouth_l, mouth_r].
    Dùng tọa độ Y của mắt để tính điểm cắt.

    Args:
        face_img_bgr: ảnh BGR crop khuôn mặt, shape (H, W, 3)
        landmarks: array shape (5, 2) — [[lx,ly],[rx,ry],[nx,ny],[ml,ml],[mr,mr]]
        expand: hệ số mở rộng tính từ eye_center_y xuống dưới

    Returns:
        ảnh BGR crop vùng periocular, shape (H', W, 3)
    """
    left_eye_y  = float(landmarks[0][1])
    right_eye_y = float(landmarks[1][1])
    eye_center_y = (left_eye_y + right_eye_y) / 2.0
    cutoff_y = min(face_img_bgr.shape[0], int(eye_center_y * expand))
    return face_img_bgr[:cutoff_y, :]


# ---------------------------------------------------------------------------
# Tiền xử lý ảnh trước FaceNet embedding
# ---------------------------------------------------------------------------

def prepare_for_embedding(img_bgr: np.ndarray, has_mask: bool = False) -> Image.Image:
    """Chuẩn bị ảnh khuôn mặt trước khi đưa vào FaceNet.

    - Không có mask: CLAHE toàn bộ mặt → RGB PIL
    - Có mask: CLAHE + blend edges → RGB PIL
      (vùng periocular đã được crop trước ở pipeline)

    Args:
        img_bgr: ảnh BGR uint8, shape (H, W, 3)
        has_mask: True nếu người đang đeo khẩu trang

    Returns:
        PIL.Image RGB, sẵn sàng đưa vào FaceNet transform
    """
    img_bgr = apply_clahe(img_bgr)
    if has_mask:
        img_bgr = blend_edges(img_bgr, edge_weight=0.15)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)
