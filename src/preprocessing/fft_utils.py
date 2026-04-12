"""FFT-based image processing utilities.

Các hàm biến đổi Fourier dùng trong pipeline:
  - fft_high_pass : làm nổi cạnh vùng mắt/lông mày trước khi tạo embedding
  - fft_low_pass  : khử JPEG artifact trước khi đưa vào training
  - fft_texture_feature : trích đặc trưng texture làm feature bổ sung cho mask classifier
"""

import numpy as np
import cv2


def fft_high_pass(img_gray: np.ndarray, cutoff: int = 30) -> np.ndarray:
    """Lọc thông cao trong miền tần số.

    Loại bỏ tần số thấp (màu nền đồng đều) → giữ lại tần số cao (cạnh mắt, lông mày).
    Dùng trước bước FaceNet embedding khi khuôn mặt đang đeo khẩu trang.

    Args:
        img_gray: ảnh grayscale uint8 hoặc float32, shape (H, W)
        cutoff: bán kính vùng tần số thấp bị loại bỏ (pixel)

    Returns:
        ảnh float32 đã lọc, shape (H, W), giá trị [0, 255]
    """
    img = img_gray.astype(np.float32)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)  # đưa tần số 0 (DC) vào giữa

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    # Mặt nạ: bỏ vùng tần số thấp tại trung tâm
    mask = np.ones((rows, cols), dtype=np.float32)
    mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 0.0

    fshift_filtered = fshift * mask
    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_filtered)))

    # Chuẩn hóa về [0, 255]
    img_back = img_back - img_back.min()
    max_val = img_back.max()
    if max_val > 0:
        img_back = img_back / max_val * 255.0
    return img_back.astype(np.float32)


def fft_low_pass(img_gray: np.ndarray, cutoff: int = 50) -> np.ndarray:
    """Lọc thông thấp trong miền tần số.

    Giữ lại tần số thấp (hình dạng tổng thể, màu da) → loại bỏ noise tần số cao
    từ JPEG artifact (block artifact). Dùng khi chuẩn bị dataset training.

    Args:
        img_gray: ảnh grayscale uint8 hoặc float32, shape (H, W)
        cutoff: bán kính vùng tần số thấp được giữ lại (pixel)

    Returns:
        ảnh float32 đã lọc, shape (H, W), giá trị [0, 255]
    """
    img = img_gray.astype(np.float32)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    # Mặt nạ: chỉ giữ vùng tần số thấp tại trung tâm
    mask = np.zeros((rows, cols), dtype=np.float32)
    mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 1.0

    fshift_filtered = fshift * mask
    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_filtered)))

    img_back = img_back - img_back.min()
    max_val = img_back.max()
    if max_val > 0:
        img_back = img_back / max_val * 255.0
    return img_back.astype(np.float32)


def fft_texture_feature(img_gray: np.ndarray, out_size: int = 32) -> np.ndarray:
    """Trích đặc trưng texture từ phổ biên độ FFT.

    Vải khẩu trang có texture tuần hoàn → tạo ra đỉnh năng lượng đặc trưng
    trong phổ FFT, khác với da mặt. Feature này dùng bổ sung cho mask classifier.

    Args:
        img_gray: ảnh grayscale uint8 hoặc float32, shape (H, W)
        out_size: kích thước resize phổ trước khi flatten

    Returns:
        vector float32 shape (out_size * out_size,)
    """
    img = img_gray.astype(np.float32)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = 20.0 * np.log(np.abs(fshift) + 1.0)  # log scale
    magnitude_resized = cv2.resize(magnitude, (out_size, out_size))
    return magnitude_resized.flatten().astype(np.float32)


def enhance_periocular(img_bgr: np.ndarray, cutoff: int = 25) -> np.ndarray:
    """Kết hợp CLAHE + FFT high-pass để tăng chất lượng vùng mắt.

    Pipeline dùng trước bước FaceNet embedding khi có khẩu trang:
      1. CLAHE cân bằng sáng tối cục bộ
      2. FFT high-pass làm nổi cạnh lông mày, mắt

    Args:
        img_bgr: ảnh BGR uint8, shape (H, W, 3)
        cutoff: ngưỡng lọc FFT

    Returns:
        ảnh BGR uint8, shape (H, W, 3) đã tăng cường
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # FFT high-pass
    high_freq = fft_high_pass(enhanced_gray, cutoff=cutoff)
    high_freq_u8 = np.clip(high_freq, 0, 255).astype(np.uint8)

    # Blend: 80% ảnh CLAHE gốc + 20% thành phần tần số cao
    blended = cv2.addWeighted(enhanced_gray, 0.80, high_freq_u8, 0.20, 0)
    return cv2.cvtColor(blended, cv2.COLOR_GRAY2BGR)
