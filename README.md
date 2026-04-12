# Hệ thống Định danh Sinh trắc học — Nhận diện Khuôn mặt có Đeo Khẩu trang

> **Đề tài Bài tập lớn — Nhận dạng và Xử lý ảnh / Sinh trắc học**
>
> Xây dựng hệ thống nhận diện khuôn mặt có khả năng hoạt động trong điều kiện người dùng **đeo khẩu trang**, kết hợp kỹ thuật xử lý ảnh cổ điển (FFT, lọc tần số, OpenCV) với mô hình học sâu (MTCNN, MobileNetV2, FaceNet).

---

## Mục lục

1. [Đặt vấn đề](#1-đặt-vấn-đề)
2. [Mục tiêu đề tài](#2-mục-tiêu-đề-tài)
3. [Pipeline tổng thể](#3-pipeline-tổng-thể)
4. [Kiến thức áp dụng](#4-kiến-thức-áp-dụng)
5. [Công nghệ và công cụ](#5-công-nghệ-và-công-cụ)
6. [Kiến trúc hệ thống](#6-kiến-trúc-hệ-thống)
7. [Giải pháp chi tiết](#7-giải-pháp-chi-tiết)
8. [Dataset](#8-dataset)
9. [Cài đặt và chạy thử](#9-cài-đặt-và-chạy-thử)
10. [Kết quả kỳ vọng](#10-kết-quả-kỳ-vọng)
11. [Hạn chế và hướng phát triển](#11-hạn-chế-và-hướng-phát-triển)

---

## 1. Đặt vấn đề

Nhận diện khuôn mặt là bài toán sinh trắc học phổ biến được ứng dụng trong kiểm soát ra vào, chấm công, xác thực danh tính. Tuy nhiên, khi người dùng **đeo khẩu trang**, phần dưới khuôn mặt bị che khuất (~60% diện tích), khiến các hệ thống truyền thống sụt giảm độ chính xác nghiêm trọng.

**Thách thức chính:**

| Vấn đề | Ảnh hưởng |
|---|---|
| Mất thông tin vùng mũi, miệng, cằm | Giảm chất lượng embedding khuôn mặt |
| Sự đa dạng của khẩu trang (màu, hình dạng) | Khó phân biệt mask/no\_mask chính xác |
| Ánh sáng, góc chụp thay đổi | Tiền xử lý ảnh phức tạp hơn |
| Dataset thiếu cân bằng | Bias mô hình về nhóm không đeo khẩu trang |

---

## 2. Mục tiêu đề tài

1. **Phát hiện khuôn mặt** trong ảnh/video (cả khi đeo và không đeo khẩu trang).
2. **Phân loại trạng thái khẩu trang**: `mask` / `no_mask` / `mask_incorrect` (đeo sai cách).
3. **Định danh danh tính** của người dùng ngay cả khi đang đeo khẩu trang, bằng cách tập trung vào **vùng mắt và trán (periocular region)**.
4. **Áp dụng kỹ thuật xử lý ảnh** (FFT, lọc tần số, OpenCV) vào các bước tiền xử lý và tăng cường dữ liệu.

---

## 3. Pipeline tổng thể

```
Ảnh đầu vào
     │
     ▼
┌─────────────────────────────┐
│  1. TIỀN XỬ LÝ ẢNH          │  OpenCV: resize, cân bằng histogram (CLAHE),
│     (Preprocessing)         │  khử nhiễu Gaussian, chuẩn hóa ánh sáng
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  2. TĂNG CƯỜNG TẦN SỐ       │  FFT: phân tích phổ tần số, lọc thông cao
│     (FFT Enhancement)       │  để làm rõ cạnh viền mắt, lông mày
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  3. PHÁT HIỆN KHUÔN MẶT     │  MTCNN (facenet-pytorch):
│     (Face Detection)        │  trả về bounding box + 5 landmarks
└────────────┬────────────────┘
             │
             ├──────────────────────────────┐
             ▼                              ▼
┌────────────────────┐         ┌────────────────────────┐
│  4a. PHÂN LOẠI     │         │  4b. CROP VÙNG MẮT     │
│      KHẨU TRANG    │         │      (Periocular ROI)   │
│  MobileNetV2       │         │  Dựa vào landmarks MTCNN│
│  mask/no_mask      │         └────────────┬───────────┘
└────────┬───────────┘                      │
         │                                  ▼
         │ (có mask)             ┌───────────────────────┐
         └──────────────────────►│  5. NHẬN DIỆN DANH   │
                                 │     TÍNH (Recognition)│
                (không mask)     │  FaceNet Embedding    │
         ┌──────────────────────►│  + SVM/kNN Classifier │
         │  (dùng toàn bộ mặt)   └────────────┬──────────┘
         │                                    │
         ▼                                    ▼
┌─────────────────────────────────────────────────────┐
│  6. KẾT QUẢ: Bounding box + Trạng thái mask         │
│     + Danh tính (nếu nhận ra) + Độ tin cậy          │
└─────────────────────────────────────────────────────┘
```

---

## 4. Kiến thức áp dụng

Mỗi kỹ thuật được chọn để giải quyết **một vấn đề cụ thể** trong pipeline. Bảng tổng quan:

| Kỹ thuật | Vấn đề cần giải quyết | Dùng ở bước nào |
|---|---|---|
| Gaussian Blur | Ảnh webcam bị nhiễu làm MTCNN detect nhầm | Trước bước detect |
| CLAHE | Vùng mắt tối, mờ khi ngược sáng | Trước bước detect + trước embedding |
| Canny Edge | Trích đặc trưng viền mắt/lông mày cho periocular | Sau crop periocular |
| Sobel | Đo độ mạnh gradient để chọn vùng ảnh sắc nét | Tiền xử lý training data |
| Dilation | Vùng khẩu trang bị cắt do bounding box nhỏ quá | Sau khi detect mask region |
| FFT lọc thông cao | Tăng rõ nét cạnh vùng mắt trước khi tạo embedding | Trước bước FaceNet khi có mask |
| FFT lọc thông thấp | Khử nhiễu tần số cao từ ảnh nén JPEG kém | Tiền xử lý dataset training |
| FFT magnitude | Texture vải khẩu trang có phổ tần số đặc trưng | Feature bổ sung cho mask classifier |
| MTCNN | Detect khuôn mặt + 5 facial landmarks | Bước 3 pipeline chính |
| MobileNetV2 | Phân loại mask/no\_mask | Bước 4a |
| FaceNet | Tạo embedding 512-D biểu diễn danh tính | Bước 5 |
| SVM | Phân loại danh tính từ embedding | Bước 5 |

---

### 4.1 Gaussian Blur — Khử nhiễu trước khi detect

**Vấn đề:** Webcam rẻ tiền, ảnh chụp ngoài trời hoặc điều kiện ánh sáng thay đổi thường có noise dạng hạt (salt-and-pepper, Gaussian noise). MTCNN dễ bị nhầm các vùng nhiễu thành khuôn mặt → sinh false positive.

**Giải pháp:** Làm mịn ảnh bằng Gaussian Blur trước khi đưa vào detector. Kernel 3×3 đủ để khử noise nhỏ mà không làm mờ các cạnh quan trọng của khuôn mặt.

```python
# Áp dụng trước MTCNN detect
blurred = cv2.GaussianBlur(img, (3, 3), sigmaX=0.8)
```

---

### 4.2 CLAHE — Cân bằng sáng tối cho vùng mắt

**Vấn đề:** Khi nhận diện qua khẩu trang, hệ thống chỉ còn dựa vào **vùng mắt và trán**. Nếu người dùng đứng ngược sáng hoặc dưới ánh đèn yếu, vùng mắt trở nên tối và mờ → FaceNet tạo ra embedding kém chất lượng → nhận diện sai.

**Giải pháp:** CLAHE (Contrast Limited Adaptive Histogram Equalization) cân bằng histogram **cục bộ theo từng ô nhỏ** (8×8 pixel), làm sáng vùng tối mà không làm cháy vùng đã sáng — khác với histogram equalization toàn cục vốn hay gây hiện tượng quá sáng.

```python
# Áp dụng sau khi crop vùng periocular (mắt + trán)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray = cv2.cvtColor(periocular_crop, cv2.COLOR_BGR2GRAY)
enhanced = clahe.apply(gray)
# enhanced có độ tương phản vùng mắt tốt hơn, đưa vào FaceNet
```

---

### 4.3 Canny Edge Detection — Trích đặc trưng viền mắt/lông mày

**Vấn đề:** Vùng periocular (mắt + lông mày) là nguồn thông tin sinh trắc học chính khi có khẩu trang. Cần làm nổi bật hình dạng mắt, viền lông mày để embedding mang nhiều thông tin danh tính hơn.

**Giải pháp:** Canny phát hiện cạnh bằng 2 bước: (1) tính gradient Sobel để tìm vùng thay đổi cường độ mạnh, (2) non-maximum suppression + hysteresis thresholding để giữ lại đúng các cạnh sắc nét. Kết quả là ảnh nhị phân thể hiện rõ đường viền mắt, lông mày.

Ảnh edge map này được **nối (concatenate) vào channel của ảnh gốc** trước khi đưa vào FaceNet, cung cấp thêm đặc trưng hình dạng bổ sung cho đặc trưng màu sắc:

```python
# Áp dụng sau CLAHE, trên vùng periocular đã crop
edges = cv2.Canny(enhanced, threshold1=40, threshold2=120)
# edges: ảnh 1 channel, giá trị 0 (nền) hoặc 255 (cạnh)

# Ghép edge map làm kênh bổ sung
periocular_rgb = cv2.cvtColor(periocular_crop, cv2.COLOR_BGR2RGB)
edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
combined = cv2.addWeighted(periocular_rgb, 0.85, edges_3ch, 0.15, 0)
```

---

### 4.4 Sobel — Đánh giá chất lượng ảnh trong preprocessing

**Vấn đề:** Dataset thu thập từ nhiều nguồn có chất lượng không đồng đều — nhiều ảnh bị mờ (blur) do chuyển động hoặc lấy nét kém. Ảnh mờ → gradient yếu → embedding không ổn định → làm nhiễu quá trình training.

**Giải pháp:** Dùng Sobel để tính **Laplacian variance** (độ sắc nét) của ảnh. Ảnh có variance thấp → bị mờ → loại khỏi dataset trước khi train.

```python
def is_sharp_enough(img_gray, threshold=80.0):
    """Trả về True nếu ảnh đủ sắc nét để đưa vào training."""
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    variance = laplacian.var()
    return variance >= threshold

# Lọc ảnh khi chuẩn bị dataset
for img_path in dataset_images:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if not is_sharp_enough(img):
        print(f"Bỏ qua (mờ): {img_path}")
        continue
    # tiếp tục xử lý
```

---

### 4.5 Dilation — Sửa bounding box khẩu trang bị cắt

**Vấn đề:** MTCNN trả về bounding box khuôn mặt, nhưng khi đeo khẩu trang, cằm thường bị cắt ra ngoài box. Nếu crop đúng bounding box của MTCNN, phần khẩu trang bị cắt bớt → mask classifier dễ nhầm `mask` thành `no_mask`.

**Giải pháp:** Mở rộng bounding box xuống dưới thêm 15–20% chiều cao trước khi crop để đảm bảo bao gồm toàn bộ khẩu trang.

```python
def expand_box(box, img_h, img_w, expand_ratio=0.18):
    """Mở rộng bounding box xuống dưới để bao trọn khẩu trang."""
    x1, y1, x2, y2 = box
    face_h = y2 - y1
    y2_expanded = min(img_h - 1, y2 + int(face_h * expand_ratio))
    return x1, y1, x2, y2_expanded
```

---

### 4.6 FFT lọc thông cao — Làm rõ cạnh vùng mắt trước embedding

**Vấn đề:** Ảnh vùng periocular sau khi crop có kích thước nhỏ (khoảng 160×70 pixel). Khi resize lên 160×160 để đưa vào FaceNet, ảnh bị mờ do nội suy (interpolation). Thông tin cạnh mắt, lông mày bị nhòa → embedding kém đặc biệt hóa.

**Giải pháp:** Trước khi resize, áp dụng **FFT lọc thông cao** để khuếch đại tần số cao (cạnh, chi tiết), bù lại phần bị mất khi resize.

```python
import numpy as np

def fft_high_pass(img_gray, cutoff=30):
    """Làm nổi cạnh viền mắt/lông mày bằng lọc thông cao trong miền tần số."""
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)          # đưa tần số 0 (DC) vào giữa phổ

    rows, cols = img_gray.shape
    crow, ccol = rows // 2, cols // 2

    # Mặt nạ: bỏ vùng tần số thấp ở trung tâm (bán kính = cutoff)
    mask = np.ones((rows, cols), np.uint8)
    mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 0

    fshift_filtered = fshift * mask
    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_filtered)))
    return img_back.astype(np.float32)

# Dùng sau crop periocular, trước khi resize + đưa vào FaceNet
enhanced = fft_high_pass(periocular_gray, cutoff=25)
```

---

### 4.7 FFT lọc thông thấp — Khử nhiễu ảnh JPEG kém chất lượng

**Vấn đề:** Ảnh tải từ internet hoặc camera an ninh thường được nén JPEG mạnh tạo ra nhiễu dạng block artifact (ô vuông). Nhiễu này là tần số cao ngẫu nhiên, không liên quan đến đặc trưng khuôn mặt, nhưng ảnh hưởng đến embedding của FaceNet.

**Giải pháp:** FFT lọc thông thấp giữ lại tần số thấp (hình dạng tổng thể, màu sắc da, vị trí mắt) và loại bỏ tần số cao ngẫu nhiên từ artifact JPEG. Khác với Gaussian blur vì có thể kiểm soát chính xác ngưỡng tần số cắt.

```python
def fft_low_pass(img_gray, cutoff=50):
    """Khử JPEG artifact bằng lọc thông thấp trong miền tần số."""
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)

    rows, cols = img_gray.shape
    crow, ccol = rows // 2, cols // 2

    # Mặt nạ: chỉ giữ vùng tần số thấp ở trung tâm
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 1

    fshift_filtered = fshift * mask
    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_filtered)))
    return img_back.astype(np.float32)
```

---

### 4.8 FFT magnitude spectrum — Feature phụ cho mask classifier

**Vấn đề:** MobileNetV2 nhìn vào màu sắc và hình dạng để phân loại mask/no\_mask. Nhưng khẩu trang màu da hoặc in họa tiết khuôn mặt có thể đánh lừa mô hình nếu chỉ dựa vào màu.

**Giải pháp:** Vải khẩu trang có **texture tuần hoàn** (sợi vải, lớp lọc) tạo ra các đỉnh năng lượng đặc trưng trong phổ FFT — khác hẳn với da mặt vốn là bề mặt không đồng đều. FFT magnitude spectrum được trích xuất làm **feature bổ sung** ghép vào vector đặc trưng của MobileNetV2.

```python
def fft_texture_feature(img_gray):
    """Phổ biên độ FFT log-scale — phân biệt texture vải vs da mặt."""
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)   # log scale tránh giá trị quá lớn
    # Resize về kích thước nhỏ cố định để làm feature vector
    magnitude_resized = cv2.resize(magnitude, (32, 32))
    return magnitude_resized.flatten()             # vector 1024 chiều
```

---

### 4.9 MTCNN — Phát hiện khuôn mặt + Facial Landmarks

**Vấn đề:** Haar Cascade (phương pháp cổ điển) bị nhiễu khi người đeo khẩu trang vì pattern học từ khuôn mặt đầy đủ. MTCNN dùng 3 mạng CNN tầng (P-Net → R-Net → O-Net) xử lý từ thô đến tinh, trả về **5 facial landmarks** (mắt trái, mắt phải, mũi, miệng trái, miệng phải) — landmark chính là cơ sở để crop vùng periocular chính xác.

```python
from facenet_pytorch import MTCNN
mtcnn = MTCNN(keep_all=True, post_process=False)

# boxes: list bounding box [x1, y1, x2, y2]
# probs: confidence score
# landmarks: [[left_eye, right_eye, nose, mouth_left, mouth_right], ...]
boxes, probs, landmarks = mtcnn.detect(pil_img, landmarks=True)
```

---

### 4.10 MobileNetV2 — Phân loại mask/no\_mask

**Vấn đề:** Cần phân loại nhanh ảnh crop khuôn mặt thành `mask` / `no_mask`, chạy được trong thời gian thực trên CPU phổ thông.

**Giải pháp:** MobileNetV2 pretrained trên ImageNet được **fine-tune** (transfer learning) bằng cách thay classification head cuối thành 2-class output. Toàn bộ backbone giữ nguyên weights ImageNet ở epoch đầu, sau đó unfreeze dần để tinh chỉnh.

```
Input 224×224 → MobileNetV2 backbone (frozen) → Global Avg Pool
→ Dropout(0.2) → Linear(1280 → 2) → Softmax → [mask, no_mask]
```

---

### 4.11 FaceNet + SVM — Nhận diện danh tính

**Vấn đề:** Nhận diện danh tính khi đeo khẩu trang — chỉ còn vùng mắt.

**Giải pháp:**
- **FaceNet (InceptionResnetV1)** pretrained trên VGGFace2 tạo embedding 512 chiều. Embedding là vector đặc trưng có tính chất: cùng người → khoảng cách cosine nhỏ, khác người → khoảng cách lớn.
- **SVM** dùng embedding làm input, phân loại danh tính. Ưu điểm: thêm người mới chỉ cần enroll embedding mới và retrain SVM — **không cần retrain FaceNet**.

```
Ảnh vùng periocular (160×160)
  → CLAHE + FFT high-pass
  → FaceNet → embedding 512-D (L2-normalized)
  → SVM.predict() → "Nguyen Van A" (confidence: 0.87)
```

---

## 5. Công nghệ và công cụ

| Lớp | Công nghệ | Vai trò |
|---|---|---|
| **Phát hiện mặt** | MTCNN (`facenet-pytorch`) | Detect bounding box + 5 facial landmarks |
| **Phân loại mask** | MobileNetV2 (PyTorch) | Transfer learning, 2-class head (mask/no\_mask) |
| **Nhận diện danh tính** | FaceNet / InceptionResnetV1 | Tạo embedding 512 chiều từ khuôn mặt |
| **Classifier** | SVM / kNN (`scikit-learn`) | Phân loại danh tính từ embedding |
| **Xử lý ảnh** | OpenCV (`cv2`) | CLAHE, Canny, Sobel, Gaussian blur |
| **Biến đổi tần số** | NumPy FFT (`np.fft`) | Lọc tần số, phân tích texture |
| **Backend API** | FastAPI + Uvicorn | REST API nhận ảnh, trả kết quả JSON |
| **Frontend** | HTML5 + JavaScript | Webcam capture, hiển thị bounding box |
| **Quản lý model** | `joblib` | Lưu/load SVM classifier đã train |

### Lý do lựa chọn

- **MTCNN thay vì Haar Cascade**: Haar cascade tạo ra quá nhiều false positive với ảnh có khẩu trang (nhầm pattern vải thành khuôn mặt). MTCNN dùng deep learning, chính xác hơn và trả về landmarks.
- **MobileNetV2 thay vì ResNet lớn**: Nhẹ, phù hợp inference real-time, đã pretrained trên ImageNet → transfer learning hiệu quả với dataset nhỏ.
- **FaceNet + SVM thay vì end-to-end**: FaceNet cho embedding chất lượng cao; SVM cho phép thêm người mới vào hệ thống mà **không cần retrain toàn bộ mạng**.
- **FFT (NumPy)**: Tích hợp trực tiếp vào pipeline Python, không cần thư viện ngoài, phù hợp cho tiền xử lý batch.

---

## 6. Kiến trúc hệ thống

```
src/
├── backend/
│   └── app.py                  # FastAPI: /predict, /enroll, /health
├── frontend/
│   ├── index.html              # Giao diện webcam + upload
│   └── main.js                 # Gửi ảnh, vẽ bounding box + identity
├── inference/
│   └── pipeline.py             # Pipeline chính: detect → classify → recognize
├── models/
│   ├── mask_classifier.py      # MobileNetV2 mask/no_mask
│   ├── recognizer.py           # FaceNet embedding + SVM matching
│   └── train_mask_detector.py  # Script training mask classifier
└── preprocessing/
    ├── fft_utils.py            # Hàm lọc tần số FFT
    └── image_utils.py          # OpenCV CLAHE, Canny, Sobel

data/
├── raw_samples/                # Ảnh mẫu test nhanh
├── raw_full/                   # Dataset đầy đủ (tải thủ công)
├── processed/                  # Ảnh đã crop + resize 160x160
├── preprocess.py               # MTCNN-based face crop pipeline
└── download_datasets.py        # Helper tải dataset mẫu

scripts/
├── enroll.py                   # Enroll khuôn mặt mới vào hệ thống
├── train_mask_classifier.py    # Train full MobileNetV2
├── train_recognizer.py         # Build SVM classifier từ embeddings
└── smoke_predict.py            # Smoke test gọi API /predict

models/                         # Weights đã train (không commit lên git)
├── mask_clf.pth                # MobileNetV2 fine-tuned
└── recognizer.joblib           # SVM + LabelEncoder
```

---

## 7. Giải pháp chi tiết

### 7.1 Tiền xử lý ảnh

Trước khi đưa vào detector, mỗi frame ảnh được xử lý:

1. **Resize** về kích thước chuẩn (nếu quá lớn) để giảm thời gian inference
2. **CLAHE** — cân bằng histogram thích nghi, cải thiện độ tương phản vùng mắt trong điều kiện ánh sáng yếu
3. **Gaussian blur nhẹ** (kernel 3×3) — khử noise salt-and-pepper từ webcam rẻ
4. **Chuẩn hóa pixel** về `[0, 1]` hoặc `[-1, 1]` tùy mô hình

### 7.2 Phân tích FFT

Ảnh grayscale của vùng khuôn mặt được biến đổi FFT để:
- **Lọc thông cao**: làm nổi bật cạnh viền mắt, lông mày → tăng chất lượng periocular embedding
- **Phân tích texture**: vải khẩu trang tạo pattern tần số cao đặc trưng, dùng làm feature bổ sung cho mask classifier

### 7.3 Chiến lược nhận diện khi có khẩu trang

```
Phát hiện mặt
     │
     ├── no_mask → Dùng toàn bộ khuôn mặt (160×160) → FaceNet embedding
     │
     └── mask    → Crop periocular (top 45% của bounding box)
                   → FFT high-pass enhancement
                   → Resize về 160×160
                   → FaceNet embedding (periocular-only)
```

Vùng periocular chứa **mắt, lông mày, trán** — các đặc trưng sinh trắc học ổn định và ít bị ảnh hưởng bởi khẩu trang.

### 7.4 Enrollment (Đăng ký khuôn mặt mới)

```bash
python scripts/enroll.py --name "Nguyen Van A" --images data/person_a/
```

Script thực hiện:
1. Detect + crop khuôn mặt từ tất cả ảnh đầu vào
2. Tạo embedding FaceNet cho từng ảnh
3. Thêm embeddings + label vào dataset
4. Retrain SVM classifier
5. Lưu `models/recognizer.joblib`

### 7.5 Tăng cường dữ liệu (Data Augmentation)

Để mô hình mask classifier robust hơn, dataset được augment:
- **Synthetic mask**: gắn ảnh khẩu trang 2D lên khuôn mặt không đeo mask (dùng facial landmarks)
- **Flip ngang**, thay đổi độ sáng, xoay nhẹ ±15°
- **Thêm nhiễu Gaussian** mô phỏng camera chất lượng thấp

---

## 8. Dataset

| Dataset | Nội dung | Link |
|---|---|---|
| **MaskedFace-Net** | ~100k ảnh mặt đeo mask (synthetic) | [GitHub](https://github.com/aqeelanwar/MaskedFace-Net) |
| **MAFA** | Ảnh mặt bị che một phần (mask, tay, vật thể) | [Link](http://www.escience.cn/people/JunweiHan/MAFA.html) |
| **LFW** (Labeled Faces in the Wild) | Ảnh mặt không mask, dùng làm negative class | [Link](http://vis-www.cs.umass.edu/lfw/) |
| **Custom** | Ảnh tự thu thập cho enrollment | Tự chụp |

Tải dataset mẫu nhanh để test:

```bash
python data/download_datasets.py --sample --out data/raw_samples
```

---

## 9. Cài đặt và chạy thử

### 9.1 Cài đặt môi trường

```bash
# Tạo virtualenv (khuyến nghị Python 3.10–3.12)
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows

# Cài dependencies
pip install -r requirements.txt
```

### 9.2 Chạy backend

```bash
uvicorn src.backend.app:app --reload --port 8000
```

### 9.3 Chạy frontend

```bash
cd src/frontend
python -m http.server 8080
# Mở trình duyệt: http://localhost:8080
```

### 9.4 Test API bằng script

```bash
# Smoke test với một ảnh
python scripts/smoke_predict.py --file data/raw_samples/sample_1.jpg
```

### 9.5 Train mask classifier (sau khi có dataset)

```bash
python scripts/train_mask_classifier.py \
  --data data/processed \
  --epochs 20 \
  --out models/mask_clf.pth
```

### 9.6 Enroll khuôn mặt mới

```bash
python scripts/enroll.py --name "Ten Nguoi Dung" --images path/to/face_images/
```

---

## 10. Kết quả kỳ vọng

| Chỉ số | Không đeo mask | Có đeo mask |
|---|---|---|
| Tỉ lệ phát hiện khuôn mặt (MTCNN) | ≥ 95% | ≥ 88% |
| Độ chính xác phân loại mask | — | ≥ 92% |
| Độ chính xác nhận diện danh tính | ≥ 90% | ≥ 75% (periocular) |
| Thời gian inference / frame | < 200 ms | < 250 ms |

*Kết quả thực tế sẽ được cập nhật sau khi train và đánh giá trên test set.*

---

## 11. Hạn chế và hướng phát triển

**Hạn chế hiện tại:**
- FaceNet được pretrain trên ảnh mặt đầy đủ (VGGFace2), periocular-only làm giảm chất lượng embedding
- Chưa xử lý trường hợp đeo kính cùng khẩu trang (che gần như toàn bộ mặt)
- Chưa có luồng video real-time liên tục (chỉ capture theo yêu cầu)

**Hướng phát triển:**
- Fine-tune FaceNet trên ảnh periocular để cải thiện nhận diện qua mask
- Tích hợp nhận diện mống mắt (iris recognition) như feature bổ sung
- Thêm stream video real-time với WebSocket
- Triển khai mô hình nhẹ hơn (MobileNet embedding) để chạy trên thiết bị edge

---

## Tài liệu tham khảo

- Schroff et al., *FaceNet: A Unified Embedding for Face Recognition and Clustering*, CVPR 2015
- Sandler et al., *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, CVPR 2018
- Zhang et al., *Joint Face Detection and Alignment using Multi-task Cascaded CNNs*, 2016 (MTCNN)
- Neto et al., *Beyond the Visible: A Survey on Cross-Spectral Face Recognition*, 2022
- Gonzalez & Woods, *Digital Image Processing*, 4th Edition (FFT, spatial filtering)
