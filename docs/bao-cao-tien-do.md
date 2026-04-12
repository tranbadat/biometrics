# Báo cáo Tiến độ — Tuần 1

**Đề tài:** Hệ thống Định danh Sinh trắc học — Nhận diện Khuôn mặt có Đeo Khẩu trang
**Ngày báo cáo:** 12/04/2026
**Giai đoạn:** Khởi động — Phân tích & Thiết kế giải pháp
**Nhóm:** Trần Bá Đạt — Ninh Đức Toàn

---

## 1. Tiến độ đã hoàn thành

### 1.1 Phân tích đề tài

- Xác định rõ bài toán: nhận diện danh tính khi người dùng **đeo khẩu trang** — phần dưới khuôn mặt bị che ~60%, các hệ thống truyền thống sụt giảm độ chính xác nghiêm trọng.
- Xác định 3 bài toán con cần giải quyết:
  1. Phát hiện khuôn mặt trong ảnh/video
  2. Phân loại trạng thái khẩu trang (mask / no\_mask)
  3. Nhận diện danh tính kể cả khi đeo khẩu trang

### 1.2 Thiết kế pipeline

Đã thiết kế pipeline xử lý hoàn chỉnh gồm 6 bước:

```
Ảnh đầu vào → Tiền xử lý → FFT → Phát hiện mặt → Phân loại mask → Nhận diện danh tính → Kết quả
```

Chi tiết kỹ thuật áp dụng tại từng bước — xem [README.md](../README.md).

### 1.3 Xây dựng codebase

Đã hoàn thành toàn bộ cấu trúc code:

| Module | File | Trạng thái |
|---|---|---|
| Tiền xử lý FFT | `src/preprocessing/fft_utils.py` | Hoàn thành |
| Tiền xử lý OpenCV | `src/preprocessing/image_utils.py` | Hoàn thành |
| Mask classifier | `src/models/mask_classifier.py` | Hoàn thành |
| Training mask | `src/models/train_mask_detector.py` | Hoàn thành |
| Face recognizer | `src/models/recognizer.py` | Hoàn thành |
| Training recognizer | `scripts/train_recognizer.py` | Hoàn thành |
| Enroll người mới | `scripts/enroll.py` | Hoàn thành |
| Inference pipeline | `src/inference/pipeline.py` | Hoàn thành |
| API Backend | `src/backend/app.py` | Hoàn thành |
| Frontend demo | `src/frontend/index.html` + `main.js` | Hoàn thành |

### 1.4 Chuẩn bị dataset

- Tải dataset mask classifier: **Face Mask 12K** (Kaggle) — 5.400 ảnh mask + 5.400 ảnh no\_mask, cân bằng hoàn hảo giữa 2 class.
- Cấu trúc `data/processed/mask/` và `data/processed/no_mask/` đã sẵn sàng để train.

### 1.5 Môi trường

- Python 3.14, PyTorch, facenet-pytorch, OpenCV, FastAPI
- Giải quyết xung đột dependency: thay `retinaface` (phụ thuộc TensorFlow, không hỗ trợ Python 3.14) bằng **MTCNN** từ `facenet-pytorch` (pure PyTorch)
- Git repository khởi tạo, remote: `git@github.com:tranbadat/biometrics.git`

---

## 2. Lý do lựa chọn giải pháp

### 2.1 MTCNN thay vì RetinaFace / Haar Cascade

**Lý do chọn MTCNN:**
- **Chạy được trên máy cá nhân không cần GPU** — MTCNN là pure PyTorch, nhẹ, inference trên CPU dưới 100ms/frame.
- RetinaFace phụ thuộc TensorFlow — không cài được trên Python 3.14 + Apple Silicon (arm64), gây xung đột dependency.
- Haar Cascade không phù hợp khi người đeo khẩu trang (pattern học từ khuôn mặt đầy đủ → nhiều false positive).
- MTCNN trả về thêm **5 facial landmarks** (vị trí mắt, mũi, miệng) — dùng để crop chính xác vùng periocular khi có khẩu trang.

### 2.2 MobileNetV2 thay vì ResNet / VGG

**Lý do chọn MobileNetV2:**
- **Train được trên laptop không có GPU** — MobileNetV2 chỉ có 3.4M tham số, train 25 epoch với 10.800 ảnh mất ~30 phút trên CPU M-series.
- ResNet-50 có 25M tham số, VGG-16 có 138M tham số — train hàng giờ đến hàng ngày trên CPU.
- Pretrained trên ImageNet — **transfer learning** hiệu quả với dataset nhỏ (~10k ảnh), không cần hàng triệu ảnh từ đầu.
- Đã được dùng rộng rãi cho bài toán binary classification trên thiết bị mobile/edge.

### 2.3 FaceNet (InceptionResnetV1) + SVM thay vì end-to-end deep learning

**Lý do chọn FaceNet + SVM:**
- **Thêm người mới không cần retrain mạng** — chỉ cần enroll vài ảnh (~20 ảnh), SVM retrain trong vài giây. End-to-end CNN sẽ phải train lại từ đầu mỗi lần thêm người.
- FaceNet pretrained trên VGGFace2 (3.3M ảnh, 9.131 người) — embedding chất lượng cao ngay cả với dataset nhỏ.
- SVM hoạt động tốt trong không gian chiều cao (512-D embedding), ít bị overfitting hơn neural network khi số mẫu nhỏ.
- Phù hợp thực tế: hệ thống chấm công / kiểm soát ra vào cần thêm/bớt người linh hoạt.

### 2.4 Chiến lược Periocular Recognition

**Lý do tập trung vùng mắt khi có khẩu trang:**
- Khi đeo khẩu trang, vùng mũi-miệng-cằm bị che → FaceNet dùng toàn bộ mặt sẽ nhận embedding méo.
- Vùng mắt + lông mày + trán (periocular) là đặc trưng sinh trắc học **ổn định và không bị ảnh hưởng** bởi khẩu trang.
- MTCNN landmarks cho biết chính xác vị trí mắt → crop periocular tự động, không cần thêm model phụ.

### 2.5 FFT và OpenCV trong tiền xử lý

**Lý do áp dụng kỹ thuật xử lý ảnh cổ điển:**
- **Yêu cầu học thuật** — đề tài yêu cầu tích hợp kiến thức xử lý ảnh (FFT, OpenCV) vào hệ thống, không chỉ dùng deep learning thuần túy.
- CLAHE cải thiện đáng kể chất lượng ảnh trong điều kiện ánh sáng yếu — thực tế quan trọng khi deploy hệ thống kiểm soát ra vào trong nhà.
- FFT high-pass làm rõ cạnh vùng mắt sau khi resize (ảnh periocular nhỏ → resize lên làm mờ) → embedding FaceNet tốt hơn.
- Lọc ảnh mờ bằng Laplacian variance giúp loại bỏ ảnh kém chất lượng khỏi dataset trước khi train.

---

## 3. Kế hoạch tuần tới

| Công việc | Ưu tiên | Ghi chú |
|---|---|---|
| Train mask classifier | Cao | Dataset đã sẵn sàng, chạy được ngay |
| Thu thập ảnh enrollment (chụp thành viên nhóm) | Cao | Cần ≥20 ảnh/người, đa dạng góc + ánh sáng |
| Train recognizer + enroll | Cao | Sau khi có ảnh |
| Chạy thử API + frontend end-to-end | Trung bình | Smoke test với webcam |
| Đánh giá kết quả (accuracy, F1, confusion matrix) | Trung bình | Xuất kết quả thực nghiệm |
| Viết báo cáo cuối | Thấp | Sau khi có kết quả thực nghiệm |

---

## 4. Phân công công việc

### Trần Bá Đạt

| Hạng mục | Chi tiết | Trạng thái |
|---|---|---|
| Thiết kế kiến trúc hệ thống | Pipeline tổng thể, lựa chọn model, công nghệ | Hoàn thành |
| Xây dựng môi trường | Cài đặt dependencies, giải quyết xung đột (retinaface → MTCNN) | Hoàn thành |
| Module tiền xử lý | `src/preprocessing/fft_utils.py`, `image_utils.py` — FFT, CLAHE, Canny, Sobel | Hoàn thành |
| Inference pipeline | `src/inference/pipeline.py` — tích hợp MTCNN + periocular crop + classifier | Hoàn thành |
| API Backend | `src/backend/app.py` — FastAPI `/predict`, `/health` | Hoàn thành |
| Training mask classifier | `src/models/train_mask_detector.py` — MobileNetV2, 2-phase training | Hoàn thành |
| Script enroll | `scripts/enroll.py` — thêm/xóa người, retrain SVM | Hoàn thành |
| Chuẩn bị dataset | Tải + sắp xếp Face Mask 12K (Kaggle), 10.800 ảnh | Hoàn thành |
| Git & quản lý code | Khởi tạo repo, `.gitignore`, remote GitHub | Hoàn thành |
| **Tuần tới** | Train mask classifier, chạy thử API end-to-end, đánh giá kết quả | Chờ |

### Ninh Đức Toàn

| Hạng mục | Chi tiết | Trạng thái |
|---|---|---|
| Nghiên cứu lý thuyết | Tổng hợp tài liệu về sinh trắc học, FaceNet, MTCNN, FFT trong xử lý ảnh | Đang làm |
| Module nhận diện danh tính | `src/models/recognizer.py` — FaceNet embedding + SVM matching | Hoàn thành |
| Training recognizer | `scripts/train_recognizer.py` — cross-validation, chọn SVM kernel | Hoàn thành |
| Thu thập ảnh enrollment | Chụp ảnh thành viên nhóm (≥20 ảnh/người, đa góc, có/không có mask) | **Cần làm** |
| Frontend demo | `src/frontend/index.html`, `main.js` — webcam, upload, vẽ bounding box | Hoàn thành |
| Viết báo cáo | Tổng hợp kết quả thực nghiệm, so sánh phương pháp, kết luận | Chờ |
| Slide thuyết trình | Thiết kế slide trình bày pipeline, kết quả, demo | Chờ |
| **Tuần tới** | Chụp ảnh enrollment, enroll thành viên, test nhận diện qua mask | Chờ |

---

### Tổng hợp phân công theo module

```
Trần Bá Đạt                         Ninh Đức Toàn
─────────────────────────────        ─────────────────────────────
Preprocessing (FFT, OpenCV)          Lý thuyết & nghiên cứu
Inference pipeline                   Frontend demo
Training mask classifier             Training recognizer
API backend                          Thu thập dataset enrollment
Môi trường & DevOps                  Báo cáo & thuyết trình
```

---

## 5. Khó khăn và cách xử lý

| Khó khăn | Cách xử lý |
|---|---|
| `retinaface` không cài được trên Python 3.14 | Thay bằng MTCNN (facenet-pytorch), không phụ thuộc TensorFlow |
| Dataset recognition không có sẵn cho từng cá nhân | Tự chụp ảnh thành viên nhóm, kết hợp LFW để demo |
| FaceNet pretrain trên ảnh không đeo mask → embedding kém khi có mask | Dùng chiến lược periocular crop — chỉ lấy vùng mắt để tạo embedding |
