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

## 5. Câu hỏi thường gặp

### Q1: Trước khi train thì có cần làm gì với ảnh không?

**Có — tiền xử lý ảnh là bắt buộc trước khi train, gồm 3 bước chính:**

**Bước 1 — Lọc ảnh kém chất lượng (Laplacian variance):**
Tính độ sắc nét bằng cách áp dụng bộ lọc Laplacian (phát hiện cạnh) lên ảnh rồi tính phương sai của kết quả. Ảnh sắc nét có nhiều cạnh → phương sai cao; ảnh mờ → phương sai thấp. Những ảnh có variance < ngưỡng bị loại khỏi tập train để tránh model học từ dữ liệu nhiễu.

```
Laplacian variance = Var( ∇²I ) = Var( I * [[0,1,0],[1,-4,1],[0,1,0]] )
```

**Bước 2 — Cân bằng sáng bằng CLAHE (Contrast Limited Adaptive Histogram Equalization):**
Ảnh trong dataset được chụp ở nhiều điều kiện ánh sáng khác nhau (trong nhà, ngoài trời, ban đêm). CLAHE chia ảnh thành các ô nhỏ (tile), cân bằng histogram cục bộ trong từng ô thay vì toàn ảnh, giới hạn mức tương phản tối đa (clip limit). Kết quả: vùng tối rõ hơn mà vùng sáng không bị overexpose — đặc biệt quan trọng cho vùng mắt/lông mày (vùng periocular thường tối do bóng đổ).

**Bước 3 — Augmentation (chỉ cho tập train, không cho val/test):**
Áp dụng các biến đổi ngẫu nhiên để tăng tính đa dạng, giúp model không bị overfitting:
- `RandomHorizontalFlip`: lật ngang (khuôn mặt đối xứng nên hợp lệ)
- `ColorJitter`: thay đổi độ sáng, tương phản, bão hòa màu ngẫu nhiên
- `RandomRotation(±15°)`: mô phỏng đầu nghiêng nhẹ
- `GaussianBlur`: mô phỏng ảnh hơi mờ do camera không lấy nét
- `Normalize`: chuẩn hóa về mean/std của ImageNet — bắt buộc vì dùng pretrained backbone ImageNet

---

### Q2: Sau khi train thì kết quả thế nào, đánh giá như thế nào?

**Kết quả mong đợi với dataset Face Mask 12K (5.400 mask + 5.400 no\_mask):**

| Metric | Phase 1 (cuối) | Phase 2 (cuối) | Mục tiêu |
|---|---|---|---|
| Train Accuracy | ~94–96% | ~97–98% | ≥ 95% |
| Val Accuracy | ~92–95% | ~95–97% | ≥ 93% |
| Val Loss | ~0.15 | ~0.10 | giảm đều |

**Cách đánh giá:**

1. **Accuracy trên val set** — tỷ lệ dự đoán đúng trên tập validation (20% dataset không dùng cho train). Metric này phù hợp vì 2 class cân bằng hoàn hảo (5.400 mỗi class).

2. **Confusion Matrix** — xem mô hình nhầm theo hướng nào: nhầm "có mask" thành "không mask" nguy hiểm hơn chiều ngược lại (bỏ sót người đeo mask) trong bài toán an ninh.

3. **Classification Report (Precision, Recall, F1):**
   - Precision = TP / (TP + FP): trong số dự đoán "mask", bao nhiêu % thực sự đeo mask
   - Recall = TP / (TP + FN): trong số người thực sự đeo mask, bao nhiêu % được phát hiện
   - F1 = harmonic mean của Precision và Recall

4. **Checkpoint lưu best val\_acc** — script train tự động lưu model có val\_acc cao nhất, không phải model ở epoch cuối cùng.

---

### Q3: Quá trình train thế nào? Tại sao phải chia 2 phase? Ý nghĩa của từng phase?

**Kỹ thuật gọi là "2-phase Transfer Learning" (Frozen → Fine-tune):**

**Vấn đề cốt lõi:** MobileNetV2 đã được pretrain trên ImageNet (1.2 triệu ảnh, 1.000 class) — backbone đã học được các đặc trưng rất tốt (cạnh, texture, hình dạng). Nếu ngay từ đầu train toàn bộ mạng với learning rate cao, các weight đã học được sẽ bị "quên" nhanh (catastrophic forgetting) trước khi classification head học được task mới.

**Phase 1 — "Làm nóng" classification head (Epoch 1–15):**
```
Backbone (pretrained, FROZEN) → features → [Dropout(0.3) → Linear(1280→2)] (TRAIN)
                                                                ↑
                                              chỉ train phần này, LR = 1e-3
```
- **Mục đích:** Để classification head (lớp Linear cuối) học cách sử dụng các đặc trưng sẵn có của backbone cho bài toán mask vs no\_mask — *mà không làm xáo trộn backbone*.
- Backbone bị đóng băng hoàn toàn (requires\_grad = False) → gradient không lan truyền ngược qua backbone.
- Sau phase này, classification head đã hội tụ khá tốt.

**Phase 2 — Fine-tune toàn bộ mạng (Epoch 16–25):**
```
Backbone (UNFREEZE) → features → classification head
↑                                ↑
LR = 1e-4 (nhỏ hơn 10×)         LR = 1e-4
```
- **Mục đích:** Tinh chỉnh toàn bộ mạng để backbone thích nghi với đặc điểm riêng của ảnh mặt người đeo khẩu trang — khác với ảnh ImageNet chung chung.
- Learning rate giảm 10× (từ 1e-3 xuống 1e-4) để tránh catastrophic forgetting — backbone chỉ "tinh chỉnh nhẹ", không bị overwrite hoàn toàn.
- Sau phase này, toàn bộ network tối ưu cho task cụ thể.

**Tóm lại:**

| | Phase 1 | Phase 2 |
|---|---|---|
| Backbone | Đóng băng | Mở đóng băng |
| Mục tiêu | Hội tụ head nhanh | Fine-tune toàn mạng |
| Learning rate | 1e-3 (lớn) | 1e-4 (nhỏ 10×) |
| Epoch | 1–15 | 16–25 |
| Nguy cơ | Head chưa học đủ | Forgetting nếu LR quá cao |

---

### Q4: Tại sao chọn 25 epoch? Không phải 10 hay 100?

**Căn cứ chọn 25 epoch dựa trên 3 yếu tố:**

**1. Quy mô dataset:**
Dataset có 10.800 ảnh, chia 80/20 → 8.640 ảnh train, 2.160 ảnh val. Với 1 epoch = 1 lần duyệt qua toàn bộ 8.640 ảnh, model cần đủ số epoch để hội tụ nhưng không overfit. Dataset càng nhỏ thì converge càng nhanh.

**2. Phân bổ theo phase:**
- Phase 1 cần ~10–15 epoch để classification head hội tụ từ random initialization đến giá trị hữu ích.
- Phase 2 chỉ cần ~8–10 epoch vì backbone đã tốt — chỉ fine-tune nhẹ.
- Tổng: 15 + 10 = 25 epoch là điểm cân bằng thực nghiệm phổ biến cho transfer learning.

**3. Dấu hiệu overfitting:**
- Nếu val\_accuracy dừng tăng trong khi train\_accuracy vẫn tăng → model bắt đầu overfit.
- Với MobileNetV2 và dataset ~10k ảnh, điều này thường xảy ra sau epoch 25–30.
- Script dùng `EarlyStopping` ngầm: lưu checkpoint của epoch có val\_acc cao nhất → nếu epoch 20 tốt hơn epoch 25, checkpoint của epoch 20 được giữ lại.

**4. Thực tế tính toán:**
- 25 epoch × ~8.640 ảnh / batch 32 ≈ 6.750 iteration
- Trên CPU Apple M-series: ~30 phút — chấp nhận được trong môi trường laptop sinh viên
- Nếu chọn 100 epoch: ~2 giờ CPU mà không chắc cải thiện thêm (có thể còn overfit hơn)

**Kết luận:** 25 epoch là con số kinh nghiệm phổ biến trong cộng đồng Transfer Learning cho dataset cỡ 10k với MobileNet. Nếu val\_acc đã plateau sớm hơn (ví dụ epoch 18), checkpoint tốt nhất vẫn được lưu — số epoch thực tế hữu ích có thể ít hơn.

---

### Q5: FFT dùng để làm gì trong hệ thống này? Không dùng được không?

**FFT trong hệ thống đóng vai trò tiền xử lý, không phải nhận diện chính:**

**High-pass FFT (dùng cho vùng periocular):**
Khi đeo khẩu trang, chỉ còn vùng mắt để nhận diện. Vùng này nhỏ → phải resize lên để FaceNet xử lý → quá trình resize làm mờ ảnh → mất detail lông mày, vân mắt. FFT high-pass lọc bỏ tần số thấp (thành phần màu nền), giữ lại tần số cao (cạnh, texture) → ảnh periocular sắc nét hơn sau resize.

```
FFT → dịch DC về tâm → đặt vùng tần số thấp = 0 → IFFT → ảnh chỉ còn cạnh
```

**Low-pass FFT (tùy chọn cho dataset):**
Lọc nhiễu JPEG trong ảnh training dataset — các ảnh tải từ internet thường có artifact JPEG ở tần số cao. Low-pass loại bỏ artifact này.

**Nếu không dùng FFT:**
Hệ thống vẫn chạy được — fallback về dùng ảnh gốc cho FaceNet. Độ chính xác nhận diện khi đeo mask có thể giảm ~2–5% với ảnh chụp điều kiện kém. FFT là bước nâng cao chất lượng, không phải thành phần bắt buộc.

---

### Q6: CLAHE khác Histogram Equalization thông thường ở điểm nào?

**Histogram Equalization (HE) thông thường:**
- Tính histogram toàn ảnh → map lại intensity để phân bố đều.
- **Nhược điểm:** Khi ảnh có vùng rất sáng và rất tối, HE khuếch đại nhiễu ở vùng tối và làm mất chi tiết vùng sáng (overexposure).

**CLAHE — Contrast Limited Adaptive HE:**
- Chia ảnh thành lưới ô nhỏ (ví dụ 8×8 tile).
- Cân bằng histogram **cục bộ** trong từng ô → phục hồi chi tiết vùng tối mà không ảnh hưởng vùng sáng.
- **Clip limit:** giới hạn độ dốc histogram tối đa → ngăn khuếch đại nhiễu quá mức.

**Tại sao CLAHE quan trọng với bài toán này:**
Khuôn mặt trong ánh sáng văn phòng thường có vùng trán sáng (đèn từ trên) và vùng mắt tối (bóng đổ). HE thông thường sẽ overexpose trán. CLAHE xử lý từng vùng riêng → vùng mắt rõ hơn → embedding FaceNet chất lượng hơn.

---

## 6. Khó khăn và cách xử lý

| Khó khăn | Cách xử lý |
|---|---|
| `retinaface` không cài được trên Python 3.14 | Thay bằng MTCNN (facenet-pytorch), không phụ thuộc TensorFlow |
| Dataset recognition không có sẵn cho từng cá nhân | Tự chụp ảnh thành viên nhóm, kết hợp LFW để demo |
| FaceNet pretrain trên ảnh không đeo mask → embedding kém khi có mask | Dùng chiến lược periocular crop — chỉ lấy vùng mắt để tạo embedding |
