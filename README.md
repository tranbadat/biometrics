# Hệ thống Nhận diện Khuôn mặt có Đeo Khẩu trang

> **Đề tài Bài tập lớn — Nhận dạng và Xử lý ảnh / Sinh trắc học**
>
> Pipeline 3 stage: **RetinaFace** detect → **YOLOv8n-cls** classify mask → **ArcFace** embedding + **Dual-slot DB** match identity. Kết hợp xử lý ảnh cổ điển (CLAHE, FFT, Sobel) với deep learning.

---

## Mục lục

1. [Bài toán & mục tiêu](#1-bài-toán--mục-tiêu)
2. [Kiến trúc pipeline](#2-kiến-trúc-pipeline)
3. [Mô hình & dữ liệu](#3-mô-hình--dữ-liệu)
4. [Tổ chức lưu trữ](#4-tổ-chức-lưu-trữ)
5. [Cài đặt](#5-cài-đặt)
6. [Cách chạy](#6-cách-chạy)
7. [Cấu trúc thư mục](#7-cấu-trúc-thư-mục)
8. [Tài liệu chi tiết](#8-tài-liệu-chi-tiết)

---

## 1. Bài toán & mục tiêu

### Bài toán
Nhận diện danh tính (1:N identification) qua webcam khi người dùng có thể **đeo khẩu trang** — tình huống ~60% diện tích khuôn mặt bị che. Đồng thời báo cáo **trạng thái mask** (đeo / không đeo).

### Mục tiêu cụ thể
1. **Detect khuôn mặt** trong ảnh / frame webcam.
2. **Phân loại mask**: `with_mask` / `without_mask`.
3. **Định danh** user qua embedding 512-D, hoạt động cho cả 2 trạng thái mask.
4. **Mask label tham gia recognition**: routing slot DB theo trạng thái — không chỉ là output phụ.

---

## 2. Kiến trúc pipeline

```
Ảnh đầu vào (webcam / upload)
        │
        ▼
   CLAHE kênh L (LAB)            ← chống loá / ngược sáng
        │
        ▼
┌──────────────────────────────┐
│ InsightFace buffalo_l        │   RetinaFace + ArcFace ResNet-50
│  - bbox, det_score           │   embedding 512-D L2-normalized
│  - normed_embedding 512-D    │
└──────────────┬───────────────┘
               │
               ▼  Crop face + expand 10% sides + 25% xuống dưới
┌──────────────────────────────┐
│ YOLOv8n-cls (custom-trained) │   2 lớp: with_mask / without_mask
│ → mask_label                  │   Train trên Face Mask 12k
└──────────────┬───────────────┘
               │
               ▼  mask_label dùng làm điều kiện routing
┌──────────────────────────────────────────┐
│ DB arcface_db.npz — DUAL SLOT             │
│   key  = "{user_id}__{mask_label}"        │
│   value = embedding 512-D                 │
│                                            │
│ Cosine match CHỈ trong slot cùng trạng    │
│ thái mask. Threshold 0.35.                │
│ Fallback: nếu slot rỗng → match toàn DB.  │
└──────────────┬───────────────────────────┘
               │
               ▼  Lookup user_id → display_name
┌──────────────────────────────────────────┐
│ arcface_names.json                       │
│   { "NV20261": "Nguyễn Văn A", ... }     │
└──────────────┬───────────────────────────┘
               │
               ▼
   JSON response: bbox + mask_label + identity (id + tên)
                  + confidences
```

### Vì sao tách dual-slot?
- Embedding ArcFace của cùng 1 người ở 2 trạng thái mask vẫn lệch ~0.4–0.5 cosine.
- Tách 2 slot độc lập (`alice__with_mask`, `alice__without_mask`) → mỗi cụm chặt → phân biệt giữa người tốt hơn.
- Mask classifier có vai trò routing thực sự, không chỉ là output phụ.

> Phân tích chi tiết, đánh giá ưu/nhược điểm, so sánh với các phương án khác: xem `docs/report-explain.md` mục 8.

---

## 3. Mô hình & dữ liệu

### Mô hình

| Mô hình | Bước trong flow | Mục đích cụ thể | Input → Output | Trạng thái | Nguồn |
|---|---|---|---|---|---|
| **RetinaFace** (`buffalo_l`) | **Stage 1** — ngay sau CLAHE | Phát hiện vị trí khuôn mặt trong ảnh, trả bbox + det_score + 5 landmark | Ảnh BGR → list bbox `[x1,y1,x2,y2]` | Pretrained, dùng nguyên (không fine-tune) | InsightFace, train trên WIDER FACE (32k ảnh) |
| **ArcFace ResNet-50** (`buffalo_l`) | **Stage 3** — chạy cùng RetinaFace trong 1 forward pass của `app.get()` | Trích xuất **vector đặc trưng 512-D** đại diện danh tính của khuôn mặt; vector này dùng để so cosine với DB | Crop face (ngầm align) → embedding `[512]` đã L2-normalize | Pretrained, dùng nguyên (không fine-tune) | InsightFace, train trên Glint360k (~17M ảnh, ~360k identities) |
| **YOLOv8n-cls** | **Stage 2** — sau khi crop face từ bbox của RetinaFace | Phân loại trạng thái khẩu trang trên crop face, output dùng làm **routing key** cho dual-slot DB | Crop face BGR → label `with_mask` / `without_mask` + confidence | **Tự train** trên Face Mask 12k | Khởi tạo từ Ultralytics ImageNet weights |

**Ghi chú quan trọng**:
- RetinaFace và ArcFace nằm chung trong gói `buffalo_l` của InsightFace → khi gọi `app.get(bgr)` **cả 2 chạy cùng lúc**, mỗi face object đã có sẵn `bbox` (từ RetinaFace) và `normed_embedding` (từ ArcFace).
- YOLOv8n-cls chạy **độc lập, sau** khi RetinaFace đã cho bbox — nhận crop face làm input, không thấy ảnh gốc.
- Cả 3 mô hình **không train nội bộ về danh tính** — recognition hoạt động ở mức **open-set**: cosine similarity giữa embedding test với embedding đã enroll trong DB. Thêm user mới chỉ cần thêm vector vào `arcface_db.npz`, không retrain.

### Augmentation khi train YOLOv8n-cls (mask classifier)

Để mask classifier robust với điều kiện thực tế (ánh sáng kém, xoay đầu, mask bị che một phần), training áp dụng augmentation **mạnh** trong `scripts/train_yolo_mask.py`:

| Loại augmentation | Tham số | Mô phỏng tình huống |
|---|---|---|
| **HSV jitter** | `hsv_h=0.015`, `hsv_s=0.7`, `hsv_v=0.6` | Ánh sáng yếu / chói / đèn vàng / ngược sáng (tăng mạnh value) |
| **Rotation** | `degrees=10` | Đầu nghiêng ±10° |
| **Translate** | `translate=0.1` | Mặt lệch tâm khung hình ±10% |
| **Scale** | `scale=0.3` | Khoảng cách camera khác nhau (zoom in/out 30%) |
| **Horizontal flip** | `fliplr=0.5` | 50% ảnh được lật ngang (mặt đối xứng nên hợp lệ) |
| **Random erasing** | `erasing=0.2` | 20% ảnh bị xoá ngẫu nhiên 1 patch — **giả lập occlusion** (tay che, vật thể chắn) |
| **Dropout** | `dropout=0.1` | Regularization, chống overfit |
| **Weight decay** | `5e-4` | Regularization L2 |
| **Early stopping** | `patience=10` | Dừng sớm nếu val loss không giảm 10 epoch |

**Augmentation có sẵn của Ultralytics** (mặc định bật, không cần khai báo):
- Mosaic augmentation (ghép 4 ảnh thành 1 → tăng đa dạng background).
- Mixup (trộn 2 ảnh + label → smooth decision boundary).
- Auto-augment (chuỗi transform tự động).

**Đã KHÔNG dùng**:
- `degrees > 15°`: xoay quá lớn → mặt lộn ngược không có trong điều kiện thực.
- `flipud`: lật dọc → mặt ngược chiều, không xảy ra trong webcam.
- Cutout patch lớn: làm mất hẳn vùng mask → label bị đảo ngược ý nghĩa.

**Kết quả thực nghiệm trên Face Mask 12k**:
- Validation accuracy: ~99% (sau 30 epoch, early stop ~epoch 20).
- Test accuracy: ~98%.
- Robust với loá sáng (HSV-V augment mạnh).
- Robust với xoay nhẹ (degrees=10).

> Có thể chạy lại training với augmentation khác bằng cách sửa `scripts/train_yolo_mask.py` rồi `python scripts/train_yolo_mask.py`. Output weights tự copy vào `models/mask_yolov8n_cls.pt`.

### Cách ArcFace xử lý ảnh có khẩu trang

**ArcFace LUÔN trích 512-D trên TOÀN BỘ crop face** — kể cả vùng đang đeo khẩu trang. Pipeline **không** segment mask ra trước khi embed; cũng **không** crop riêng vùng mắt (periocular).

```
Input  : crop face 112×112 (đã align bằng 5 landmark từ RetinaFace)
         ─ chứa cả vùng mắt + mũi + miệng (hoặc khẩu trang nếu có)
ArcFace: ResNet-50 → mọi pixel đi qua convolution như nhau
Output : vector 512-D L2-normalized
```

**Tại sao vẫn nhận ra được khi đeo mask?**
1. **Training data đa dạng**: Glint360k chứa ảnh có occlusion (kính, tay che mặt, mask một phần) → mạng đã ngầm học cách **giảm trọng số đặc trưng** từ vùng dễ bị che, **tăng trọng số** từ vùng ổn định (mắt, lông mày, trán, hình dạng đầu).
2. **Convolution có receptive field rộng**: lớp sâu của ResNet-50 nhìn cả vùng lớn → nếu vùng cằm/miệng bị che, các neuron vẫn rút được đặc trưng từ phần còn lại.
3. **Loss ArcFace ép cụm chặt**: train với additive angular margin → embedding cùng người (kể cả ở trạng thái khác nhau) bị kéo về gần nhau trên hypersphere, **ngay cả khi 60% mặt bị che**.

**Hệ quả thực tế** (giá trị cosine similarity giữa 2 embedding):

| Trường hợp | Cosine ≈ | Ghi chú |
|---|---|---|
| Cùng người, cùng trạng thái mask | **0.65–0.80** | Cụm rất chặt, dễ phân biệt |
| **Cùng người, khác trạng thái mask** | **0.40–0.50** | **"Drift"** — vẫn vượt threshold 0.35 nên match đúng, nhưng kém chặt |
| Khác người, cả 2 không mask | 0.10–0.25 | Phân tách rõ |
| Khác người, cả 2 đeo mask | 0.20–0.35 | Gần threshold → dễ false accept |

**Cách hiểu drift**: tưởng tượng embedding 512-D nằm trên mặt cầu đơn vị. Embedding của Alice "đeo mask" và Alice "không mask" là **2 điểm khác nhau** trên cầu, lệch một góc tương đương cosine ~0.45. Cùng người mà lệch ~60° góc → đáng kể, đủ để khiến **centroid trộn chung** (single-slot) trở thành vector "ở giữa", không đại diện tốt cho cụm nào.

→ Đây là cơ sở cho **dual-slot DB**: tách 2 cụm riêng → mỗi cụm chặt → cosine intra-class trong slot tăng từ ~0.45 lên ~0.65 → biên phân biệt với người khác rộng hơn.

**Hệ quả phụ — vùng khẩu trang vẫn ảnh hưởng**:
- Mask màu/họa tiết/loại khác nhau **vẫn tạo ra noise** trong embedding (vì pixel mask vẫn đi qua convolution).
- Đó là lý do thực nghiệm thấy: cùng 1 người đeo 2 loại mask khác nhau → cosine có thể giảm xuống ~0.5; ngược lại 2 người khác nhau cùng đeo mask đen → cosine có thể tăng lên ~0.3 (vẫn dưới threshold nhưng gần hơn so với khi không mask).

> **Phương án thay thế đã cân nhắc và bỏ qua**: crop riêng vùng periocular (mắt + trán) trước khi embed. Bỏ qua vì: (i) ArcFace pretrained trên ảnh full-face, embed periocular crop sẽ giảm chất lượng đáng kể; (ii) cần train periocular model chuyên dụng, vượt phạm vi đề tài. Xem `docs/report-explain.md` mục 8 (phương án D) để biết thêm.

### Vì sao KHÔNG cắt vùng khẩu trang trước khi embed?

Câu hỏi tự nhiên: "ArcFace dù sao cũng bị nhiễu bởi pixel khẩu trang — tại sao không tô đen / cắt vùng đó đi rồi mới embed?". Đã cân nhắc và **quyết định không làm**, vì:

#### 1. ArcFace expect input đã được train phân phối — tô đen là **out-of-distribution**
- ArcFace train trên crop 112×112 với khuôn mặt **tự nhiên** (kể cả có mask thật). Network học cách rút đặc trưng từ phân phối ảnh đó.
- Nếu thay vùng khẩu trang bằng **pixel đen / xám / blur**, đó là phân phối **chưa từng thấy** trong training → activation các lớp conv tạo ra pattern lạ → embedding có thể **xấu hơn** so với để nguyên mask.
- Glint360k đã có ảnh đeo mask thật → ArcFace ngầm học cách xử lý mask vải → **giữ nguyên mask hoạt động tốt hơn tô đen**.

#### 2. Vùng khẩu trang KHÔNG hoàn toàn vô dụng — nó là **tín hiệu occlusion**
- Network sâu nhận biết "khu vực này là vải mask" → tự ngầm **giảm trọng số** đặc trưng từ đó.
- Nếu xoá hẳn → mất luôn tín hiệu này → network không biết phân biệt "đang occluded" vs "đang là một bộ phận khuôn mặt khác thường".

#### 3. Cắt mask cần **segmentation chính xác** — thêm điểm fail
- Biên khẩu trang không phải đường thẳng (có dây đeo, gấp vải, bóng đổ).
- Cần thêm 1 model segmentation (vd U²-Net hoặc Mask R-CNN) — nặng, chậm, bias mới.
- Mỗi 1 pixel segment sai → 1 vùng nhỏ bị tô nhầm → embedding shift theo cách không kiểm soát.

#### 4. Phương án "crop chỉ vùng periocular" cũng có vấn đề riêng
- Mất bố cục không gian ArcFace expect (mắt nằm ở vị trí cố định trên 112×112) → resize crop periocular lên 112×112 sẽ kéo giãn → embedding kém chất lượng.
- Cần **fine-tune ArcFace** trên ảnh periocular — phải có dataset chuyên dụng + GPU + thời gian.
- Thực nghiệm trong nghiên cứu (Neto et al. 2022): periocular embedding **cần model riêng**, dùng ArcFace full-face cho periocular crop **giảm 10–15% accuracy**.

#### 5. Dual-slot DB là cách **hợp lý hơn** để giải quyết drift
Thay vì can thiệp vào embedding bên trong, ta chấp nhận drift mask vs no_mask, và **xử lý ở cấp DB**:
- Lưu 2 centroid riêng cho mỗi user → mỗi cụm chặt → cosine intra-class cao.
- Không can thiệp vào ArcFace → giữ nguyên chất lượng pretrained.
- Không cần segmentation → đơn giản hơn, ít điểm fail.

#### Tóm tắt trade-off

| Phương án | Ưu | Nhược | Quyết định |
|---|---|---|---|
| Tô đen vùng mask | Trực giác hợp lý | Out-of-distribution → embedding xấu hơn | ❌ Bỏ |
| Inpaint (GAN tô lại) | Có thể "khôi phục" mặt | Hallucinate đặc trưng giả → identity sai | ❌ Bỏ |
| Crop periocular | Tránh hẳn vùng mask | Cần fine-tune ArcFace, mất bố cục | ❌ Bỏ |
| Multi-region + fusion | SOTA học thuật | Phức tạp, cần train model mới | ❌ Bỏ (vượt phạm vi) |
| **Giữ nguyên + Dual-slot DB** | **Đơn giản, không can thiệp model, tận dụng pretrained** | Cosine giữa "cùng người, khác trạng thái mask" chỉ còn ~0.4–0.5 (so với ~0.7 khi cùng trạng thái) | ✅ **Chọn** |

> **Kết luận**: cách "thông minh" hơn là **để pixel mask đi qua ArcFace như nó vốn được train**, rồi xử lý drift ở **cấp database** (dual-slot) thay vì cấp **input** (cắt/tô đen).

**Lý do không train ArcFace**: pretrained Glint360k đã đủ tổng quát cho open-set recognition; train mới cần GPU lớn, dataset triệu ảnh — vượt tầm đồ án. Thay vào đó tập trung vào **kiến trúc pipeline** + **dual-slot DB** + **xử lý ảnh cổ điển**.

### Dữ liệu

| Loại | Mục đích | Khối lượng |
|---|---|---|
| Glint360k (pretrained) | ArcFace embedding | 17M ảnh, không tải ảnh gốc |
| WIDER FACE (pretrained) | RetinaFace detect | 32k ảnh, không tải ảnh gốc |
| **Face Mask 12k** (Kaggle) | Train YOLOv8n-cls | ~12k ảnh, 2 lớp WithMask/WithoutMask |
| **Custom enrollment** | DB người dùng dự án | 5–10 ảnh / user × 2 trạng thái |
| **MFR2** (benchmark) | Đánh giá dual-slot vs single-slot | 53 identities, 269 ảnh |

### Kỹ thuật xử lý ảnh

| Kỹ thuật | Vị trí | Mục đích |
|---|---|---|
| CLAHE (LAB-L) | Trước detect | Cân bằng sáng cục bộ |
| Bbox expansion | Sau detect, trước YOLO | Bao đủ vùng cằm/khẩu trang |
| FFT high-pass | Tiền xử lý periocular | Khuếch đại cạnh (tuỳ chọn) |
| FFT low-pass | Tiền xử lý dataset | Khử JPEG artifact |
| FFT magnitude | Feature texture | Phân biệt vải vs da |
| Sobel / Laplacian variance | Lọc dataset train | Loại ảnh mờ |

---

## 4. Tổ chức lưu trữ

```
2D-Regonization-Mask/
├── data/
│   └── faces/                          ← ảnh thô enroll
│       └── {user_id}/frame_*.jpg
└── models/
    ├── arcface_db.npz                  ← embeddings (binary, key={id}__{mask})
    ├── arcface_names.json              ← mapping user_id → họ tên (UTF-8 JSON)
    └── mask_yolov8n_cls.pt             ← weights YOLO mask classifier
```

### Cơ chế mapping 3 cấp

```
display_name "Nguyễn Văn A"
    ↓ arcface_names.json
user_id "NV20261"
    ↓ + mask_label
slot_key "NV20261__with_mask"
    ↓ arcface_db.npz
embedding [512] float32
```

### Quản trị
| Thao tác | Cách làm |
|---|---|
| Thêm user | POST `/enroll` với `name` + `user_id` + ảnh |
| Sửa họ tên | Edit `models/arcface_names.json` → restart |
| Reset DB | `rm models/arcface_db.npz models/arcface_names.json && rm -rf data/faces/` |
| Xem danh sách | GET `/health` |

---

## 5. Cài đặt

```bash
# Tạo virtualenv (Python 3.10–3.14)
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows

# Cài dependencies
pip install -r requirements.txt
```

**Lần đầu chạy** InsightFace sẽ tự tải `buffalo_l` (~280 MB) về `~/.insightface/models/`.

---

## 6. Cách chạy

### 6.1 Backend
```bash
uvicorn src.backend.app:app --reload --port 8000
```

### 6.2 Frontend
```bash
cd src/frontend && python -m http.server 8080
# Mở http://localhost:8080
```

### 6.3 Test API
```bash
# Health check
curl http://localhost:8000/health

# Predict 1 ảnh
curl -X POST http://localhost:8000/predict -F "file=@/tmp/test.jpg" | python -m json.tool

# Enroll user mới
curl -X POST http://localhost:8000/enroll \
  -F "name=Nguyễn Văn A" \
  -F "user_id=NV20261" \
  -F "files=@photo1.jpg" \
  -F "files=@photo2.jpg"
```

> **Khuyến nghị enroll**: cho mỗi user chụp **3–5 ảnh không mask + 3–5 ảnh đeo mask** ở các góc/sáng khác nhau để cả 2 slot đều có dữ liệu chất lượng.

### 6.4 Train mask classifier (đã có sẵn weights, chỉ chạy nếu muốn re-train)
```bash
python scripts/train_yolo_mask.py
```
Output: `models/mask_yolov8n_cls.pt`

### 6.5 Benchmark dual-slot vs single-slot (trên MFR2)
```bash
python scripts/benchmark_mfr2.py
```
Output: `docs/benchmark-results.md`

### 6.6 Smoke test pipeline
```bash
python scripts/smoke_predict.py --file data/raw_samples/sample_1.jpg
```

---

## 7. Cấu trúc thư mục

```
src/
├── backend/
│   └── app.py                  # FastAPI: /health, /predict, /enroll
├── frontend/
│   ├── index.html              # Webcam capture + upload UI
│   └── main.js                 # Vẽ bbox + render kết quả
├── inference/
│   └── pipeline_yolo.py        # Pipeline chính (đang dùng)
├── models/                     # Code wrappers (legacy, tham khảo)
└── preprocessing/
    ├── fft_utils.py            # FFT high/low/magnitude
    └── image_utils.py          # CLAHE, Canny, Sobel

scripts/
├── benchmark_mfr2.py           # Đánh giá single vs dual slot
├── train_yolo_mask.py          # Train YOLOv8n-cls
├── enroll.py                   # CLI enroll (legacy, dùng /enroll API thay thế)
└── smoke_predict.py            # Smoke test

models/                         # Weights + DB (không commit)
├── mask_yolov8n_cls.pt         # YOLO mask classifier (đã train)
├── arcface_db.npz              # Dual-slot embedding DB
└── arcface_names.json          # Mapping user_id → họ tên

data/
├── faces/{user_id}/*.jpg       # Ảnh enroll
├── kaggle_raw/Face Mask Dataset/  # Dataset train mask
├── yolo_cls/                   # Format chuẩn cho YOLO
└── mfr2/mfr2/                  # Dataset benchmark (tải bằng gdown)

docs/
├── report-explain.md           # Báo cáo chi tiết kỹ thuật
├── benchmark-results.md        # Kết quả benchmark MFR2
├── pipeline-comparison.md      # So sánh pipeline cũ vs mới
└── train-guideline.md          # Hướng dẫn training
```

---

## 8. Tài liệu chi tiết

| File | Nội dung |
|---|---|
| `docs/report-explain.md` | Giải thích đầy đủ kỹ thuật, mô hình, kiến trúc, câu hỏi phản biện, FAQ |
| `docs/benchmark-results.md` | Kết quả benchmark dual-slot vs single-slot trên MFR2, phân tích trung thực |
| `docs/pipeline-comparison.md` | So sánh pipeline cũ (MTCNN+FaceNet) vs mới |
| `docs/train-guideline.md` | Hướng dẫn train YOLO mask classifier |
| `docs/bao-cao-tien-do.md` | Báo cáo tiến độ |

### Endpoint API

| Method | Path | Mô tả |
|---|---|---|
| GET | `/health` | Trạng thái + danh sách user đã enroll |
| POST | `/predict` | Nhận ảnh → trả bbox + mask + identity |
| POST | `/enroll` | Đăng ký user mới với `name` + `user_id` + ảnh |

### Schema response `/predict`
```json
{
  "predictions": [
    {
      "box": [x1, y1, x2, y2],
      "detection_score": 0.97,
      "label": "with_mask",
      "confidence": 1.00,
      "identity": "NV20261",
      "identity_name": "Nguyễn Văn A",
      "identity_confidence": 0.52
    }
  ]
}
```

---

## Hạn chế & hướng phát triển

**Hạn chế**
- Chưa có anti-spoofing (chiếu ảnh / video có thể qua mặt).
- Chưa benchmark dual-slot trên dataset webcam thực (MFR2 không phản ánh điều kiện thực — xem `benchmark-results.md`).
- DB lưu file phẳng, không scale lên >10k user.
- Threshold cosine (0.35) chọn thực nghiệm, chưa tối ưu theo EER.

**Hướng phát triển**
- Liveness detection (Silent-Face-Anti-Spoofing).
- Face alignment trước embedding cho ảnh nghiêng.
- WebSocket video stream realtime.
- Frontend wizard 2 bước cho enroll (chụp không mask → đeo mask).
- Fine-tune ArcFace trên ảnh masked nội bộ nếu thu thập đủ data.

---

## Tài liệu tham khảo

- Deng et al., *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*, CVPR 2019
- Deng et al., *RetinaFace: Single-stage Dense Face Localisation in the Wild*, CVPR 2020
- Jocher et al., *Ultralytics YOLOv8*, 2023
- InsightFace project: https://github.com/deepinsight/insightface
- MFR2 dataset (Anwar & Raychowdhury, *Masked Face Recognition for Secure Authentication*, 2020)
- Gonzalez & Woods, *Digital Image Processing*, 4th Edition (FFT, spatial filtering)
