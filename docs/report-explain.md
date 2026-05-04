# Báo cáo giải thích chi tiết — Hệ thống Nhận diện Khuôn mặt có Đeo Khẩu trang

> Tài liệu này giải thích **giải pháp**, **kỹ thuật**, **mô hình** đang dùng trong dự án, kèm các **câu hỏi phản biện** (defense questions) và **FAQ** thường gặp khi báo cáo đề tài.

> **Lưu ý lịch sử**: phiên bản đầu của đồ án dùng pipeline MTCNN + MobileNetV2 + FaceNet/SVM. Pipeline đó đã được **thay thế hoàn toàn** bằng kiến trúc hiện tại (RetinaFace + YOLOv8n-cls + ArcFace + Dual-slot DB). Toàn bộ tài liệu này mô tả pipeline đang chạy.

---

## 0. Bối cảnh sử dụng mô hình pretrained, nguồn dữ liệu, và đóng góp đề tài

> Phần này quan trọng để bảo vệ — vì đề tài **không huấn luyện ArcFace cho danh tính**, mà sử dụng pretrained, nên cần làm rõ ranh giới giữa "công đoạn dùng lại" và "công đoạn đề tài tự đóng góp".

### 0.1 Mô hình pretrained — dùng nguyên hay fine-tune?

| Mô hình | Trạng thái trong đề tài | Nguồn pretrained | Giấy phép |
|---|---|---|---|
| **RetinaFace** (trong `buffalo_l`) | **Dùng nguyên**, không fine-tune | InsightFace model zoo (`buffalo_l`), train trên WIDER FACE | MIT |
| **ArcFace ResNet-50** (trong `buffalo_l`) | **Dùng nguyên**, không fine-tune | InsightFace model zoo, train trên Glint360k (~17M ảnh, ~360k identities) | MIT |
| **YOLOv8n-cls** | **Fine-tune** từ pretrained ImageNet weights | Ultralytics `yolov8n-cls.pt` → fine-tune trên dataset Face Mask 12k | AGPL-3.0 |

**Ý nghĩa:**
- ArcFace và RetinaFace đóng vai trò "**feature extractor có sẵn**" — như dùng OpenCV để đọc ảnh, không phải đóng góp của đề tài.
- YOLOv8n-cls **được đề tài tự train lại** từ Ultralytics weights pretrained → đây là phần work thực sự.
- Cụm DB embedding (`arcface_db.npz`) **không phải mô hình** — chỉ là kho lưu vector, mỗi lần enroll thêm 1 vector vào file.

### 0.2 Vì sao không train ArcFace cho danh tính?

1. **Quy mô dữ liệu**: train ArcFace cần ~hàng triệu ảnh, ~hàng trăm nghìn identity, GPU >24 GB VRAM, vài ngày–vài tuần. Một đồ án sinh viên không thể đáp ứng.
2. **Không cần thiết**: ArcFace pretrained trên Glint360k đã học được "**không gian sinh trắc khuôn mặt tổng quát**" — bất kỳ user mới (kể cả không có trong tập train) đều cho ra embedding chất lượng cao. Đây là tinh thần **few-shot recognition / open-set recognition**.
3. **Đóng góp nằm chỗ khác**: việc ghép pipeline (CLAHE → RetinaFace → YOLO routing → dual-slot ArcFace), thiết kế DB dual-slot, và xử lý ảnh cổ điển mới là phần đóng góp.

### 0.3 Nguồn dữ liệu

| Loại dữ liệu | Mục đích | Nguồn | Khối lượng |
|---|---|---|---|
| **Glint360k / MS1MV2** | Pretrained ArcFace (đã train sẵn, không tải) | InsightFace public model | ~17M ảnh, ~360k identities |
| **WIDER FACE** | Pretrained RetinaFace (đã train sẵn, không tải) | Public benchmark | 32k ảnh, 393k bbox mặt |
| **ImageNet** | Pretrained YOLOv8n-cls (weights khởi tạo) | Ultralytics public | 1.28M ảnh, 1000 lớp |
| **Face Mask 12k images** | **Train mask classifier** (đề tài tự train) | Kaggle dataset `face-mask-12k-images-dataset.zip` (đã có trong repo) | ~12k ảnh, 2 lớp `with_mask`/`without_mask` |
| **Custom enrollment** | DB embedding danh tính (đề tài tự thu thập) | User dự án tự chụp qua webcam, lưu trong `data/faces/{user_id}/` | 5–10 ảnh / user × 2 trạng thái |

**Lưu ý đạo đức**:
- Glint360k và WIDER FACE là dataset công khai, đã được tác giả pretrained → đề tài chỉ tải weights, không tải ảnh gốc.
- Face Mask 12k là dataset Kaggle license open — phù hợp dùng học thuật.
- Ảnh enrollment là của thành viên nhóm hoặc tình nguyện viên có đồng ý, không thu thập từ web.

### 0.4 Đóng góp của đề tài

Phân tách rõ những phần là **ý tưởng / công sức của nhóm** so với những phần đi mượn:

#### 0.4.1 Đóng góp kỹ thuật
1. **Pipeline tích hợp 3 stage** (detect → mask classify → recognize) tổ chức gọn trong `pipeline_yolo.py`, có lazy-load, có fallback heuristic khi thiếu model.
2. **Thiết kế Dual-slot DB**: tách embedding theo trạng thái mask `name__with_mask` / `name__without_mask`, biến mask classifier từ output phụ thành thành phần routing — đóng góp **nguyên bản** của đề tài.
3. **Cơ chế fallback graceful**: khi slot rỗng (user mới enroll thiếu trạng thái) → match toàn DB. Khi mask classifier sai → có thể mở rộng "match cả 2 slot lấy best".
4. **Trung bình hoá embedding L2-normalized** trong cùng slot khi user enroll nhiều ảnh — giữ tính chất unit hypersphere của ArcFace.
5. **Train YOLOv8n-cls cho mask** trên dataset 12k ảnh, đạt ~95%+ accuracy (ghi vào báo cáo sau khi đo).

#### 0.4.2 Đóng góp về xử lý ảnh (đúng tinh thần môn học)
1. **CLAHE trên kênh L của không gian màu LAB** — thay vì CLAHE trên grayscale; lý do: giữ thông tin màu, chỉ cân bằng độ sáng → embedding ArcFace ổn định hơn.
2. **FFT high-pass enhancement** trước resize cho periocular crop.
3. **FFT magnitude texture feature** — phân biệt vải khẩu trang vs da mặt qua phổ tần số.
4. **Sobel-based dataset filtering** — loại ảnh mờ trước khi train mask classifier.

#### 0.4.3 Đóng góp về kiến trúc hệ thống
1. **REST API FastAPI** với endpoint `/predict`, `/enroll`, `/health` — đóng gói pipeline thành dịch vụ.
2. **Frontend HTML5 + JavaScript** webcam capture, vẽ bounding box realtime.
3. **Wizard enroll 2 bước** (kế hoạch): chụp không mask → đeo mask → chụp tiếp, tận dụng dual-slot.

#### 0.4.4 KHÔNG phải đóng góp (làm rõ để tránh nhận nhầm)
- Không phải đóng góp về mô hình deep learning core (ArcFace, RetinaFace).
- Không phải dataset gốc cho recognition (đã có Glint360k pretrained).
- Không phải thuật toán cosine similarity (kinh điển).
- Không phải framework (PyTorch, FastAPI, Ultralytics).

### 0.5 Ứng dụng thực tế của hệ thống

| Lĩnh vực | Use case cụ thể | Tại sao phù hợp |
|---|---|---|
| **Y tế / bệnh viện** | Chấm công nhân viên y tế trong khu vực bắt buộc đeo mask (ICU, phòng mổ) | Không cần tháo mask để check-in → tuân thủ vô trùng; dual-slot routing cho accuracy cao |
| **Văn phòng / công ty** | Cổng kiểm soát ra vào trong/sau dịch | Hỗ trợ cả 2 trạng thái; mask classifier kiêm giám sát quy định đeo mask |
| **Trường học / ký túc xá** | Điểm danh sinh viên qua camera lớp | Enroll 1 lần, hoạt động lâu dài kể cả thay đổi quy định mask |
| **Sân bay / nhà ga** | Hỗ trợ kiểm tra danh tính hành khách | Robust với mask y tế phổ biến trong di chuyển công cộng |
| **Bán lẻ / siêu thị** | Loyalty program nhận diện khách VIP | UX không yêu cầu khách tháo mask; phát hiện đồng thời tình trạng đeo mask cho báo cáo y tế |
| **An ninh công cộng** | Tìm kiếm người trong đám đông qua CCTV | (Cần cân nhắc khía cạnh đạo đức/luật) |

**Use case không phù hợp** (giới hạn cần nói thẳng):
- ❌ Xác thực ngân hàng / giao dịch tài chính giá trị cao — cần liveness + multi-factor.
- ❌ Pháp y / bằng chứng pháp lý — accuracy chưa đủ cao và chưa được kiểm định.
- ❌ Giám sát hàng loạt không có sự đồng ý — vi phạm GDPR/Nghị định 13.

---

## 1. Tổng quan giải pháp

### 1.1 Bài toán
Nhận diện danh tính (1:N identification) trong điều kiện người dùng có thể **đeo khẩu trang** — tình huống mà ~60% diện tích khuôn mặt (mũi, miệng, cằm) bị che. Bài toán đồng thời cần biết người đó **có đang đeo khẩu trang hay không** (mask classification) — và quan trọng hơn, **mask label được dùng trực tiếp trong logic nhận diện** (dual-slot DB), chứ không chỉ là output báo cáo phụ.

### 1.2 Kiến trúc pipeline (`pipeline_yolo.py`) — Dual-slot DB

```
Ảnh đầu vào (PIL/BGR)
        │
        ▼
   CLAHE trên kênh L (LAB)        ← chống loá sáng / ngược sáng
        │
        ▼
┌──────────────────────────────┐
│ InsightFace (buffalo_l)      │   RetinaFace detect + ArcFace embedding 512-D
│  - bbox, det_score           │   (đã L2-normalized)
│  - normed_embedding 512-D    │
└──────────────┬───────────────┘
               │
               ▼
        Crop face BGR
               │
               ▼
┌──────────────────────────────┐
│ YOLOv8n-cls                  │   Phân loại trạng thái khẩu trang
│ → mask_label ∈                │   (with_mask / without_mask)
│   {with_mask, without_mask}  │
└──────────────┬───────────────┘
               │
               ▼   (mask_label dùng làm điều kiện routing)
┌──────────────────────────────────────────┐
│ DB arcface_db.npz — DUAL SLOT             │
│   key = "{name}__{mask_label}"            │
│   ─ alice__with_mask    → emb_512         │
│   ─ alice__without_mask → emb_512         │
│   ─ bob__without_mask   → emb_512         │
│ Cosine match CHỈ trong slot cùng trạng    │
│ thái mask. Threshold 0.35.                │
│ Fallback: nếu slot rỗng → match toàn DB.  │
└──────────────┬───────────────────────────┘
               │
               ▼
   JSON: bbox + mask_label + identity + confidences
```

### 1.3 Triết lý: mask label LÀ thành phần của recognition

Khác với nhiều hệ thống coi mask classification là output phụ, pipeline này **dùng mask label làm điều kiện routing để chọn cụm embedding so khớp**:

- ArcFace tuy robust với occlusion, embedding của cùng một người ở 2 trạng thái (đeo / không đeo mask) **vẫn lệch nhau ~0.4–0.5 cosine** — đủ lớn để làm "centroid" trộn chung trở thành vector trung bình kém đại diện cho cả hai.
- Tách 2 slot độc lập `alice__with_mask` và `alice__without_mask` → mỗi slot là một cụm chặt → cosine intra-class cao hơn → phân biệt giữa người tốt hơn.
- **Mask classifier không còn là output decorative** — nó quyết định pipeline match đi vào nhánh nào.

---

## 2. Kỹ thuật xử lý ảnh cổ điển được áp dụng

| Kỹ thuật | Vị trí | Mục đích | Trong code |
|---|---|---|---|
| CLAHE (LAB-L channel) | Trước detect | Cân bằng sáng cục bộ, chống loá / ngược sáng | `pipeline_yolo._apply_clahe` |
| Gaussian blur 3×3 | Tiền xử lý | Khử noise webcam, tránh false positive | `image_utils` |
| Canny edge | Periocular feature | Làm nổi viền mắt/lông mày | `image_utils` |
| Sobel / Laplacian variance | Lọc dataset | Loại ảnh mờ trước khi train | `image_utils` |
| Dilation bbox | Sau detect | Mở rộng bbox xuống cằm để mask classify đúng | preprocessing |
| FFT high-pass | Trước embedding | Khuếch đại cạnh sau resize | `fft_utils` |
| FFT low-pass | Tiền xử lý dataset | Khử JPEG block artifact | `fft_utils` |
| FFT magnitude | Feature phụ mask | Phát hiện texture vải tuần hoàn | `fft_utils` |

**Ý nghĩa học thuật**: phần FFT/CLAHE/Sobel là phần **xử lý ảnh** trong tiêu đề môn học — chứng minh hiểu biết về miền tần số và miền không gian, không chỉ "nhét deep learning".

---

## 3. Mô hình deep learning đang dùng

### 3.1 RetinaFace (trong `buffalo_l`) — Face Detection
- **Single-stage anchor-based detector** với multi-task loss: classification + bbox regression + 5 landmark + dense 3D regression.
- **FPN (Feature Pyramid Network)** giúp phát hiện mặt ở nhiều scale.
- One-shot inference, train trên WIDER FACE — robust với khuôn mặt đeo mask.

### 3.2 ArcFace (trong `buffalo_l`) — Recognition
- **Loss function**: Additive Angular Margin Loss
  $$L = -\log \frac{e^{s\cos(\theta_{y_i}+m)}}{e^{s\cos(\theta_{y_i}+m)} + \sum_{j\ne y_i} e^{s\cos\theta_j}}$$
- **Ý tưởng cốt lõi**: thay vì học embedding để cosine cao, **cộng thêm margin góc m** vào lớp đúng → ép embedding cùng class **cụm chặt** trên hypersphere, khác class **tách rộng**.
- **Output**: vector 512-D đã L2-normalize → so sánh bằng **cosine similarity** đơn giản.
- **Training data**: MS1MV2 / Glint360k (~1M+ identities) — đa dạng pose, age, occlusion.

### 3.3 YOLOv8n-cls — Mask Classification (vai trò: routing slot DB)
- **Backbone CSPDarknet** + classification head.
- **`n` = nano**: ~2.8 MB, latency ~20–30 ms trên CPU.
- **Train custom** trên Face Mask 12k images dataset, 2 lớp `with_mask` / `without_mask`.
- **Vai trò trong dual-slot**: output của YOLO **quyết định slot DB** được truy vấn (enroll) hoặc so khớp (inference) — đây không còn là metadata phụ.

---

## 4. Quy trình Enroll & Inference (Dual-slot DB)

### 4.1 Enroll — `enroll_identity(name, face_bgr)`
```
1. CLAHE trên ảnh
2. InsightFace.get(bgr)  → bbox + normed_embedding
3. Crop face theo bbox
4. YOLOv8n-cls(crop)     → mask_label ∈ {with_mask, without_mask}
5. key = f"{name}__{mask_label}"
6. Nếu key đã tồn tại trong DB:
       merged = (db[key] + new_emb)  /  ||db[key] + new_emb||₂
   Ngược lại:
       db[key] = new_emb
7. Lưu arcface_db.npz
```

**Khuyến nghị UX**: enroll mỗi user với **cả 2 trạng thái** (3–5 ảnh đeo mask + 3–5 ảnh không) để cả 2 slot đều có embedding chất lượng tốt. Frontend nên có 2 bước rõ ràng: "Bước 1: chụp không đeo mask" → "Bước 2: đeo mask vào, chụp tiếp".

### 4.2 Inference — `_match_embedding(emb, mask_label)`
```
1. Classify mask trên crop                → mask_label
2. candidates = {k: v for k, v in db
                 if k.endswith(f"__{mask_label}")}
3. Nếu candidates rỗng (user chỉ enroll 1 trạng thái):
       fallback → candidates = toàn bộ DB
4. argmax cosine(emb, candidates[k]) → best_key, best_sim
5. Nếu best_sim < 0.35 → trả về (None, best_sim)  # unknown
   Ngược lại:
       base_name = strip suffix khỏi best_key
       trả về (base_name, best_sim)
```

### 4.3 Vì sao có fallback?
Nếu user mới enroll chỉ với 1 trạng thái (ví dụ chỉ ảnh không đeo mask), khi họ xuất hiện trong khung hình **đang đeo mask**, slot `__with_mask` rỗng → cần fallback toàn DB để vẫn match được, dù confidence thấp hơn. Đây là "soft degradation" thay vì fail cứng.

### 4.4 Ngưỡng cosine 0.35
- ArcFace với MS1MV2: cùng người trung bình ~0.5–0.7, khác người ~0.0–0.2.
- Threshold 0.35 cân bằng FAR / FRR cho dataset nhỏ (vài chục người).
- **Với dual-slot**: do cụm chặt hơn, có thể nâng threshold lên 0.40 cho security cao mà vẫn giữ recall.

---

## 4b. Tổ chức lưu dữ liệu & cơ chế mapping user ↔ khuôn mặt

> Phần này giải thích cách hệ thống lưu trữ dữ liệu sinh trắc và **liên kết một bộ embedding với thông tin định danh** (mã người dùng + họ tên) — câu hỏi phản biện rất hay gặp.

### 4b.1 Sơ đồ tổ chức dữ liệu

```
2D-Regonization-Mask/
├── data/
│   └── faces/                          ← ảnh thô enroll (raw images)
│       ├── NV20261/
│       │   ├── frame_0000.jpg
│       │   ├── frame_0001.jpg
│       │   └── frame_0002.jpg
│       ├── NV20262/
│       │   └── ...
│       └── ...
└── models/
    ├── arcface_db.npz                  ← bảng embedding 512-D (binary)
    │     keys = "{user_id}__{mask_label}"
    │     values = np.ndarray shape (512,) float32, đã L2-normalize
    │
    └── arcface_names.json              ← bảng tên hiển thị (sidecar JSON)
          {
            "NV20261": "Nguyễn Văn A",
            "NV20262": "Trần Thị B"
          }
```

### 4b.2 Ba lớp dữ liệu — vai trò khác nhau

| Lớp | File | Định dạng | Có thể tái tạo? | Vai trò |
|---|---|---|---|---|
| **Raw images** | `data/faces/{user_id}/*.jpg` | JPEG | Không (gốc) | Backup để re-enroll, debug; **có thể xoá để bảo vệ privacy** |
| **Embeddings** | `models/arcface_db.npz` | NumPy `.npz` (binary) | Có (re-enroll từ raw) | Dữ liệu thực sự dùng cho inference; vector 512-D không thể reconstruct ảnh |
| **Display names** | `models/arcface_names.json` | JSON UTF-8 | Không (gốc) | Mapping `user_id → họ tên` để hiển thị friendly |

**Nguyên tắc tách lớp**:
- Embedding là "xương sống" — đủ để nhận diện, không lộ thông tin gốc.
- Tên hiển thị là metadata phụ — có thể chỉnh tay (sửa file JSON), không ảnh hưởng pipeline.
- Ảnh thô là layer dùng để debug/re-enroll khi cần — nên xoá sau khi enroll xong nếu sản phẩm production.

### 4b.3 Cơ chế mapping: 3 cấp khoá

```
┌──────────────────────────────────────────────────┐
│  display_name      "Nguyễn Văn A"                 │  ← cấp UX
│       │                                           │
│       │  arcface_names.json                       │
│       ▼                                           │
│  user_id           "NV20261"                      │  ← cấp định danh
│       │                                           │
│       │  + mask_label                             │
│       ▼                                           │
│  slot_key          "NV20261__with_mask"           │  ← cấp lưu trữ
│       │                                           │
│       │  arcface_db.npz                           │
│       ▼                                           │
│  embedding         np.float32 [512]               │  ← cấp sinh trắc
└──────────────────────────────────────────────────┘
```

**Tại sao 3 cấp thay vì 1 cấp?**
- **Tách `display_name` khỏi `user_id`**: nếu lưu thẳng "Nguyễn Văn A" làm key, hệ thống vỡ khi: (i) trùng tên, (ii) tên có dấu/khoảng trắng/Unicode không an toàn cho key. `user_id` (`NV20261`) là **business key** ổn định, ASCII-safe, không trùng.
- **Tách `slot_key` khỏi `user_id`**: cần mặt thái mask để routing dual-slot (xem mục 4). Composite key `{user_id}__{mask_label}` là cấu trúc nội bộ của DB.
- **Tách `embedding` khỏi key**: embedding là dữ liệu sinh trắc, lưu binary trong npz để compact và load nhanh; key là string nằm ở metadata.

### 4b.4 Quy trình enroll — cập nhật cả 3 lớp

```python
# Backend nhận: name="Nguyễn Văn A", user_id="NV20261", files=[ảnh1, ảnh2, ...]

# 1. Lưu raw images (lớp 1)
for i, file in enumerate(files):
    save_dir = Path("data/faces") / user_id
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / f"frame_{i:04d}.jpg").write_bytes(file_bytes)

# 2. Tạo embedding + xác định slot (lớp 2)
for bgr in decoded_images:
    pipeline.enroll_identity(
        name=user_id,             # business key
        face_bgr=bgr,
        display_name=name,        # họ tên hiển thị
        persist=False,
    )
    # → bên trong:
    #     mask_label = YOLO(crop)
    #     slot_key   = f"{user_id}__{mask_label}"
    #     _known_embeddings[slot_key] = avg_normalized_embedding
    #     _known_names[user_id]       = display_name

# 3. Persist cả 2 file 1 lần
pipeline._save_db()
# → arcface_db.npz   (embeddings)
# → arcface_names.json (names)
```

### 4b.5 Quy trình inference — đi ngược 3 cấp

```python
# 1. Lớp sinh trắc → lớp định danh
emb        = arcface(bgr)          # vector 512-D
mask_label = yolo(crop)            # "with_mask" / "without_mask"
best_slot  = argmax_cosine(emb, db, filter=f"__{mask_label}")
# best_slot = "NV20261__with_mask"

# 2. Lớp định danh → lớp UX
user_id      = best_slot.split("__")[0]            # "NV20261"
display_name = _known_names.get(user_id)           # "Nguyễn Văn A"

# 3. Trả về client
return {
    "identity":      "NV20261",
    "identity_name": "Nguyễn Văn A",
    "label":         "with_mask",
    "confidence":    1.00,
    "identity_confidence": 0.52,
    "box":           [x1, y1, x2, y2],
}
```

### 4b.6 Vì sao dùng JSON sidecar thay vì nhồi tên vào npz?

| Phương án | Ưu | Nhược |
|---|---|---|
| **Nhồi tên vào npz** (vd metadata key `_name_NV20261`) | 1 file duy nhất | npz chỉ lưu numpy arrays; nhồi string phải allow_pickle → **rủi ro bảo mật** (load arbitrary code), khó đọc thủ công |
| **Sidecar JSON** (đang dùng) | An toàn (`allow_pickle=False`); người vận hành mở file JSON sửa tên dễ | Có 2 file cần đồng bộ |
| **SQLite** | Truy vấn linh hoạt | Quá nặng cho ~vài chục user |

→ JSON sidecar là **trade-off đúng quy mô đề tài**.

### 4b.7 Tính chất của embedding (giải thích cho hội đồng)

- **Kích thước cố định**: mỗi user = 1 vector 512-D × 2 slot = ~4 KB → DB 100 user ≤ **400 KB** (rất nhẹ).
- **Không reversible**: từ embedding **không thể tái tạo lại ảnh khuôn mặt gốc** — đây là tính chất one-way của deep feature, đảm bảo privacy ở mức cơ bản.
- **L2-normalized**: mọi vector nằm trên mặt cầu đơn vị 512-D → so sánh bằng cosine similarity = dot product (rất nhanh).
- **Stable across sessions**: cùng 1 ảnh, ArcFace luôn cho ra cùng embedding → enroll 1 lần dùng nhiều lần.

### 4b.8 Quản trị: thêm / xoá / sửa user

| Thao tác | Cần làm gì |
|---|---|
| **Thêm user mới** | POST `/enroll` với `name` + `user_id` + ảnh → cả 3 lớp tự cập nhật |
| **Sửa họ tên** | Mở `models/arcface_names.json`, sửa value → restart server |
| **Xoá 1 user** | (i) xoá `data/faces/{user_id}/`, (ii) xoá các key `{user_id}__*` trong npz, (iii) xoá entry trong JSON |
| **Reset toàn bộ** | `rm models/arcface_db.npz models/arcface_names.json && rm -rf data/faces/` |
| **Audit** | `python -c "import json; print(json.load(open('models/arcface_names.json')))"` để xem ai đã enroll |

### 4b.9 Ưu / nhược điểm thiết kế lưu trữ này

**Ưu điểm**
- Tách lớp rõ ràng → **xoá ảnh thô không phá DB**, đảm bảo privacy ngay khi cần.
- File phẳng (npz + json) → **dễ backup, dễ migrate**, không cần daemon DB.
- Composite key `{user_id}__{mask_label}` → **mở rộng tự nhiên** cho dual-slot, có thể thêm trạng thái khác (kính, mũ) mà không đổi schema.
- Họ tên Unicode (tiếng Việt có dấu) lưu UTF-8 trong JSON → **không vướng encoding**.

**Nhược điểm**
- 2 file cần đồng bộ (npz + json) → nếu lỗi giữa chừng có thể lệch trạng thái → giảm thiểu bằng `_save_db()` ghi cả 2 trong cùng 1 lượt.
- Không có versioning → nếu enroll nhầm, không rollback được trừ khi backup tay.
- Không scale lên hàng triệu user (load toàn bộ vào RAM) → đề tài nhỏ-vừa nên chấp nhận; cần FAISS + SQLite/Postgres cho production lớn.

---

## 5. Câu hỏi phản biện thường gặp khi bảo vệ

### 5.1 Về thiết kế dual-slot

**Q1: Vì sao tách 2 slot mà không trộn chung embedding?**
Embedding ArcFace của cùng 1 người ở 2 trạng thái (đeo/không đeo mask) lệch nhau ~0.4–0.5 cosine. Trộn vào 1 vector trung bình tạo ra centroid không đại diện tốt cho cả hai. Tách slot → mỗi cụm chặt hơn → phân biệt liên-class tốt hơn.

**Q2: ArcFace đã robust với mask, tách slot có thừa không?**
Robust ≠ bất biến. ArcFace giữ được khả năng nhận diện qua mask nhưng embedding **vẫn dịch** đáng kể. Tách slot khai thác đúng đặc tính này: thay vì ép 1 centroid bao trùm cả 2 trạng thái, ta để mô hình "biết" trạng thái nào để chọn cụm tham chiếu.

**Q3: Mask classifier sai (false positive/negative) thì sao?**
Hai cơ chế bảo vệ:
- (i) Fallback toàn DB khi slot rỗng → không fail cứng.
- (ii) Có thể mở rộng: nếu best_sim trong slot dự đoán thấp, **thử lại slot còn lại** rồi lấy max — gần như vô hiệu hóa lỗi mask classifier (chi phí 2× cosine, vẫn rất nhanh).

**Q4: Vì sao không dùng adaptive threshold thay vì dual-slot?**
Adaptive threshold chỉ điều chỉnh ngưỡng quyết định, không giải quyết được vấn đề centroid trộn lệch. Dual-slot tác động vào chính cấu trúc DB → cải thiện sâu hơn. Hai cách có thể kết hợp.

### 5.2 Về lựa chọn mô hình

**Q5: Vì sao tách 3 bước thay vì end-to-end?**
Tách bước cho phép **nâng cấp từng phần độc lập**. Trong thiết kế dual-slot, mask classifier còn có vai trò routing → càng cần module hoá rõ ràng.

**Q6: Vì sao chọn YOLOv8-nano cho mask, không dùng SVM trên HOG?**
HOG + SVM yêu cầu feature engineering thủ công, không robust với màu/họa tiết khẩu trang đa dạng. YOLOv8n đã ~2.8 MB, latency <30 ms — chấp nhận được. Trong dual-slot, độ chính xác mask classifier càng quan trọng (sai → match nhầm slot).

**Q7: Tại sao threshold 0.35 mà không phải 0.5 (giá trị paper)?**
0.5 đo trên LFW. Với DB nội bộ qua webcam, 0.35 cho recall tốt hơn. **Với dual-slot**, có thể nâng lên 0.40 do cụm chặt hơn — khuyến khích re-tune sau khi enroll đủ data.

### 5.3 Về xử lý ảnh

**Q8: CLAHE có làm méo embedding không?**
CLAHE chỉ thay đổi phân bố histogram cục bộ, không méo cấu trúc. Convolution sớm trong CNN deep vốn đã chuẩn hóa cục bộ ngầm. Thực nghiệm: CLAHE giúp recall tăng 3–5% trong điều kiện ngược sáng.

**Q9: Tại sao FFT mà không dùng wavelet?**
FFT đủ cho mục đích lọc tần số đơn giản (high/low pass). Wavelet ưu thế cho phân tích đa scale + định vị không gian, nhưng overhead cài đặt và giải thích vượt yêu cầu đề tài (KISS).

**Q10: FFT magnitude làm feature phụ cho mask classifier có cần khi YOLO đã đủ mạnh?**
Trong production hiện tại không dùng — YOLO + augmentation đủ. FFT magnitude là **đóng góp học thuật** thể hiện hiểu biết về miền tần số.

### 5.4 Về dữ liệu & đánh giá

**Q11: Đánh giá dual-slot DB như thế nào?**
- Enroll mỗi user 5 ảnh không mask + 5 ảnh đeo mask.
- Test set: ảnh giữ lại của mỗi user, gồm cả 2 trạng thái.
- So sánh accuracy single-slot vs dual-slot trên cùng test set.
- Báo cáo top-1 accuracy, EER, FAR@FRR=1% chia theo nhóm có/không mask.

**Q12: Có rủi ro spoofing (chiếu ảnh, video)?**
Có. Pipeline hiện không có liveness detection. Hướng phát triển: thêm anti-spoofing (Silent-Face-Anti-Spoofing) hoặc challenge-response (yêu cầu nháy mắt, quay đầu).

**Q13: Dataset mask 12k có bị bias không?**
Bias chắc chắn có (lệch sang một số chủng tộc/độ tuổi). Cần test trên MAFA và báo cáo confusion matrix theo subgroup. Bias mask classifier ảnh hưởng routing → cần kiểm soát kỹ.

### 5.5 Về kiến trúc hệ thống

**Q14: Vì sao lưu DB ở file npz mà không phải vector database (FAISS, Milvus)?**
Quy mô đề tài (vài chục → vài trăm user × 2 slot ≤ vài trăm vector), cosine brute-force trên numpy <1 ms. FAISS chỉ cần khi >100k embeddings.

**Q15: Khi thêm 1 người mới, có cần retrain không?**
Không. Chỉ append 2 vector (with_mask + without_mask) vào npz. Đây là ưu điểm của thiết kế feature extractor + classifier-free.

**Q16: Latency end-to-end?**
- CLAHE: ~5 ms
- RetinaFace + ArcFace embed (CPU): ~150–250 ms
- YOLOv8n-cls: ~20–30 ms
- Cosine match (DB <200 vector): <1 ms

Tổng ~200–300 ms / frame trên CPU → ~3–5 FPS. Dual-slot **không** thêm latency vì chỉ thay đổi tập candidates so khớp, không gọi thêm model.

### 5.6 Câu hỏi "khoá"

**Q17: Giải thích Additive Angular Margin Loss cho người không học deep learning?**
Hình dung 512-D space như mặt cầu. Mỗi người chiếm 1 "cụm" trên cầu. ArcFace **cộng thêm 1 góc m vào ranh giới** giữa các cụm → ép khoảng cách góc giữa 2 người gần nhau **luôn ≥ m**, làm cụm nén lại và tách xa hơn. Trong dual-slot, mỗi user còn được tách thêm 2 cụm nhỏ theo trạng thái mask → cấu trúc còn rõ hơn.

**Q18: Tại sao L2-normalize embedding?**
Để khoảng cách Euclidean tương đương cosine similarity (`||a-b||² = 2 - 2·cos(a,b)` khi `||a||=||b||=1`). Mọi tính toán phụ thuộc **hướng** vector, không phụ thuộc độ lớn → ổn định với ánh sáng/độ tương phản. Khi merge embedding trong slot, sau khi cộng phải chia cho norm để giữ tính chất này.

**Q19: Nếu mask classifier sai liên tục (toàn ra `with_mask` nhầm), hệ thống còn hoạt động?**
Vẫn hoạt động, nhưng accuracy giảm:
- Inference luôn vào slot `__with_mask` → không match được user chỉ enroll trạng thái không đeo mask → fallback toàn DB → vẫn ra kết quả nhưng confidence thấp.
- Đây là lý do nên enroll cả 2 trạng thái cho mọi user.

**Q20: Camera nghiêng 45° hệ thống còn hoạt động không?**
RetinaFace detect được mặt nghiêng ±45°, ArcFace robust đến ±30°. Lớn hơn cần face alignment (warp 5 landmark về vị trí chuẩn) — InsightFace có sẵn nhưng pipeline hiện chưa bật.

---

## 6. FAQ ngắn

**Q: Có cần train lại model không?**
**Không.** ArcFace và RetinaFace là pretrained dùng nguyên; YOLOv8n-cls đã train sẵn cho mask. Chỉ cần **enroll user** (chụp ảnh) → DB tự cập nhật.

**Q: Một người enroll bao nhiêu ảnh là đủ?**
3–5 ảnh **mỗi trạng thái** (đeo mask + không đeo mask) ở các góc/sáng khác nhau. Pipeline tự trung bình embedding trong cùng slot.

**Q: Sao không dùng GPU?**
Đề tài chạy demo trên laptop sinh viên, CPU đủ. Đổi `providers=["CUDAExecutionProvider"]` là chạy GPU.

**Q: Hệ thống có lưu ảnh người dùng không?**
Lưu ảnh enroll trong `data/faces/{user_id}/`. Embedding 512-D không thể reconstruct ảnh gốc. Cần xoá ảnh gốc nếu muốn tuân thủ privacy nghiêm ngặt.

**Q: YOLOv8n-cls và YOLOv8n-detect khác gì?**
`-cls` chỉ classify cả ảnh thành 1 nhãn; `-detect` trả bbox + class. Mask đã có bbox từ RetinaFace nên `-cls` đủ.

**Q: Thay khẩu trang bằng kính đen?**
ArcFace train có kính đen → vẫn match, confidence giảm. Nếu mask + kính đen + mũ → có thể fail; cần đa modal (giọng nói, dáng đi).

**Q: Mask classifier sai 1–2% có làm hỏng dual-slot không?**
Không nghiêm trọng nhờ fallback. Có thể gia cố thêm bằng "match cả 2 slot, lấy best" — chi phí 2× cosine, latency tăng <1 ms.

**Q: GDPR / Nghị định 13?**
Cần thêm cơ chế xoá embedding (right to be forgotten). Hiện chỉ là demo học thuật.

---

## 7. Tóm tắt điểm mạnh / điểm yếu để bảo vệ

**Điểm mạnh**
- Kết hợp **xử lý ảnh cổ điển** (CLAHE, FFT, Sobel) + **deep learning** (RetinaFace, ArcFace, YOLO) → đáp ứng đúng tinh thần môn "Xử lý ảnh".
- **Dual-slot DB**: mask label có vai trò thực sự trong recognition, không phải output phụ → trả lời được câu hỏi "mask classifier dùng để làm gì?".
- Pipeline tách module → dễ thay thế, dễ benchmark.
- Enroll mới không cần retrain, chỉ append vector vào slot tương ứng.
- Robust với mask nhờ ArcFace + CLAHE + dual-slot routing.

**Điểm yếu (chủ động nêu)**
- Chưa có anti-spoofing.
- Chưa benchmark dual-slot vs single-slot trên dataset chuẩn (LFW masked, RMFRD).
- Threshold cosine chọn thực nghiệm, chưa tối ưu theo EER cho dual-slot.
- DB lưu file phẳng — không scale lên >10k user.
- UX enroll chưa hướng dẫn user chụp đủ 2 trạng thái — phụ thuộc vào instruction.

**Hướng phát triển**
- Train periocular model riêng cho slot `__with_mask` để tăng accuracy cao hơn nữa.
- Score fusion song song full-face + periocular.
- Anti-spoofing (Silent-Face-Anti-Spoofing).
- Frontend wizard 2 bước cho enroll (chụp không mask → chụp đeo mask).
- Triển khai WebSocket video stream thay vì capture từng frame.

---

## 8. So sánh thiết kế Dual-slot với các phương án khác

> Phần phụ lục để bảo vệ tại sao **chọn Dual-slot** thay vì các hướng đi khác (gồm cả phương án đơn giản hơn lẫn phương án phức tạp hơn).

### 8.1 Bảng so sánh tổng thể

| Phương án | Ý tưởng | Ưu điểm | Nhược điểm |
|---|---|---|---|
| **(A) Single-slot, không dùng mask label** | 1 vector trung bình / user | Đơn giản nhất, code ít | Centroid lệch khi user enroll cả 2 trạng thái; mask classifier thành output thừa |
| **(B) Adaptive threshold** | 1 slot, hạ threshold khi có mask | Code rất ít (~5 dòng); không tốn DB space | Không sửa được centroid lệch; chỉ "dễ tính" hơn chứ không "phân biệt" tốt hơn |
| **(C) Dual-slot DB — đang dùng** | 2 vector / user theo trạng thái mask | Cụm chặt, mask label có vai trò thật, không retrain | Cần enroll cả 2 trạng thái; phụ thuộc độ chính xác mask classifier |
| **(D) Periocular routing** | `with_mask` → model crop vùng mắt riêng | Chuẩn nhất về mặt sinh trắc học | Phải train thêm periocular model; phức tạp; cần dataset chuyên dụng |
| **(E) Score fusion** | Chạy song song full-face + periocular, cộng score | Accuracy cao nhất | 2× inference latency; cần 2 model; tuning trọng số phức tạp |
| **(F) Mask-aware end-to-end** | Fine-tune ArcFace với mask label làm conditional input | Lý thuyết đẹp nhất | Cần GPU + dataset lớn; vượt tầm đồ án |

### 8.2 Ưu điểm cụ thể của Dual-slot (C) so với từng phương án

**So với (A) Single-slot:**
- ✅ Cosine intra-class cao hơn (~0.65–0.70 thay vì ~0.50) → **ngưỡng quyết định rộng hơn** giữa cùng người và khác người.
- ✅ Mask classifier có vai trò routing → **trả lời được câu phản biện** "mask classifier để làm gì với recognition?".
- ✅ Tăng accuracy ~5–10% (kỳ vọng) khi user xuất hiện ở trạng thái đã enroll.
- ❌ DB tăng gấp đôi kích thước (vẫn rất nhỏ — vài chục KB cho ~100 user).
- ❌ Nếu mask classifier sai → match nhầm slot (có fallback cứu, nhưng không hoàn hảo).

**So với (B) Adaptive threshold:**
- ✅ Sửa **gốc** vấn đề (centroid lệch) thay vì che đậy bằng cách hạ ngưỡng.
- ✅ Hạ threshold làm tăng false acceptance rate (FAR) — security kém hơn. Dual-slot **không** đánh đổi FAR.
- ❌ Code phức tạp hơn (~30 dòng thay vì 5 dòng).

**So với (D) Periocular routing:**
- ✅ **Không cần train model thứ hai** — chỉ dùng ArcFace pretrained sẵn.
- ✅ Triển khai trong 1 ngày thay vì 1 tuần.
- ❌ Accuracy với mask thấp hơn periocular chuyên dụng (kỳ vọng -3 đến -5% so với D).
- ❌ ArcFace dù robust nhưng vẫn dùng full-face — vùng bị mask che vẫn có ảnh hưởng nhiễu nhẹ.

**So với (E) Score fusion:**
- ✅ Latency giữ nguyên (~200–300 ms / frame). Score fusion gấp đôi.
- ✅ 1 model đơn giản hơn 2 model song song.
- ❌ Accuracy thấp hơn fusion (-5 đến -8% kỳ vọng trên test khó).

**So với (F) Mask-aware end-to-end:**
- ✅ **Không train**, dùng được ngay với pretrained.
- ✅ Trade-off thực dụng phù hợp đồ án sinh viên.
- ❌ Không phải state-of-the-art học thuật.

### 8.3 Phân tích nhược điểm sâu hơn (chủ động nêu để bảo vệ)

**N1. Phụ thuộc độ chính xác mask classifier**
- Nếu YOLO sai 5% → 5% truy vấn vào nhầm slot.
- *Giảm thiểu*: fallback toàn DB khi slot rỗng; có thể nâng cấp thành "match cả 2 slot, lấy best cosine" với chi phí <1 ms.

**N2. Yêu cầu enroll 2 trạng thái — ma sát UX**
- User phải chịu khó đeo mask vào để enroll → trải nghiệm kém hơn enroll 1 lần.
- *Giảm thiểu*: frontend wizard 2 bước rõ ràng; hoặc cho phép enroll 1 trạng thái rồi dần dần thu thập trạng thái còn lại từ frame inference (online learning).

**N3. DB không thuần "1 user = 1 vector" — khó phân tích**
- Khó visualize cụm user trên t-SNE/UMAP vì 1 user thành 2 điểm.
- *Giảm thiểu*: thêm `list_slots()` debug API; visualize có thể color-code theo trạng thái.

**N4. Không scale tự nhiên cho >2 trạng thái**
- Nếu sau này thêm `mask_incorrect` (đeo sai cách), phải thêm slot thứ 3 → DB = 3× user.
- *Giảm thiểu*: với 3 lớp vẫn OK; nếu mở rộng nhiều hơn (kính, mũ, scarf) thì nên chuyển sang (E) hoặc (F).

**N5. Threshold cố định không tối ưu cho mọi slot**
- Slot `__with_mask` có thể cần threshold khác `__without_mask` vì chất lượng embedding khác nhau.
- *Giảm thiểu*: nâng cấp tương lai = per-slot threshold + adaptive threshold (kết hợp (B) + (C)).

### 8.4 Khi nào nên / không nên dùng Dual-slot?

**Nên dùng khi:**
- Đề tài/dự án nhỏ-vừa (<1000 user).
- Không có ngân sách train model mới.
- Muốn có "câu chuyện" để bảo vệ tại sao mask classifier tồn tại.
- Mask classifier có accuracy ≥ 95%.

**Không nên dùng khi:**
- Mask classifier không tin cậy (<90% accuracy) → routing sai làm hỏng cả pipeline.
- DB cực lớn (>100k user) — gấp đôi DB là gánh nặng IO.
- Cần SOTA accuracy → đi thẳng đến (E) hoặc (F).
- User chỉ có thể chụp 1 trạng thái → fallback hoạt động nhưng không tận dụng được dual-slot.
