# Hướng dẫn Train YOLOv8-cls cho Mask Detection

Pipeline 3-stage: **MediaPipe (detect) → YOLOv8-cls (mask classify) → InsightFace ArcFace (recognize)**.
Tài liệu này tập trung vào stage 2 — train classifier YOLOv8 trên dataset Face Mask 12k.

---

## 1. Tổng quan

| Stage | Model | Train? | Lý do |
|---|---|---|---|
| Detect | MediaPipe BlazeFace | ❌ Pretrained | Robust với loá sáng, real-time |
| Classify mask | YOLOv8n-cls | ✅ Train trên dataset của bạn | Custom classes WithMask/WithoutMask |
| Recognize | InsightFace `buffalo_l` | ❌ Pretrained | Robust với occlusion |

Chỉ cần train YOLOv8-cls. ArcFace đăng ký bằng enrollment (không cần training).

---

## 2. Chuẩn bị dependencies

Đã có trong `requirements.txt`:
```
ultralytics
mediapipe
insightface
onnxruntime
```

Cài (đã thực hiện):
```bash
.venv/bin/pip install ultralytics mediapipe insightface onnxruntime
```

Verify:
```bash
.venv/bin/python -c "import ultralytics; print(ultralytics.__version__)"
```

---

## 3. Chuẩn bị dataset

### Cấu trúc đầu vào (đã có sẵn từ Kaggle)

```
data/kaggle_raw/Face Mask Dataset/
├── Train/
│   ├── WithMask/
│   └── WithoutMask/
├── Validation/
│   ├── WithMask/
│   └── WithoutMask/
└── Test/
    ├── WithMask/
    └── WithoutMask/
```

Nếu chưa giải nén:
```bash
unzip face-mask-12k-images-dataset.zip -d data/kaggle_raw/
```

### Cấu trúc YOLOv8-cls yêu cầu

YOLOv8-cls cần tên thư mục lowercase: `train/`, `val/`, `test/`. Script
`scripts/train_yolo_mask.py` tự tạo symlinks vào `data/yolo_cls/`:

```
data/yolo_cls/
├── train/{WithMask, WithoutMask}/   → symlink
├── val/{WithMask, WithoutMask}/     → symlink
└── test/{WithMask, WithoutMask}/    → symlink
```

---

## 4. Chạy training

```bash
.venv/bin/python scripts/train_yolo_mask.py
```

Script sẽ tự:
1. Tạo symlinks `data/yolo_cls/`
2. Tải pretrained `yolov8n-cls.pt` (~3MB) lần đầu
3. Train 30 epochs
4. Copy weights tốt nhất → `models/mask_yolov8n_cls.pt`

---

## 5. Cấu hình quan trọng

### Device — chỉnh ở `scripts/train_yolo_mask.py:48`

```python
device="cpu"     # mặc định, mọi máy đều chạy được
device="mps"     # Mac Apple Silicon (M1/M2/M3) — nhanh hơn 5-10x
device="0"       # NVIDIA GPU đầu tiên (CUDA)
device="0,1"     # Multi-GPU
```

**Khuyến nghị:**
- Mac M1/M2/M3 → đổi sang `device="mps"`
- Có GPU NVIDIA → `device="0"`
- Không có GPU → giữ `device="cpu"` (chậm ~45-60 phút)

### Thời gian training dự kiến (30 epochs, 12k ảnh, batch=64)

| Hardware | Thời gian |
|---|---|
| CPU (M2 Mac) | 45-60 phút |
| MPS (M2 Mac) | 8-12 phút |
| GPU T4 / RTX 3060 | 5-8 phút |
| GPU A100 | 2-3 phút |

### Batch size

Nếu Out-Of-Memory:
```python
batch=32   # giảm từ 64
batch=16   # giảm thêm nếu vẫn OOM
```

### Augmentation chống loá sáng (đã set sẵn)

```python
hsv_h=0.015          # nhiễu hue nhẹ
hsv_s=0.7            # saturation rộng
hsv_v=0.6            # ★ value cao — giả lập loá/tối
erasing=0.2          # random erasing — giả lập occlusion
degrees=10           # xoay nhẹ
scale=0.3            # zoom in/out
fliplr=0.5           # lật ngang
```

`hsv_v=0.6` là key parameter chống false negative do loá. Có thể tăng lên
`0.7-0.8` nếu môi trường deploy có ánh sáng cực kỳ biến động.

---

## 6. Theo dõi training

Trong khi train, ultralytics log ra `runs/mask_yolov8n_cls/`:

```
runs/mask_yolov8n_cls/
├── weights/
│   ├── best.pt          # checkpoint accuracy cao nhất trên val
│   └── last.pt          # checkpoint epoch cuối
├── results.csv          # loss + accuracy theo epoch
├── confusion_matrix.png
└── results.png
```

Mở `results.png` xem loss/accuracy curves.

**Mong đợi:** trên dataset 12k clean này, val accuracy >98% sau 15-20 epochs.

---

## 7. Verify model sau training

```bash
.venv/bin/python -c "
from ultralytics import YOLO
m = YOLO('models/mask_yolov8n_cls.pt')
r = m.predict('data/kaggle_raw/Face Mask Dataset/Test/WithMask/10.png', imgsz=224)
print('Classes:', m.names)
print('Top1:', m.names[int(r[0].probs.top1)], 'conf:', float(r[0].probs.top1conf))
"
```

Đánh giá toàn bộ test set:
```bash
.venv/bin/python -c "
from ultralytics import YOLO
m = YOLO('models/mask_yolov8n_cls.pt')
metrics = m.val(data='data/yolo_cls', split='test')
print('Top1 accuracy:', metrics.top1)
"
```

---

## 8. Hard negative mining (cải thiện chống loá thực tế)

Sau khi train xong, nếu deploy gặp false negative do ánh sáng đặc biệt:

1. **Thu thập failure cases** — log ảnh model dự đoán sai từ camera production
2. **Label lại** vào folder phù hợp:
   ```
   data/kaggle_raw/Face Mask Dataset/Train/WithMask/   ← thêm ảnh khó
   data/kaggle_raw/Face Mask Dataset/Train/WithoutMask/
   ```
3. **Retrain** từ checkpoint cũ:
   ```python
   model = YOLO("models/mask_yolov8n_cls.pt")  # load weights cũ
   model.train(data="data/yolo_cls", epochs=10, ...)
   ```

---

## 9. Re-train / fine-tune

Train lại từ đầu:
```bash
rm -rf runs/mask_yolov8n_cls data/yolo_cls
.venv/bin/python scripts/train_yolo_mask.py
```

Fine-tune từ weights hiện tại — sửa trong `train_yolo_mask.py`:
```python
model = YOLO("models/mask_yolov8n_cls.pt")  # thay vì yolov8n-cls.pt
model.train(..., epochs=10, lr0=0.001)      # learning rate thấp hơn
```

---

## 10. Tích hợp vào pipeline

Sau khi có `models/mask_yolov8n_cls.pt`, pipeline tự động dùng:

```python
from PIL import Image
from src.inference.pipeline_yolo import infer_pil

img = Image.open("test.jpg")
print(infer_pil(img))
```

`pipeline_yolo.py` lazy-load weights từ đường dẫn `models/mask_yolov8n_cls.pt`
mặc định, không cần thay đổi gì sau khi train.

---

## 11. Troubleshooting

| Lỗi | Nguyên nhân | Cách fix |
|---|---|---|
| `FileNotFoundError: yolov8n-cls.pt` | Không có internet lần đầu | Download manual từ ultralytics releases |
| `CUDA out of memory` | Batch quá lớn | Giảm `batch=32` hoặc `batch=16` |
| `MPS backend out of memory` | Mac M1/M2 RAM thấp | Giảm `batch` + `imgsz=160` |
| Symlink lỗi trên Windows | OS không support | Đổi `symlink_to` thành `shutil.copytree` |
| Val accuracy thấp (<90%) | Augmentation quá mạnh | Giảm `hsv_v=0.4`, `erasing=0.1` |
| Val accuracy cao nhưng deploy kém | Domain gap | Hard negative mining (mục 8) |
| Python 3.14 import lỗi | Package chưa hỗ trợ | Tạo venv mới với Python 3.11 |

---

## 12. Tham khảo

- Ultralytics YOLOv8 docs: https://docs.ultralytics.com
- Dataset: https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset
- MediaPipe Face Detection: https://developers.google.com/mediapipe/solutions/vision/face_detector
- InsightFace: https://github.com/deepinsight/insightface
