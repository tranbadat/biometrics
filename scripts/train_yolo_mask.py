"""Train YOLOv8-cls trên Face Mask 12k dataset.

Dataset đã có sẵn cấu trúc folder phù hợp YOLOv8-cls:
  data/kaggle_raw/Face Mask Dataset/
    Train/{WithMask, WithoutMask}/
    Validation/{WithMask, WithoutMask}/
    Test/{WithMask, WithoutMask}/

YOLOv8-cls yêu cầu tên thư mục: train/ val/ test/ (lowercase).
Script này tạo symlinks chuẩn rồi chạy training với augmentation
mạnh để chống false negative do loá sáng.
"""
from pathlib import Path
import shutil

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "kaggle_raw" / "Face Mask Dataset"
YOLO_DIR = PROJECT_ROOT / "data" / "yolo_cls"
WEIGHTS_OUT = PROJECT_ROOT / "models" / "mask_yolov8n_cls.pt"


def prepare_dataset() -> None:
    """Tạo cấu trúc train/val/test cho YOLOv8-cls."""
    mapping = {"Train": "train", "Validation": "val", "Test": "test"}
    YOLO_DIR.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in mapping.items():
        src = RAW_DIR / src_name
        dst = YOLO_DIR / dst_name
        if dst.exists():
            print(f"  Skip {dst_name}/ (đã tồn tại)")
            continue
        print(f"  Copy {src_name} → {dst_name} ...")
        shutil.copytree(src, dst)
    print(f"Dataset ready at: {YOLO_DIR.resolve()}")


def train() -> None:
    model = YOLO("yolov8n-cls.pt")  # ~3MB, đủ cho real-time
    model.train(
        data=str(YOLO_DIR.resolve()),
        epochs=30,
        imgsz=224,
        batch=64,
        device="mps",            # đổi "0" nếu có GPU CUDA, hoặc "mps" nếu M-series
        project=str(PROJECT_ROOT / "runs"),
        name="mask_yolov8n_cls",
        # ── Augmentation chống loá sáng ────────────────────────────────
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.6,               # tăng mạnh value augmentation
        degrees=10,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        erasing=0.2,             # random erasing — giả lập occlusion
        # ── Regularization ─────────────────────────────────────────────
        dropout=0.1,
        weight_decay=5e-4,
        patience=10,
    )

    best = PROJECT_ROOT / "runs" / "mask_yolov8n_cls" / "weights" / "best.pt"
    if best.exists():
        WEIGHTS_OUT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best, WEIGHTS_OUT)
        print(f"Saved best weights → {WEIGHTS_OUT}")


if __name__ == "__main__":
    prepare_dataset()
    train()
