"""FastAPI backend — pipeline MediaPipe + YOLOv8-cls + InsightFace ArcFace.

Switched từ pipeline cũ (MTCNN + CNN + FaceNet/SVM) ngày 2026-04-30.
Xem docs/pipeline-comparison.md để biết lý do chuyển đổi.
"""
from pathlib import Path
from typing import List

import cv2
import io
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from src.inference import pipeline_yolo as pipeline

app = FastAPI(title="Face+Mask Recognition API (YOLO pipeline)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "pipeline": "mediapipe+yolov8-cls+arcface",
        "enrolled": pipeline.list_identities(),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    results = pipeline.infer_pil(img)
    print(f"[predict] {len(results)} face(s):")
    for i, r in enumerate(results):
        print(f"  #{i}: label={r['label']} ({r['confidence']:.2f}) "
              f"identity={r['identity']} ({r['identity_confidence']:.2f}) "
              f"box={r['box']}")
    return {"predictions": results}


@app.post("/enroll")
async def enroll_face(
    name: str    = Form(..., description="Họ và tên người dùng"),
    user_id: str = Form(..., description="Mã người dùng — dùng làm tên thư mục"),
    files: List[UploadFile] = File(..., description="Danh sách ảnh khuôn mặt"),
):
    """Đăng ký khuôn mặt mới — dùng InsightFace ArcFace embedding.

    - Lưu ảnh vào data/faces/{user_id}/
    - Tính embedding ArcFace cho mỗi ảnh, trung bình hoá, lưu vào models/arcface_db.npz
    """
    if not name.strip():
        raise HTTPException(400, "name không được để trống")
    if not user_id.strip():
        raise HTTPException(400, "user_id không được để trống")
    if not files:
        raise HTTPException(400, "Cần ít nhất 1 ảnh")

    save_dir = Path("data/faces") / user_id
    save_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    enrolled = 0
    failures: list[str] = []

    for i, f in enumerate(files):
        data = await f.read()
        if not data:
            continue
        path = save_dir / f"frame_{i:04d}.jpg"
        path.write_bytes(data)
        saved += 1

        # Decode ảnh → BGR cho InsightFace
        arr = np.frombuffer(data, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            failures.append(f"frame_{i:04d}: decode failed")
            continue

        ok = pipeline.enroll_identity(user_id, bgr, persist=False, display_name=name)
        if ok:
            enrolled += 1
        else:
            failures.append(f"frame_{i:04d}: no face detected")

    if enrolled == 0:
        raise HTTPException(400, f"Không trích xuất được embedding nào. Failures: {failures}")

    # Persist 1 lần sau khi merge xong tất cả ảnh
    pipeline._save_db()

    return {
        "status": "ok",
        "name": name,
        "user_id": user_id,
        "saved": saved,
        "enrolled": enrolled,
        "failures": failures,
        "total_identities": len(pipeline.list_identities()),
    }
