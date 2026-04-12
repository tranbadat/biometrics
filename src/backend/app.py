from pathlib import Path
from typing import List

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from src.inference import pipeline

app = FastAPI(title="Face+Mask Recognition API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    results = pipeline.infer_pil(img)
    return {"predictions": results}


@app.post("/enroll")
async def enroll_face(
    name: str    = Form(..., description="Họ và tên người dùng"),
    user_id: str = Form(..., description="Mã người dùng — dùng làm tên thư mục"),
    files: List[UploadFile] = File(..., description="Danh sách ảnh khuôn mặt"),
):
    """Đăng ký khuôn mặt mới vào hệ thống.

    - Lưu ảnh vào data/faces/{user_id}/
    - Chạy FaceNet embedding + retrain SVM
    """
    if not name.strip():
        raise HTTPException(400, "name không được để trống")
    if not user_id.strip():
        raise HTTPException(400, "user_id không được để trống")
    if len(files) == 0:
        raise HTTPException(400, "Cần ít nhất 1 ảnh")

    # Lưu ảnh vào disk
    save_dir = Path("data/faces") / user_id
    save_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for i, f in enumerate(files):
        data = await f.read()
        if not data:
            continue
        path = save_dir / f"frame_{i:04d}.jpg"
        path.write_bytes(data)
        saved_paths.append(str(path))

    if not saved_paths:
        raise HTTPException(400, "Không lưu được ảnh nào")

    # Chạy enroll pipeline
    enroll_result = {"saved": len(saved_paths), "name": name, "user_id": user_id}
    try:
        import sys
        sys.path.insert(0, ".")
        from scripts.enroll import (
            load_database, get_mtcnn, get_facenet,
            action_enroll, retrain_svm, save_database
        )
        import torch

        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        model_path = "models/recognizer.joblib"
        data       = load_database(model_path)
        mtcnn      = get_mtcnn()
        facenet    = get_facenet(device)

        data = action_enroll(data, name, str(save_dir), mtcnn, facenet, device)

        if len(set(data["labels"])) >= 2:
            data["clf"], data["le"] = retrain_svm(data["embeddings"], data["labels"])
            data["class_names"]     = data["le"].classes_.tolist()

        save_database(data, model_path)

        # Reload recognizer trong pipeline
        pipeline._RECOG = None

        enroll_result["status"]  = "ok"
        enroll_result["persons"] = list(set(data["labels"]))

    except Exception as e:
        # Ảnh vẫn đã được lưu — chỉ bước embedding/SVM thất bại
        enroll_result["status"]  = "saved_only"
        enroll_result["warning"] = str(e)
        enroll_result["hint"]    = (
            "Ảnh đã lưu thành công. Chạy thủ công: "
            f"python scripts/enroll.py --name '{name}' --images data/faces/{user_id}/"
        )

    return enroll_result
