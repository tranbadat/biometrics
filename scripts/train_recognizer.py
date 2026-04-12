"""Train FaceNet + SVM nhận diện danh tính.

Dataset structure:
    data/faces/
    ├── nguyen_van_a/
    │   ├── img1.jpg
    │   └── ...
    ├── tran_thi_b/
    │   └── ...
    └── ...

Pipeline:
    1. Với mỗi ảnh: MTCNN detect + crop khuôn mặt (bỏ qua ảnh không detect được)
    2. Tiền xử lý: CLAHE + filter ảnh mờ
    3. FaceNet (InceptionResnetV1) tạo embedding 512-D, L2-normalize
    4. Chia train/test (stratified)
    5. Train SVM (RBF kernel, probability=True)
    6. Đánh giá: accuracy, classification report
    7. Lưu {clf, le, class_names} vào joblib

Usage:
    python scripts/train_recognizer.py --data data/faces --out models/recognizer.joblib
"""

import argparse
import os
import sys
from pathlib import Path
from glob import glob

import numpy as np
from tqdm import tqdm
from PIL import Image

# Thêm project root vào path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_image_paths(data_dir: str):
    """Thu thập (path, identity_label) từ cây thư mục data_dir/{person}/*.jpg"""
    paths, labels = [], []
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f'Thư mục không tồn tại: {data_dir}')

    for person_dir in sorted(data_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        imgs = [p for ext in ('*.jpg', '*.jpeg', '*.png')
                for p in person_dir.glob(ext)]
        if not imgs:
            print(f'  [warn] Không có ảnh trong {person_dir.name}, bỏ qua')
            continue
        for p in imgs:
            paths.append(str(p))
            labels.append(person_dir.name)

    return paths, labels


def detect_and_crop(paths, labels, min_face_size: int = 40):
    """MTCNN detect khuôn mặt, trả về list PIL crops đã align.

    Ảnh không detect được mặt sẽ bị bỏ qua.
    """
    try:
        from facenet_pytorch import MTCNN
    except ImportError:
        print('[error] facenet-pytorch chưa được cài: pip install facenet-pytorch')
        sys.exit(1)

    mtcnn = MTCNN(
        image_size=160,
        margin=20,
        min_face_size=min_face_size,
        keep_all=False,        # lấy mặt có confidence cao nhất
        post_process=False,
    )

    crops, crop_labels = [], []
    skipped = 0

    for path, label in tqdm(zip(paths, labels), total=len(paths),
                             desc='MTCNN detect', unit='img'):
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            print(f'  [warn] Không đọc được {path}: {e}')
            skipped += 1
            continue

        face = mtcnn(img)  # Tensor (3, 160, 160) hoặc None

        if face is None:
            skipped += 1
            continue

        crops.append(face)
        crop_labels.append(label)

    print(f'  Detect thành công: {len(crops)}/{len(paths)} '
          f'(bỏ {skipped} ảnh không tìm thấy mặt)')
    return crops, crop_labels


def extract_embeddings(face_tensors):
    """FaceNet tạo embedding 512-D L2-normalized từ list tensor khuôn mặt."""
    try:
        import torch
        from facenet_pytorch import InceptionResnetV1
    except ImportError:
        print('[error] facenet-pytorch hoặc torch chưa cài')
        sys.exit(1)

    device = (
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    embeddings = []
    batch_size = 32
    with torch.no_grad():
        for i in tqdm(range(0, len(face_tensors), batch_size),
                      desc='FaceNet embed', unit='batch'):
            batch = torch.stack(face_tensors[i:i + batch_size]).to(device)
            # MTCNN post_process=False → pixel [0,255]; normalize về [-1,1]
            batch = (batch - 127.5) / 128.0
            emb = model(batch)                           # (B, 512)
            # L2 normalize
            emb = emb / emb.norm(dim=1, keepdim=True)
            embeddings.append(emb.cpu().numpy())

    return np.vstack(embeddings)


def train_svm(X: np.ndarray, y: np.ndarray, kernel: str = 'rbf'):
    """Train SVM với cross-validation để chọn C tốt nhất."""
    try:
        from sklearn.svm import SVC
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.preprocessing import LabelEncoder
    except ImportError:
        print('[error] scikit-learn chưa cài: pip install scikit-learn')
        sys.exit(1)

    le = LabelEncoder()
    Y  = le.fit_transform(y)

    # Cross-validation để tìm C tốt nhất
    best_C, best_score = 1.0, 0.0
    for C in [0.1, 1.0, 5.0, 10.0]:
        clf_try = SVC(kernel=kernel, C=C, probability=True)
        cv = StratifiedKFold(n_splits=min(5, np.bincount(Y).min()), shuffle=True,
                             random_state=42)
        scores = cross_val_score(clf_try, X, Y, cv=cv, scoring='accuracy')
        print(f'  SVM C={C:5.1f}: cv_acc = {scores.mean():.4f} ± {scores.std():.4f}')
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_C     = C

    print(f'  → Chọn C={best_C}, train trên toàn bộ tập')
    clf = SVC(kernel=kernel, C=best_C, probability=True)
    clf.fit(X, Y)
    return clf, le


def evaluate(clf, le, X_test, y_test_encoded):
    """In classification report và confusion matrix."""
    try:
        from sklearn.metrics import classification_report, confusion_matrix
    except ImportError:
        return

    preds = clf.predict(X_test)
    class_names = le.classes_

    print('\nClassification report:')
    print(classification_report(y_test_encoded, preds,
                                target_names=class_names, digits=4))

    cm = confusion_matrix(y_test_encoded, preds)
    print('Confusion matrix:')
    header = '        ' + '  '.join(f'{c[:8]:>8}' for c in class_names)
    print(header)
    for i, row in enumerate(cm):
        print(f'{class_names[i][:8]:>8}  ' + '  '.join(f'{v:>8}' for v in row))


def main():
    parser = argparse.ArgumentParser(
        description='Train FaceNet+SVM nhận diện danh tính'
    )
    parser.add_argument('--data', required=True,
                        help='Thư mục faces với subfolders theo tên người')
    parser.add_argument('--out', default='models/recognizer.joblib',
                        help='Đường dẫn lưu model (default: models/recognizer.joblib)')
    parser.add_argument('--kernel', choices=['rbf', 'linear'], default='rbf',
                        help='SVM kernel (default: rbf)')
    parser.add_argument('--min-face', type=int, default=40,
                        help='Kích thước mặt tối thiểu để detect (default: 40px)')
    parser.add_argument('--test-split', type=float, default=0.2,
                        help='Tỉ lệ test split (default: 0.2)')
    args = parser.parse_args()

    # ── Load paths ───────────────────────────────────────────────────────────
    print(f'Dataset: {args.data}')
    paths, labels = load_image_paths(args.data)
    print(f'Tổng ảnh: {len(paths)} | Số người: {len(set(labels))}')
    for person in sorted(set(labels)):
        count = labels.count(person)
        print(f'  {person}: {count} ảnh')

    if len(set(labels)) < 2:
        print('[error] Cần ít nhất 2 người để train')
        sys.exit(1)

    # ── Detect & embed ───────────────────────────────────────────────────────
    face_tensors, face_labels = detect_and_crop(paths, labels, args.min_face)
    if len(face_tensors) == 0:
        print('[error] Không detect được khuôn mặt nào')
        sys.exit(1)

    print('Trích xuất embeddings...')
    X = extract_embeddings(face_tensors)

    # ── Train/test split ─────────────────────────────────────────────────────
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.preprocessing import LabelEncoder

    le_temp = LabelEncoder()
    Y_all   = le_temp.fit_transform(face_labels)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_split,
                                 random_state=42)
    train_idx, test_idx = next(sss.split(X, Y_all))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train = [face_labels[i] for i in train_idx]
    y_test  = [face_labels[i] for i in test_idx]

    print(f'Train: {len(X_train)} | Test: {len(X_test)}')

    # ── Train SVM ────────────────────────────────────────────────────────────
    print('\nTrain SVM (cross-validation để chọn C)...')
    clf, le = train_svm(X_train, y_train, kernel=args.kernel)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    y_test_encoded = le.transform(y_test)
    evaluate(clf, le, X_test, y_test_encoded)

    # ── Lưu model ────────────────────────────────────────────────────────────
    try:
        import joblib
    except ImportError:
        print('[error] joblib chưa cài: pip install joblib')
        sys.exit(1)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    joblib.dump({
        'clf':         clf,
        'le':          le,
        'class_names': le.classes_.tolist(),
        'embedding_dim': X.shape[1],
    }, args.out)
    print(f'\nModel đã lưu tại: {args.out}')
    print(f'Số danh tính: {len(le.classes_)} — {list(le.classes_)}')


if __name__ == '__main__':
    main()
