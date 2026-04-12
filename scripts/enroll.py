"""Enroll (đăng ký) khuôn mặt mới vào hệ thống nhận diện.

Thêm một người mới vào recognizer mà không cần retrain FaceNet —
chỉ cần tạo embeddings mới + retrain SVM (nhanh, dưới 1 giây).

Usage:
    # Lần đầu (chưa có model):
    python scripts/enroll.py --name "Nguyen Van A" --images data/faces/nguyen_van_a/

    # Thêm người thứ 2 vào model đã có:
    python scripts/enroll.py --name "Tran Thi B" --images data/faces/tran_thi_b/ \\
        --model models/recognizer.joblib

    # Xem danh sách người đã enroll:
    python scripts/enroll.py --list --model models/recognizer.joblib

    # Xóa một người khỏi hệ thống:
    python scripts/enroll.py --remove "Nguyen Van A" --model models/recognizer.joblib
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Tiện ích load/save
# ---------------------------------------------------------------------------

def load_database(model_path: str) -> dict:
    """Load embedding database từ joblib.

    Trả về dict:
        {
          'embeddings': np.ndarray (N, 512),
          'labels':     list[str],
          'clf':        sklearn classifier hoặc None,
          'le':         LabelEncoder hoặc None,
        }
    """
    try:
        import joblib
    except ImportError:
        print('[error] joblib chưa cài: pip install joblib')
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f'  Model chưa tồn tại tại {model_path}, tạo database mới')
        return {'embeddings': np.empty((0, 512), dtype=np.float32),
                'labels': [], 'clf': None, 'le': None}

    data = joblib.load(model_path)
    # Hỗ trợ format cũ (chỉ có clf + le) và format mới (có thêm embeddings)
    if 'embeddings' not in data:
        print('  [warn] Model cũ không có raw embeddings — không thể thêm người mới')
        print('  Hãy chạy train_recognizer.py lại với --out để tạo model mới')
        sys.exit(1)

    n = len(data['labels'])
    persons = sorted(set(data['labels']))
    print(f'  Đã load: {n} embeddings, {len(persons)} người: {persons}')
    return data


def save_database(data: dict, model_path: str) -> None:
    """Lưu database + retrain SVM."""
    try:
        import joblib
    except ImportError:
        print('[error] joblib chưa cài')
        sys.exit(1)

    os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
    joblib.dump(data, model_path)
    print(f'  Đã lưu: {model_path}')


# ---------------------------------------------------------------------------
# MTCNN + FaceNet
# ---------------------------------------------------------------------------

def get_mtcnn():
    try:
        from facenet_pytorch import MTCNN
        return MTCNN(image_size=160, margin=20, min_face_size=40,
                     keep_all=False, post_process=False)
    except ImportError:
        print('[error] facenet-pytorch chưa cài: pip install facenet-pytorch')
        sys.exit(1)


def get_facenet(device: str):
    try:
        from facenet_pytorch import InceptionResnetV1
        return InceptionResnetV1(pretrained='vggface2').eval().to(device)
    except ImportError:
        print('[error] facenet-pytorch chưa cài')
        sys.exit(1)


def embed_images(image_paths: list, mtcnn, facenet, device: str) -> np.ndarray:
    """Detect khuôn mặt + tạo embeddings từ list ảnh."""
    import torch

    face_tensors = []
    skipped = 0

    for path in tqdm(image_paths, desc='  Detect mặt', unit='img'):
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            print(f'    [warn] Không đọc được {path}: {e}')
            skipped += 1
            continue

        face = mtcnn(img)
        if face is None:
            print(f'    [warn] Không detect được mặt: {Path(path).name}')
            skipped += 1
            continue
        face_tensors.append(face)

    if not face_tensors:
        return np.empty((0, 512), dtype=np.float32)

    print(f'  Detect thành công: {len(face_tensors)}/{len(image_paths)} '
          f'(bỏ {skipped})')

    embeddings = []
    batch_size = 16
    with torch.no_grad():
        for i in tqdm(range(0, len(face_tensors), batch_size),
                      desc='  FaceNet embed', unit='batch'):
            batch = torch.stack(face_tensors[i:i + batch_size]).to(device)
            batch = (batch - 127.5) / 128.0
            emb = facenet(batch)
            emb = emb / emb.norm(dim=1, keepdim=True)
            embeddings.append(emb.cpu().numpy())

    return np.vstack(embeddings).astype(np.float32)


# ---------------------------------------------------------------------------
# Retrain SVM
# ---------------------------------------------------------------------------

def retrain_svm(embeddings: np.ndarray, labels: list, kernel: str = 'rbf'):
    """Retrain SVM từ toàn bộ embedding database."""
    try:
        from sklearn.svm import SVC
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import cross_val_score, StratifiedKFold
    except ImportError:
        print('[error] scikit-learn chưa cài')
        sys.exit(1)

    le = LabelEncoder()
    Y  = le.fit_transform(labels)

    persons = le.classes_
    counts  = np.bincount(Y)
    n_splits = min(5, counts.min())

    if n_splits < 2:
        # Chưa đủ ảnh để cross-validate, dùng C mặc định
        print('  Chưa đủ ảnh để cross-validate, dùng C=1.0')
        best_C = 1.0
    else:
        best_C, best_score = 1.0, 0.0
        for C in [0.1, 1.0, 5.0, 10.0]:
            clf_try = SVC(kernel=kernel, C=C, probability=True)
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            scores = cross_val_score(clf_try, embeddings, Y, cv=cv, scoring='accuracy')
            print(f'    C={C:5.1f}: cv_acc = {scores.mean():.4f}')
            if scores.mean() > best_C:
                best_score = scores.mean()
                best_C     = C

    clf = SVC(kernel=kernel, C=best_C, probability=True)
    clf.fit(embeddings, Y)
    print(f'  SVM train xong. Danh tính ({len(persons)}): {list(persons)}')
    return clf, le


# ---------------------------------------------------------------------------
# Main actions
# ---------------------------------------------------------------------------

def action_list(data: dict) -> None:
    persons = sorted(set(data['labels']))
    if not persons:
        print('Chưa có ai được enroll.')
        return
    print(f'\nDanh sách người đã enroll ({len(persons)} người):')
    for person in persons:
        count = data['labels'].count(person)
        print(f'  {person}: {count} embeddings')


def action_remove(data: dict, name: str) -> dict:
    if name not in data['labels']:
        print(f'[warn] "{name}" không có trong hệ thống')
        return data

    keep = [i for i, l in enumerate(data['labels']) if l != name]
    data['embeddings'] = data['embeddings'][keep]
    data['labels']     = [data['labels'][i] for i in keep]
    print(f'  Đã xóa "{name}" ({len(data["labels"])} embeddings còn lại)')
    return data


def action_enroll(data: dict, name: str, images_dir: str,
                  mtcnn, facenet, device: str) -> dict:
    """Thêm người mới hoặc thêm ảnh cho người đã có."""
    images_dir = Path(images_dir)
    if not images_dir.exists():
        print(f'[error] Thư mục không tồn tại: {images_dir}')
        sys.exit(1)

    image_paths = [str(p) for ext in ('*.jpg', '*.jpeg', '*.png')
                   for p in images_dir.glob(ext)]
    if not image_paths:
        print(f'[error] Không có ảnh trong {images_dir}')
        sys.exit(1)

    print(f'\nEnroll "{name}" từ {len(image_paths)} ảnh...')
    new_embeddings = embed_images(image_paths, mtcnn, facenet, device)

    if len(new_embeddings) == 0:
        print('[error] Không tạo được embedding nào — kiểm tra lại ảnh đầu vào')
        sys.exit(1)

    new_labels = [name] * len(new_embeddings)

    # Ghép vào database
    if data['embeddings'].shape[0] == 0:
        data['embeddings'] = new_embeddings
    else:
        data['embeddings'] = np.vstack([data['embeddings'], new_embeddings])

    data['labels'].extend(new_labels)

    existing = data['labels'].count(name) - len(new_labels)
    if existing > 0:
        print(f'  "{name}" đã có {existing} embeddings cũ, thêm {len(new_embeddings)} mới')
    else:
        print(f'  Thêm mới "{name}" với {len(new_embeddings)} embeddings')

    return data


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Enroll khuôn mặt mới vào hệ thống nhận diện',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--name', help='Tên người cần enroll')
    parser.add_argument('--images', help='Thư mục chứa ảnh khuôn mặt')
    parser.add_argument('--model', default='models/recognizer.joblib',
                        help='Đường dẫn model (default: models/recognizer.joblib)')
    parser.add_argument('--list', action='store_true',
                        help='Hiển thị danh sách người đã enroll')
    parser.add_argument('--remove', metavar='NAME',
                        help='Xóa một người khỏi hệ thống')
    parser.add_argument('--kernel', choices=['rbf', 'linear'], default='rbf')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()

    # ── Device ──────────────────────────────────────────────────────────────
    if args.device == 'auto':
        try:
            import torch
            device = ('cuda' if torch.cuda.is_available()
                      else 'mps' if torch.backends.mps.is_available()
                      else 'cpu')
        except ImportError:
            device = 'cpu'
    else:
        device = args.device

    # ── Load database ────────────────────────────────────────────────────────
    print(f'Model: {args.model}')
    data = load_database(args.model)

    # ── Action: list ─────────────────────────────────────────────────────────
    if args.list:
        action_list(data)
        return

    # ── Action: remove ───────────────────────────────────────────────────────
    if args.remove:
        data = action_remove(data, args.remove)
        if len(set(data['labels'])) >= 2:
            print('Retrain SVM sau khi xóa...')
            data['clf'], data['le'] = retrain_svm(
                data['embeddings'], data['labels'], args.kernel
            )
        else:
            data['clf'] = data['le'] = None
            print('[warn] Còn ít hơn 2 người — SVM chưa thể train')
        save_database(data, args.model)
        return

    # ── Action: enroll ───────────────────────────────────────────────────────
    if not args.name or not args.images:
        parser.print_help()
        print('\n[error] Cần --name và --images để enroll')
        sys.exit(1)

    print(f'Device: {device}')
    mtcnn   = get_mtcnn()
    facenet = get_facenet(device)

    data = action_enroll(data, args.name, args.images, mtcnn, facenet, device)

    # Retrain SVM
    if len(set(data['labels'])) >= 2:
        print('\nRetrain SVM...')
        data['clf'], data['le'] = retrain_svm(
            data['embeddings'], data['labels'], args.kernel
        )
        data['class_names'] = data['le'].classes_.tolist()
    else:
        persons = list(set(data['labels']))
        print(f'\n[info] Hiện có 1 người ({persons[0]}). '
              f'Enroll thêm ít nhất 1 người nữa để SVM hoạt động.')
        data['clf'] = data['le'] = None

    save_database(data, args.model)
    print('\nEnroll hoàn tất!')
    action_list(data)


if __name__ == '__main__':
    main()
