"""Training script: MobileNetV2 mask/no_mask classifier.

Dataset structure expected:
    {data}/
    ├── mask/
    │   ├── img001.jpg
    │   └── ...
    └── no_mask/
        ├── img001.jpg
        └── ...

Chiến lược training 2 phase:
  Phase 1 (epoch 1 → unfreeze_epoch): backbone MobileNetV2 đóng băng,
      chỉ train classification head mới. LR cao hơn.
  Phase 2 (epoch unfreeze_epoch+1 → end): mở backbone, fine-tune toàn bộ.
      LR giảm 10 lần để không phá vỡ features đã học.

Usage:
    python src/models/train_mask_detector.py \\
        --data data/processed \\
        --epochs 25 \\
        --unfreeze-epoch 8 \\
        --out models/mask_clf.pth
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.models as tv_models
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from tqdm import tqdm

try:
    from sklearn.metrics import classification_report, confusion_matrix
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

try:
    import cv2
    from src.preprocessing.image_utils import is_sharp_enough
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(num_classes: int = 2, freeze_backbone: bool = True) -> nn.Module:
    """Tải MobileNetV2 pretrained, thay classification head thành num_classes chiều."""
    weights = tv_models.MobileNet_V2_Weights.IMAGENET1K_V1
    model = tv_models.mobilenet_v2(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def unfreeze_backbone(model: nn.Module) -> None:
    """Mở backbone để fine-tune toàn bộ (Phase 2)."""
    for param in model.features.parameters():
        param.requires_grad = True


# ---------------------------------------------------------------------------
# Dataset & Transforms
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transform() -> T.Compose:
    """Augmentation cho training set."""
    return T.Compose([
        T.Resize((256, 256)),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.08),
        T.RandomRotation(degrees=15),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.3),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform() -> T.Compose:
    """Không augment cho validation set."""
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def filter_blurry_indices(dataset: ImageFolder, threshold: float = 80.0):
    """Lọc bỏ ảnh blur bằng Laplacian variance (dùng Sobel).

    Ảnh mờ → gradient yếu → variance thấp → không đưa vào training.

    Args:
        dataset: ImageFolder đã load
        threshold: ngưỡng Laplacian variance (≥80 là ảnh đủ nét)

    Returns:
        list index của ảnh đủ sắc nét
    """
    if not _HAS_CV2:
        print('[warn] OpenCV không có sẵn — bỏ qua bước lọc ảnh mờ')
        return list(range(len(dataset)))

    sharp_indices = []
    skipped = 0
    for i, (path, _) in enumerate(dataset.samples):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped += 1
            continue
        if is_sharp_enough(img, threshold=threshold):
            sharp_indices.append(i)
        else:
            skipped += 1

    print(f'  Lọc ảnh mờ: giữ {len(sharp_indices)}/{len(dataset.samples)} '
          f'(bỏ {skipped})')
    return sharp_indices


def compute_class_weights(dataset: ImageFolder, indices: list) -> torch.Tensor:
    """Tính class weights để bù class imbalance.

    Nếu dataset mất cân bằng (vd: 8000 mask vs 2000 no_mask),
    CrossEntropyLoss sẽ bị bias về class đa số.
    Weight = total / (num_classes × count_per_class).
    """
    labels = [dataset.targets[i] for i in indices]
    counts = np.bincount(labels)
    total  = len(labels)
    weights = total / (len(counts) * counts.astype(np.float32))
    print(f'  Class counts: {dict(zip(dataset.classes, counts))}')
    print(f'  Class weights: {dict(zip(dataset.classes, weights.round(3)))}')
    return torch.tensor(weights, dtype=torch.float32)


def split_indices(n: int, val_ratio: float, seed: int = 42):
    """Tách indices thành train/val, có seed để reproducible."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = int(n * val_ratio)
    return perm[n_val:], perm[:n_val]  # train_idx, val_idx


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, epoch: int):
    model.train()
    total_loss = correct = total = 0

    pbar = tqdm(loader, desc=f'  Train ep{epoch}', leave=False, unit='batch')
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += bs

        pbar.set_postfix(loss=f'{loss.item():.4f}',
                         acc=f'{correct / total:.4f}')

    return total_loss / total, correct / total


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device, epoch: int):
    model.eval()
    total_loss = correct = total = 0
    all_preds  = []
    all_labels = []

    pbar = tqdm(loader, desc=f'  Val   ep{epoch}', leave=False, unit='batch')
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)

        logits = model(imgs)
        loss   = criterion(logits, labels)
        preds  = logits.argmax(1)

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        correct    += (preds == labels).sum().item()
        total      += bs

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Train MobileNetV2 mask/no_mask classifier'
    )
    parser.add_argument('--data', required=True,
                        help='Dataset root với subfolders mask/ và no_mask/')
    parser.add_argument('--epochs', type=int, default=25,
                        help='Tổng số epoch (default: 25)')
    parser.add_argument('--unfreeze-epoch', type=int, default=8,
                        help='Epoch bắt đầu unfreeze backbone (default: 8)')
    parser.add_argument('--batch', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate Phase 1 (default: 1e-3)')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Tỉ lệ validation split (default: 0.2)')
    parser.add_argument('--blur-threshold', type=float, default=80.0,
                        help='Laplacian variance threshold để lọc ảnh mờ (default: 80)')
    parser.add_argument('--no-filter-blur', action='store_true',
                        help='Bỏ qua bước lọc ảnh mờ')
    parser.add_argument('--out', default='models/mask_clf.pth',
                        help='Đường dẫn lưu model tốt nhất (default: models/mask_clf.pth)')
    parser.add_argument('--device', default='auto',
                        help='cpu | cuda | mps | auto (default: auto)')
    parser.add_argument('--workers', type=int, default=2,
                        help='Số DataLoader workers (default: 2)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # ── Device ──────────────────────────────────────────────────────────────
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    print(f'Device: {device}')

    # ── Dataset ─────────────────────────────────────────────────────────────
    if not os.path.isdir(args.data):
        raise FileNotFoundError(f'Dataset không tồn tại: {args.data}')

    # Tạo 2 dataset cùng folder nhưng khác transform (để train/val tách biệt)
    train_ds_full = ImageFolder(args.data, transform=get_train_transform())
    val_ds_full   = ImageFolder(args.data, transform=get_val_transform())

    class_names = train_ds_full.classes
    num_classes = len(class_names)
    print(f'Classes ({num_classes}): {class_names}')
    print(f'Tổng samples: {len(train_ds_full)}')

    # Lọc ảnh mờ
    if not args.no_filter_blur:
        print('Lọc ảnh mờ (Laplacian variance)...')
        valid_idx = filter_blurry_indices(train_ds_full, args.blur_threshold)
    else:
        valid_idx = list(range(len(train_ds_full)))

    # Tách train/val
    train_idx, val_idx = split_indices(len(valid_idx), args.val_split, args.seed)
    # Ánh xạ lại qua valid_idx
    train_idx = [valid_idx[i] for i in train_idx]
    val_idx   = [valid_idx[i] for i in val_idx]

    print(f'Train: {len(train_idx)} | Val: {len(val_idx)}')

    # Class weights (tính trên train set)
    print('Tính class weights...')
    class_weights = compute_class_weights(train_ds_full, train_idx).to(device)

    train_loader = DataLoader(
        Subset(train_ds_full, train_idx),
        batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=(device != 'cpu'),
    )
    val_loader = DataLoader(
        Subset(val_ds_full, val_idx),
        batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=(device != 'cpu'),
    )

    # ── Model ────────────────────────────────────────────────────────────────
    print('Tải MobileNetV2 (pretrained ImageNet)...')
    model = build_model(num_classes=num_classes, freeze_backbone=True).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Phase 1: chỉ train head
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.unfreeze_epoch
    )

    # ── Training ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    best_val_acc  = 0.0
    best_epoch    = 0
    phase         = 1

    print(f'\nPhase 1: train head ({args.unfreeze_epoch} epoch), '
          f'Phase 2: fine-tune toàn bộ ({args.epochs - args.unfreeze_epoch} epoch)')
    print('─' * 70)

    for epoch in range(1, args.epochs + 1):

        # Chuyển sang Phase 2
        if epoch == args.unfreeze_epoch + 1 and phase == 1:
            phase = 2
            print(f'\n[Epoch {epoch}] → Phase 2: unfreeze backbone, LR giảm 10×')
            unfreeze_backbone(model)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.unfreeze_epoch
            )

        t0 = time.time()
        cur_lr = optimizer.param_groups[0]['lr']

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc, val_preds, val_labels = eval_one_epoch(
            model, val_loader, criterion, device, epoch
        )
        scheduler.step()

        elapsed = time.time() - t0
        marker = ' *' if val_acc > best_val_acc else ''

        print(f'Ep {epoch:3d}/{args.epochs} | ph{phase} | lr={cur_lr:.2e} | '
              f'train loss={train_loss:.4f} acc={train_acc:.4f} | '
              f'val loss={val_loss:.4f} acc={val_acc:.4f} | '
              f'{elapsed:.1f}s{marker}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'classes':     class_names,
                'val_acc':     val_acc,
            }, args.out)

    # ── Kết quả cuối ─────────────────────────────────────────────────────────
    print('\n' + '─' * 70)
    print(f'Best val accuracy: {best_val_acc:.4f} (epoch {best_epoch})')
    print(f'Model đã lưu tại: {args.out}')

    if _HAS_SKLEARN and val_preds:
        print('\nClassification report (last epoch):')
        print(classification_report(val_labels, val_preds,
                                    target_names=class_names, digits=4))
        cm = confusion_matrix(val_labels, val_preds)
        print('Confusion matrix:')
        header = '        ' + '  '.join(f'{c:>8}' for c in class_names)
        print(header)
        for i, row in enumerate(cm):
            print(f'{class_names[i]:>8}  ' + '  '.join(f'{v:>8}' for v in row))


if __name__ == '__main__':
    main()
