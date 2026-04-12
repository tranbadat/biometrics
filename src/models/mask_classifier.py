import os
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    import torchvision.models as tv_models
    from PIL import Image
    _HAS_TORCH = True
except Exception:
    torch = None
    nn = None
    T = None
    tv_models = None
    Image = None
    _HAS_TORCH = False

_WARNED_NO_WEIGHTS = False


class MaskClassifier:
    """MobileNetV2 mask/no_mask classifier.

    Chỉ dùng model khi đã có file weights đã train. Nếu không có weights,
    tự động dùng heuristic thay vì dùng model với random head (predict bừa).

    Weights tạo ra bởi:
        python src/models/train_mask_detector.py --data data/processed --out models/mask_clf.pth
    """

    def __init__(self, weights_path: str = 'models/mask_clf.pth', device: str = 'cpu'):
        global _WARNED_NO_WEIGHTS
        self.device = device
        self.weights_path = weights_path
        self.model = None
        self.classes = ['mask', 'no_mask']  # default, overridden by checkpoint

        if not _HAS_TORCH:
            return

        if not weights_path or not os.path.exists(weights_path):
            if not _WARNED_NO_WEIGHTS:
                print(
                    f'[MaskClassifier] Không tìm thấy weights tại "{weights_path}". '
                    f'Dùng heuristic thay thế.\n'
                    f'  → Để train: python src/models/train_mask_detector.py '
                    f'--data data/processed --out {weights_path}'
                )
                _WARNED_NO_WEIGHTS = True
            return

        # Load checkpoint
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

        # Hỗ trợ 2 format: dict checkpoint (từ train_mask_detector.py) hoặc state_dict thẳng
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            state_dict   = checkpoint['model_state']
            self.classes = checkpoint.get('classes', self.classes)
            val_acc      = checkpoint.get('val_acc', None)
            if val_acc is not None:
                print(f'[MaskClassifier] Loaded "{weights_path}" '
                      f'(val_acc={val_acc:.4f}, classes={self.classes})')
        else:
            # state_dict thẳng (format cũ)
            state_dict = checkpoint

        num_classes = len(self.classes)
        self.model  = self._build_model(num_classes)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _build_model(self, num_classes: int = 2):
        weights = tv_models.MobileNet_V2_Weights.IMAGENET1K_V1
        m = tv_models.mobilenet_v2(weights=weights)
        in_features = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes),
        )
        return m

    def predict(self, pil_img):
        """Phân loại mask/no_mask từ PIL.Image crop khuôn mặt.

        Returns:
            (label: 'mask'|'no_mask', confidence: float)
        """
        if self.model is not None:
            x = self.transform(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(x)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
                idx    = int(probs.argmax())
                label  = self.classes[idx] if idx < len(self.classes) else 'no_mask'
                return label, float(probs[idx])

        # Heuristic: phân tích màu sắc vùng mũi-miệng
        # Khẩu trang thường có màu trắng/xanh → saturation thấp, brightness cao
        arr = np.array(pil_img.convert('RGB')).astype(np.float32)
        if arr.ndim != 3:
            return 'no_mask', 0.6

        # Lấy nửa dưới ảnh (vùng mũi-miệng)
        h = arr.shape[0]
        lower = arr[h // 2:, :, :]

        r, g, b = lower[:, :, 0], lower[:, :, 1], lower[:, :, 2]
        brightness  = (r + g + b) / 3.0
        max_rgb     = np.maximum(np.maximum(r, g), b)
        min_rgb     = np.minimum(np.minimum(r, g), b)
        saturation  = np.where(max_rgb > 0, (max_rgb - min_rgb) / max_rgb, 0)

        mean_bright = float(brightness.mean())
        mean_sat    = float(saturation.mean())

        # Khẩu trang y tế: bright > 160, sat < 0.25
        # Khẩu trang vải màu: sat thấp hơn da mặt
        if mean_bright > 155 and mean_sat < 0.30:
            return 'mask', 0.65
        elif mean_sat < 0.20:
            return 'mask', 0.60
        else:
            return 'no_mask', 0.70


__all__ = ['MaskClassifier']
