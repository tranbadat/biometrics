try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from PIL import Image
    _HAS_TORCH = True
except Exception:
    torch = None
    nn = None
    T = None
    Image = None
    _HAS_TORCH = False


class MaskClassifier:
    """Lightweight mask classifier wrapper.

    If PyTorch is available, loads a MobileNetV2 backbone and a 2-class head.
    If not available, falls back to a simple heuristic based on channel statistics.
    """

    def __init__(self, weights_path: str = None, device: str = 'cpu'):
        self.device = device
        self.weights_path = weights_path
        if _HAS_TORCH:
            self.model = self._build_model()
            if weights_path and os.path.exists(weights_path):
                self.model.load_state_dict(torch.load(weights_path, map_location=device))
            self.model.to(self.device)
            self.model.eval()
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.model = None

    def _build_model(self):
        import torchvision.models as models
        m = models.mobilenet_v2(pretrained=True)
        in_features = m.classifier[1].in_features
        # Replace classifier with 2-class head
        m.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, 2))
        return m

    def predict(self, pil_img):
        """Predict label and confidence for a PIL.Image crop.

        Returns: (label: 'mask'|'no_mask', confidence: float)
        """
        if _HAS_TORCH and self.model is not None:
            x = self.transform(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                idx = int(probs.argmax())
                label = 'mask' if idx == 0 else 'no_mask'
                return label, float(probs[idx])
        else:
            # Fallback heuristic (mean green channel)
            arr = np.array(pil_img.convert('RGB'))
            mean_g = float(arr[:, :, 1].mean()) if arr.ndim == 3 else 0.0
            label = 'mask' if mean_g < 100 else 'no_mask'
            conf = 0.6 if label == 'mask' else 0.6
            return label, conf


import os
import numpy as np

__all__ = ['MaskClassifier']
