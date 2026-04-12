"""Face recognizer: embedding extraction (facenet) and optional classifier matching.

Provides a lightweight wrapper that uses `facenet-pytorch` for embeddings when
available and an sklearn classifier saved with `joblib` for ID matching.
"""
import os
import numpy as np

try:
    import torch
    from facenet_pytorch import InceptionResnetV1
    _HAS_FACENET = True
except Exception:
    torch = None
    InceptionResnetV1 = None
    _HAS_FACENET = False

try:
    import joblib
    from sklearn.preprocessing import LabelEncoder
    _HAS_SKLEARN = True
except Exception:
    joblib = None
    LabelEncoder = None
    _HAS_SKLEARN = False


class Recognizer:
    def __init__(self, classifier_path: str = 'models/recognizer.joblib', device: str = 'cpu'):
        self.device = device
        self.classifier_path = classifier_path
        self.classifier = None
        self.label_encoder = None
        if _HAS_FACENET:
            self.embed_model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        else:
            self.embed_model = None

        if _HAS_SKLEARN and os.path.exists(classifier_path):
            try:
                data = joblib.load(classifier_path)
                # expected to be a dict {'clf': clf, 'le': label_encoder}
                if isinstance(data, dict) and 'clf' in data and 'le' in data:
                    self.classifier = data['clf']
                    self.label_encoder = data['le']
                else:
                    # older format: just clf with classes_
                    self.classifier = data
            except Exception:
                self.classifier = None

    def embed(self, pil_img):
        """Return L2-normalized embedding vector from a PIL image crop.

        If facenet is unavailable, returns None.
        """
        if not _HAS_FACENET or self.embed_model is None:
            return None
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        import torchvision.transforms as T
        transf = T.Compose([T.Resize((160, 160)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
        x = transf(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.embed_model(x)
        vec = emb.cpu().numpy().reshape(-1)
        # L2 normalize
        vec = vec / np.linalg.norm(vec)
        return vec

    def match(self, pil_img):
        """Return (label, confidence) if classifier is available, else (None, 0.0)."""
        if self.classifier is None:
            return None, 0.0
        vec = self.embed(pil_img)
        if vec is None:
            return None, 0.0
        try:
            probs = None
            if hasattr(self.classifier, 'predict_proba'):
                probs = self.classifier.predict_proba([vec])[0]
                idx = int(probs.argmax())
                label = self.classifier.classes_[idx] if hasattr(self.classifier, 'classes_') else str(idx)
                conf = float(probs[idx])
            else:
                pred = self.classifier.predict([vec])[0]
                label = pred
                conf = 1.0
            # decode label if label_encoder present
            if self.label_encoder is not None:
                label = str(self.label_encoder.inverse_transform([label])[0])
            return label, conf
        except Exception:
            return None, 0.0


__all__ = ['Recognizer']
