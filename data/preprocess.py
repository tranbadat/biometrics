"""Preprocessing utilities: detect faces (Haar cascade) and crop/resize for training/inference.

Usage example:
  python data/preprocess.py --input data/raw_samples --output data/processed --size 160
"""
import argparse
import os
import cv2


CASCADE_URL = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'


def ensure_cascade(path='haarcascade_frontalface_default.xml'):
    if os.path.exists(path):
        return path
    # download cascade
    import requests
    r = requests.get(CASCADE_URL)
    r.raise_for_status()
    with open(path, 'wb') as f:
        f.write(r.content)
    return path


def process_folder(input_dir, output_dir, size=160, cascade_path=None):
    cascade_path = cascade_path or ensure_cascade()
    detector = cv2.CascadeClassifier(cascade_path)
    os.makedirs(output_dir, exist_ok=True)
    idx = 0
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            path = os.path.join(root, fname)
            img = cv2.imread(path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            if len(faces) == 0:
                continue
            for (x, y, w, h) in faces:
                crop = img[y:y+h, x:x+w]
                resized = cv2.resize(crop, (size, size))
                out_path = os.path.join(output_dir, f'face_{idx:06d}.jpg')
                cv2.imwrite(out_path, resized)
                idx += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--size', type=int, default=160)
    parser.add_argument('--cascade', default=None)
    args = parser.parse_args()
    process_folder(args.input, args.output, size=args.size, cascade_path=args.cascade)


if __name__ == '__main__':
    main()
