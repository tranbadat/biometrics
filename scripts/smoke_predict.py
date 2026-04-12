"""Simple smoke-test that posts an image to the local /predict endpoint.

Usage:
  python scripts/smoke_predict.py --file path/to/image.jpg

If the file doesn't exist, run:
  python data/download_datasets.py --sample --out data/raw_samples
"""
import argparse
import os
import requests

API = 'http://localhost:8000/predict'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='Image file to send', default='data/raw_samples/sample_1.jpg')
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f'File not found: {args.file}')
        print('Run: python data/download_datasets.py --sample --out data/raw_samples')
        return

    with open(args.file, 'rb') as f:
        files = {'file': (os.path.basename(args.file), f, 'image/jpeg')}
        print('Sending request to', API)
        r = requests.post(API, files=files)
        try:
            print('Status:', r.status_code)
            print(r.json())
        except Exception:
            print('Response text:', r.text)

if __name__ == '__main__':
    main()
