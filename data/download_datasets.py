"""Dataset download helper and small-sample fetcher.

This script provides:
- instructions and placeholders for large public datasets (manual download recommended)
- a small-sample downloader to fetch a few example images for quick testing/PoC

Usage:
  python data/download_datasets.py --sample

For full datasets, follow instructions printed by `--list`.
"""
import argparse
import os
import requests
from tqdm import tqdm

SAMPLE_IMAGES = [
    "https://upload.wikimedia.org/wikipedia/commons/8/89/Portrait_Placeholder.png",
    "https://upload.wikimedia.org/wikipedia/commons/3/3f/Fronalpstock_big.jpg",
]

DATASET_LINKS = {
    "MaskedFaceNet": "https://github.com/aqeelanwar/MaskedFace-Net (follow repo to download)",
    "MAFA": "http://www.escience.cn/people/JunweiHan/MAFA.html",
    "WIDERFace": "http://shuoyang1213.me/WIDERFACE/",
}


def download_file(url, dest_path):
    r = requests.get(url, stream=True)
    r.raise_for_status()
    total = int(r.headers.get('content-length', 0))
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, 'wb') as f:
        for data in tqdm(r.iter_content(chunk_size=8192), total=(total // 8192), unit='KB'):
            f.write(data)


def download_sample(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i, url in enumerate(SAMPLE_IMAGES):
        dest = os.path.join(out_dir, f'sample_{i+1}.jpg')
        print(f'Downloading {url} -> {dest}')
        download_file(url, dest)


def list_datasets():
    print("Public datasets and links:")
    for name, link in DATASET_LINKS.items():
        print(f'- {name}: {link}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', action='store_true', help='Download small sample images for PoC')
    parser.add_argument('--out', default='data/raw_samples', help='Output directory for samples')
    parser.add_argument('--list', action='store_true', help='List known dataset links and hints')
    args = parser.parse_args()

    if args.list:
        list_datasets()
    if args.sample:
        download_sample(args.out)
    if not args.sample and not args.list:
        parser.print_help()


if __name__ == '__main__':
    main()
