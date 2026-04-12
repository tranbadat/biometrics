Dataset helper notes

Public datasets (links / quick notes):

- MaskedFace-Net: https://github.com/aqeelanwar/MaskedFace-Net — large; follow repo instructions to download
- MAFA: http://www.escience.cn/people/JunweiHan/MAFA.html — face mask dataset
- WIDERFace: http://shuoyang1213.me/WIDERFACE/ — wide-scale face detection dataset

Quick PoC:

1. Download a few sample images for testing:

```bash
python data/download_datasets.py --sample --out data/raw_samples
```

2. Preprocess (face crop & resize):

```bash
python data/preprocess.py --input data/raw_samples --output data/processed --size 160
```

3. After verification, download full datasets manually and place raw images under `data/raw_full/` then run the same `preprocess.py` to prepare training data.

Notes:
- Large public datasets often require manual acceptance or Google Drive downloads; the helper script only fetches small samples and prints links.
- For production training, use RetinaFace/MTCNN for robust face detection instead of Haar cascades.

RetinaFace installation (optional, recommended for better detection):

```bash
# On many systems, install via pip
pip install retinaface

# If installation fails due to native deps, consider using the prebuilt wheel or follow the project instructions:
# https://github.com/serengil/retinaface
```

If you plan to use `retinaface` for inference, install `torch`/`torchvision` as well (see project `requirements.txt`).
