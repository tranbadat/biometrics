# Hướng dẫn Dataset

## Cấu trúc thư mục

```
data/
├── processed/          # Ảnh đã sẵn sàng để train (tạo bằng các bước bên dưới)
│   ├── mask/           # Ảnh khuôn mặt đeo khẩu trang
│   └── no_mask/        # Ảnh khuôn mặt không đeo khẩu trang
├── faces/              # Ảnh cho recognition, mỗi subfolder = 1 người
│   ├── nguyen_van_a/
│   └── tran_thi_b/
└── raw_samples/        # Ảnh mẫu nhỏ để test nhanh
```

> **Lưu ý:** Các thư mục `processed/`, `faces/`, `raw_samples/` bị bỏ qua bởi `.gitignore`
> vì dung lượng lớn. Mỗi thành viên tự tải về theo hướng dẫn bên dưới.

---

## Dataset 1 — Mask Classifier (mask / no\_mask)

### Nguồn: Kaggle — Face Mask 12K Images Dataset

- **Link:** https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset
- **Tác giả:** Ashish Jangra
- **Số lượng:** ~12.000 ảnh (5.400 mask + 5.400 no\_mask sau khi merge Train + Validation)
- **License:** CC BY-SA 4.0
- **Mô tả:** Ảnh khuôn mặt chụp thực tế, đã được cắt sát mặt, cân bằng tốt giữa 2 class

### Cách tải và cài đặt

**Bước 1 — Cài Kaggle CLI và cấu hình API key**

```bash
pip install kaggle
```

Tạo API key tại https://www.kaggle.com/settings → Account → API → Create New Token.
File `kaggle.json` tải về, đặt vào:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

**Bước 2 — Tải dataset**

```bash
# Chạy từ thư mục gốc project
kaggle datasets download -d ashishjangra27/face-mask-12k-images-dataset
```

**Bước 3 — Giải nén và sắp xếp**

```bash
unzip -q face-mask-12k-images-dataset.zip -d data/kaggle_raw

mkdir -p data/processed/mask data/processed/no_mask

# Merge Train + Validation vào processed/
cp -r "data/kaggle_raw/Face Mask Dataset/Train/WithMask/."      data/processed/mask/
cp -r "data/kaggle_raw/Face Mask Dataset/Train/WithoutMask/."   data/processed/no_mask/
cp -r "data/kaggle_raw/Face Mask Dataset/Validation/WithMask/." data/processed/mask/
cp -r "data/kaggle_raw/Face Mask Dataset/Validation/WithoutMask/." data/processed/no_mask/

# Dọn file tạm
rm face-mask-12k-images-dataset.zip
rm -rf data/kaggle_raw
```

**Bước 4 — Kiểm tra**

```bash
echo "mask:    $(ls data/processed/mask | wc -l) ảnh"
echo "no_mask: $(ls data/processed/no_mask | wc -l) ảnh"
# Kết quả mong đợi: mask: 5400 | no_mask: 5400
```

**Bước 5 — Train**

```bash
python src/models/train_mask_detector.py \
    --data data/processed \
    --epochs 25 \
    --out models/mask_clf.pth
```

---

## Dataset 2 — Face Recognition (enrollment)

### Cách A: Dùng LFW (Labeled Faces in the Wild)

- **Link:** http://vis-www.cs.umass.edu/lfw/
- **Mô tả:** ~13.000 ảnh khuôn mặt người nổi tiếng, mỗi người 1 subfolder
- **Phù hợp:** demo nhận diện nhiều người, không cần tự chụp

```bash
wget http://vis-www.cs.umass.edu/lfw/lfw.tgz
tar -xzf lfw.tgz
# Chọn những người có nhiều ảnh (≥10 ảnh) để recognition đủ chất lượng
mkdir -p data/faces
# Ví dụ: lấy 5 người để demo
for name in "George_W_Bush" "Colin_Powell" "Tony_Blair" "Donald_Rumsfeld" "Gerhard_Schroeder"; do
    cp -r lfw/$name data/faces/
done
rm -rf lfw lfw.tgz
```

### Cách B: Tự chụp ảnh (khuyến nghị cho bài tập lớn)

Chụp ảnh khuôn mặt của các thành viên nhóm — vừa thực tế hơn, vừa có ảnh đeo khẩu trang để test nhận diện qua mask.

```
data/faces/
├── nguyen_van_a/       # ≥ 20 ảnh, đa dạng góc + ánh sáng
│   ├── no_mask_01.jpg
│   ├── no_mask_02.jpg
│   ├── mask_01.jpg     # có ảnh đeo khẩu trang để test
│   └── ...
└── tran_thi_b/
    └── ...
```

**Gợi ý khi chụp:**
- Ít nhất 20 ảnh/người
- Đa dạng góc mặt: thẳng, nghiêng trái/phải, ngửa nhẹ
- Đa dạng ánh sáng: trong nhà, ngoài trời, đèn bên cạnh
- Chụp cả lúc đeo và không đeo khẩu trang

**Enroll và train:**

```bash
# Enroll từng người
python scripts/enroll.py --name "Nguyen Van A" --images data/faces/nguyen_van_a/
python scripts/enroll.py --name "Tran Thi B"   --images data/faces/tran_thi_b/

# Hoặc train từ toàn bộ thư mục cùng lúc
python scripts/train_recognizer.py --data data/faces --out models/recognizer.joblib
```

---

## Dataset 3 — Tùy chọn bổ sung

| Dataset | Nội dung | Link |
|---|---|---|
| **MaskedFace-Net** | ~70k ảnh mask synthetic, chất lượng cao | https://github.com/aqeelanwar/MaskedFace-Net |
| **MAFA** | Mặt bị che một phần (mask, tay, vật thể) | http://www.escience.cn/people/JunweiHan/MAFA.html |
| **WIDERFace** | 32k ảnh detect mặt đa góc độ | http://shuoyang1213.me/WIDERFACE/ |

---

## Tải ảnh mẫu nhanh (không cần Kaggle)

Dùng script có sẵn để tải vài ảnh test ngay:

```bash
python data/download_datasets.py --sample --out data/raw_samples
python scripts/smoke_predict.py --file data/raw_samples/sample_1.jpg
```
