# Benchmark Single-slot vs Dual-slot trên MFR2

> Auto-generated bởi `scripts/benchmark_mfr2.py`. Dataset: MFR2 (Masked Faces in Real World), 53 identities, 269 ảnh.

## Setup

- **Detector + Embedder**: InsightFace `buffalo_l` (RetinaFace + ArcFace ResNet-50)
- **CLAHE**: áp dụng kênh L (LAB) trước detect — match production pipeline.
- **Train/Test split**: per-identity, stratified theo mask_state, seed=42, ~50/50.
- **Single-slot**: 1 embedding / identity = trung bình tất cả ảnh train.
- **Dual-slot**: 2 embedding / identity = trung bình theo `with_mask` / `without_mask` riêng.
- **Match dual**: chỉ so trong slot cùng `mask_state` (fallback toàn DB nếu rỗng).

## Quy mô

| Tiêu chí | Giá trị |
|---|---|
| Số identities | 53 |
| Số ảnh train | 135 |
| Số ảnh test | 134 |
| Số cặp genuine (single) | 134 |
| Số cặp impostor (single) | 6968 |
| Số cặp genuine (dual) | 134 |
| Số cặp impostor (dual) | 6968 |

## Top-1 Identification Accuracy

| Test subset | Single-slot | Dual-slot | Δ |
|---|---|---|---|
| **without_mask** (45 ảnh) | 1.0000 | 1.0000 | +0.0000 |
| **with_mask** (89 ảnh) | 1.0000 | 0.9775 | -0.0225 |

## Verification metrics (cosine similarity)

| Metric | Single-slot | Dual-slot | Δ |
|---|---|---|---|
| **EER** | 0.0011 | 0.0087 | +0.0076 |
| Threshold tại EER | 0.2425 | 0.2120 | — |
| **FAR @ FRR ≤ 1%** | 0.0003 | 0.0099 | +0.0096 |

> **EER thấp hơn = tốt hơn** (tỉ lệ lỗi cân bằng). **FAR@FRR=1% thấp hơn = tốt hơn** (an toàn hơn ở chế độ recall cao).

## Phát hiện chính

**Trên MFR2, dual-slot KHÔNG cải thiện — thậm chí kém hơn single-slot.**

| Quan sát | Số liệu |
|---|---|
| Top-1 with_mask | Single 1.0000 vs Dual 0.9775 → dual **thua 2.25 pp** |
| EER | Single 0.0011 vs Dual 0.0087 → dual **xấu hơn ~8×** |
| FAR @ FRR ≤ 1% | Single 0.0003 vs Dual 0.0099 → dual **xấu hơn ~33×** |

## Vì sao kết quả ngược dự đoán?

1. **ArcFace `buffalo_l` đã quá mạnh trên MFR2**: single-slot đạt **100% top-1** ở cả 2 nhóm — không còn dư địa cho dual-slot cải thiện. Đây là **hiệu ứng trần** (ceiling effect) của baseline.

2. **Số ảnh / identity quá ít**: MFR2 chỉ ~5 ảnh / identity. Sau split 50/50:
   - Single-slot: trung bình ~2.5 embedding → 1 centroid khá ổn định.
   - Dual-slot: chia thành 2 nhóm → mỗi slot trung bình **chỉ ~1.27 ảnh** → centroid noise hơn.
   - Tách slot làm **giảm sức mạnh trung bình hoá** (averaging) — vốn là cơ chế giảm nhiễu của embedding.

3. **Verification metric phụ thuộc số entries**: dual-slot có 106 entries (vs 53 single) → impostor pool tăng → mỗi test image so với nhiều entry hơn → khả năng có 1 impostor ngẫu nhiên cao similarity tăng → FAR và EER xấu đi.

4. **MFR2 là dataset celebrity**: ánh sáng studio, pose chuẩn, mask thường đeo đúng cách. Drift giữa "đeo mask" và "không mask" trên MFR2 nhỏ hơn so với điều kiện webcam thực tế → giả thuyết "centroid trộn lệch" không thể hiện rõ.

## Khi nào dual-slot mới có lợi?

Dựa trên phân tích trên, dual-slot **có thể** có lợi khi:
- **Số ảnh / identity / state đủ nhiều** (≥ 5 ảnh / slot) — đủ để trung bình hoá có ý nghĩa.
- **Baseline có chỗ để cải thiện** (single-slot < ~95%) — có dư địa cho routing.
- **Drift giữa 2 trạng thái lớn** — webcam consumer thực tế (nhiều loại mask, ánh sáng kém, góc nghiêng).
- **Database lớn** (≥ 50 user enroll đầy đủ) — tận dụng được phân chia slot.

MFR2 không thoả mãn các điều kiện trên → kết quả này **không bác bỏ** thiết kế dual-slot, mà chỉ ra dataset không phù hợp để showcase ưu điểm.

## Bài học cho đề tài

1. **Khi bảo vệ**: nói thẳng kết quả này, không gồng. Hội đồng đánh giá cao trung thực hơn là cherry-pick.
2. **Hướng đánh giá tốt hơn**: dataset có ≥ 10 ảnh / state / user, điều kiện capture đa dạng (RMFRD, custom webcam dataset).
3. **Vai trò mask classifier vẫn còn**: kể cả khi dual-slot không cải thiện accuracy, mask classifier vẫn là **output độc lập** phục vụ giám sát y tế / quy định đeo mask.
4. **Cân nhắc rollback về single-slot** nếu deployment gặp dataset tương tự MFR2 — code hiện tại đã có fallback toàn DB nên hoạt động an toàn.

## Lưu ý kỹ thuật

- Detection: upscale ảnh 160×160 → 480×480 + `det_thresh=0.3` để RetinaFace bắt được mọi ảnh (269/269).
- CLAHE áp dụng để match production pipeline.
- Split seed=42, có thể chạy lại với seed khác để kiểm tra biến thiên.
