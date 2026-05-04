"""Benchmark Single-slot vs Dual-slot DB trên MFR2 (Masked Faces in Real World).

Evaluation theo Q11 trong docs/report-explain.md:
- Top-1 identification accuracy (chia theo nhóm có/không mask)
- EER (Equal Error Rate) qua sweep threshold
- FAR @ FRR=1%

Usage:
    python scripts/benchmark_mfr2.py
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

ROOT = Path(__file__).resolve().parents[1]
MFR2 = ROOT / "data" / "mfr2" / "mfr2"
LABELS = MFR2 / "mfr2_labels.txt"
OUT = ROOT / "docs" / "benchmark-results.md"


# ── Parse labels ──────────────────────────────────────────────────────────
def parse_labels() -> dict[tuple[str, int], str]:
    """Trả về dict[(identity, image_num)] -> 'with_mask' | 'without_mask'."""
    out: dict[tuple[str, int], str] = {}
    for line in LABELS.read_text().strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        identity, num, mask_type = parts
        state = "without_mask" if mask_type == "no-mask" else "with_mask"
        out[(identity, int(num))] = state
    return out


# ── CLAHE (giống pipeline) ────────────────────────────────────────────────
def apply_clahe(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# ── Load all embeddings ───────────────────────────────────────────────────
def collect_embeddings(app: FaceAnalysis, labels: dict) -> list[dict]:
    """Trả về list[{identity, num, mask_state, emb, path}]."""
    samples = []
    skipped = 0
    for (identity, num), mask_state in sorted(labels.items()):
        path = MFR2 / identity / f"{identity}_{num:04d}.png"
        if not path.exists():
            skipped += 1
            continue
        bgr = cv2.imread(str(path))
        if bgr is None:
            skipped += 1
            continue
        # MFR2 ảnh 160x160 (đã pre-crop) → upscale để RetinaFace detect được
        h, w = bgr.shape[:2]
        if max(h, w) < 320:
            scale = 480 / max(h, w)
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        bgr = apply_clahe(bgr)
        faces = app.get(bgr)
        if not faces:
            skipped += 1
            continue
        face = max(faces, key=lambda f: float(f.det_score))
        samples.append({
            "identity": identity,
            "num": num,
            "mask_state": mask_state,
            "emb": face.normed_embedding.astype(np.float32),
            "path": str(path),
        })
    print(f"  Trích xuất: {len(samples)}/{len(labels)} embedding (skip {skipped})")
    return samples


# ── Train/test split per identity, stratified theo mask_state ────────────
def split_train_test(samples: list[dict], seed: int = 42) -> tuple[list, list]:
    rng = np.random.default_rng(seed)
    by_id: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_id[s["identity"]].append(s)

    train, test = [], []
    for identity, items in by_id.items():
        # Tách theo mask_state để stratify
        with_mask = [x for x in items if x["mask_state"] == "with_mask"]
        no_mask = [x for x in items if x["mask_state"] == "without_mask"]

        for group in (with_mask, no_mask):
            if len(group) == 0:
                continue
            idx = rng.permutation(len(group))
            shuffled = [group[i] for i in idx]
            half = max(1, len(shuffled) // 2)
            # Train = nửa đầu, test = nửa sau (đảm bảo cả 2 phía đều có ít nhất 1)
            if len(shuffled) == 1:
                # Identity chỉ có 1 ảnh trong group: dùng làm train, không có test ở group này
                train.extend(shuffled)
            else:
                train.extend(shuffled[:half])
                test.extend(shuffled[half:])
    return train, test


# ── Build DBs ─────────────────────────────────────────────────────────────
def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def build_single_slot(train: list[dict]) -> dict[str, np.ndarray]:
    """1 vector / identity = trung bình tất cả embedding train, L2-normalize."""
    by_id: dict[str, list[np.ndarray]] = defaultdict(list)
    for s in train:
        by_id[s["identity"]].append(s["emb"])
    return {k: l2_normalize(np.mean(v, axis=0)) for k, v in by_id.items()}


def build_dual_slot(train: list[dict]) -> dict[str, np.ndarray]:
    """1 vector / (identity, mask_state) — 2 slot / user."""
    by_key: dict[str, list[np.ndarray]] = defaultdict(list)
    for s in train:
        key = f"{s['identity']}__{s['mask_state']}"
        by_key[key].append(s["emb"])
    return {k: l2_normalize(np.mean(v, axis=0)) for k, v in by_key.items()}


# ── Match ─────────────────────────────────────────────────────────────────
def match_single(emb: np.ndarray, db: dict) -> tuple[str, float]:
    best_id, best_sim = None, -1.0
    for ident, ref in db.items():
        sim = float(np.dot(emb, ref))
        if sim > best_sim:
            best_id, best_sim = ident, sim
    return best_id, best_sim


def match_dual(emb: np.ndarray, mask_state: str, db: dict) -> tuple[str, float]:
    candidates = {k: v for k, v in db.items() if k.endswith(f"__{mask_state}")}
    if not candidates:
        candidates = db
    best_key, best_sim = None, -1.0
    for k, ref in candidates.items():
        sim = float(np.dot(emb, ref))
        if sim > best_sim:
            best_key, best_sim = k, sim
    if best_key is None:
        return None, best_sim
    return best_key.split("__", 1)[0], best_sim


# ── Metrics: EER và FAR@FRR=1% ────────────────────────────────────────────
def compute_eer_far_at_frr(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float, float]:
    """Trả về (eer, threshold_at_eer, far_at_frr_1pct)."""
    thrs = np.linspace(-1.0, 1.0, 4001)
    fars, frrs = [], []
    for t in thrs:
        far = float(np.mean(impostor >= t))  # impostor pass = false accept
        frr = float(np.mean(genuine < t))    # genuine reject = false reject
        fars.append(far)
        frrs.append(frr)
    fars = np.array(fars)
    frrs = np.array(frrs)
    diff = np.abs(fars - frrs)
    idx_eer = int(np.argmin(diff))
    eer = (fars[idx_eer] + frrs[idx_eer]) / 2.0
    thr_eer = float(thrs[idx_eer])

    # FAR @ FRR ≤ 1%
    valid = np.where(frrs <= 0.01)[0]
    far_at_frr1 = float(fars[valid].min()) if len(valid) > 0 else 1.0
    return eer, thr_eer, far_at_frr1


def collect_pairs(test: list[dict], db: dict, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Sinh similarity (genuine, impostor) bằng cách so từng test_emb với từng entry trong db."""
    genuine, impostor = [], []
    for s in test:
        for key, ref in db.items():
            ident = key.split("__")[0] if mode == "dual" else key
            # Với dual-slot, chỉ so trong slot cùng mask_state (giống inference)
            if mode == "dual" and not key.endswith(f"__{s['mask_state']}"):
                continue
            sim = float(np.dot(s["emb"], ref))
            if ident == s["identity"]:
                genuine.append(sim)
            else:
                impostor.append(sim)
    return np.array(genuine), np.array(impostor)


# ── Pipeline đánh giá ─────────────────────────────────────────────────────
def evaluate(samples: list[dict]) -> dict:
    train, test = split_train_test(samples)
    print(f"  Train: {len(train)}, Test: {len(test)}")
    print(f"  Train identities: {len(set(s['identity'] for s in train))}")
    print(f"  Test identities:  {len(set(s['identity'] for s in test))}")

    # Lọc test: chỉ giữ identity có trong train (open-set là chuyện khác)
    train_ids = set(s["identity"] for s in train)
    test = [s for s in test if s["identity"] in train_ids]

    db_single = build_single_slot(train)
    db_dual = build_dual_slot(train)
    print(f"  DB single-slot: {len(db_single)} entries")
    print(f"  DB dual-slot:   {len(db_dual)} entries")

    # Top-1 accuracy chia theo mask_state
    def top1(predict_fn) -> dict:
        results = {"with_mask": [0, 0], "without_mask": [0, 0]}  # [correct, total]
        for s in test:
            pred, _ = predict_fn(s)
            ok = (pred == s["identity"])
            results[s["mask_state"]][0] += int(ok)
            results[s["mask_state"]][1] += 1
        return {
            k: (v[0] / v[1] if v[1] > 0 else 0.0, v[1])
            for k, v in results.items()
        }

    acc_single = top1(lambda s: match_single(s["emb"], db_single))
    acc_dual = top1(lambda s: match_dual(s["emb"], s["mask_state"], db_dual))

    # EER & FAR@FRR=1%
    g_single, i_single = collect_pairs(test, db_single, "single")
    g_dual, i_dual = collect_pairs(test, db_dual, "dual")

    eer_s, thr_s, far_s = compute_eer_far_at_frr(g_single, i_single)
    eer_d, thr_d, far_d = compute_eer_far_at_frr(g_dual, i_dual)

    return {
        "n_train": len(train),
        "n_test": len(test),
        "n_identities": len(train_ids),
        "single": {
            "acc": acc_single,
            "eer": eer_s,
            "thr_eer": thr_s,
            "far_at_frr1": far_s,
            "n_genuine": len(g_single),
            "n_impostor": len(i_single),
        },
        "dual": {
            "acc": acc_dual,
            "eer": eer_d,
            "thr_eer": thr_d,
            "far_at_frr1": far_d,
            "n_genuine": len(g_dual),
            "n_impostor": len(i_dual),
        },
    }


# ── Output Markdown ───────────────────────────────────────────────────────
def write_report(r: dict) -> None:
    s, d = r["single"], r["dual"]
    md = f"""# Benchmark Single-slot vs Dual-slot trên MFR2

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
| Số identities | {r["n_identities"]} |
| Số ảnh train | {r["n_train"]} |
| Số ảnh test | {r["n_test"]} |
| Số cặp genuine (single) | {s["n_genuine"]} |
| Số cặp impostor (single) | {s["n_impostor"]} |
| Số cặp genuine (dual) | {d["n_genuine"]} |
| Số cặp impostor (dual) | {d["n_impostor"]} |

## Top-1 Identification Accuracy

| Test subset | Single-slot | Dual-slot | Δ |
|---|---|---|---|
| **without_mask** ({s["acc"]["without_mask"][1]} ảnh) | {s["acc"]["without_mask"][0]:.4f} | {d["acc"]["without_mask"][0]:.4f} | {d["acc"]["without_mask"][0] - s["acc"]["without_mask"][0]:+.4f} |
| **with_mask** ({s["acc"]["with_mask"][1]} ảnh) | {s["acc"]["with_mask"][0]:.4f} | {d["acc"]["with_mask"][0]:.4f} | {d["acc"]["with_mask"][0] - s["acc"]["with_mask"][0]:+.4f} |

## Verification metrics (cosine similarity)

| Metric | Single-slot | Dual-slot | Δ |
|---|---|---|---|
| **EER** | {s["eer"]:.4f} | {d["eer"]:.4f} | {d["eer"] - s["eer"]:+.4f} |
| Threshold tại EER | {s["thr_eer"]:.4f} | {d["thr_eer"]:.4f} | — |
| **FAR @ FRR ≤ 1%** | {s["far_at_frr1"]:.4f} | {d["far_at_frr1"]:.4f} | {d["far_at_frr1"] - s["far_at_frr1"]:+.4f} |

> **EER thấp hơn = tốt hơn** (tỉ lệ lỗi cân bằng). **FAR@FRR=1% thấp hơn = tốt hơn** (an toàn hơn ở chế độ recall cao).

## Diễn giải

- Cột **Δ** dương ở Top-1 → dual-slot tốt hơn.
- Cột **Δ** âm ở EER / FAR@FRR=1% → dual-slot tốt hơn.
- Nếu cải thiện chủ yếu rơi vào nhóm **with_mask** → đúng giả thuyết: routing theo mask state giúp ArcFace xử lý tốt hơn vùng bị occlusion.
- Nếu nhóm **without_mask** không đổi hoặc giảm nhẹ → bình thường: ArcFace pretrained vốn rất mạnh với ảnh thường, dual-slot không thêm lợi ích đáng kể.

## Lưu ý

- MFR2 chỉ có ~5 ảnh / identity → DB sau split khá nhỏ; biên độ noise cao.
- Một số identity chỉ có ảnh `with_mask` hoặc chỉ `without_mask` → fallback toàn DB hoạt động.
- ArcFace `buffalo_l` đã được train trên dữ liệu có occlusion → baseline single-slot vốn đã khá mạnh.
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(md, encoding="utf-8")
    print(f"\n  Đã lưu: {OUT}")


def main() -> None:
    print("== MFR2 Benchmark: Single-slot vs Dual-slot ==\n")
    print("[1/4] Load InsightFace buffalo_l ...")
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(320, 320), det_thresh=0.3)

    print("[2/4] Parse labels ...")
    labels = parse_labels()
    print(f"  {len(labels)} (identity, image_num) entries")

    print("[3/4] Trích xuất embedding (có thể mất 1-2 phút) ...")
    samples = collect_embeddings(app, labels)

    print("[4/4] Đánh giá ...")
    results = evaluate(samples)

    # In ngắn ra console
    s, d = results["single"], results["dual"]
    print(f"\n=== Top-1 Accuracy ===")
    print(f"  without_mask: single={s['acc']['without_mask'][0]:.4f}  dual={d['acc']['without_mask'][0]:.4f}")
    print(f"  with_mask   : single={s['acc']['with_mask'][0]:.4f}  dual={d['acc']['with_mask'][0]:.4f}")
    print(f"\n=== EER ===")
    print(f"  single = {s['eer']:.4f}")
    print(f"  dual   = {d['eer']:.4f}")
    print(f"\n=== FAR @ FRR ≤ 1% ===")
    print(f"  single = {s['far_at_frr1']:.4f}")
    print(f"  dual   = {d['far_at_frr1']:.4f}")

    write_report(results)


if __name__ == "__main__":
    main()
