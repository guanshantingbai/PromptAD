# Multi-Prototype Training Guide

## Why Re-train?

Baseline weights were trained with **single-anchor system**:
- Averaged `text_features` (1 normal + 1 abnormal)
- Saved as `text_features` tensor shape [2, dim]

New **multi-prototype system**:
- Multiple `normal_prototypes` [K_normal, dim]
- Multiple `abnormal_prototypes` [M_abnormal, dim]
- Incompatible checkpoint structure

**You must re-train with the new multi-prototype code.**

---

## Quick Start

### 1. Train Single Class (Quick Test)

```bash
# Train carpet with multi-prototypes (k=2, ~10-15 minutes)
./train_single_class.sh
```

This trains:
- 3 normal prototypes (multi-modal normal manifold)
- 6 abnormal prototypes (structured abnormal directions)
- 100 epochs
- Saves to `result/mvtec/k_2/checkpoint/`

### 2. Train All MVTec Classes

```bash
# Train all 15 MVTec classes (takes ~3-4 hours)
./train_all_mvtec.sh
```

### 3. Custom Training

```bash
python train_cls.py \
    --dataset mvtec \
    --class_name bottle \
    --k-shot 2 \
    --n_pro 3 \              # Number of normal prototypes
    --n_pro_ab 6 \           # Number of abnormal prototypes
    --Epoch 100 \
    --lr 0.002 \
    --resolution 256 \
    --gpu-id 0
```

---

## Multi-Prototype Configuration

### Recommended Settings

| Setting | Value | Reason |
|---------|-------|--------|
| `n_pro` | 3-5 | Capture multi-modal normal distributions |
| `n_pro_ab` | 4-8 | Diverse abnormal semantic directions |
| `Epoch` | 100 | Sufficient convergence |
| `lr` | 0.002 | Standard for prompt tuning |
| `resolution` | 256 | Faster than 400, good performance |

### Ablation Studies (Optional)

Try different prototype numbers:

```bash
# More normal prototypes (5)
python train_cls.py --n_pro 5 --n_pro_ab 6 ...

# More abnormal prototypes (8)
python train_cls.py --n_pro 3 --n_pro_ab 8 ...

# Fewer prototypes (baseline-like)
python train_cls.py --n_pro 1 --n_pro_ab 4 ...
```

---

## After Training

### Test with Diagnostics

```bash
# Test single class with prototype diagnostics
python test_cls_with_diagnostics.py \
    --dataset mvtec \
    --class_name carpet \
    --k-shot 2 \
    --n_pro 3 \
    --n_pro_ab 6 \
    --save-diagnostics True
```

### Check Diagnostics Output

Look for:
- ✅ Balanced normal prototype usage (20-40% each)
- ✅ Low abnormal collapse (mean cosine < 0.5)
- ✅ Good normal-abnormal separation (mean < 0.3)

---

## Checkpoint Structure

### Old (Baseline)
```python
{
    'text_features': [2, dim],        # Averaged single anchors
    'feature_gallery1': [...],        # Visual memory bank (removed)
    'feature_gallery2': [...],        # Visual memory bank (removed)
}
```

### New (Multi-Prototype)
```python
{
    'normal_prototypes': [K_normal, dim],      # e.g., [3, 768]
    'abnormal_prototypes': [M_abnormal, dim],  # e.g., [6, 768]
}
```

---

## Training Time Estimates

| Setup | Time per Class | Total (15 classes) |
|-------|---------------|-------------------|
| k=1, 50 epochs | ~5 min | ~1.5 hours |
| k=2, 100 epochs | ~10 min | ~3 hours |
| k=4, 100 epochs | ~12 min | ~4 hours |

GPU: Single RTX 3090 / A100

---

## Monitoring Training

Check logs during training:
```
Epoch 95/100:
  i_roc: 0.94 (improving)
  Normal proto usage: balanced
  Loss: converging
```

If issues:
- Prototype collapse → Increase `n_pro` or `n_pro_ab`
- Poor performance → Check learning rate or epochs
- Out of memory → Reduce `batch-size` or `resolution`

---

## Next Steps

1. ✅ Run `./train_single_class.sh` to test
2. ✅ Check diagnostics output
3. ✅ If good, run `./train_all_mvtec.sh`
4. ✅ Compare with baseline results
