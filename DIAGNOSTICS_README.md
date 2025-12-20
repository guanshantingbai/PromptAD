# Multi-Prototype Diagnostics for PromptAD

Lightweight inference-time diagnostics for analyzing multi-prototype behavior.

## Features

✅ **No training code changes** - Analysis-only during inference  
✅ **No backprop** - Pure statistics collection with `@torch.no_grad()`  
✅ **Minimal overhead** - Reuses existing model computations  
✅ **Comprehensive metrics** - 4 key diagnostic categories

## What it analyzes

### 1. Normal Prototype Usage Frequency
- Tracks which normal prototype achieves `argmax` similarity for each patch
- Reports normalized frequencies per prototype
- Identifies imbalanced usage (some prototypes dominating)

### 2. Abnormal Prototype Collapse Check
- Computes pairwise cosine similarity between all abnormal prototypes
- High similarity (>0.9) indicates mode collapse
- Reports mean/min/max similarity

### 3. Normal vs Abnormal Angular Separation
- Measures cosine similarity between abnormal prototypes and mean normal
- Lower values = better separation
- Reports distribution statistics

### 4. Patch-Level Dominance (Optional)
- Per-image metric: does single prototype dominate >80% of patches?
- Indicates whether multi-prototype structure is actually utilized

## Usage

### Quick Start (Classification)

```bash
# Run test with diagnostics on MVTec carpet
python test_cls_with_diagnostics.py \
    --dataset mvtec \
    --class_name carpet \
    --k-shot 2 \
    --save-diagnostics True
```

### Quick Start (Segmentation)

```bash
# Run test with diagnostics on MVTec bottle
python test_seg_with_diagnostics.py \
    --dataset mvtec \
    --class_name bottle \
    --k-shot 2 \
    --save-diagnostics True
```

### Programmatic Usage

```python
from utils.prototype_diagnostics import PrototypeDiagnostics

# Initialize
diagnostics = PrototypeDiagnostics()

# During inference loop
for data in dataloader:
    visual_features = model.encode_image(data)
    
    # Collect statistics (no backprop)
    diagnostics.update_from_inference(
        visual_features=visual_features,
        normal_prototypes=model.normal_prototypes,
        abnormal_prototypes=model.abnormal_prototypes,
        logit_scale=model.model.logit_scale
    )

# Print summary
diagnostics.print_summary(
    normal_prototypes=model.normal_prototypes,
    abnormal_prototypes=model.abnormal_prototypes
)

# Export to JSON
stats = diagnostics.export_statistics(
    normal_prototypes=model.normal_prototypes,
    abnormal_prototypes=model.abnormal_prototypes
)
```

## Output Format

### Console Output Example

```
================================================================================
MULTI-PROTOTYPE DIAGNOSTICS SUMMARY
================================================================================

[1] Normal Prototype Usage Frequency:
   Prototype  0:  34.23% (12,345 patches)
   Prototype  1:  28.91% (10,432 patches)
   Prototype  2:  36.86% (13,298 patches)
   Range: 28.91% - 36.86%
   Balance score (1 - max_freq): 63.14%

[2] Abnormal Prototype Collapse Check:
   Pairwise cosine similarity (higher = more collapse):
     Mean: 0.3421
     Min:  0.0823
     Max:  0.7654
     Pairs analyzed: 15

[3] Normal vs Abnormal Angular Separation:
   Cosine similarity (lower = better separation):
     Each abnormal to mean normal:
       Mean: 0.1234
       Min:  -0.0521
       Max:  0.3456
       Std:  0.0987
     All normal-abnormal pairs:
       Mean: 0.1567
       Min:  -0.0823
       Max:  0.4231

[4] Patch-Level Dominance (single prototype dominates patches):
   Mean dominance: 45.67%
   Median dominance: 43.21%
   Std: 0.1234
   Images with >80% dominance: 23/150 (15.3%)
   Images with >90% dominance: 5/150 (3.3%)

================================================================================
```

### JSON Output

Saved to `result/<dataset>/k_<shot>/diagnostics/<class>_k<shot>_diagnostics.json`:

```json
{
  "normal_prototype_frequencies": [0.3423, 0.2891, 0.3686],
  "abnormal_prototype_frequencies": [0.1234, 0.2345, ...],
  "abnormal_collapse": {
    "mean": 0.3421,
    "min": 0.0823,
    "max": 0.7654,
    "num_pairs": 15
  },
  "normal_abnormal_separation": {
    "mean_to_mean_normal": 0.1234,
    "min_to_mean_normal": -0.0521,
    "max_to_mean_normal": 0.3456,
    "std_to_mean_normal": 0.0987,
    "all_pairs_mean": 0.1567,
    "all_pairs_min": -0.0823,
    "all_pairs_max": 0.4231
  },
  "dominance_statistics": {
    "mean": 0.4567,
    "median": 0.4321,
    "min": 0.2145,
    "max": 0.9823,
    "std": 0.1234,
    "images_above_80pct": 23,
    "images_above_90pct": 5,
    "total_images": 150
  },
  "total_patches": 36075,
  "total_images": 150
}
```

## Interpreting Results

### Healthy Multi-Prototype System
- ✅ Normal prototypes: balanced usage (30-40% each)
- ✅ Abnormal collapse: mean cosine < 0.5
- ✅ Normal-abnormal separation: mean cosine < 0.3
- ✅ Dominance: <20% images with >80% dominance

### Warning Signs
- ⚠️ One normal prototype >60% usage → under-utilization
- ⚠️ Abnormal collapse: mean cosine > 0.7 → mode collapse
- ⚠️ Normal-abnormal separation: mean > 0.5 → poor separation
- ⚠️ Dominance: >50% images with >80% → not using multi-modal structure

## Files

- `utils/prototype_diagnostics.py` - Core diagnostics class
- `test_cls_with_diagnostics.py` - Classification test with diagnostics
- `test_seg_with_diagnostics.py` - Segmentation test with diagnostics

## Requirements

No additional dependencies beyond existing PromptAD requirements.

## Performance

Negligible overhead:
- Reuses `model.encode_image()` already computed for inference
- Only adds argmax operations and simple statistics
- ~1-2% slower than normal inference
