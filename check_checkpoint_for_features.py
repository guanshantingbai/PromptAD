#!/usr/bin/env python3
"""
Check if prompt1 checkpoints contain feature_gallery for quick fusion testing.
"""

import torch
from pathlib import Path


def check_checkpoint_contents(ckpt_path):
    """Check what's inside a checkpoint."""
    print(f"\n{'='*80}")
    print(f"Checkpoint: {ckpt_path.name}")
    print('='*80)
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    print("\nKeys in checkpoint:")
    for key in ckpt.keys():
        if isinstance(ckpt[key], torch.Tensor):
            print(f"  {key:<30} {ckpt[key].shape}")
        else:
            print(f"  {key:<30} {type(ckpt[key])}")
    
    # Check for feature galleries
    has_feature_gallery = False
    for key in ckpt.keys():
        if 'feature_gallery' in key.lower():
            has_feature_gallery = True
            print(f"\n✓ Found: {key}")
    
    if not has_feature_gallery:
        print("\n✗ No feature_gallery found in checkpoint")
        print("  → Cannot do quick fusion testing")
        print("  → Need to add memory bank and re-test")
    
    return has_feature_gallery


if __name__ == "__main__":
    # Check a few representative checkpoints
    ckpt_dir = Path("result/prompt1/mvtec/k_2/checkpoint")
    
    test_classes = ['bottle', 'screw', 'transistor']  # Best, improved, degraded
    
    print("Checking prompt1 checkpoints for feature galleries...")
    print("="*80)
    
    has_features = []
    for cls_name in test_classes:
        ckpt_path = ckpt_dir / f"CLS-Seed_111-{cls_name}-check_point.pt"
        if ckpt_path.exists():
            has_feat = check_checkpoint_contents(ckpt_path)
            has_features.append(has_feat)
        else:
            print(f"\n✗ Not found: {ckpt_path}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if any(has_features):
        print("\n✓ Feature galleries found!")
        print("  → Can implement quick fusion testing")
        print("  → Load features and compute visual scores")
    else:
        print("\n✗ No feature galleries in checkpoints")
        print("\nREASON: You removed memory bank code, so training didn't save features.")
        print("\nOPTIONS:")
        print("  A. Add memory bank back to model → re-test (no retraining needed!)")
        print("  B. Use baseline's feature_gallery + prompt1's semantic branch")
        print("  C. Accept theoretical prediction without validation")
        print("\nRECOMMENDATION: Option B (mixed checkpoint fusion)")
        print("  - Use baseline checkpoint for visual features")
        print("  - Use prompt1 checkpoint for semantic features (prototypes)")
        print("  - Fuse at test time → verify improvement")
