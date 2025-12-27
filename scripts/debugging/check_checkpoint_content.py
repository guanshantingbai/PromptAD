#!/usr/bin/env python3
"""
检查checkpoint中保存了哪些内容
"""
import torch
from pathlib import Path

checkpoint_path = "result/prompt1/mvtec/k_2/checkpoint/CLS-Seed_111-bottle-check_point.pt"

print(f"Loading: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("\n" + "="*80)
print("Checkpoint Keys:")
print("="*80)
for key in checkpoint.keys():
    value = checkpoint[key]
    if isinstance(value, torch.Tensor):
        print(f"  {key:<30} {str(value.shape):<20} {value.dtype}")
    else:
        print(f"  {key:<30} {type(value).__name__}")

print("\n" + "="*80)
print("Checking for memory bank (feature_gallery):")
print("="*80)
has_gallery = False
for key in checkpoint.keys():
    if 'gallery' in key.lower() or 'memory' in key.lower():
        print(f"  ✓ Found: {key}")
        has_gallery = True

if not has_gallery:
    print("  ✗ No feature gallery found in checkpoint")
    print("  → Memory bank was not saved during training")

print("\n" + "="*80)
print("Prototypes information:")
print("="*80)
for key in checkpoint.keys():
    if 'prototype' in key.lower():
        value = checkpoint[key]
        print(f"  {key}: {value.shape}")
