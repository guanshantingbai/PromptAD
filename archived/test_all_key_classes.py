#!/usr/bin/env python3
"""测试所有6个关键类别的语义性能"""
import subprocess
import sys

KEY_CLASSES = ['screw', 'toothbrush', 'hazelnut', 'capsule', 'pill', 'metal_nut']
K_SHOT = 2

print("="*80)
print(f"测试6个关键类别的纯语义性能 (k={K_SHOT})")
print("="*80)

results = {}

for cls_name in KEY_CLASSES:
    print(f"\n[{KEY_CLASSES.index(cls_name)+1}/6] 测试 {cls_name}...")
    
    cmd = [
        "python", "test_cls.py",
        "--dataset", "mvtec",
        "--class_name", cls_name,
        "--k-shot", str(K_SHOT),
        "--semantic-only", "True",
        "--vis", "False",
        "--n_pro", "3",
        "--n_pro_ab", "4",
        "--root-dir", "result/prompt1_fixed",
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # 解析结果
    for line in result.stdout.split('\n'):
        if "Pixel-AUROC:" in line:
            auroc_str = line.split("Pixel-AUROC:")[-1].strip()
            auroc = float(auroc_str)
            results[cls_name] = auroc
            print(f"  ✓ {cls_name}: {auroc:.2f}%")
            break
    else:
        print(f"  ✗ {cls_name}: 解析失败")
        results[cls_name] = None

print("\n" + "="*80)
print("测试完成！")
print("="*80)

# 对比baseline和期望
BASELINE_SEMANTIC = {
    "screw": 66.42, "toothbrush": 69.58, "hazelnut": 80.11,
    "capsule": 73.69, "pill": 85.50, "metal_nut": 85.56,
}

EXPECTED_SEMANTIC = {
    "screw": 79.57, "toothbrush": 89.44, "hazelnut": 91.14,
    "capsule": 80.65, "pill": 86.12, "metal_nut": 88.71,
}

print(f"\n{'类别':<12} {'Baseline':<10} {'期望':<10} {'实际':<10} {'vs Base':<10} {'vs 期望':<10} {'状态'}")
print("-" * 80)

for cls_name in KEY_CLASSES:
    if results[cls_name] is None:
        continue
    
    baseline = BASELINE_SEMANTIC[cls_name]
    expected = EXPECTED_SEMANTIC[cls_name]
    actual = results[cls_name]
    
    vs_base = actual - baseline
    vs_exp = actual - expected
    
    if actual >= expected * 0.98:
        status = "✅ 达标"
    elif actual > baseline:
        status = "⚠️  可接受"
    else:
        status = "❌ 需重训"
    
    print(f"{cls_name:<12} {baseline:<10.2f} {expected:<10.2f} {actual:<10.2f} {vs_base:+<10.2f} {vs_exp:+<10.2f} {status}")

print("="*80)
