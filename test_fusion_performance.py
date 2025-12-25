#!/usr/bin/env python3
"""测试所有6个关键类别的融合性能（不使用--semantic-only）"""
import subprocess
import sys

KEY_CLASSES = ['screw', 'toothbrush', 'hazelnut', 'capsule', 'pill', 'metal_nut']
K_SHOT = 2

print("="*80)
print(f"测试6个关键类别的融合性能 (k={K_SHOT})")
print("="*80)

results = {}

for cls_name in KEY_CLASSES:
    print(f"\n[{KEY_CLASSES.index(cls_name)+1}/6] 测试 {cls_name}...")
    
    cmd = [
        "python", "test_cls.py",
        "--dataset", "mvtec",
        "--class_name", cls_name,
        "--k-shot", str(K_SHOT),
        # 不加 --semantic-only，测试融合性能
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
        if result.returncode != 0:
            print(f"     错误: {result.stderr[:200]}")
        results[cls_name] = None

print("\n" + "="*80)
print("测试完成！")
print("="*80)

# 对比数据
BASELINE_FUSION = {
    "screw": 58.66, "toothbrush": 98.89, "hazelnut": 99.93,
    "capsule": 79.94, "pill": 95.61, "metal_nut": 100.00,
}

FIXED_SEMANTIC = {
    "screw": 77.35, "toothbrush": 89.17, "hazelnut": 90.86,
    "capsule": 82.21, "pill": 84.56, "metal_nut": 89.74,
}

print(f"\n{'类别':<12} {'Baseline融合':<13} {'修复后语义':<13} {'修复后融合':<13} {'融合vs Base':<12} {'融合vs语义':<12} {'状态'}")
print("-" * 95)

for cls_name in KEY_CLASSES:
    if results[cls_name] is None:
        continue
    
    baseline_fus = BASELINE_FUSION[cls_name]
    fixed_sem = FIXED_SEMANTIC[cls_name]
    fixed_fus = results[cls_name]
    
    vs_base = fixed_fus - baseline_fus
    vs_sem = fixed_fus - fixed_sem
    
    if fixed_fus > baseline_fus + 2:
        status = "✅ 融合改进"
    elif fixed_fus > baseline_fus:
        status = "✅ 略有改进"
    elif fixed_fus > baseline_fus - 2:
        status = "⚠️  基本持平"
    else:
        status = "❌ 融合下降"
    
    print(f"{cls_name:<12} {baseline_fus:<13.2f} {fixed_sem:<13.2f} {fixed_fus:<13.2f} {vs_base:+<12.2f} {vs_sem:+<12.2f} {status}")

# 计算平均
valid_results = [v for v in results.values() if v is not None]
if valid_results:
    avg_fixed_fus = sum(valid_results) / len(valid_results)
    avg_baseline_fus = sum(BASELINE_FUSION[k] for k in KEY_CLASSES if results[k] is not None) / len(valid_results)
    avg_fixed_sem = sum(FIXED_SEMANTIC[k] for k in KEY_CLASSES if results[k] is not None) / len(valid_results)
    
    print("-" * 95)
    print(f"{'平均':<12} {avg_baseline_fus:<13.2f} {avg_fixed_sem:<13.2f} {avg_fixed_fus:<13.2f} {avg_fixed_fus - avg_baseline_fus:+<12.2f} {avg_fixed_fus - avg_fixed_sem:+<12.2f}")

print("="*80)
