#!/usr/bin/env python3
"""
测试修复后的Memory Bank实现
对比修复前后的融合性能
"""
import subprocess
import sys

KEY_CLASSES = ['screw', 'toothbrush', 'hazelnut', 'capsule', 'pill', 'metal_nut']
K_SHOT = 2

print("="*80)
print("测试修复后的Memory Bank - 融合性能评估")
print("="*80)
print("\n🔧 修复内容：")
print("  • Memory Bank构建方式：改为先收集所有features，再一次性build")
print("  • 参照baseline的正确实现")
print(f"\n📊 测试配置：")
print(f"  • 数据集: MVTec-AD")
print(f"  • 关键类别: {len(KEY_CLASSES)}个")
print(f"  • K-shot: {K_SHOT}")
print(f"  • Checkpoint目录: result/prompt1_fixed")
print("\n" + "="*80)

# 先测试一个类别验证修复是否生效
print("\n【步骤1】快速验证 - 测试单个类别（screw）")
print("-"*80)

test_class = 'screw'
print(f"\n测试 {test_class} 的融合性能...")

cmd = [
    "python", "test_cls.py",
    "--dataset", "mvtec",
    "--class_name", test_class,
    "--k-shot", str(K_SHOT),
    "--vis", "False",
    "--n_pro", "3",
    "--n_pro_ab", "4",
    "--root-dir", "result/prompt1_fixed",
]

result = subprocess.run(cmd, capture_output=True, text=True)

# 解析结果
fusion_score = None
for line in result.stdout.split('\n'):
    if "Memory bank built:" in line:
        print(f"  ✓ {line.strip()}")
    if "Pixel-AUROC:" in line:
        auroc_str = line.split("Pixel-AUROC:")[-1].strip()
        fusion_score = float(auroc_str)
        print(f"  ✓ 融合AUROC: {fusion_score:.2f}%")
        break

if fusion_score is None:
    print(f"  ✗ 测试失败")
    print("\n错误输出:")
    print(result.stderr[:500])
    sys.exit(1)

# 对比数据
BASELINE_FUSION = {"screw": 58.66}
FIXED_SEMANTIC = {"screw": 77.35}

baseline_fus = BASELINE_FUSION[test_class]
fixed_sem = FIXED_SEMANTIC[test_class]

print(f"\n📊 快速对比:")
print(f"  • Baseline融合: {baseline_fus:.2f}%")
print(f"  • 修复后语义: {fixed_sem:.2f}%")
print(f"  • 修复后融合: {fusion_score:.2f}%")
print(f"  • 融合 vs Baseline: {fusion_score - baseline_fus:+.2f}%")
print(f"  • 融合 vs 纯语义: {fusion_score - fixed_sem:+.2f}%")

if fusion_score > baseline_fus:
    print(f"  ✅ 融合后超越baseline！")
elif fusion_score > baseline_fus - 2:
    print(f"  ⚠️  融合后与baseline接近")
else:
    print(f"  ❌ 融合后不如baseline")

# 判断是否继续测试所有类别
print("\n" + "="*80)
print("【步骤2】完整测试 - 测试所有6个关键类别")
print("-"*80)

response = input("\n是否继续测试其余5个类别? (y/n): ")
if response.lower() != 'y':
    print("\n已停止。")
    print(f"\n💡 如需测试所有类别，运行: python test_fusion_performance.py")
    sys.exit(0)

# 测试所有类别
print("\n开始测试所有类别...")
results = {test_class: fusion_score}  # 已经测试过screw

for cls_name in KEY_CLASSES:
    if cls_name == test_class:
        continue
    
    print(f"\n[{KEY_CLASSES.index(cls_name)+1}/6] 测试 {cls_name}...")
    
    cmd = [
        "python", "test_cls.py",
        "--dataset", "mvtec",
        "--class_name", cls_name,
        "--k-shot", str(K_SHOT),
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

# 完整对比
print("\n\n" + "="*80)
print("完整对比结果")
print("="*80)

BASELINE_FUSION_ALL = {
    "screw": 58.66, "toothbrush": 98.89, "hazelnut": 99.93,
    "capsule": 79.94, "pill": 95.61, "metal_nut": 100.00,
}

FIXED_SEMANTIC_ALL = {
    "screw": 77.35, "toothbrush": 89.17, "hazelnut": 90.86,
    "capsule": 82.21, "pill": 84.56, "metal_nut": 89.74,
}

print(f"\n{'类别':<12} {'Baseline融合':<13} {'修复后语义':<13} {'修复后融合':<13} {'融合vs Base':<12} {'融合vs语义':<12} {'状态'}")
print("-" * 100)

for cls_name in KEY_CLASSES:
    if results[cls_name] is None:
        continue
    
    baseline_fus = BASELINE_FUSION_ALL[cls_name]
    fixed_sem = FIXED_SEMANTIC_ALL[cls_name]
    fixed_fus = results[cls_name]
    
    vs_base = fixed_fus - baseline_fus
    vs_sem = fixed_fus - fixed_sem
    
    if fixed_fus > baseline_fus + 2:
        status = "✅ 显著改进"
    elif fixed_fus > baseline_fus:
        status = "✅ 略有改进"
    elif fixed_fus > baseline_fus - 2:
        status = "⚠️  基本持平"
    else:
        status = "❌ 需优化"
    
    print(f"{cls_name:<12} {baseline_fus:<13.2f} {fixed_sem:<13.2f} {fixed_fus:<13.2f} {vs_base:+<12.2f} {vs_sem:+<12.2f} {status}")

# 计算平均
valid_results = [v for v in results.values() if v is not None]
if valid_results:
    avg_fixed_fus = sum(valid_results) / len(valid_results)
    avg_baseline_fus = sum(BASELINE_FUSION_ALL[k] for k in KEY_CLASSES if results.get(k) is not None) / len(valid_results)
    avg_fixed_sem = sum(FIXED_SEMANTIC_ALL[k] for k in KEY_CLASSES if results.get(k) is not None) / len(valid_results)
    
    print("-" * 100)
    print(f"{'平均':<12} {avg_baseline_fus:<13.2f} {avg_fixed_sem:<13.2f} {avg_fixed_fus:<13.2f} {avg_fixed_fus - avg_baseline_fus:+<12.2f} {avg_fixed_fus - avg_fixed_sem:+<12.2f}")

print("\n" + "="*80)
print("测试完成！")
print("="*80)
