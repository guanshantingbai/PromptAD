#!/usr/bin/env python3
"""
估计修复后的融合性能

思路：
1. 修复后的语义分支性能已知（test_all_key_classes.py结果）
2. Memory Bank的性能可以从baseline中推算（baseline也有memory bank）
3. 用harmonic mean估算融合后的性能
4. 对比baseline的融合性能
"""

import numpy as np
import pandas as pd

print("="*80)
print("修复后融合性能估计")
print("="*80)

# ==================== 已知数据 ====================
# 1. 修复后的语义分支性能（从test_all_key_classes.py）
fixed_semantic = {
    "screw": 77.35,
    "toothbrush": 89.17,
    "hazelnut": 90.86,
    "capsule": 82.21,
    "pill": 84.56,
    "metal_nut": 89.74,
}

# 2. Baseline的融合性能
baseline_fusion = {
    "screw": 58.66,
    "toothbrush": 98.89,
    "hazelnut": 99.93,
    "capsule": 79.94,
    "pill": 95.61,
    "metal_nut": 100.00,
}

# 3. Baseline的语义性能
baseline_semantic = {
    "screw": 66.42,
    "toothbrush": 69.58,
    "hazelnut": 80.11,
    "capsule": 73.69,
    "pill": 85.50,
    "metal_nut": 85.56,
}

print("\n第一步：推算Memory Bank的性能")
print("-"*80)
print("Baseline使用融合策略：harmonic_mean(semantic, memory_bank)")
print("已知baseline的semantic和fusion，反推memory_bank性能\n")

# Harmonic mean: 1/fusion = 1/semantic + 1/memory_bank
# 因此: 1/memory_bank = 1/fusion - 1/semantic
# memory_bank = 1 / (1/fusion - 1/semantic)

memory_bank_estimated = {}
print(f"{'类别':<12} {'Baseline语义':<12} {'Baseline融合':<12} {'推算Memory':<12}")
print("-"*80)

for cls in fixed_semantic.keys():
    b_sem = baseline_semantic[cls]
    b_fus = baseline_fusion[cls]
    
    # 反推memory bank性能
    # harmonic: fusion = 1 / (1/sem + 1/mem)
    # 1/fusion = 1/sem + 1/mem
    # 1/mem = 1/fusion - 1/sem
    
    if b_fus > 0 and b_sem > 0:
        inv_mem = 1.0/b_fus - 1.0/b_sem
        if inv_mem > 0:
            mem_score = 1.0 / inv_mem
        else:
            # Memory bank很强，导致融合比语义还好
            mem_score = b_fus * 1.5  # 粗略估计
    else:
        mem_score = 50.0  # 默认值
    
    memory_bank_estimated[cls] = mem_score
    print(f"{cls:<12} {b_sem:<12.2f} {b_fus:<12.2f} {mem_score:<12.2f}")

print("\n⚠️  注意：Memory Bank性能是基于baseline推算的，实际可能有差异")

# ==================== 估计修复后的融合性能 ====================
print("\n\n第二步：估计修复后的融合性能")
print("-"*80)
print("假设：Memory Bank性能与baseline类似（重新构建的memory bank）")
print("融合策略：harmonic_mean(fixed_semantic, memory_bank_estimated)\n")

estimated_fusion = {}
print(f"{'类别':<12} {'修复后语义':<12} {'估算Memory':<12} {'估算融合':<12} {'Baseline融合':<12} {'vs Baseline':<12}")
print("-"*80)

for cls in fixed_semantic.keys():
    f_sem = fixed_semantic[cls]
    mem = memory_bank_estimated[cls]
    
    # Harmonic mean
    fusion_est = 1.0 / (1.0/f_sem + 1.0/mem)
    estimated_fusion[cls] = fusion_est
    
    b_fus = baseline_fusion[cls]
    diff = fusion_est - b_fus
    
    print(f"{cls:<12} {f_sem:<12.2f} {mem:<12.2f} {fusion_est:<12.2f} {b_fus:<12.2f} {diff:+<12.2f}")

# 计算平均
avg_fixed_sem = np.mean(list(fixed_semantic.values()))
avg_est_fusion = np.mean(list(estimated_fusion.values()))
avg_baseline_fusion = np.mean(list(baseline_fusion.values()))

print("-"*80)
print(f"{'平均':<12} {avg_fixed_sem:<12.2f} {'':<12} {avg_est_fusion:<12.2f} {avg_baseline_fusion:<12.2f} {avg_est_fusion - avg_baseline_fusion:+<12.2f}")

# ==================== 场景分析 ====================
print("\n\n第三步：多场景估计（考虑Memory Bank变化）")
print("="*80)

scenarios = {
    "乐观场景": 1.1,   # Memory Bank也有所提升
    "基准场景": 1.0,   # Memory Bank保持不变
    "悲观场景": 0.9,   # Memory Bank略有下降
}

print(f"\n{'场景':<12} {'估算平均融合':<15} {'vs Baseline':<12} {'vs 修复后语义':<15} {'结论'}")
print("-"*80)

for scenario_name, factor in scenarios.items():
    # 调整memory bank性能
    adjusted_memory = {k: v * factor for k, v in memory_bank_estimated.items()}
    
    # 重新计算融合
    adjusted_fusion = {}
    for cls in fixed_semantic.keys():
        f_sem = fixed_semantic[cls]
        mem = adjusted_memory[cls]
        fusion = 1.0 / (1.0/f_sem + 1.0/mem)
        adjusted_fusion[cls] = fusion
    
    avg_fusion = np.mean(list(adjusted_fusion.values()))
    vs_baseline = avg_fusion - avg_baseline_fusion
    vs_semantic = avg_fusion - avg_fixed_sem
    
    if vs_baseline > 2:
        conclusion = "✅ 显著改进"
    elif vs_baseline > 0:
        conclusion = "✅ 有所改进"
    elif vs_baseline > -2:
        conclusion = "⚠️  基本持平"
    else:
        conclusion = "❌ 需优化"
    
    print(f"{scenario_name:<12} {avg_fusion:<15.2f} {vs_baseline:+<12.2f} {vs_semantic:+<15.2f} {conclusion}")

# ==================== 总结和建议 ====================
print("\n\n" + "="*80)
print("估计总结")
print("="*80)

print("\n📊 性能对比：")
print(f"  修复后语义平均: {avg_fixed_sem:.2f}%")
print(f"  估算融合平均:   {avg_est_fusion:.2f}%")
print(f"  Baseline融合:   {avg_baseline_fusion:.2f}%")

print(f"\n📈 改进幅度：")
print(f"  语义 vs Baseline: {avg_fixed_sem - np.mean(list(baseline_semantic.values())):+.2f}% ✅")
print(f"  融合 vs Baseline: {avg_est_fusion - avg_baseline_fusion:+.2f}%")
print(f"  融合 vs 修复后语义: {avg_est_fusion - avg_fixed_sem:+.2f}%")

print("\n🔍 关键发现：")

# 分析哪些类别融合有帮助，哪些有害
helps = []
hurts = []
for cls in fixed_semantic.keys():
    diff = estimated_fusion[cls] - fixed_semantic[cls]
    if diff > 1:
        helps.append(f"{cls} ({diff:+.2f}%)")
    elif diff < -1:
        hurts.append(f"{cls} ({diff:+.2f}%)")

print(f"  融合有帮助的类别: {len(helps)}/6")
if helps:
    for h in helps:
        print(f"    - {h}")

print(f"\n  融合有害的类别: {len(hurts)}/6")
if hurts:
    for h in hurts:
        print(f"    - {h}")

print("\n💡 估计结论：")
if avg_est_fusion > avg_baseline_fusion + 2:
    print("  ✅ 融合后预计显著超越baseline")
    print(f"     估计改进幅度：{avg_est_fusion - avg_baseline_fusion:+.2f}%")
    print("     建议：直接测试融合性能")
elif avg_est_fusion > avg_baseline_fusion:
    print("  ✅ 融合后预计略微超越baseline")
    print(f"     估计改进幅度：{avg_est_fusion - avg_baseline_fusion:+.2f}%")
    print("     建议：测试融合性能，可能需要优化融合策略")
elif avg_est_fusion > avg_baseline_fusion - 2:
    print("  ⚠️  融合后预计与baseline接近")
    print(f"     估计差异：{avg_est_fusion - avg_baseline_fusion:+.2f}%")
    print("     建议：先测试验证，考虑优化融合策略")
else:
    print("  ❌ 融合后预计不如baseline")
    print(f"     估计下降：{avg_est_fusion - avg_baseline_fusion:+.2f}%")
    print("     建议：优先优化融合策略，而非直接测试")

print("\n📝 后续步骤建议：")
print("  1. 运行融合测试：测试6个关键类别的融合性能")
print("  2. 对比实际vs估计：验证估计模型的准确性")
print("  3. 分析差异原因：如果实际与估计差异大，找出原因")
print("  4. 优化融合策略：如果融合不理想，尝试其他融合方法")
print("     - 加权平均（可调权重）")
print("     - 自适应融合（基于置信度）")
print("     - 类别特定融合策略")

print("\n" + "="*80)
