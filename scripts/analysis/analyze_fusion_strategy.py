#!/usr/bin/env python3
"""
重新分析融合策略的影响

关键洞察：
1. Harmonic mean: score = 1/(1/a + 1/b)
2. 当a < b时，score会更接近a（被较小值主导）
3. 当a > b时，score会更接近b（被较小值主导）
4. Harmonic mean对"短板"敏感

问题：如果语义分支很强（90%），但memory bank很弱（50%），
     融合后会是多少？答案：约60%（被拖累）
"""

import numpy as np

def harmonic_fusion(score1, score2):
    """Harmonic mean fusion"""
    return 1.0 / (1.0/score1 + 1.0/score2)

print("="*80)
print("理解Harmonic Mean融合的特性")
print("="*80)

# 示例
print("\n示例：不同性能组合的融合结果")
print("-"*80)
print(f"{'语义分支':<12} {'Memory Bank':<12} {'融合结果':<12} {'分析'}")
print("-"*80)

test_cases = [
    (90, 90, "两者都强"),
    (90, 50, "语义强，Memory弱"),
    (50, 90, "语义弱，Memory强"),
    (80, 60, "语义中等强，Memory中等弱"),
    (70, 70, "两者都中等"),
]

for sem, mem, desc in test_cases:
    fused = harmonic_fusion(sem, mem)
    print(f"{sem:<12.1f} {mem:<12.1f} {fused:<12.2f} {desc}")

print("\n💡 关键发现：Harmonic mean被短板主导！")
print("   如果memory bank比语义分支弱，会拉低整体性能")

# ==================== 真实问题分析 ====================
print("\n\n" + "="*80)
print("分析Baseline的情况")
print("="*80)

baseline_data = {
    "screw": {"semantic": 66.42, "fusion": 58.66},
    "toothbrush": {"semantic": 69.58, "fusion": 98.89},
    "hazelnut": {"semantic": 80.11, "fusion": 99.93},
    "capsule": {"semantic": 73.69, "fusion": 79.94},
    "pill": {"semantic": 85.50, "fusion": 95.61},
    "metal_nut": {"semantic": 85.56, "fusion": 100.00},
}

print("\n分析：Baseline的Memory Bank相对于语义分支的强弱")
print("-"*80)
print(f"{'类别':<12} {'语义':<8} {'融合':<8} {'Memory推算':<12} {'Memory vs 语义':<15}")
print("-"*80)

for cls, data in baseline_data.items():
    sem = data['semantic']
    fus = data['fusion']
    
    # 反推memory: 1/fus = 1/sem + 1/mem => mem = 1/(1/fus - 1/sem)
    if fus >= sem:
        # fusion >= semantic，说明memory更强
        inv_mem = 1.0/fus - 1.0/sem
        if inv_mem > 1e-6:
            mem = 1.0 / inv_mem
            status = f"强得多 (+{mem-sem:.1f})"
        else:
            mem = fus * 2  # 粗略估计
            status = "非常强"
    else:
        # fusion < semantic，说明memory较弱
        inv_mem = 1.0/fus - 1.0/sem
        mem = 1.0 / inv_mem
        status = f"弱 ({mem-sem:.1f})"
    
    print(f"{cls:<12} {sem:<8.2f} {fus:<8.2f} {mem:<12.2f} {status:<15}")

print("\n📊 重要发现：")
print("  • screw: Memory Bank弱，拖累了语义分支")
print("  • toothbrush, hazelnut, pill, metal_nut: Memory Bank强，提升了性能")
print("  • capsule: Memory Bank略强")

# ==================== 修复后的预测 ====================
print("\n\n" + "="*80)
print("修复后的现实预测")
print("="*80)

fixed_semantic = {
    "screw": 77.35,
    "toothbrush": 89.17,
    "hazelnut": 90.86,
    "capsule": 82.21,
    "pill": 84.56,
    "metal_nut": 89.74,
}

print("\n假设：Memory Bank性能保持baseline水平（重新构建的memory bank类似）")
print("\n场景1：直接使用baseline推算的Memory Bank性能")
print("-"*80)
print(f"{'类别':<12} {'修复语义':<10} {'Baseline Memory':<15} {'预测融合':<10} {'vs Baseline融合':<15} {'vs 修复语义':<12}")
print("-"*80)

scenario1_results = {}
for cls in fixed_semantic.keys():
    f_sem = fixed_semantic[cls]
    b_data = baseline_data[cls]
    b_sem = b_data['semantic']
    b_fus = b_data['fusion']
    
    # 反推baseline的memory性能
    if b_fus >= b_sem:
        inv_mem = 1.0/b_fus - 1.0/b_sem
        if inv_mem > 1e-6:
            b_mem = 1.0 / inv_mem
        else:
            b_mem = b_fus * 1.5
    else:
        inv_mem = 1.0/b_fus - 1.0/b_sem
        b_mem = 1.0 / abs(inv_mem)
    
    # 预测修复后的融合（使用相同的memory性能）
    pred_fus = harmonic_fusion(f_sem, b_mem)
    
    vs_baseline_fus = pred_fus - b_fus
    vs_fixed_sem = pred_fus - f_sem
    
    scenario1_results[cls] = pred_fus
    
    print(f"{cls:<12} {f_sem:<10.2f} {b_mem:<15.2f} {pred_fus:<10.2f} {vs_baseline_fus:+<15.2f} {vs_fixed_sem:+<12.2f}")

avg_fixed_sem = np.mean(list(fixed_semantic.values()))
avg_pred_fus = np.mean(list(scenario1_results.values()))
avg_baseline_fus = np.mean([d['fusion'] for d in baseline_data.values()])

print("-"*80)
print(f"{'平均':<12} {avg_fixed_sem:<10.2f} {'':<15} {avg_pred_fus:<10.2f} {avg_pred_fus - avg_baseline_fus:+<15.2f} {avg_pred_fus - avg_fixed_sem:+<12.2f}")

print("\n⚠️  警告：这个预测假设Memory Bank性能非常高（>100），不现实！")
print("   实际上，异常检测分数应该在0-100范围内")

# ==================== 更合理的预测 ====================
print("\n\n场景2：更合理的Memory Bank性能假设")
print("-"*80)
print("假设：Memory Bank在不同类别上的性能变化范围为 50-95%")
print()

# 基于baseline的fusion vs semantic比例，估算memory的相对强度
print(f"{'类别':<12} {'修复语义':<10} {'假设Memory':<12} {'预测融合':<10} {'vs Baseline':<12} {'vs 修复语义':<12}")
print("-"*80)

scenario2_results = {}
for cls in fixed_semantic.keys():
    f_sem = fixed_semantic[cls]
    b_data = baseline_data[cls]
    b_sem = b_data['semantic']
    b_fus = b_data['fusion']
    
    # 基于baseline的fusion/semantic比例，估算memory的相对强度
    ratio = b_fus / b_sem
    
    if ratio > 1.1:
        # Memory很强，假设在90-95
        assumed_mem = 92.0
    elif ratio > 0.95:
        # Memory略强或持平，假设在75-85
        assumed_mem = 80.0
    else:
        # Memory较弱，假设在50-70
        assumed_mem = 60.0
    
    # 预测融合
    pred_fus = harmonic_fusion(f_sem, assumed_mem)
    
    vs_baseline = pred_fus - b_fus
    vs_semantic = pred_fus - f_sem
    
    scenario2_results[cls] = pred_fus
    
    print(f"{cls:<12} {f_sem:<10.2f} {assumed_mem:<12.2f} {pred_fus:<10.2f} {vs_baseline:+<12.2f} {vs_semantic:+<12.2f}")

avg_pred_fus2 = np.mean(list(scenario2_results.values()))
print("-"*80)
print(f"{'平均':<12} {avg_fixed_sem:<10.2f} {'':<12} {avg_pred_fus2:<10.2f} {avg_pred_fus2 - avg_baseline_fus:+<12.2f} {avg_pred_fus2 - avg_fixed_sem:+<12.2f}")

# ==================== 总结 ====================
print("\n\n" + "="*80)
print("预测总结与建议")
print("="*80)

print("\n📊 不同场景下的预测：")
print(f"  修复后纯语义:  {avg_fixed_sem:.2f}%")
print(f"  Baseline融合:  {avg_baseline_fus:.2f}%")
print(f"  场景2预测融合: {avg_pred_fus2:.2f}%")

print(f"\n📈 预期改进：")
print(f"  修复语义 vs Baseline语义: {avg_fixed_sem - np.mean([d['semantic'] for d in baseline_data.values()]):+.2f}% ✅")
print(f"  预测融合 vs Baseline融合: {avg_pred_fus2 - avg_baseline_fus:+.2f}%")

print("\n🎯 核心问题：Harmonic Mean的短板效应")
print("  • 如果Memory Bank弱于语义分支，会显著拉低整体性能")
print("  • 语义分支从76.81% → 85.65% (+8.84%)")  
print("  • 但如果Memory Bank仍是60-80%，融合后可能降至75-80%")
print("  • Harmonic mean让较弱分支主导结果！")

print("\n💡 建议的下一步：")
print("  1. ✅ 先测试实际融合性能（验证预测）")
print("  2. 📊 如果融合不如纯语义，考虑改进策略：")
print("     a) 加权平均：给语义分支更高权重")
print("        fusion = 0.7 * semantic + 0.3 * memory")
print("     b) 自适应融合：基于置信度动态调整")
print("        if confidence_high: use semantic")
print("        else: use fusion")
print("     c) 类别特定策略：")
print("        - screw等困难类别：只用语义")
print("        - 简单类别：可用融合")
print("  3. 🔍 分析Memory Bank在修复后的实际性能")
print("     - 可能需要重新训练Memory Bank")
print("     - 或优化Memory Bank的构建方式")

print("\n" + "="*80)
