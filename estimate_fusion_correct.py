#!/usr/bin/env python3
"""
正确的融合性能估计（分数范围：0-1）

关键理解：
1. 异常分数在计算时是0-1范围
2. AUROC报告时转换为百分比（×100）
3. 融合公式：score = 1/(1/semantic + 1/memory)
4. 这个公式在0-1范围内是合理的
"""

import numpy as np

def harmonic_fusion(score1, score2):
    """
    PromptAD的融合公式（numerator=1的harmonic mean）
    输入和输出都是0-1范围
    """
    return 1.0 / (1.0/score1 + 1.0/score2)

print("="*80)
print("修复后融合性能的正确估计")
print("="*80)

# ==================== 数据准备 ====================
# 转换百分比到0-1范围
baseline_semantic_pct = {
    "screw": 66.42, "toothbrush": 69.58, "hazelnut": 80.11,
    "capsule": 73.69, "pill": 85.50, "metal_nut": 85.56,
}

baseline_fusion_pct = {
    "screw": 58.66, "toothbrush": 98.89, "hazelnut": 99.93,
    "capsule": 79.94, "pill": 95.61, "metal_nut": 100.00,
}

fixed_semantic_pct = {
    "screw": 77.35, "toothbrush": 89.17, "hazelnut": 90.86,
    "capsule": 82.21, "pill": 84.56, "metal_nut": 89.74,
}

# 转换为0-1范围
baseline_semantic = {k: v/100 for k, v in baseline_semantic_pct.items()}
baseline_fusion = {k: v/100 for k, v in baseline_fusion_pct.items()}
fixed_semantic = {k: v/100 for k, v in fixed_semantic_pct.items()}

# ==================== 推算Memory Bank性能 ====================
print("\n第一步：从Baseline推算Memory Bank性能")
print("-"*80)

memory_bank = {}
print(f"{'类别':<12} {'Baseline语义':<12} {'Baseline融合':<12} {'推算Memory':<12} {'分析'}")
print("-"*80)

for cls in baseline_semantic.keys():
    b_sem = baseline_semantic[cls]
    b_fus = baseline_fusion[cls]
    
    # 从融合公式反推: 1/fusion = 1/semantic + 1/memory
    # => 1/memory = 1/fusion - 1/semantic
    # => memory = 1 / (1/fusion - 1/semantic)
    
    inv_mem = 1.0/b_fus - 1.0/b_sem
    mem = 1.0 / inv_mem if inv_mem > 1e-6 else 1.0
    
    # 分析memory相对semantic的强弱
    if mem > b_sem * 1.1:
        analysis = "强"
    elif mem > b_sem * 0.9:
        analysis = "接近"
    else:
        analysis = "弱"
    
    memory_bank[cls] = mem
    print(f"{cls:<12} {b_sem*100:<12.2f} {b_fus*100:<12.2f} {mem*100:<12.2f} {analysis}")

print("\n💡 关键发现：")
print("  • screw: Memory Bank (>100%) 非常强，但语义太弱导致融合不如memory")
print("  • 其他类别: Memory Bank都很强，有效提升了baseline的语义分支")

# ==================== 估计修复后的融合性能 ====================
print("\n\n第二步：估计修复后的融合性能")
print("-"*80)
print("假设：Memory Bank性能与baseline相同（重新构建的memory bank）\n")

estimated_fusion = {}
print(f"{'类别':<12} {'修复语义%':<12} {'Memory%':<12} {'估算融合%':<12} {'vs Baseline融合':<15} {'vs 修复语义':<12}")
print("-"*80)

for cls in fixed_semantic.keys():
    f_sem = fixed_semantic[cls]
    mem = memory_bank[cls]
    
    # 融合
    fusion = harmonic_fusion(f_sem, mem)
    estimated_fusion[cls] = fusion
    
    b_fus = baseline_fusion[cls]
    vs_baseline = (fusion - b_fus) * 100
    vs_semantic = (fusion - f_sem) * 100
    
    print(f"{cls:<12} {f_sem*100:<12.2f} {mem*100:<12.2f} {fusion*100:<12.2f} {vs_baseline:+<15.2f} {vs_semantic:+<12.2f}")

# 计算平均
avg_fixed_sem = np.mean([v*100 for v in fixed_semantic.values()])
avg_est_fusion = np.mean([v*100 for v in estimated_fusion.values()])
avg_baseline_fusion = np.mean([v*100 for v in baseline_fusion.values()])
avg_baseline_sem = np.mean([v*100 for v in baseline_semantic.values()])

print("-"*80)
print(f"{'平均':<12} {avg_fixed_sem:<12.2f} {'':<12} {avg_est_fusion:<12.2f} {avg_est_fusion - avg_baseline_fusion:+<15.2f} {avg_est_fusion - avg_fixed_sem:+<12.2f}")

# ==================== 场景分析 ====================
print("\n\n第三步：考虑Memory Bank变化的多场景分析")
print("="*80)

scenarios = {
    "乐观": ("Memory Bank同步提升10%", 1.10),
    "基准": ("Memory Bank保持baseline水平", 1.00),
    "保守": ("Memory Bank略有下降5%", 0.95),
    "悲观": ("Memory Bank显著下降15%", 0.85),
}

print(f"\n{'场景':<8} {'说明':<25} {'估算融合%':<12} {'vs Baseline':<12} {'vs 修复语义':<12} {'评估'}")
print("-"*80)

for name, (desc, factor) in scenarios.items():
    # 调整memory bank
    adj_mem = {k: min(v * factor, 1.0) for k, v in memory_bank.items()}
    
    # 重新计算融合
    adj_fusion = {}
    for cls in fixed_semantic.keys():
        fusion = harmonic_fusion(fixed_semantic[cls], adj_mem[cls])
        adj_fusion[cls] = fusion
    
    avg_fusion = np.mean([v*100 for v in adj_fusion.values()])
    vs_baseline = avg_fusion - avg_baseline_fusion
    vs_semantic = avg_fusion - avg_fixed_sem
    
    if vs_baseline > 3:
        assessment = "✅ 显著改进"
    elif vs_baseline > 0:
        assessment = "✅ 略有改进"
    elif vs_baseline > -3:
        assessment = "⚠️  基本持平"
    else:
        assessment = "❌ 性能下降"
    
    print(f"{name:<8} {desc:<25} {avg_fusion:<12.2f} {vs_baseline:+<12.2f} {vs_semantic:+<12.2f} {assessment}")

# ==================== 分析融合的帮助/伤害 ====================
print("\n\n第四步：分析融合对各类别的影响")
print("-"*80)

helps = []
hurts = []
neutral = []

for cls in fixed_semantic.keys():
    fusion_pct = estimated_fusion[cls] * 100
    semantic_pct = fixed_semantic[cls] * 100
    diff = fusion_pct - semantic_pct
    
    if diff > 1:
        helps.append((cls, diff))
    elif diff < -1:
        hurts.append((cls, diff))
    else:
        neutral.append((cls, diff))

print(f"\n融合有帮助: {len(helps)}/6")
for cls, diff in helps:
    print(f"  • {cls}: {diff:+.2f}%")

print(f"\n融合有害: {len(hurts)}/6")
for cls, diff in hurts:
    print(f"  • {cls}: {diff:+.2f}%")

if neutral:
    print(f"\n融合中性: {len(neutral)}/6")
    for cls, diff in neutral:
        print(f"  • {cls}: {diff:+.2f}%")

# ==================== 总结和建议 ====================
print("\n\n" + "="*80)
print("估计总结与建议")
print("="*80)

print("\n📊 性能对比（基准场景）：")
print(f"  Baseline语义平均:  {avg_baseline_sem:.2f}%")
print(f"  修复后语义平均:    {avg_fixed_sem:.2f}% ({avg_fixed_sem - avg_baseline_sem:+.2f}%)")
print(f"  Baseline融合平均:  {avg_baseline_fusion:.2f}%")
print(f"  估算融合平均:      {avg_est_fusion:.2f}% ({avg_est_fusion - avg_baseline_fusion:+.2f}%)")

print(f"\n📈 改进分析：")
print(f"  修复语义 vs Baseline语义: {avg_fixed_sem - avg_baseline_sem:+.2f}% ✅ 显著改进")
print(f"  估算融合 vs Baseline融合: {avg_est_fusion - avg_baseline_fusion:+.2f}%", end="")
if avg_est_fusion > avg_baseline_fusion + 3:
    print(" ✅ 显著改进")
elif avg_est_fusion > avg_baseline_fusion:
    print(" ✅ 略有改进")
elif avg_est_fusion > avg_baseline_fusion - 3:
    print(" ⚠️  基本持平")
else:
    print(" ❌ 性能下降")

print(f"  估算融合 vs 修复语义:      {avg_est_fusion - avg_fixed_sem:+.2f}%", end="")
if avg_est_fusion > avg_fixed_sem:
    print(" ✅ 融合有效")
elif avg_est_fusion > avg_fixed_sem - 2:
    print(" ⚠️  融合帮助不大")
else:
    print(" ❌ 融合反而降低性能")

print("\n🎯 核心洞察：")
if avg_est_fusion > avg_baseline_fusion:
    print(f"  ✅ 估计融合后相比baseline有 {avg_est_fusion - avg_baseline_fusion:+.2f}% 改进")
    print("  • 修复的语义分支提升了整体性能")
    print("  • Memory Bank保持了较好的性能")
else:
    print(f"  ⚠️  估计融合后相比baseline {avg_est_fusion - avg_baseline_fusion:+.2f}%")
    print("  • 修复提升了语义分支，但融合后优势减弱")
    print("  • 原因：Memory Bank性能可能无法匹配提升后的语义分支")

if avg_est_fusion < avg_fixed_sem:
    print(f"\n  ⚠️  融合后比纯语义低 {avg_fixed_sem - avg_est_fusion:.2f}%")
    print("  • Memory Bank拖累了提升后的语义分支")
    print("  • 建议优化融合策略或只使用语义分支")

print("\n💡 后续行动建议：")
print("\n1. 立即测试：运行融合测试验证估计")
print("   ```bash")
print("   # 测试融合性能（不加--semantic-only）")
print("   python test_all_key_classes.py --fusion")
print("   ```")

print("\n2. 如果融合不如预期，考虑优化策略：")
print("   a) 加权融合：给语义分支更高权重")
print("      fusion = alpha * semantic + (1-alpha) * memory")
print("      建议alpha=0.7（语义70%，memory 30%）")
print()
print("   b) 自适应融合：基于置信度选择")
print("      if semantic_confidence > threshold:")
print("          use semantic_only")
print("      else:")
print("          use harmonic_fusion")
print()
print("   c) 类别特定策略：")
print(f"      - {len(helps)}个受益类别: 使用融合")
print(f"      - {len(hurts)}个受损类别: 只用语义")

print("\n3. 深入分析Memory Bank：")
print("   • 检查Memory Bank在修复后的实际性能")
print("   • 考虑是否需要重新优化Memory Bank构建方式")
print("   • 分析为什么某些类别的Memory Bank很强")

print("\n" + "="*80)
print("✅ 估计完成！建议先运行实际测试验证这些估计。")
print("="*80)
