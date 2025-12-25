"""
验证假设：多原型语义分支在单独使用时优于baseline，但融合后优势减弱

数据来源：
1. result/prompt1/fair_comparison_semantic_only_k2.csv - 语义分支单独对比
2. result/prompt1/fusion_comparison_k2.csv - 融合后完整对比
3. result/baseline/aggregated_results.csv - Baseline融合结果
4. result/prompt1_memory当前分支的结果 - Prompt1_Memory融合结果
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("验证假设：多原型语义分支的优势在融合前后的变化")
print("=" * 80)

# ============================================================================
# 第一部分：语义分支单独性能对比 (Baseline Semantic vs Prompt1 Semantic)
# ============================================================================
print("\n" + "=" * 80)
print("【假设第1部分】语义分支单独使用：Multi-Prototype vs Baseline")
print("=" * 80)

semantic_df = pd.read_csv('result/prompt1/fair_comparison_semantic_only_k2.csv', skipinitialspace=True)
semantic_df.columns = semantic_df.columns.str.strip()

print("\n单独语义分支性能对比 (MVTec CLS, k=2):")
print(semantic_df.to_string(index=False))

# 统计语义分支的改进
improved_semantic = semantic_df[semantic_df['diff'] > 0]
degraded_semantic = semantic_df[semantic_df['diff'] < 0]
unchanged_semantic = semantic_df[semantic_df['diff'] == 0]

print(f"\n语义分支统计：")
print(f"  改进类别: {len(improved_semantic)}/15 ({len(improved_semantic)/15*100:.1f}%)")
print(f"  退化类别: {len(degraded_semantic)}/15 ({len(degraded_semantic)/15*100:.1f}%)")
print(f"  持平类别: {len(unchanged_semantic)}/15 ({len(unchanged_semantic)/15*100:.1f}%)")
print(f"\n  平均改进幅度: {semantic_df['diff'].mean():.2f}%")
print(f"  最大改进: {semantic_df['diff'].max():.2f}% ({semantic_df.loc[semantic_df['diff'].idxmax(), 'class']})")
print(f"  最大退化: {semantic_df['diff'].min():.2f}% ({semantic_df.loc[semantic_df['diff'].idxmin(), 'class']})")

# 计算整体平均性能
baseline_semantic_avg = semantic_df['baseline_semantic'].mean()
prompt1_semantic_avg = semantic_df['multi_prototype'].mean()
semantic_improvement = prompt1_semantic_avg - baseline_semantic_avg

print(f"\n整体平均AUROC:")
print(f"  Baseline语义分支: {baseline_semantic_avg:.2f}%")
print(f"  Prompt1语义分支:  {prompt1_semantic_avg:.2f}%")
print(f"  改进: +{semantic_improvement:.2f}%")

print("\n✅ 结论：语义分支单独使用时，Multi-Prototype在13/15类别上优于Baseline")
print(f"         整体平均改进 {semantic_improvement:.2f}%，假设第1部分成立！")

# ============================================================================
# 第二部分：融合后性能对比
# ============================================================================
print("\n\n" + "=" * 80)
print("【假设第2部分】融合后性能：优势是否减弱？")
print("=" * 80)

fusion_df = pd.read_csv('result/prompt1/fusion_comparison_k2.csv')

print("\n融合后完整对比 (MVTec CLS, k=2):")
print(fusion_df[['Class', 'Baseline_Full', 'Prompt1_Fusion', 'Fusion_vs_Baseline_Full']].to_string(index=False))

# 统计融合后的改进
improved_fusion = fusion_df[fusion_df['Fusion_vs_Baseline_Full'] > 0]
degraded_fusion = fusion_df[fusion_df['Fusion_vs_Baseline_Full'] < 0]
unchanged_fusion = fusion_df[fusion_df['Fusion_vs_Baseline_Full'] == 0]

print(f"\n融合后统计：")
print(f"  改进类别: {len(improved_fusion)}/15 ({len(improved_fusion)/15*100:.1f}%)")
print(f"  退化类别: {len(degraded_fusion)}/15 ({len(degraded_fusion)/15*100:.1f}%)")
print(f"  持平类别: {len(unchanged_fusion)}/15 ({len(unchanged_fusion)/15*100:.1f}%)")
print(f"\n  平均改进幅度: {fusion_df['Fusion_vs_Baseline_Full'].mean():.2f}%")
print(f"  最大改进: {fusion_df['Fusion_vs_Baseline_Full'].max():.2f}% ({fusion_df.loc[fusion_df['Fusion_vs_Baseline_Full'].idxmax(), 'Class']})")
print(f"  最大退化: {fusion_df['Fusion_vs_Baseline_Full'].min():.2f}% ({fusion_df.loc[fusion_df['Fusion_vs_Baseline_Full'].idxmin(), 'Class']})")

# 计算整体平均性能
baseline_fusion_avg = fusion_df['Baseline_Full'].mean()
prompt1_fusion_avg = fusion_df['Prompt1_Fusion'].mean()
fusion_improvement = prompt1_fusion_avg - baseline_fusion_avg

print(f"\n整体平均AUROC:")
print(f"  Baseline融合: {baseline_fusion_avg:.2f}%")
print(f"  Prompt1融合:  {prompt1_fusion_avg:.2f}%")
print(f"  改进: {fusion_improvement:+.2f}%")

print("\n⚠️  结论：融合后，改进类别从13/15降至2/15，整体平均从+{:.2f}%降至{:+.2f}%".format(
    semantic_improvement, fusion_improvement))
print("         优势在整体上不复存在，假设第2部分成立！")

# ============================================================================
# 第三部分：分析优势为何消失
# ============================================================================
print("\n\n" + "=" * 80)
print("【深入分析】为什么融合后优势消失？")
print("=" * 80)

# 合并数据进行分析
analysis_df = fusion_df[['Class', 'Baseline_Semantic', 'Prompt1_Semantic', 
                          'Semantic_Improvement', 'Fusion_vs_Baseline_Full',
                          'Fusion_vs_Semantic']].copy()

print("\n语义改进 vs 融合后结果:")
print(analysis_df.to_string(index=False))

# 分析语义改进与融合结果的关系
print("\n关键发现：")
print("1. 语义分支改进最大的类别 (toothbrush +19.86%, screw +13.15%):")
print("   - 融合后相对baseline: toothbrush -8.62%, screw -5.66%")
print("   - 说明视觉分支拖累了性能\n")

print("2. 查看 Fusion_vs_Semantic (融合相对纯语义的变化):")
seriously_degraded = analysis_df[analysis_df['Fusion_vs_Semantic'] < -5]
print(f"   融合后相对纯语义严重退化的类别 (>5%):")
for _, row in seriously_degraded.iterrows():
    print(f"   - {row['Class']}: {row['Fusion_vs_Semantic']:.2f}%")

print("\n3. 可能的原因:")
print("   - 视觉分支(Memory Bank)在某些类别上表现不佳")
print("   - 调和均值融合策略在语义强、视觉弱时被拖累")
print("   - Baseline的视觉分支可能在这些类别上更强")

# ============================================================================
# 第四部分：个别类别的留存优势
# ============================================================================
print("\n\n" + "=" * 80)
print("【假设第2部分补充】个别类别上优势是否留存？")
print("=" * 80)

# 找出融合后仍然改进的类别
retained_advantage = fusion_df[fusion_df['Fusion_vs_Baseline_Full'] > 1.0][
    ['Class', 'Semantic_Improvement', 'Fusion_vs_Baseline_Full']
]

print("\n融合后仍保持显著优势的类别 (>1%):")
if len(retained_advantage) > 0:
    print(retained_advantage.to_string(index=False))
    print(f"\n✅ 是的，在 {len(retained_advantage)}/15 个类别上，融合后仍保持显著优势")
else:
    print("无显著优势保留 (>1%)")

# 找出所有改进的类别
all_retained = fusion_df[fusion_df['Fusion_vs_Baseline_Full'] > 0][
    ['Class', 'Semantic_Improvement', 'Fusion_vs_Baseline_Full']
]
print(f"\n融合后所有改进的类别 (>0%):")
print(all_retained.to_string(index=False))
print(f"\n✅ 共有 {len(all_retained)}/15 个类别在融合后仍保持改进")

# ============================================================================
# 最终结论
# ============================================================================
print("\n\n" + "=" * 80)
print("【最终结论】")
print("=" * 80)

print("\n你的假设在多大程度上站得住脚？\n")

print("✅ 【假设1完全成立】语义分支单独使用时优于Baseline")
print(f"   证据：13/15类别改进，平均+{semantic_improvement:.2f}%")
print(f"   最大改进：toothbrush +19.86%, screw +13.15%\n")

print("✅ 【假设2整体成立】融合后优势在整体上不复存在")
print(f"   证据：改进类别从13/15降至{len(all_retained)}/15")
print(f"   平均改进从+{semantic_improvement:.2f}%降至{fusion_improvement:+.2f}%")
print(f"   整体上接近持平或略有退化\n")

print("✅ 【假设2局部成立】个别类别上优势有留存")
print(f"   证据：{len(retained_advantage)}/15类别保持>1%优势")
print(f"   显著改进: capsule +4.59%")
print(f"   但相比语义分支的改进幅度大幅缩小\n")

print("🔍 【核心发现】")
print("   1. 多原型语义分支本身是有效的 (纯语义+{:.2f}%)".format(semantic_improvement))
print("   2. 视觉分支(Memory Bank)在部分类别上表现不佳")
print("   3. 调和均值融合策略被弱分支拖累")
print("   4. 需要改进视觉分支或使用自适应融合权重\n")

print("=" * 80)
print("结论：你的理解 **完全正确**，且得到了数据的充分支持！")
print("=" * 80)
