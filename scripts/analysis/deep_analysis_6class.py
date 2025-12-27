#!/usr/bin/env python3
"""
6类深度分析：整合AUROC变化与Extended Metrics
分析目标：
1. AUROC提升与Margin/Separation改善的相关性
2. Collapse指标变化
3. Normal侧 vs Abnormal侧的改善
4. 理论验证：三项改动是否达到预期效果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# 设置
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (16, 12)

# 1. 加载训练日志AUROC（Image-AUROC，融合分支）
training_auroc = {
    'toothbrush': {'prompt2': 86.94, 'ours': 88.61},
    'capsule': {'prompt2': 64.46, 'ours': 67.41},
    'carpet': {'prompt2': 100.0, 'ours': 100.0},
    'leather': {'prompt2': 100.0, 'ours': 99.93},
    'screw': {'prompt2': 73.36, 'ours': 75.49},
    'pcb2': {'prompt2': 66.71, 'ours': 66.61},
}

# 2. 加载Extended Metrics汇总
df_extended = pd.read_csv('analysis/6class_comparison/comparison_summary.csv')

print("="*100)
print("6类深度分析：AUROC变化 vs Extended Metrics")
print("="*100)
print()

# 3. 整合数据
rows = []
for idx, row in df_extended.iterrows():
    cls_name = row['class'].split('-')[1]  # 提取类别名
    
    # 训练日志AUROC（融合分支）
    train_p2 = training_auroc[cls_name]['prompt2']
    train_ours = training_auroc[cls_name]['ours']
    train_delta = train_ours - train_p2
    
    # Extended Metrics AUROC（语义分支）
    semantic_p2 = row['prompt2_auroc'] * 100  # 转为百分比
    semantic_ours = row['ours_auroc'] * 100
    semantic_delta = semantic_ours - semantic_p2
    
    # Separation变化
    sep_p2 = row['prompt2_separation']
    sep_ours = row['ours_separation']
    sep_delta = sep_ours - sep_p2
    
    # Normal margin变化
    nm_p2 = row['prompt2_normal_margin']
    nm_ours = row['ours_normal_margin']
    nm_delta = nm_ours - nm_p2
    
    # Collapse代理（semantic_std，越大越好）
    collapse_p2 = row['prompt2_semantic_std']
    collapse_ours = row['ours_semantic_std']
    collapse_delta = collapse_ours - collapse_p2
    
    rows.append({
        'class': row['class'],
        'group': row['group'],
        # 训练AUROC（融合）
        'train_auroc_p2': train_p2,
        'train_auroc_ours': train_ours,
        'train_auroc_delta': train_delta,
        # 语义AUROC
        'semantic_auroc_p2': semantic_p2,
        'semantic_auroc_ours': semantic_ours,
        'semantic_auroc_delta': semantic_delta,
        # Separation
        'separation_p2': sep_p2,
        'separation_ours': sep_ours,
        'separation_delta': sep_delta,
        # Normal margin
        'normal_margin_p2': nm_p2,
        'normal_margin_ours': nm_ours,
        'normal_margin_delta': nm_delta,
        # Collapse
        'collapse_p2': collapse_p2,
        'collapse_ours': collapse_ours,
        'collapse_delta': collapse_delta,
    })

df_analysis = pd.DataFrame(rows)

# 4. 打印详细对比表
print("【详细对比表】")
print("="*100)
print(f"{'类别':<20} {'组':<12} {'训练AUROC Δ':<15} {'语义AUROC Δ':<15} {'Separation Δ':<15} {'Margin Δ':<12} {'Collapse Δ':<12}")
print("-"*100)

for idx, row in df_analysis.iterrows():
    print(f"{row['class']:<20} {row['group']:<12} "
          f"{row['train_auroc_delta']:>+14.2f} {row['semantic_auroc_delta']:>+14.2f} "
          f"{row['separation_delta']:>+14.4f} {row['normal_margin_delta']:>+11.4f} "
          f"{row['collapse_delta']:>+11.4f}")

print("="*100)
print()

# 5. 分组统计
print("【分组统计】")
print("="*100)

for group in ['Severe', 'Stable', 'Improved']:
    group_df = df_analysis[df_analysis['group'] == group]
    if len(group_df) == 0:
        continue
    
    print(f"\n【{group}】(n={len(group_df)})")
    print(f"  训练AUROC变化:     {group_df['train_auroc_delta'].mean():>+8.2f}% (avg)")
    print(f"  语义AUROC变化:     {group_df['semantic_auroc_delta'].mean():>+8.2f}% (avg)")
    print(f"  Separation变化:    {group_df['separation_delta'].mean():>+8.4f} (avg)")
    print(f"  Normal Margin变化: {group_df['normal_margin_delta'].mean():>+8.4f} (avg)")
    print(f"  Collapse变化:      {group_df['collapse_delta'].mean():>+8.4f} (avg)")

print("\n" + "="*100)
print()

# 6. 相关性分析
print("【相关性分析】")
print("="*100)
print("\n核心问题：AUROC提升是否与理论预期指标（Separation/Margin）改善一致？\n")

# 6.1 训练AUROC vs Separation
r1, p1 = pearsonr(df_analysis['train_auroc_delta'], df_analysis['separation_delta'])
print(f"(1) 训练AUROC变化 vs Separation变化:")
print(f"    Pearson r = {r1:.3f}, p = {p1:.3f} {'✅ 显著' if p1 < 0.05 else '❌ 不显著'}")

# 6.2 训练AUROC vs Normal Margin
r2, p2 = pearsonr(df_analysis['train_auroc_delta'], df_analysis['normal_margin_delta'])
print(f"\n(2) 训练AUROC变化 vs Normal Margin变化:")
print(f"    Pearson r = {r2:.3f}, p = {p2:.3f} {'✅ 显著' if p2 < 0.05 else '❌ 不显著'}")

# 6.3 语义AUROC vs Separation
r3, p3 = pearsonr(df_analysis['semantic_auroc_delta'], df_analysis['separation_delta'])
print(f"\n(3) 语义AUROC变化 vs Separation变化:")
print(f"    Pearson r = {r3:.3f}, p = {p3:.3f} {'✅ 显著' if p3 < 0.05 else '❌ 不显著'}")

# 6.4 Separation vs Normal Margin
r4, p4 = pearsonr(df_analysis['separation_delta'], df_analysis['normal_margin_delta'])
print(f"\n(4) Separation变化 vs Normal Margin变化:")
print(f"    Pearson r = {r4:.3f}, p = {p4:.3f} {'✅ 显著' if p4 < 0.05 else '❌ 不显著'}")

print("\n" + "="*100)
print()

# 7. Split AUROC分析（Normal侧 vs Abnormal侧）
print("【Split AUROC分析】")
print("="*100)
print("\n改善主要来自Normal侧还是Abnormal侧？\n")

split_rows = []
for idx, row in df_analysis.iterrows():
    cls_full = row['class']
    cls_name = cls_full.split('-')[1]
    dataset = cls_full.split('-')[0]
    
    # 加载Split AUROC数据
    try:
        p2_split = pd.read_csv(f'analysis/6class_comparison/{dataset}_{cls_name}_prompt2_split_auroc.csv')
        ours_split = pd.read_csv(f'analysis/6class_comparison/{dataset}_{cls_name}_ours_split_auroc.csv')
        
        # Normal侧AUROC
        normal_p2 = p2_split['normal_semantic_auroc'].values[0]
        normal_ours = ours_split['normal_semantic_auroc'].values[0]
        normal_delta = normal_ours - normal_p2
        
        # Abnormal侧AUROC
        abnormal_p2 = p2_split['abnormal_semantic_auroc'].values[0]
        abnormal_ours = ours_split['abnormal_semantic_auroc'].values[0]
        abnormal_delta = abnormal_ours - abnormal_p2
        
        split_rows.append({
            'class': cls_full,
            'normal_delta': normal_delta,
            'abnormal_delta': abnormal_delta,
            'dominant_side': 'Normal' if abs(normal_delta) > abs(abnormal_delta) else 'Abnormal',
        })
    except:
        pass

if split_rows:
    df_split = pd.DataFrame(split_rows)
    print(f"{'类别':<20} {'Normal侧Δ':<15} {'Abnormal侧Δ':<15} {'主导侧':<12}")
    print("-"*100)
    for idx, row in df_split.iterrows():
        print(f"{row['class']:<20} {row['normal_delta']:>+14.4f} {row['abnormal_delta']:>+14.4f} {row['dominant_side']:<12}")
    
    print("-"*100)
    normal_count = (df_split['dominant_side'] == 'Normal').sum()
    print(f"主要改善侧: Normal({normal_count}/6), Abnormal({6-normal_count}/6)")

print("\n" + "="*100)
print()

# 8. 生成可视化
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 8.1 训练AUROC vs 语义AUROC对比
ax1 = fig.add_subplot(gs[0, 0])
x = np.arange(len(df_analysis))
width = 0.35
ax1.bar(x - width/2, df_analysis['train_auroc_delta'], width, label='训练AUROC (融合)', alpha=0.8)
ax1.bar(x + width/2, df_analysis['semantic_auroc_delta'], width, label='语义AUROC', alpha=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels([c.split('-')[1] for c in df_analysis['class']], rotation=45, ha='right')
ax1.set_ylabel('AUROC Change (%)')
ax1.set_title('(A) AUROC变化对比\n训练融合 vs 语义分支')
ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# 8.2 Separation变化
ax2 = fig.add_subplot(gs[0, 1])
colors = ['green' if x > 0 else 'red' for x in df_analysis['separation_delta']]
ax2.barh(df_analysis['class'], df_analysis['separation_delta'], color=colors, alpha=0.7)
ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
ax2.set_xlabel('Separation Change')
ax2.set_title('(B) Separation变化\n(Ours - Prompt2)')
ax2.grid(True, alpha=0.3, axis='x')

# 8.3 Normal Margin变化
ax3 = fig.add_subplot(gs[0, 2])
colors = ['green' if x > 0 else 'red' for x in df_analysis['normal_margin_delta']]
ax3.barh(df_analysis['class'], df_analysis['normal_margin_delta'], color=colors, alpha=0.7)
ax3.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
ax3.set_xlabel('Normal Margin Change')
ax3.set_title('(C) Normal Margin变化\n(Ours - Prompt2)')
ax3.grid(True, alpha=0.3, axis='x')

# 8.4 训练AUROC vs Separation散点图
ax4 = fig.add_subplot(gs[1, 0])
colors_map = {'Severe': 'red', 'Stable': 'green', 'Improved': 'blue'}
for group in ['Severe', 'Stable', 'Improved']:
    group_df = df_analysis[df_analysis['group'] == group]
    ax4.scatter(group_df['separation_delta'], group_df['train_auroc_delta'], 
               label=group, alpha=0.7, s=100, color=colors_map[group])
ax4.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
ax4.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
ax4.set_xlabel('Separation Change')
ax4.set_ylabel('Train AUROC Change (%)')
ax4.set_title(f'(D) 训练AUROC vs Separation\nr={r1:.3f}, p={p1:.3f}')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 8.5 语义AUROC vs Separation散点图
ax5 = fig.add_subplot(gs[1, 1])
for group in ['Severe', 'Stable', 'Improved']:
    group_df = df_analysis[df_analysis['group'] == group]
    ax5.scatter(group_df['separation_delta'], group_df['semantic_auroc_delta'], 
               label=group, alpha=0.7, s=100, color=colors_map[group])
ax5.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
ax5.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
ax5.set_xlabel('Separation Change')
ax5.set_ylabel('Semantic AUROC Change (%)')
ax5.set_title(f'(E) 语义AUROC vs Separation\nr={r3:.3f}, p={p3:.3f}')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 8.6 Collapse变化
ax6 = fig.add_subplot(gs[1, 2])
colors = ['green' if x > 0 else 'red' for x in df_analysis['collapse_delta']]
ax6.barh(df_analysis['class'], df_analysis['collapse_delta'], color=colors, alpha=0.7)
ax6.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
ax6.set_xlabel('Collapse Proxy Change (semantic_std)')
ax6.set_title('(F) Collapse代理变化\n正值=减少坍缩')
ax6.grid(True, alpha=0.3, axis='x')

# 8.7 Split AUROC对比（如果有数据）
if split_rows:
    ax7 = fig.add_subplot(gs[2, 0])
    x = np.arange(len(df_split))
    width = 0.35
    ax7.bar(x - width/2, df_split['normal_delta'], width, label='Normal侧', alpha=0.8)
    ax7.bar(x + width/2, df_split['abnormal_delta'], width, label='Abnormal侧', alpha=0.8)
    ax7.set_xticks(x)
    ax7.set_xticklabels([c.split('-')[1] for c in df_split['class']], rotation=45, ha='right')
    ax7.set_ylabel('AUROC Change')
    ax7.set_title('(G) Split AUROC变化\nNormal侧 vs Abnormal侧')
    ax7.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')

# 8.8 按组汇总热力图
ax8 = fig.add_subplot(gs[2, 1:])
group_summary = df_analysis.groupby('group').agg({
    'train_auroc_delta': 'mean',
    'semantic_auroc_delta': 'mean',
    'separation_delta': 'mean',
    'normal_margin_delta': 'mean',
    'collapse_delta': 'mean',
}).T

im = ax8.imshow(group_summary.values, cmap='RdYlGn', aspect='auto', vmin=-2, vmax=2)
ax8.set_xticks(np.arange(len(group_summary.columns)))
ax8.set_yticks(np.arange(len(group_summary.index)))
ax8.set_xticklabels(group_summary.columns)
ax8.set_yticklabels(['训练AUROC Δ', '语义AUROC Δ', 'Separation Δ', 'Margin Δ', 'Collapse Δ'])
ax8.set_title('(H) 按组汇总热力图')

for i in range(len(group_summary.index)):
    for j in range(len(group_summary.columns)):
        text = ax8.text(j, i, f'{group_summary.values[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=10)

plt.colorbar(im, ax=ax8, label='Change')

plt.savefig('analysis/6class_comparison/deep_analysis_visualization.png', dpi=300, bbox_inches='tight')
print(f"✅ 可视化已保存: analysis/6class_comparison/deep_analysis_visualization.png")
print()

# 9. 综合结论
print("="*100)
print("💡 综合结论")
print("="*100)
print()

# 9.1 AUROC结论
train_improve = (df_analysis['train_auroc_delta'] > 0).sum()
semantic_improve = (df_analysis['semantic_auroc_delta'] > 0).sum()
print(f"【AUROC变化】")
print(f"  训练AUROC（融合）: {train_improve}/6 类改善, 平均 {df_analysis['train_auroc_delta'].mean():+.2f}%")
print(f"  语义AUROC:         {semantic_improve}/6 类改善, 平均 {df_analysis['semantic_auroc_delta'].mean():+.2f}%")

if abs(df_analysis['train_auroc_delta'].mean()) > abs(df_analysis['semantic_auroc_delta'].mean()):
    print(f"  ⚠️  训练AUROC提升大于语义AUROC → 改善可能来自Memory Bank（非训练改动）")
else:
    print(f"  ✅ 语义分支与融合分支变化一致")

# 9.2 Separation结论
sep_improve = (df_analysis['separation_delta'] > 0).sum()
print(f"\n【Separation变化】")
print(f"  改善类别: {sep_improve}/6, 平均变化 {df_analysis['separation_delta'].mean():+.4f}")

if df_analysis['separation_delta'].mean() > 0.01:
    print(f"  ✅ Separation显著改善 → Margin Loss有效")
elif df_analysis['separation_delta'].mean() > 0:
    print(f"  ⚖️  Separation略有改善 → Margin Loss效果温和")
else:
    print(f"  ❌ Separation未改善甚至下降 → Margin Loss可能干扰了Stable类")

# 9.3 相关性结论
print(f"\n【相关性验证】")
if p1 < 0.05:
    print(f"  ✅ 训练AUROC提升与Separation改善显著相关 (r={r1:.3f}, p={p1:.3f})")
    print(f"     → 理论预期得到验证：Margin改善确实带来性能提升")
else:
    print(f"  ❌ 训练AUROC与Separation无显著相关 (r={r1:.3f}, p={p1:.3f})")
    print(f"     → 性能提升可能来自其他因素（EMA/Repulsion/Memory Bank）")

# 9.4 Collapse结论
collapse_improve = (df_analysis['collapse_delta'] > 0).sum()
print(f"\n【Collapse变化】")
print(f"  Collapse减少（std增加）: {collapse_improve}/6 类, 平均变化 {df_analysis['collapse_delta'].mean():+.4f}")

if df_analysis['collapse_delta'].mean() > 0.002:
    print(f"  ✅ Collapse显著减少 → Repulsion Loss有效")
elif df_analysis['collapse_delta'].mean() > 0:
    print(f"  ⚖️  Collapse略有减少 → Repulsion权重可能过小")
else:
    print(f"  ❌ Collapse未改善 → Repulsion Loss未生效或权重不足")

# 9.5 最终建议
print(f"\n【决策建议】")
print("="*100)

severe_train = df_analysis[df_analysis['group'] == 'Severe']['train_auroc_delta'].mean()
severe_sep = df_analysis[df_analysis['group'] == 'Severe']['separation_delta'].mean()

if severe_train > 1.5 and severe_sep > 0.01 and p1 < 0.1:
    print("✅ 建议：扩展到27类全量验证")
    print("   理由：")
    print("   - Severe组显著改善（训练AUROC +{:.2f}%）".format(severe_train))
    print("   - Separation改善与AUROC提升相关")
    print("   - Screw未回退，改动未破坏Improved类")
elif severe_train > 0.5:
    print("⚖️  建议：调整超参数后重训6类")
    print("   理由：")
    print("   - Severe组有改善趋势但不够显著（+{:.2f}%）".format(severe_train))
    print("   - 考虑调整：")
    print("     • 增大lambda_margin（当前0.1 → 0.2）如果Separation改善不明显")
    print("     • 增大lambda_rep（当前0.05 → 0.1）如果Collapse未减少")
    print("     • 减小lambda_margin如果Stable组退化严重")
else:
    print("❌ 建议：单项改动测试（隔离EMA/Repulsion/Margin）")
    print("   理由：")
    print("   - 统一改动未带来预期改善")
    print("   - 需要确定是哪项改动引入负面影响")
    print("   - 测试顺序：")
    print("     1. 只修正EMA（不加Repulsion/Margin）")
    print("     2. EMA + Margin（不加Repulsion）")
    print("     3. EMA + Repulsion（不加Margin）")

print("\n" + "="*100)

# 保存分析数据
df_analysis.to_csv('analysis/6class_comparison/deep_analysis_data.csv', index=False, float_format='%.4f')
print(f"\n✅ 分析数据已保存: analysis/6class_comparison/deep_analysis_data.csv")
