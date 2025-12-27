#!/usr/bin/env python3
"""
5版本对比分析脚本
版本: Baseline, Prompt2, Ours_v1, Ours_v2, Ours_v3
重点: 验证自适应Repulsion策略是否优于统一配置
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# 设置样式
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (20, 12)

# 类别分组
CLASS_GROUPS = {
    'mvtec-toothbrush': 'Severe',
    'mvtec-capsule': 'Severe',
    'visa-pcb2': 'Severe',
    'mvtec-carpet': 'Stable',
    'mvtec-leather': 'Stable',
    'mvtec-screw': 'Improved'
}

# 读取adaptive repulsion配置
with open('analysis/controlled_comparison/adaptive_repulsion_config.json', 'r') as f:
    adaptive_config = json.load(f)

print("=" * 80)
print("5版本对比分析: Baseline vs Prompt2 vs v1 vs v2 vs v3")
print("=" * 80)

# 步骤1: 加载所有评估数据
data_dir = Path('analysis/5version_comparison')
versions = ['Baseline', 'Prompt2', 'Ours_v1', 'Ours_v2', 'Ours_v3']

all_data = []
for class_name in CLASS_GROUPS.keys():
    row = {'class': class_name, 'group': CLASS_GROUPS[class_name]}
    
    for version in versions:
        prefix = f"{class_name}_{version}"
        
        # 读取split_auroc
        auroc_file = data_dir / f"{prefix}_split_auroc.csv"
        if auroc_file.exists():
            auroc_df = pd.read_csv(auroc_file)
            train_auroc = auroc_df[auroc_df['split'] == 'train']['auroc'].values[0] * 100
            row[f'train_{version}'] = train_auroc
        
        # 读取margin_stats
        margin_file = data_dir / f"{prefix}_margin_stats.csv"
        if margin_file.exists():
            margin_df = pd.read_csv(margin_file)
            separation = margin_df['separation'].values[0]
            row[f'sep_{version}'] = separation
        
        # 读取semantic_contrib
        sem_file = data_dir / f"{prefix}_semantic_contrib.csv"
        if sem_file.exists():
            sem_df = pd.read_csv(sem_file)
            sem_auroc = sem_df['semantic_auroc'].values[0]
            row[f'sem_{version}'] = sem_auroc
    
    all_data.append(row)

df = pd.DataFrame(all_data)

# 计算变化量 (相对Prompt2 baseline)
for version in ['Ours_v1', 'Ours_v2', 'Ours_v3']:
    df[f'train_delta_{version}'] = df[f'train_{version}'] - df['train_Prompt2']
    df[f'sep_delta_{version}'] = df[f'sep_{version}'] - df['sep_Prompt2']
    df[f'sem_delta_{version}'] = df[f'sem_{version}'] - df['sem_Prompt2']

# 步骤2: 详细对比表
print("\n【详细对比表】")
print("=" * 80)
print(f"{'类别':<20} {'组':<8} {'训练Δv1':<10} {'训练Δv2':<10} {'训练Δv3':<10} {'SepΔv3':<10}")
print("-" * 80)
for _, row in df.iterrows():
    print(f"{row['class']:<20} {row['group']:<8} "
          f"{row['train_delta_Ours_v1']:>8.2f}% "
          f"{row['train_delta_Ours_v2']:>8.2f}% "
          f"{row['train_delta_Ours_v3']:>8.2f}% "
          f"{row['sep_delta_Ours_v3']:>8.4f}")
print("=" * 80)

# 步骤3: 分组统计
print("\n【分组对比】")
print("=" * 80)
for group in ['Severe', 'Stable', 'Improved']:
    group_df = df[df['group'] == group]
    print(f"\n【{group}】(n={len(group_df)})")
    
    print("  训练AUROC变化:")
    for version in ['Ours_v1', 'Ours_v2', 'Ours_v3']:
        mean_delta = group_df[f'train_delta_{version}'].mean()
        print(f"    {version:8s}: {mean_delta:>+6.2f}%")
    
    print("  Separation变化:")
    for version in ['Ours_v1', 'Ours_v2', 'Ours_v3']:
        mean_sep = group_df[f'sep_delta_{version}'].mean()
        print(f"    {version:8s}: {mean_sep:>+8.4f}")

# 步骤4: v3 vs v2重点对比
print("\n" + "=" * 80)
print("【v3 vs v2 关键对比】(验证自适应策略)")
print("=" * 80)

comparison_data = []
for _, row in df.iterrows():
    class_name = row['class']
    lambda_rep_v3 = adaptive_config['class_lambda_rep'][class_name]
    
    improvement = {
        'class': class_name,
        'lambda_v3': lambda_rep_v3,
        'train_v2': row['train_delta_Ours_v2'],
        'train_v3': row['train_delta_Ours_v3'],
        'train_improve': row['train_delta_Ours_v3'] - row['train_delta_Ours_v2'],
        'sep_v2': row['sep_delta_Ours_v2'],
        'sep_v3': row['sep_delta_Ours_v3'],
        'sep_improve': row['sep_delta_Ours_v3'] - row['sep_delta_Ours_v2']
    }
    comparison_data.append(improvement)

comp_df = pd.DataFrame(comparison_data)

print(f"{'类别':<20} {'λ_v3':<8} {'训练Δv2':<10} {'训练Δv3':<10} {'改善':<10} {'分离改善':<10}")
print("-" * 80)
for _, row in comp_df.iterrows():
    train_marker = "✅" if row['train_improve'] > 0 else "⚠️" if row['train_improve'] < -1 else "→"
    sep_marker = "✅" if row['sep_improve'] > 0.01 else "→"
    print(f"{row['class']:<20} {row['lambda_v3']:<8.2f} "
          f"{row['train_v2']:>8.2f}% "
          f"{row['train_v3']:>8.2f}% "
          f"{row['train_improve']:>+7.2f}% {train_marker}  "
          f"{row['sep_improve']:>+7.4f} {sep_marker}")

print("=" * 80)

# 步骤5: 总体统计
print("\n【总体统计】")
print("=" * 80)

overall_stats = {
    'v1': {
        'train_mean': df['train_delta_Ours_v1'].mean(),
        'sep_mean': df['sep_delta_Ours_v1'].mean(),
        'improve_count': (df['train_delta_Ours_v1'] > 0).sum()
    },
    'v2': {
        'train_mean': df['train_delta_Ours_v2'].mean(),
        'sep_mean': df['sep_delta_Ours_v2'].mean(),
        'improve_count': (df['train_delta_Ours_v2'] > 0).sum()
    },
    'v3': {
        'train_mean': df['train_delta_Ours_v3'].mean(),
        'sep_mean': df['sep_delta_Ours_v3'].mean(),
        'improve_count': (df['train_delta_Ours_v3'] > 0).sum()
    }
}

for version, stats in overall_stats.items():
    print(f"\n{version}:")
    print(f"  平均训练AUROC提升: {stats['train_mean']:+.2f}%")
    print(f"  平均Separation变化: {stats['sep_mean']:+.4f}")
    print(f"  改善类别数: {stats['improve_count']}/6")

# 步骤6: 假设验证
print("\n" + "=" * 80)
print("【假设验证】")
print("=" * 80)

print("\n假设: 类别自适应Repulsion优于统一配置")

# H1: v3整体性能 >= v2
h1_result = overall_stats['v3']['train_mean'] >= overall_stats['v2']['train_mean']
print(f"\nH1: v3平均AUROC ≥ v2")
print(f"  v2: {overall_stats['v2']['train_mean']:+.2f}%")
print(f"  v3: {overall_stats['v3']['train_mean']:+.2f}%")
print(f"  {'✅ 成立' if h1_result else '❌ 不成立'}")

# H2: toothbrush改善
toothbrush_data = df[df['class'] == 'mvtec-toothbrush'].iloc[0]
h2_improve = toothbrush_data['train_delta_Ours_v3'] - toothbrush_data['train_delta_Ours_v2']
h2_result = h2_improve > 3.0  # 预期改善 > 3%
print(f"\nH2: toothbrush改善 > 3%")
print(f"  v2: {toothbrush_data['train_delta_Ours_v2']:+.2f}%")
print(f"  v3: {toothbrush_data['train_delta_Ours_v3']:+.2f}%")
print(f"  改善: {h2_improve:+.2f}%")
print(f"  {'✅ 成立' if h2_result else '❌ 不成立'}")

# H3: Stable类Separation改善
stable_sep_v2 = df[df['group'] == 'Stable']['sep_delta_Ours_v2'].mean()
stable_sep_v3 = df[df['group'] == 'Stable']['sep_delta_Ours_v3'].mean()
h3_result = stable_sep_v3 > stable_sep_v2
print(f"\nH3: Stable类Separation改善")
print(f"  v2: {stable_sep_v2:+.4f}")
print(f"  v3: {stable_sep_v3:+.4f}")
print(f"  {'✅ 成立' if h3_result else '❌ 不成立'}")

# 步骤7: 可视化
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# Plot 1: 训练AUROC对比
ax = axes[0, 0]
x = np.arange(len(df))
width = 0.2
ax.bar(x - width, df['train_delta_Ours_v1'], width, label='v1', alpha=0.8)
ax.bar(x, df['train_delta_Ours_v2'], width, label='v2', alpha=0.8)
ax.bar(x + width, df['train_delta_Ours_v3'], width, label='v3 (Adaptive)', alpha=0.8)
ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax.set_xlabel('Class')
ax.set_ylabel('Training AUROC Change (%)')
ax.set_title('Training AUROC Change (vs Prompt2)')
ax.set_xticks(x)
ax.set_xticklabels(df['class'].str.replace('mvtec-', '').str.replace('visa-', ''), rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 2: Separation对比
ax = axes[0, 1]
ax.bar(x - width, df['sep_delta_Ours_v1'], width, label='v1', alpha=0.8)
ax.bar(x, df['sep_delta_Ours_v2'], width, label='v2', alpha=0.8)
ax.bar(x + width, df['sep_delta_Ours_v3'], width, label='v3 (Adaptive)', alpha=0.8)
ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax.set_xlabel('Class')
ax.set_ylabel('Separation Change')
ax.set_title('Separation Change (vs Prompt2)')
ax.set_xticks(x)
ax.set_xticklabels(df['class'].str.replace('mvtec-', '').str.replace('visa-', ''), rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 3: v3 vs v2改善
ax = axes[0, 2]
improvements = df['train_delta_Ours_v3'] - df['train_delta_Ours_v2']
colors = ['green' if x > 0 else 'red' for x in improvements]
ax.bar(x, improvements, color=colors, alpha=0.6)
ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax.set_xlabel('Class')
ax.set_ylabel('Improvement (%)')
ax.set_title('v3 vs v2: Training AUROC Improvement')
ax.set_xticks(x)
ax.set_xticklabels(df['class'].str.replace('mvtec-', '').str.replace('visa-', ''), rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)

# Plot 4: Lambda_rep分布
ax = axes[1, 0]
lambda_values = [adaptive_config['class_lambda_rep'][c] for c in df['class']]
ax.bar(x, lambda_values, color='purple', alpha=0.6)
ax.set_xlabel('Class')
ax.set_ylabel('lambda_rep')
ax.set_title('Adaptive Repulsion Weights (v3)')
ax.set_xticks(x)
ax.set_xticklabels(df['class'].str.replace('mvtec-', '').str.replace('visa-', ''), rotation=45, ha='right')
ax.set_ylim([0, 0.12])
for i, v in enumerate(lambda_values):
    ax.text(i, v + 0.005, f'{v:.2f}', ha='center', fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Plot 5: 分组统计
ax = axes[1, 1]
groups = ['Severe', 'Stable', 'Improved']
v1_means = [df[df['group'] == g]['train_delta_Ours_v1'].mean() for g in groups]
v2_means = [df[df['group'] == g]['train_delta_Ours_v2'].mean() for g in groups]
v3_means = [df[df['group'] == g]['train_delta_Ours_v3'].mean() for g in groups]
x_pos = np.arange(len(groups))
ax.bar(x_pos - width, v1_means, width, label='v1', alpha=0.8)
ax.bar(x_pos, v2_means, width, label='v2', alpha=0.8)
ax.bar(x_pos + width, v3_means, width, label='v3 (Adaptive)', alpha=0.8)
ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax.set_xlabel('Group')
ax.set_ylabel('Mean Training AUROC Change (%)')
ax.set_title('Group-wise Performance')
ax.set_xticks(x_pos)
ax.set_xticklabels(groups)
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 6: 语义AUROC对比
ax = axes[1, 2]
ax.bar(x - width, df['sem_delta_Ours_v1'], width, label='v1', alpha=0.8)
ax.bar(x, df['sem_delta_Ours_v2'], width, label='v2', alpha=0.8)
ax.bar(x + width, df['sem_delta_Ours_v3'], width, label='v3 (Adaptive)', alpha=0.8)
ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax.set_xlabel('Class')
ax.set_ylabel('Semantic AUROC Change (%)')
ax.set_title('Semantic AUROC Change (vs Prompt2)')
ax.set_xticks(x)
ax.set_xticklabels(df['class'].str.replace('mvtec-', '').str.replace('visa-', ''), rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('analysis/5version_comparison/5version_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ 可视化已保存: analysis/5version_comparison/5version_analysis.png")

# 保存数据
df.to_csv('analysis/5version_comparison/5version_data.csv', index=False)
comp_df.to_csv('analysis/5version_comparison/v3_vs_v2_comparison.csv', index=False)
print("✅ 数据已保存: analysis/5version_comparison/5version_data.csv")

# 步骤8: 结论
print("\n" + "=" * 80)
print("💡 结论")
print("=" * 80)

if h1_result and h2_result:
    print("✅ 类别自适应Repulsion策略验证成功！")
    print(f"  - 整体性能: v3 ({overall_stats['v3']['train_mean']:+.2f}%) > v2 ({overall_stats['v2']['train_mean']:+.2f}%)")
    print(f"  - toothbrush修复: 改善 {h2_improve:+.2f}%")
    print(f"  - 推荐: 使用v3配置扩展到27类")
elif h1_result:
    print("⚖️ v3整体优于v2，但toothbrush改善不足预期")
    print(f"  - 可能需要进一步调优λ_rep值")
else:
    print("❌ v3未达预期，需要重新审视策略")
    print("  - 建议: 尝试EMA-only或其他方案")

print("=" * 80)
