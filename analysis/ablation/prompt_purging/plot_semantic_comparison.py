#!/usr/bin/env python3
"""
生成Prompt-Purging对Semantic AUROC影响的柱形图
MVTec K=2 CLS任务
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
baseline_path = '/home/zju/mywork/PromptAD/result/baseline/mvtec/k_2/csv/Seed_111-results.csv'
purged_path = '/home/zju/mywork/PromptAD/result/promptpurging/mvtec/k_2/csv/Seed_111-results.csv'

df_baseline = pd.read_csv(baseline_path, index_col=0)
df_purged = pd.read_csv(purged_path, index_col=0)

# 提取数据
classes = [cls.replace('mvtec-', '') for cls in df_baseline.index]
baseline_semantic = df_baseline['semantic_i_roc'].values
purged_semantic = df_purged['semantic_i_roc'].values
delta = purged_semantic - baseline_semantic

# 设置图形
fig, ax = plt.subplots(figsize=(14, 6))

# 设置柱形位置
x = np.arange(len(classes))

# 为每个柱子设置颜色（正值绿色，负值红色）
colors = ['#27ae60' if d > 0 else '#e74c3c' if d < 0 else '#95a5a6' for d in delta]

# 绘制delta柱形
bars = ax.bar(x, delta, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

# 在柱形上标注数值
for i, (d, bar) in enumerate(zip(delta, bars)):
    y_pos = d + 0.3 if d > 0 else d - 0.3
    va = 'bottom' if d > 0 else 'top'
    ax.text(i, y_pos, f'{d:+.1f}%', ha='center', va=va, 
            fontsize=9, fontweight='bold')

# 添加零线
ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)

# 设置轴标签和标题
ax.set_xlabel('MVTec Classes', fontsize=12, fontweight='bold')
ax.set_ylabel('Semantic I-AUROC Change (%)', fontsize=12, fontweight='bold')
ax.set_title('Impact of Prompt-Purging on Semantic Branch Performance (MVTec K=2)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3, linestyle='--')

# 设置Y轴范围
y_max = max(abs(delta.min()), abs(delta.max())) + 1
ax.set_ylim([-y_max, y_max])

# 添加平均提升线
avg_delta = delta.mean()
ax.axhline(y=avg_delta, color='#9b59b6', linestyle='--', linewidth=2, alpha=0.7, 
           label=f'Average: {avg_delta:+.2f}%')
ax.legend(loc='upper right', fontsize=11)

plt.tight_layout()

# 保存图形
output_path = './semantic_comparison_bar_chart.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ 柱形图已保存到: {output_path}")

# 显示统计信息
avg_baseline = baseline_semantic.mean()
avg_purged = purged_semantic.mean()
print(f"\n=== 统计摘要 ===")
print(f"Baseline平均: {avg_baseline:.2f}%")
print(f"Purged平均: {avg_purged:.2f}%")
print(f"平均提升: {avg_purged - avg_baseline:+.2f}%")
print(f"\n提升最大的3个类别:")
top_3 = sorted(zip(classes, delta), key=lambda x: x[1], reverse=True)[:3]
for cls, d in top_3:
    print(f"  {cls}: {d:+.2f}%")
print(f"\n下降的类别:")
neg = [(cls, d) for cls, d in zip(classes, delta) if d < 0]
for cls, d in neg:
    print(f"  {cls}: {d:+.2f}%")

plt.show()
