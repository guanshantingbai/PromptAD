"""
可视化批量测试结果
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
data = {
    'Class': ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 
              'leather', 'metal_nut', 'pill', 'screw', 'tile', 
              'toothbrush', 'transistor', 'wood', 'zipper'],
    'Baseline': [100.00, 96.38, 79.94, 100.00, 99.08, 99.93,
                 100.00, 100.00, 95.61, 58.66, 100.00,
                 98.89, 89.79, 99.82, 96.40],
    'Prompt1_Memory': [99.52, 93.83, 86.78, 100.00, 98.33, 99.88,
                       99.97, 99.22, 92.05, 68.96, 99.82,
                       95.56, 90.58, 98.77, 92.28]
}

df = pd.DataFrame(data)
df['Difference'] = df['Prompt1_Memory'] - df['Baseline']
df = df.sort_values('Difference', ascending=False)

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 图1: 性能对比条形图
ax1 = axes[0, 0]
x = np.arange(len(df))
width = 0.35
bars1 = ax1.bar(x - width/2, df['Baseline'], width, label='Baseline', alpha=0.8)
bars2 = ax1.bar(x + width/2, df['Prompt1_Memory'], width, label='Prompt1_Memory', alpha=0.8)
ax1.set_xlabel('Class', fontsize=12)
ax1.set_ylabel('AUROC (%)', fontsize=12)
ax1.set_title('MVTec CLS (k=2): Baseline vs Prompt1_Memory', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(df['Class'], rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.axhline(y=94.30, color='blue', linestyle='--', linewidth=1, label='Baseline Avg')
ax1.axhline(y=94.37, color='orange', linestyle='--', linewidth=1, label='Prompt1 Avg')
ax1.axhline(y=95.7, color='red', linestyle='--', linewidth=2, label='Paper Claimed')

# 图2: 差异瀑布图
ax2 = axes[0, 1]
colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in df['Difference']]
bars = ax2.barh(df['Class'], df['Difference'], color=colors, alpha=0.7)
ax2.set_xlabel('Difference (%)', fontsize=12)
ax2.set_ylabel('Class', fontsize=12)
ax2.set_title('Performance Change (Prompt1_Memory - Baseline)', fontsize=14, fontweight='bold')
ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax2.grid(axis='x', alpha=0.3)
# 添加数值标签
for i, (cls, val) in enumerate(zip(df['Class'], df['Difference'])):
    ax2.text(val + 0.3 if val > 0 else val - 0.3, i, f'{val:+.2f}', 
             va='center', ha='left' if val > 0 else 'right', fontsize=9)

# 图3: 性能分布散点图
ax3 = axes[1, 0]
improved = df[df['Difference'] > 0]
degraded = df[df['Difference'] < 0]
unchanged = df[df['Difference'] == 0]

ax3.scatter(improved['Baseline'], improved['Prompt1_Memory'], 
           s=100, alpha=0.7, c='green', label=f'Improved ({len(improved)})')
ax3.scatter(degraded['Baseline'], degraded['Prompt1_Memory'], 
           s=100, alpha=0.7, c='red', label=f'Degraded ({len(degraded)})')
ax3.scatter(unchanged['Baseline'], unchanged['Prompt1_Memory'], 
           s=100, alpha=0.7, c='gray', label=f'Unchanged ({len(unchanged)})')

# 添加对角线(y=x)
ax3.plot([50, 100], [50, 100], 'k--', alpha=0.3, linewidth=1)
ax3.set_xlabel('Baseline AUROC (%)', fontsize=12)
ax3.set_ylabel('Prompt1_Memory AUROC (%)', fontsize=12)
ax3.set_title('Performance Distribution', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)
ax3.set_xlim(55, 102)
ax3.set_ylim(55, 102)

# 添加类别标签
for _, row in df.iterrows():
    ax3.annotate(row['Class'], (row['Baseline'], row['Prompt1_Memory']),
                fontsize=7, alpha=0.6, xytext=(2, 2), textcoords='offset points')

# 图4: 三层位置关系
ax4 = axes[1, 1]
baseline_claimed = 95.7
baseline_actual = 94.30
prompt1_actual = 94.37

positions = [baseline_actual, prompt1_actual, baseline_claimed]
labels = ['Baseline\n(Reproduced)\n94.30%', 
          'Prompt1_Memory\n(This Work)\n94.37%',
          'Baseline\n(Paper Claimed)\n95.7%']
colors_pos = ['#3498db', '#e74c3c', '#2ecc71']

bars = ax4.barh(labels, positions, color=colors_pos, alpha=0.7, edgecolor='black', linewidth=2)
ax4.set_xlabel('Average AUROC (%)', fontsize=12)
ax4.set_title('The "Embarrassing" Position Analysis', fontsize=14, fontweight='bold')
ax4.set_xlim(93, 97)
ax4.grid(axis='x', alpha=0.3)

# 添加范围标注
ax4.axvspan(94.2, 97.2, alpha=0.1, color='green', label='Paper Claimed Range')
ax4.axvline(x=baseline_claimed, color='green', linestyle='--', linewidth=2, label='Paper Target')

# 添加数值标签
for bar, val in zip(bars, positions):
    ax4.text(val + 0.1, bar.get_y() + bar.get_height()/2, 
            f'{val:.2f}%', va='center', fontsize=10, fontweight='bold')

# 添加差异箭头
ax4.annotate('', xy=(prompt1_actual, 1), xytext=(baseline_actual, 1),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
ax4.text((baseline_actual + prompt1_actual)/2, 1.3, '+0.07%', 
        ha='center', fontsize=9, color='black', fontweight='bold')

ax4.annotate('', xy=(baseline_claimed, 1), xytext=(prompt1_actual, 1),
            arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
ax4.text((prompt1_actual + baseline_claimed)/2, 0.7, '-1.33%', 
        ha='center', fontsize=9, color='red', fontweight='bold')

plt.tight_layout()
plt.savefig('test_logs/batch_testing/results_visualization.png', dpi=300, bbox_inches='tight')
print("可视化图表已保存到: test_logs/batch_testing/results_visualization.png")

# 创建第二个图：假设保留率分析
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

# 从之前的分析中我们知道的语义改进数据（部分）
semantic_improvements = {
    'screw': 13.15,  # 已知
    'capsule': 9.0,  # 估计
    'toothbrush': 19.86,  # 从之前分析得知
    'hazelnut': 11.03,  # 从之前分析得知
}

fusion_improvements = {
    'screw': 10.30,
    'capsule': 6.84,
    'toothbrush': -3.33,  # 实际退化
    'hazelnut': -0.05,   # 几乎持平
}

preservation_rates = {}
for cls in semantic_improvements:
    sem_imp = semantic_improvements[cls]
    fus_imp = fusion_improvements[cls]
    if sem_imp != 0:
        preservation_rates[cls] = (fus_imp / sem_imp) * 100
    else:
        preservation_rates[cls] = 0

# 图5: 保留率分析
ax5 = axes2[0]
classes = list(preservation_rates.keys())
rates = list(preservation_rates.values())
colors_rates = ['green' if r > 50 else 'orange' if r > 0 else 'red' for r in rates]

bars = ax5.bar(classes, rates, color=colors_rates, alpha=0.7, edgecolor='black')
ax5.set_ylabel('Preservation Rate (%)', fontsize=12)
ax5.set_xlabel('Class', fontsize=12)
ax5.set_title('Semantic Improvement Preservation in Fusion', fontsize=14, fontweight='bold')
ax5.axhline(y=75, color='blue', linestyle='--', linewidth=2, label='Expected 75%')
ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax5.legend()
ax5.grid(axis='y', alpha=0.3)

# 添加数值标签
for bar, val in zip(bars, rates):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=10)

# 图6: 语义改进 vs 融合改进
ax6 = axes2[1]
sem_vals = [semantic_improvements[c] for c in classes]
fus_vals = [fusion_improvements[c] for c in classes]

ax6.scatter(sem_vals, fus_vals, s=150, alpha=0.7, c=colors_rates, edgecolor='black')
ax6.plot([min(sem_vals), max(sem_vals)], 
         [min(sem_vals), max(sem_vals)], 
         'k--', alpha=0.3, label='100% Preservation')
ax6.plot([min(sem_vals), max(sem_vals)], 
         [0.75*min(sem_vals), 0.75*max(sem_vals)], 
         'b--', alpha=0.5, label='75% Preservation')

ax6.set_xlabel('Semantic Improvement (%)', fontsize=12)
ax6.set_ylabel('Fusion Improvement (%)', fontsize=12)
ax6.set_title('Semantic vs Fusion Improvement', fontsize=14, fontweight='bold')
ax6.legend()
ax6.grid(alpha=0.3)

# 添加类别标签
for i, cls in enumerate(classes):
    ax6.annotate(cls, (sem_vals[i], fus_vals[i]),
                fontsize=9, xytext=(5, 5), textcoords='offset points')

plt.tight_layout()
plt.savefig('test_logs/batch_testing/preservation_analysis.png', dpi=300, bbox_inches='tight')
print("保留率分析图表已保存到: test_logs/batch_testing/preservation_analysis.png")

print("\n" + "="*80)
print("关键统计数据:")
print("="*80)
avg_preservation = np.mean(list(preservation_rates.values()))
print(f"平均保留率: {avg_preservation:.1f}%")
print(f"Screw保留率: {preservation_rates['screw']:.1f}% (最可靠，baseline最低)")
print(f"Capsule保留率: {preservation_rates['capsule']:.1f}%")
print(f"\n注意: Toothbrush和Hazelnut的负保留率说明融合反而降低了性能，")
print("      这可能是因为baseline已经很高(98-99%)，零样本语义分支反而成为瓶颈。")
print("="*80)
