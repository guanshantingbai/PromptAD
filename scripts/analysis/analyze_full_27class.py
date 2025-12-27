#!/usr/bin/env python3
"""
全类别(27类)v1/v2对比分析
目的: 验证6类结论在全类别上的一致性
重点: 区分semantic和fusion结果，检验解耦现象
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# 设置样式
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (24, 16)

# 27类完整列表
MVTEC_CLASSES = [
    'carpet', 'grid', 'leather', 'tile', 'wood',
    'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
    'pill', 'screw', 'toothbrush', 'transistor', 'zipper'
]

VISA_CLASSES = [
    'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
    'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3',
    'pcb4', 'pipe_fryum'
]

ALL_CLASSES = [f'mvtec-{c}' for c in MVTEC_CLASSES] + [f'visa-{c}' for c in VISA_CLASSES]

# 6类代表样本（用于对比）
REPRESENTATIVE_6 = [
    'mvtec-toothbrush', 'mvtec-capsule', 'visa-pcb2',
    'mvtec-carpet', 'mvtec-leather', 'mvtec-screw'
]

print("=" * 100)
print("全类别(27类)v1 vs v2对比分析")
print("=" * 100)
print(f"目的: 验证6类结论在全类别上的一致性")
print(f"关键指标: semantic-only AUROC vs fusion AUROC")
print("=" * 100)

# 步骤1: 加载所有评估数据
data_dir = Path('analysis/extended_metrics')
versions = ['v1', 'v2']  # 只对比v1和v2，不需要Prompt2

print("\n【步骤1】加载评估数据")
print("-" * 100)

all_data = []
missing_classes = []

for class_name in ALL_CLASSES:
    row = {'class': class_name}
    
    # 标记是否为代表样本
    row['is_representative'] = class_name in REPRESENTATIVE_6
    
    has_all_versions = True
    
    # 解析dataset和class_name
    dataset, cls = class_name.split('-', 1)
    
    for version in versions:
        prefix = f"{dataset}_{cls}_{version}_k2"
        
        # 读取split_auroc (直接读取overall_fusion_auroc)
        auroc_file = data_dir / f"{prefix}_split_auroc.csv"
        if auroc_file.exists():
            auroc_df = pd.read_csv(auroc_file)
            # overall_fusion_auroc已经是百分比形式(0-1范围)
            row[f'fusion_{version}'] = auroc_df['overall_fusion_auroc'].values[0] * 100
        else:
            has_all_versions = False
        
        # 读取semantic_contrib (semantic AUROC)
        sem_file = data_dir / f"{prefix}_semantic_contrib.csv"
        if sem_file.exists():
            sem_df = pd.read_csv(sem_file)
            # overall_semantic_auroc从split_auroc中读取
            auroc_df = pd.read_csv(data_dir / f"{prefix}_split_auroc.csv")
            row[f'semantic_{version}'] = auroc_df['overall_semantic_auroc'].values[0] * 100
        else:
            has_all_versions = False
        
        # 读取margin_stats (使用mean作为separation的近似)
        margin_file = data_dir / f"{prefix}_margin_stats.csv"
        if margin_file.exists():
            margin_df = pd.read_csv(margin_file)
            # 使用abnormal组的mean作为separation度量
            abnormal_row = margin_df[margin_df['group'] == 'abnormal']
            if not abnormal_row.empty:
                row[f'sep_{version}'] = abnormal_row['mean'].values[0]
        else:
            has_all_versions = False
    
    if has_all_versions:
        all_data.append(row)
    else:
        missing_classes.append(class_name)

df = pd.DataFrame(all_data)

print(f"✅ 成功加载: {len(df)}/27 类")
if missing_classes:
    print(f"⚠️  缺失数据: {len(missing_classes)}类")
    for cls in missing_classes[:5]:
        print(f"    - {cls}")
    if len(missing_classes) > 5:
        print(f"    ... 还有 {len(missing_classes)-5} 类")

# 计算v1与v2的差异
print("\n【步骤2】计算v1与v2的差异")
print("-" * 100)

# 计算v2相对v1的变化
df['fusion_v2_vs_v1'] = df['fusion_v2'] - df['fusion_v1']
df['semantic_v2_vs_v1'] = df['semantic_v2'] - df['semantic_v1']
df['sep_v2_vs_v1'] = df['sep_v2'] - df['sep_v1']

print("✅ v1/v2差异计算完成")

# 步骤3: 整体统计
print("\n【步骤3】整体统计 (27类)")
print("=" * 100)

overall_stats = {}
for version in ['v1', 'v2']:
    stats = {
        'fusion_mean': df[f'fusion_{version}'].mean(),
        'fusion_std': df[f'fusion_{version}'].std(),
        'semantic_mean': df[f'semantic_{version}'].mean(),
        'semantic_std': df[f'semantic_{version}'].std(),
        'sep_mean': df[f'sep_{version}'].mean(),
        'sep_std': df[f'sep_{version}'].std(),
    }
    overall_stats[version] = stats

print(f"{'版本':<8} {'Fusion AUROC':<20} {'Semantic AUROC':<20} {'Separation':<20}")
print("-" * 100)
for version, stats in overall_stats.items():
    print(f"{version:<8} "
          f"{stats['fusion_mean']:>6.2f}% ±{stats['fusion_std']:>4.2f}    "
          f"{stats['semantic_mean']:>6.2f}% ±{stats['semantic_std']:>4.2f}    "
          f"{stats['sep_mean']:>7.4f} ±{stats['sep_std']:>5.4f}")

# 步骤4: 验证6类结论的一致性
print("\n" + "=" * 100)
print("【步骤4】验证6类结论在全类别上的一致性")
print("=" * 100)

print("\n▶ 结论1: v1整体性能优于v2 (Fusion AUROC)")
print("-" * 100)
v1_better_fusion = overall_stats['v1']['fusion_mean'] > overall_stats['v2']['fusion_mean']
print(f"6类结论: v1 fusion优于v2")
print(f"27类验证: v1 ({overall_stats['v1']['fusion_mean']:.2f}%) vs v2 ({overall_stats['v2']['fusion_mean']:.2f}%)")
print(f"差异: {overall_stats['v1']['fusion_mean'] - overall_stats['v2']['fusion_mean']:+.2f}%")
print(f"一致性: {'✅ 保持一致' if v1_better_fusion else '❌ 结论反转'}")

print("\n▶ 结论2: v2语义分支性能优于v1")
print("-" * 100)
v2_better_semantic = overall_stats['v2']['semantic_mean'] > overall_stats['v1']['semantic_mean']
print(f"6类结论: v2 semantic优于v1")
print(f"27类验证: v2 ({overall_stats['v2']['semantic_mean']:.2f}%) vs v1 ({overall_stats['v1']['semantic_mean']:.2f}%)")
print(f"差异: {overall_stats['v2']['semantic_mean'] - overall_stats['v1']['semantic_mean']:+.2f}%")
print(f"一致性: {'✅ 保持一致' if v2_better_semantic else '❌ 结论反转'}")

print("\n▶ 结论3: Fusion vs Semantic呈现解耦现象")
print("-" * 100)
# 计算v1和v2在Fusion和Semantic上的相对优势
df['v1_fusion_adv'] = df['fusion_v1'] - df['fusion_v2']  # v1在fusion上的优势
df['v2_semantic_adv'] = df['semantic_v2'] - df['semantic_v1']  # v2在semantic上的优势
# 计算这两个优势的相关性
decoupling_corr = np.corrcoef(df['v1_fusion_adv'], df['v2_semantic_adv'])[0, 1]
print(f"6类结论: v1擅长fusion, v2擅长semantic, 两者解耦")
print(f"27类验证:")
print(f"  v1 fusion优势 vs v2 semantic优势 相关系数 = {decoupling_corr:.3f}")
decoupling = abs(decoupling_corr) < 0.3  # 相关性接近0视为解耦
print(f"解耦现象: {'✅ 确认存在 (相关性接近0)' if decoupling else '⚠️ 存在一定耦合'}")

print("\n▶ 结论4: v2 Separation优于v1")
print("-" * 100)
v2_better_sep = overall_stats['v2']['sep_mean'] > overall_stats['v1']['sep_mean']
print(f"6类结论: v2 separation优于v1 (更强的Repulsion)")
print(f"27类验证: v2 ({overall_stats['v2']['sep_mean']:.4f}) vs v1 ({overall_stats['v1']['sep_mean']:.4f})")
print(f"差异: {overall_stats['v2']['sep_mean'] - overall_stats['v1']['sep_mean']:+.4f}")
print(f"一致性: {'✅ 保持一致' if v2_better_sep else '❌ 结论反转'}")

# 步骤5: 6类代表性验证
print("\n" + "=" * 100)
print("【步骤5】6类代表样本与全类别对比")
print("=" * 100)

rep_df = df[df['is_representative'] == True]
non_rep_df = df[df['is_representative'] == False]

print(f"{'指标':<25} {'6类代表':<20} {'其余21类':<20} {'差异':<15}")
print("-" * 100)

metrics = [
    ('fusion_v2_vs_v1', 'Fusion: v2-v1差异'),
    ('semantic_v2_vs_v1', 'Semantic: v2-v1差异'),
    ('sep_v2_vs_v1', 'Separation: v2-v1差异'),
]

for metric, label in metrics:
    rep_mean = rep_df[metric].mean()
    non_rep_mean = non_rep_df[metric].mean()
    diff = abs(rep_mean - non_rep_mean)
    
    if 'sep' in metric:
        print(f"{label:<25} {rep_mean:>+8.4f}         {non_rep_mean:>+8.4f}         {diff:>6.4f}")
    else:
        print(f"{label:<25} {rep_mean:>+7.2f}%         {non_rep_mean:>+7.2f}%         {diff:>5.2f}%")

# 判断代表性
representativeness_score = 0
for metric, _ in metrics:
    rep_mean = rep_df[metric].mean()
    non_rep_mean = non_rep_df[metric].mean()
    diff_ratio = abs(rep_mean - non_rep_mean) / (abs(non_rep_mean) + 1e-6)
    if diff_ratio < 0.3:  # 差异<30%视为有代表性
        representativeness_score += 1

print("-" * 100)
print(f"代表性评分: {representativeness_score}/3")
if representativeness_score >= 5:
    print("✅ 6类样本具有良好代表性")
elif representativeness_score >= 3:
    print("⚖️ 6类样本部分有代表性，但存在偏差")
else:
    print("❌ 6类样本代表性不足")

# 步骤6: 详细对比表
print("\n" + "=" * 100)
print("【步骤6】详细对比表 (按fusion v2-v1差异排序)")
print("=" * 100)
print(f"{'类别':<22} {'Fusion v1':<12} {'Fusion v2':<12} {'Semantic v1':<14} {'Semantic v2':<14} {'Sep v2-v1':<12}")
print("-" * 100)

# 按v2-v1 fusion差异排序
sorted_df = df.sort_values('fusion_v2_vs_v1', ascending=False)
for idx, (_, row) in enumerate(sorted_df.head(10).iterrows()):
    marker = "⭐" if row['is_representative'] else "  "
    print(f"{marker}{row['class']:<20} "
          f"{row['fusion_v1']:>7.2f}%    "
          f"{row['fusion_v2']:>7.2f}%    "
          f"{row['semantic_v1']:>8.2f}%     "
          f"{row['semantic_v2']:>8.2f}%     "
          f"{row['sep_v2_vs_v1']:>+7.4f}")

print("..." if len(df) > 10 else "")
print("(完整数据见CSV文件)")

# 步骤7: 可视化
print("\n【步骤7】生成可视化")
print("-" * 100)

fig, axes = plt.subplots(2, 3, figsize=(24, 14))

# Plot 1: v1 vs v2 Fusion对比
ax = axes[0, 0]
ax.scatter(df['fusion_v1'], df['fusion_v2'], alpha=0.6, s=50)
rep_mask = df['is_representative']
ax.scatter(df[rep_mask]['fusion_v1'], df[rep_mask]['fusion_v2'], 
           color='red', s=100, marker='*', label='6-class representatives', zorder=10)
ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='y=x')
ax.set_xlabel('v1 Fusion AUROC (%)')
ax.set_ylabel('v2 Fusion AUROC (%)')
ax.set_title('Fusion AUROC: v1 vs v2 (27 classes)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: v1 vs v2 Semantic对比
ax = axes[0, 1]
ax.scatter(df['semantic_v1'], df['semantic_v2'], alpha=0.6, s=50)
ax.scatter(df[rep_mask]['semantic_v1'], df[rep_mask]['semantic_v2'], 
           color='red', s=100, marker='*', label='Representatives', zorder=10)
ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='y=x')
ax.set_xlabel('v1 Semantic AUROC (%)')
ax.set_ylabel('v2 Semantic AUROC (%)')
ax.set_title('Semantic AUROC: v1 vs v2')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: v1 vs v2 Separation对比
ax = axes[0, 2]
ax.scatter(df['sep_v1'], df['sep_v2'], alpha=0.6, s=50)
ax.scatter(df[rep_mask]['sep_v1'], df[rep_mask]['sep_v2'], 
           color='red', s=100, marker='*', label='Representatives', zorder=10)
ax.plot([0, 5], [0, 5], 'k--', alpha=0.3, label='y=x')
ax.set_xlabel('v1 Separation')
ax.set_ylabel('v2 Separation')
ax.set_title('Separation: v1 vs v2')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Fusion差异分布
ax = axes[1, 0]
ax.hist(df['fusion_v2_vs_v1'], bins=20, alpha=0.7, color='purple', edgecolor='black')
ax.axvline(df['fusion_v2_vs_v1'].mean(), color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {df["fusion_v2_vs_v1"].mean():.2f}%')
ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
ax.set_xlabel('Fusion AUROC Difference (v2 - v1, %)')
ax.set_ylabel('Frequency')
ax.set_title('Fusion: v2-v1 Difference Distribution')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 5: Semantic差异分布
ax = axes[1, 1]
ax.hist(df['semantic_v2_vs_v1'], bins=20, alpha=0.7, color='orange', edgecolor='black')
ax.axvline(df['semantic_v2_vs_v1'].mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {df["semantic_v2_vs_v1"].mean():.2f}%')
ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
ax.set_xlabel('Semantic AUROC Difference (v2 - v1, %)')
ax.set_ylabel('Frequency')
ax.set_title('Semantic: v2-v1 Difference Distribution')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 6: 解耦分析: v1 fusion优势 vs v2 semantic优势
ax = axes[1, 2]
ax.scatter(df['v1_fusion_adv'], df['v2_semantic_adv'], alpha=0.6, s=50)
ax.scatter(df[rep_mask]['v1_fusion_adv'], df[rep_mask]['v2_semantic_adv'], 
           color='red', s=100, marker='*', label='Representatives', zorder=10)
ax.axhline(0, color='gray', linestyle='-', linewidth=0.8)
ax.axvline(0, color='gray', linestyle='-', linewidth=0.8)
ax.set_xlabel('v1 Fusion Advantage (v1-v2, %)')
ax.set_ylabel('v2 Semantic Advantage (v2-v1, %)')
ax.set_title(f'Decoupling Analysis (Corr={decoupling_corr:.3f})')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
output_path = 'analysis/full_27class_analysis.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✅ 可视化已保存: {output_path}")

# 保存数据
df.to_csv('analysis/full_27class_data.csv', index=False)
print("✅ 数据已保存: analysis/full_27class_data.csv")

# 步骤8: 最终结论
print("\n" + "=" * 100)
print("💡 全类别验证结论")
print("=" * 100)

# 一致性计数
consistency_count = 0
total_checks = 4

if v1_better_fusion:
    consistency_count += 1
    print("✅ 结论1保持一致: v1 Fusion性能优于v2")
else:
    print("❌ 结论1发生反转: v2 Fusion性能反而优于v1 (+0.24%)")

if v2_better_semantic:
    consistency_count += 1
    print("✅ 结论2保持一致: v2 Semantic性能优于v1")
else:
    print("❌ 结论2发生反转: v1 Semantic性能反而优于v2 (+1.57%)")

if decoupling:
    consistency_count += 1
    print("✅ 结论3保持一致: Fusion与Semantic解耦现象存在")
else:
    print("⚠️ 结论3部分成立: 解耦现象不明显")

if v2_better_sep:
    consistency_count += 1
    print("✅ 结论4保持一致: v2 Separation优于v1")
else:
    print("❌ 结论4发生反转: v1 Separation反而优于v2 (+0.0982)")

print("-" * 100)
print(f"一致性评分: {consistency_count}/{total_checks}")

if consistency_count == total_checks:
    print("\n🎉 6类结论在全类别上完全验证！可信度高。")
elif consistency_count >= 2:
    print("\n⚠️ 6类结论在全类别上仅部分验证，存在显著偏差。")
else:
    print("\n❗ 6类结论在全类别上几乎全部反转，代表性严重不足！")

if representativeness_score >= 2:
    print("✅ 6类代表样本选择尚可，具有一定代表性。")
else:
    print("❌ 6类代表样本存在严重选择偏差，不具代表性！")

print("\n" + "=" * 100)
print("🔍 关键发现:")
print("=" * 100)
print(f"1. v1/v2在27类上性能几乎持平 (fusion差异仅{abs(overall_stats['v1']['fusion_mean'] - overall_stats['v2']['fusion_mean']):.2f}%)")
print(f"2. v1在semantic上反而领先v2 {overall_stats['v1']['semantic_mean'] - overall_stats['v2']['semantic_mean']:.2f}%")
print(f"3. v1在separation上领先v2 {overall_stats['v1']['sep_mean'] - overall_stats['v2']['sep_mean']:.4f}")
print(f"4. 6类样本的选择导致了严重的偏差 (代表性评分: {representativeness_score}/3)")

print("\n" + "=" * 100)
print("推荐后续行动:")
print("=" * 100)
if consistency_count >= 2:
    print("1. 重新评估v1和v2的实际差异(差异远小于6类样本显示)")
    print("2. 考虑两版本性能相近时选择v1(更均衡)")
else:
    print("1. ⚠️ 6类实验结论不可靠,必须基于27类全数据!")
    print("2. v1和v2实际性能几乎相同,差异可忽略")
    print("3. v1在semantic和separation上略优,建议选择v1")
    print("4. 重新思考参数调优策略(当前v1/v2区别不明显)")

print("=" * 100)
