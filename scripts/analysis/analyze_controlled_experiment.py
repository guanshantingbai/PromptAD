#!/usr/bin/env python3
"""
受控实验分析：对比4个版本
- Baseline (n_pro=3)
- Prompt2 (n_pro=1, 原始EMA)
- Ours_v1 (EMA修正 + Repulsion + Margin)
- Ours_v2 (EMA修正 + Repulsion, 无Margin) ← 受控实验

验证假设：
1. 移除Margin Loss后，Stable类Separation不再下降
2. 增强Repulsion (0.1) 后，Collapse减少更明显
3. 语义AUROC提升更稳定
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (18, 14)

# 训练日志AUROC（会在训练完成后更新）
training_auroc = {
    'toothbrush': {'prompt2': 86.94, 'v1': 88.61, 'v2': None},
    'capsule': {'prompt2': 64.46, 'v1': 67.41, 'v2': None},
    'carpet': {'prompt2': 100.0, 'v1': 100.0, 'v2': None},
    'leather': {'prompt2': 100.0, 'v1': 99.93, 'v2': None},
    'screw': {'prompt2': 73.36, 'v1': 75.49, 'v2': None},
    'pcb2': {'prompt2': 66.71, 'v1': 66.61, 'v2': None},
}

def load_extended_metrics(class_key, version):
    """加载Extended Metrics"""
    dataset, cls = class_key.split('-')
    
    # Split AUROC
    split_file = f'analysis/controlled_comparison/{dataset}_{cls}_{version}_split_auroc.csv'
    margin_file = f'analysis/controlled_comparison/{dataset}_{cls}_{version}_margin_stats.csv'
    
    try:
        df_split = pd.read_csv(split_file)
        df_margin = pd.read_csv(margin_file)
        
        normal_row = df_margin[df_margin['group'] == 'normal']
        abnormal_row = df_margin[df_margin['group'] == 'abnormal']
        
        return {
            'semantic_auroc': df_split['overall_semantic_auroc'].values[0] * 100,
            'fusion_auroc': df_split['overall_fusion_auroc'].values[0] * 100,
            'separation': normal_row['mean'].values[0] - abnormal_row['mean'].values[0],
            'normal_margin': normal_row['mean'].values[0],
        }
    except:
        return None


# 分析类别
classes = ['mvtec-toothbrush', 'mvtec-capsule', 'visa-pcb2', 
           'mvtec-carpet', 'mvtec-leather', 'mvtec-screw']

print("="*100)
print("受控实验分析：4版本对比")
print("="*100)
print()

# 提取训练AUROC（v2需要手动更新）
print("【步骤1】提取训练日志AUROC（v2版本）")
print("-"*100)
print("请从训练日志中提取v2的Image-AUROC:")

for cls in classes:
    cls_name = cls.split('-')[1]
    dataset = cls.split('-')[0]
    log_file = f'logs/controlled_exp/{dataset}_{cls_name}_k2.log'
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in reversed(lines):
                if 'Image-AUROC:' in line:
                    auroc = float(line.split('Image-AUROC:')[1].strip())
                    training_auroc[cls_name]['v2'] = auroc
                    print(f"  {cls:<25} v2 AUROC: {auroc:.2f}")
                    break
    except:
        print(f"  {cls:<25} v2 AUROC: [未找到日志]")

print()

# 加载Extended Metrics
print("【步骤2】加载Extended Metrics")
print("-"*100)

rows = []
for cls in classes:
    baseline = load_extended_metrics(cls, 'baseline')
    prompt2 = load_extended_metrics(cls, 'prompt2')
    v1 = load_extended_metrics(cls, 'ours_v1')
    v2 = load_extended_metrics(cls, 'ours_v2')
    
    if not all([baseline, prompt2, v1, v2]):
        print(f"⚠️  {cls}: 数据不完整，跳过")
        continue
    
    cls_name = cls.split('-')[1]
    
    # 性能组
    if cls in ['mvtec-toothbrush', 'mvtec-capsule', 'visa-pcb2']:
        group = 'Severe'
    elif cls in ['mvtec-carpet', 'mvtec-leather']:
        group = 'Stable'
    else:
        group = 'Improved'
    
    rows.append({
        'class': cls,
        'group': group,
        # 训练AUROC
        'train_p2': training_auroc[cls_name]['prompt2'],
        'train_v1': training_auroc[cls_name]['v1'],
        'train_v2': training_auroc[cls_name]['v2'],
        # 语义AUROC
        'sem_p2': prompt2['semantic_auroc'],
        'sem_v1': v1['semantic_auroc'],
        'sem_v2': v2['semantic_auroc'],
        # Separation
        'sep_p2': prompt2['separation'],
        'sep_v1': v1['separation'],
        'sep_v2': v2['separation'],
        # Normal Margin
        'nm_p2': prompt2['normal_margin'],
        'nm_v1': v1['normal_margin'],
        'nm_v2': v2['normal_margin'],
    })

df = pd.DataFrame(rows)

if len(df) == 0:
    print("❌ 无有效数据，请先运行评估")
    exit(1)

print(f"✅ 加载了 {len(df)} 个类别的数据")
print()

# 计算变化量
df['train_delta_v1'] = df['train_v1'] - df['train_p2']
df['train_delta_v2'] = df['train_v2'] - df['train_p2']
df['sem_delta_v1'] = df['sem_v1'] - df['sem_p2']
df['sem_delta_v2'] = df['sem_v2'] - df['sem_p2']
df['sep_delta_v1'] = df['sep_v1'] - df['sep_p2']
df['sep_delta_v2'] = df['sep_v2'] - df['sep_p2']

# 打印详细对比
print("【步骤3】详细对比表")
print("="*100)
print(f"{'类别':<20} {'组':<10} {'训练Δv1':<12} {'训练Δv2':<12} {'SepΔv1':<12} {'SepΔv2':<12}")
print("-"*100)

for idx, row in df.iterrows():
    print(f"{row['class']:<20} {row['group']:<10} "
          f"{row['train_delta_v1']:>+11.2f} {row['train_delta_v2']:>+11.2f} "
          f"{row['sep_delta_v1']:>+11.4f} {row['sep_delta_v2']:>+11.4f}")

print("="*100)
print()

# 分组统计
print("【步骤4】分组对比")
print("="*100)

for group in ['Severe', 'Stable', 'Improved']:
    group_df = df[df['group'] == group]
    if len(group_df) == 0:
        continue
    
    print(f"\n【{group}】(n={len(group_df)})")
    print(f"  训练AUROC变化:")
    print(f"    v1 (全改动):   {group_df['train_delta_v1'].mean():>+8.2f}%")
    print(f"    v2 (EMA+Rep):  {group_df['train_delta_v2'].mean():>+8.2f}%")
    print(f"  语义AUROC变化:")
    print(f"    v1:            {group_df['sem_delta_v1'].mean():>+8.2f}%")
    print(f"    v2:            {group_df['sem_delta_v2'].mean():>+8.2f}%")
    print(f"  Separation变化:")
    print(f"    v1 (有Margin): {group_df['sep_delta_v1'].mean():>+8.4f}")
    print(f"    v2 (无Margin): {group_df['sep_delta_v2'].mean():>+8.4f}")

print("\n" + "="*100)
print()

# 假设验证
print("【步骤5】假设验证")
print("="*100)
print()

# 假设1：移除Margin后，Stable组Separation不再下降
stable_sep_v1 = df[df['group'] == 'Stable']['sep_delta_v1'].mean()
stable_sep_v2 = df[df['group'] == 'Stable']['sep_delta_v2'].mean()

print(f"假设1：移除Margin Loss后，Stable类Separation不再严重下降")
print(f"  Stable组Separation变化:")
print(f"    v1 (有Margin): {stable_sep_v1:+.4f}")
print(f"    v2 (无Margin): {stable_sep_v2:+.4f}")

if stable_sep_v2 > stable_sep_v1 and stable_sep_v2 > -0.02:
    print(f"  ✅ 假设成立：v2的Separation下降显著减轻")
elif stable_sep_v2 > stable_sep_v1:
    print(f"  ⚖️  部分成立：v2略好于v1，但仍有下降")
else:
    print(f"  ❌ 假设不成立：v2的Separation仍然下降")

print()

# 假设2：增强Repulsion后，整体性能更稳定
overall_train_v1 = df['train_delta_v1'].mean()
overall_train_v2 = df['train_delta_v2'].mean()

print(f"假设2：增强Repulsion后，整体训练AUROC提升更明显")
print(f"  整体训练AUROC变化:")
print(f"    v1 (Rep=0.05): {overall_train_v1:+.2f}%")
print(f"    v2 (Rep=0.10): {overall_train_v2:+.2f}%")

if overall_train_v2 > overall_train_v1 + 0.5:
    print(f"  ✅ 假设成立：v2显著优于v1")
elif overall_train_v2 > overall_train_v1:
    print(f"  ⚖️  部分成立：v2略优于v1")
else:
    print(f"  ❌ 假设不成立：v2未优于v1")

print()

# 假设3：v2相对v1，改善/退化类别数量
improve_v1 = (df['train_delta_v1'] > 0).sum()
improve_v2 = (df['train_delta_v2'] > 0).sum()

print(f"假设3：v2的改善类别比例不低于v1")
print(f"  改善类别数:")
print(f"    v1: {improve_v1}/{len(df)}")
print(f"    v2: {improve_v2}/{len(df)}")

if improve_v2 >= improve_v1:
    print(f"  ✅ 假设成立")
else:
    print(f"  ❌ 假设不成立")

print()
print("="*100)
print()

# 可视化
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. 训练AUROC变化对比
ax = axes[0, 0]
x = np.arange(len(df))
width = 0.35
ax.bar(x - width/2, df['train_delta_v1'], width, label='v1 (全改动)', alpha=0.8)
ax.bar(x + width/2, df['train_delta_v2'], width, label='v2 (EMA+Rep)', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([c.split('-')[1] for c in df['class']], rotation=45, ha='right')
ax.set_ylabel('Train AUROC Change (%)')
ax.set_title('(A) 训练AUROC变化对比')
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 2. 语义AUROC变化对比
ax = axes[0, 1]
ax.bar(x - width/2, df['sem_delta_v1'], width, label='v1', alpha=0.8)
ax.bar(x + width/2, df['sem_delta_v2'], width, label='v2', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels([c.split('-')[1] for c in df['class']], rotation=45, ha='right')
ax.set_ylabel('Semantic AUROC Change (%)')
ax.set_title('(B) 语义AUROC变化对比')
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 3. Separation变化对比（关键）
ax = axes[0, 2]
ax.bar(x - width/2, df['sep_delta_v1'], width, label='v1 (有Margin)', alpha=0.8, color='red')
ax.bar(x + width/2, df['sep_delta_v2'], width, label='v2 (无Margin)', alpha=0.8, color='green')
ax.set_xticks(x)
ax.set_xticklabels([c.split('-')[1] for c in df['class']], rotation=45, ha='right')
ax.set_ylabel('Separation Change')
ax.set_title('(C) Separation变化对比 [关键验证]')
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 4. 按组汇总 - 训练AUROC
ax = axes[1, 0]
group_train_v1 = df.groupby('group')['train_delta_v1'].mean()
group_train_v2 = df.groupby('group')['train_delta_v2'].mean()
x_group = np.arange(len(group_train_v1))
ax.bar(x_group - width/2, group_train_v1, width, label='v1', alpha=0.8)
ax.bar(x_group + width/2, group_train_v2, width, label='v2', alpha=0.8)
ax.set_xticks(x_group)
ax.set_xticklabels(group_train_v1.index)
ax.set_ylabel('Avg Train AUROC Change (%)')
ax.set_title('(D) 按组汇总 - 训练AUROC')
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 5. 按组汇总 - Separation
ax = axes[1, 1]
group_sep_v1 = df.groupby('group')['sep_delta_v1'].mean()
group_sep_v2 = df.groupby('group')['sep_delta_v2'].mean()
ax.bar(x_group - width/2, group_sep_v1, width, label='v1 (有Margin)', alpha=0.8, color='red')
ax.bar(x_group + width/2, group_sep_v2, width, label='v2 (无Margin)', alpha=0.8, color='green')
ax.set_xticks(x_group)
ax.set_xticklabels(group_sep_v1.index)
ax.set_ylabel('Avg Separation Change')
ax.set_title('(E) 按组汇总 - Separation [关键]')
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 6. Stable组专项对比
ax = axes[1, 2]
stable_data = df[df['group'] == 'Stable']
metrics = ['训练AUROC', '语义AUROC', 'Separation×10']
v1_vals = [stable_data['train_delta_v1'].mean(), 
           stable_data['sem_delta_v1'].mean(),
           stable_data['sep_delta_v1'].mean() * 10]
v2_vals = [stable_data['train_delta_v2'].mean(),
           stable_data['sem_delta_v2'].mean(),
           stable_data['sep_delta_v2'].mean() * 10]
x_metric = np.arange(len(metrics))
ax.bar(x_metric - width/2, v1_vals, width, label='v1', alpha=0.8)
ax.bar(x_metric + width/2, v2_vals, width, label='v2', alpha=0.8)
ax.set_xticks(x_metric)
ax.set_xticklabels(metrics)
ax.set_ylabel('Change')
ax.set_title('(F) Stable组专项对比')
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('analysis/controlled_comparison/controlled_experiment_analysis.png', dpi=300, bbox_inches='tight')
print(f"✅ 可视化已保存: analysis/controlled_comparison/controlled_experiment_analysis.png")
print()

# 保存数据
df.to_csv('analysis/controlled_comparison/controlled_experiment_data.csv', index=False, float_format='%.4f')
print(f"✅ 数据已保存: analysis/controlled_comparison/controlled_experiment_data.csv")
print()

# 最终结论
print("="*100)
print("💡 受控实验结论")
print("="*100)
print()

if stable_sep_v2 > -0.02 and overall_train_v2 > overall_train_v1:
    print("✅ 受控实验成功！")
    print("   - 移除Margin Loss后，Stable类Separation下降显著减轻")
    print("   - 增强Repulsion后，整体性能提升更明显")
    print("   → 建议：采用v2配置（EMA+Rep），扩展到27类验证")
elif stable_sep_v2 > stable_sep_v1:
    print("⚖️  部分成功")
    print("   - Separation下降有所改善，但可能需要进一步调整")
    print("   → 建议：考虑单独测试EMA-only配置")
else:
    print("❌ 受控实验未达预期")
    print("   - Separation仍然下降或性能未改善")
    print("   → 建议：重新审视EMA和Repulsion的实现")

print()
print("="*100)
