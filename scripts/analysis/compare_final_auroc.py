#!/usr/bin/env python3
"""
最终AUROC对比：同时对比两个来源的测试集AUROC
1. results.csv中的Image AUROC (i_roc) - 标准评估
2. split_auroc.csv中的overall_fusion_auroc - 扩展评估
"""
import pandas as pd
import numpy as np

print("=" * 100)
print("测试集AUROC完整对比: v1 vs v2")
print("=" * 100)

# === 读取标准评估结果 (results.csv) ===
v1_mvtec_std = pd.read_csv('result/v1_ema_rep05_margin01/mvtec/k_2/csv/Seed_111-results.csv')
v1_visa_std = pd.read_csv('result/v1_ema_rep05_margin01/visa/k_2/csv/Seed_111-results.csv')
v2_mvtec_std = pd.read_csv('result/v2_ema_rep10_nomargin/mvtec/k_2/csv/Seed_111-results.csv')
v2_visa_std = pd.read_csv('result/v2_ema_rep10_nomargin/visa/k_2/csv/Seed_111-results.csv')

std_v1 = pd.concat([v1_mvtec_std, v1_visa_std], ignore_index=True)
std_v2 = pd.concat([v2_mvtec_std, v2_visa_std], ignore_index=True)
std_v1.columns = ['class', 'i_roc_v1', 'p_roc_v1']
std_v2.columns = ['class', 'i_roc_v2', 'p_roc_v2']

# === 读取扩展评估结果 (split_auroc.csv) ===
from pathlib import Path
import glob

ext_data = []
for v in ['v1', 'v2']:
    files = glob.glob(f'analysis/extended_metrics/*_{v}_k2_split_auroc.csv')
    for f in files:
        df = pd.read_csv(f)
        fname = Path(f).stem
        parts = fname.split('_')
        dataset = parts[0]
        cls = '_'.join(parts[1:-3])  # 处理multi-word类名
        
        ext_data.append({
            'class': f'{dataset}-{cls}',
            'version': v,
            'fusion_auroc': df['overall_fusion_auroc'].values[0] * 100,
            'semantic_auroc': df['overall_semantic_auroc'].values[0] * 100,
        })

ext_df = pd.DataFrame(ext_data)
ext_v1 = ext_df[ext_df['version'] == 'v1'][['class', 'fusion_auroc', 'semantic_auroc']]
ext_v2 = ext_df[ext_df['version'] == 'v2'][['class', 'fusion_auroc', 'semantic_auroc']]
ext_v1.columns = ['class', 'fusion_v1', 'semantic_v1']
ext_v2.columns = ['class', 'fusion_v2', 'semantic_v2']

# === 合并所有数据 ===
df = std_v1.merge(std_v2, on='class')
df = df.merge(ext_v1, on='class')
df = df.merge(ext_v2, on='class')

df['i_roc_diff'] = df['i_roc_v2'] - df['i_roc_v1']
df['fusion_diff'] = df['fusion_v2'] - df['fusion_v1']
df['semantic_diff'] = df['semantic_v2'] - df['semantic_v1']

print("\n【指标对比说明】")
print("=" * 100)
print("1. Image AUROC (i_roc): 标准评估的Image-level异常检测性能")
print("2. Fusion AUROC: 扩展评估的Memory Bank + Semantic fusion分数在测试集上的AUROC")
print("3. Semantic AUROC: 扩展评估的纯Semantic分数在测试集上的AUROC")
print("=" * 100)

print("\n【整体统计】")
print("=" * 100)
print(f"{'指标':<20} {'v1均值':<12} {'v2均值':<12} {'差异(v2-v1)':<15}")
print("-" * 100)
print(f"{'Image AUROC':<20} {df['i_roc_v1'].mean():>7.2f}%     {df['i_roc_v2'].mean():>7.2f}%     {df['i_roc_diff'].mean():>+7.2f}%")
print(f"{'Fusion AUROC':<20} {df['fusion_v1'].mean():>7.2f}%     {df['fusion_v2'].mean():>7.2f}%     {df['fusion_diff'].mean():>+7.2f}%")
print(f"{'Semantic AUROC':<20} {df['semantic_v1'].mean():>7.2f}%     {df['semantic_v2'].mean():>7.2f}%     {df['semantic_diff'].mean():>+7.2f}%")

print("\n【关键发现】")
print("=" * 100)
i_roc_diff = df['i_roc_diff'].mean()
fusion_diff = df['fusion_diff'].mean()
semantic_diff = df['semantic_diff'].mean()

if abs(i_roc_diff) < 0.5:
    print(f"✅ Image AUROC (标准评估): v1和v2几乎相同 (差异{i_roc_diff:+.2f}%)")
elif i_roc_diff > 0:
    print(f"�� Image AUROC (标准评估): v2优于v1 (+{i_roc_diff:.2f}%)")
else:
    print(f"📉 Image AUROC (标准评估): v1优于v2 ({i_roc_diff:+.2f}%)")

if abs(fusion_diff) < 0.5:
    print(f"✅ Fusion AUROC (扩展评估): v1和v2几乎相同 (差异{fusion_diff:+.2f}%)")
elif fusion_diff > 0:
    print(f"📈 Fusion AUROC (扩展评估): v2优于v1 (+{fusion_diff:.2f}%)")
else:
    print(f"📉 Fusion AUROC (扩展评估): v1优于v2 ({fusion_diff:+.2f}%)")

if abs(semantic_diff) < 0.5:
    print(f"✅ Semantic AUROC (扩展评估): v1和v2几乎相同 (差异{semantic_diff:+.2f}%)")
elif semantic_diff > 0:
    print(f"📈 Semantic AUROC (扩展评估): v2优于v1 (+{semantic_diff:.2f}%)")
else:
    print(f"📉 Semantic AUROC (扩展评估): v1优于v2 ({semantic_diff:+.2f}%)")

print("\n【详细对比 - Top 10类按Image AUROC差异排序】")
print("=" * 100)
print(f"{'类别':<20} {'v1 I-AUROC':<10} {'v2 I-AUROC':<10} {'I-Δ':<8} {'v1 Fusion':<10} {'v2 Fusion':<10} {'F-Δ':<8}")
print("-" * 100)

df_sorted = df.sort_values('i_roc_diff', ascending=False)
for _, row in df_sorted.head(10).iterrows():
    print(f"{row['class']:<20} "
          f"{row['i_roc_v1']:>7.2f}%   {row['i_roc_v2']:>7.2f}%   "
          f"{row['i_roc_diff']:>+6.2f}%  "
          f"{row['fusion_v1']:>7.2f}%   {row['fusion_v2']:>7.2f}%   "
          f"{row['fusion_diff']:>+6.2f}%")

print("\n... (查看完整数据请打开CSV文件)")

df.to_csv('analysis/comprehensive_auroc_comparison.csv', index=False)
print(f"\n✅ 完整数据已保存: analysis/comprehensive_auroc_comparison.csv")
print("=" * 100)
