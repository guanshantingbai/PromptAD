#!/usr/bin/env python3
"""
Prototype Similarity 消融实验分析
分析K-shot样本间学习到的原型的相似度与性能的关系
"""

import pandas as pd
import numpy as np

# 读取跨数据集总结
df_summary = pd.read_csv('analysis/ablation/prototype_similarity/cross_dataset_summary.csv')

# 读取MVTec K=2详细数据
df_mvtec_k2 = pd.read_csv('analysis/ablation/prototype_similarity/mvtec/baseline_k2_similarity.csv')

# 读取baseline k=2的性能数据
df_perf = pd.read_csv('result/baseline/mvtec/k_2/csv/Seed_111-results.csv', index_col=0)

print('=' * 80)
print('Prototype Similarity 消融实验分析')
print('=' * 80)
print()

# 1. 跨数据集对比
print('📊 1. 跨数据集 Prototype Similarity 对比')
print('-' * 80)
print(f'{"数据集":<15} {"平均相似度":<15} {"标准差":<12} {"平均AUROC":<15} {"Pearson相关":<15}')
print('-' * 80)
for _, row in df_summary.iterrows():
    print(f'{row["dataset"]:<15} {row["mean_sim"]:.4f}          {row["std_sim"]:.4f}       {row["mean_auroc"]:.2f}%         r={row["pearson_r"]:.3f} (p={row["pearson_p"]:.3f})')

print()
print('✅ **关键发现**:')
print('   1. MVTec平均相似度: 0.669 (K=2)')
print('   2. ViSA平均相似度: 0.746 (K=2) - 更高的原型一致性')
print('   3. Pearson相关系数:')
print('      - MVTec K=2: r=0.504 (p=0.055) - 中等正相关，接近显著')
print('      - ViSA K=2: r=0.181 (p=0.573) - 弱相关')
print()

# 2. MVTec K=2 类别级分析
print('📊 2. MVTec K=2 类别级 Prototype Similarity')
print('-' * 80)

# 合并相似度和性能数据
df_mvtec_k2['class_full'] = 'mvtec-' + df_mvtec_k2['class']
df_combined = df_mvtec_k2.set_index('class_full').join(df_perf[['i_roc', 'semantic_i_roc']])

# 按相似度排序
df_sorted = df_combined.sort_values('mean_sim', ascending=False)

print(f'{"类别":<15} {"相似度":<12} {"Fusion AUROC":<15} {"Semantic AUROC":<15}')
print('-' * 80)
for idx, row in df_sorted.iterrows():
    cls = idx.replace('mvtec-', '')
    print(f'{cls:<15} {row["mean_sim"]:.4f}       {row["i_roc"]:.2f}%          {row["semantic_i_roc"]:.2f}%')

print()

# 3. 相似度与性能的关系
high_sim = df_combined[df_combined['mean_sim'] > 0.7]
low_sim = df_combined[df_combined['mean_sim'] <= 0.7]

print('📊 3. 相似度与性能的关系')
print('-' * 80)
print(f'高相似度类别 (>0.7, {len(high_sim)}/15):')
print(f'   平均Fusion AUROC: {high_sim["i_roc"].mean():.2f}%')
print(f'   平均Semantic AUROC: {high_sim["semantic_i_roc"].mean():.2f}%')
print()
print(f'低相似度类别 (≤0.7, {len(low_sim)}/15):')
print(f'   平均Fusion AUROC: {low_sim["i_roc"].mean():.2f}%')
print(f'   平均Semantic AUROC: {low_sim["semantic_i_roc"].mean():.2f}%')
print()
print(f'性能差距: Fusion {high_sim["i_roc"].mean() - low_sim["i_roc"].mean():+.2f}%, Semantic {high_sim["semantic_i_roc"].mean() - low_sim["semantic_i_roc"].mean():+.2f}%')
print()

# 4. 极端案例分析
print('📊 4. 典型案例分析')
print('-' * 80)
top3 = df_sorted.head(3)
bottom3 = df_sorted.tail(3)

print('✨ Top 3 高相似度类别:')
for idx, row in top3.iterrows():
    cls = idx.replace('mvtec-', '')
    print(f'   {cls:<15}: 相似度={row["mean_sim"]:.4f}, Fusion={row["i_roc"]:.2f}%, Semantic={row["semantic_i_roc"]:.2f}%')

print()
print('⚠️  Bottom 3 低相似度类别:')
for idx, row in bottom3.iterrows():
    cls = idx.replace('mvtec-', '')
    print(f'   {cls:<15}: 相似度={row["mean_sim"]:.4f}, Fusion={row["i_roc"]:.2f}%, Semantic={row["semantic_i_roc"]:.2f}%')

print()

# 5. 结论
print('=' * 80)
print('🎯 结论: Prototype Similarity 的意义')
print('=' * 80)
print()
print('1️⃣  **相似度反映原型一致性**:')
print('    高相似度(>0.7)表示K个样本学到的原型接近')
print('    低相似度表示样本间差异大或学习不稳定')
print()
print('2️⃣  **中等正相关关系** (MVTec K=2, r=0.504):')
print('    高相似度类别倾向于更好的性能')
print('    但不是决定性因素(p=0.055, 接近显著)')
print()
print('3️⃣  **类别特性差异**:')
print('    - 高相似度: zipper(0.835), transistor(0.792), wood(0.796)')
print('    - 低相似度: toothbrush(0.385), cable(0.462), capsule(0.477)')
print()
print('4️⃣  **Few-shot学习的稳定性指标**:')
print('    相似度可作为评估Few-shot学习质量的指标')
print('    低相似度可能需要更多样本或更强正则化')
print()
print('=' * 80)
print()
print('💡 **论文陈述建议**:')
print()
print('   "We analyze the prototype similarity across K-shot samples to assess')
print('   the consistency of learned representations. On MVTec (K=2), we observe')
print('   a moderate positive correlation (r=0.504, p=0.055) between prototype')
print('   similarity and detection performance. High-similarity classes like')
print('   zipper (0.835) achieve better AUROC (97.06%) compared to low-similarity')
print('   classes like toothbrush (0.385, AUROC 93.89%), suggesting that')
print('   consistent prototypes across few-shot samples contribute to more')
print('   reliable anomaly detection. This metric provides insights into')
print('   few-shot learning stability and can guide sample selection strategies."')
print()
print('=' * 80)
