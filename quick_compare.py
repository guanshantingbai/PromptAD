#!/usr/bin/env python3
import pandas as pd
import numpy as np

baseline = pd.read_csv('result/baseline/mvtec/k_2/csv/Seed_111-results.csv', index_col=0)
cleaned = pd.read_csv('result/phase1_cleaned/mvtec/k_2/csv/Seed_111-results.csv', index_col=0)

print('='*80)
print('Phase 1 Results: Baseline vs Cleaned Prompts')
print('='*80)
print()

# 准备对比数据
comparison = pd.DataFrame({
    'Baseline_Semantic': baseline['semantic_i_roc'],
    'Cleaned_Semantic': cleaned['semantic_i_roc'],
    'Delta_Semantic': cleaned['semantic_i_roc'] - baseline['semantic_i_roc'],
    'Baseline_Memory': baseline['memory_i_roc'],
    'Cleaned_Memory': cleaned['memory_i_roc'],
    'Delta_Memory': cleaned['memory_i_roc'] - baseline['memory_i_roc'],
})

# 按 Semantic Delta 排序
comparison = comparison.sort_values('Delta_Semantic', ascending=False)

print('Semantic AUROC Comparison (sorted by improvement):')
print('-'*80)
print(f"{'Class':<20} {'Baseline':>10} {'Cleaned':>10} {'Delta':>10} {'Status':>8}")
print('-'*80)

for cls, row in comparison.iterrows():
    delta = row['Delta_Semantic']
    if delta > 1.0:
        status = '✓✓'
    elif delta > 0:
        status = '✓'
    elif delta < -1.0:
        status = '✗✗'
    elif delta < 0:
        status = '✗'
    else:
        status = '='
    
    cls_short = cls.replace('mvtec-', '')
    print(f'{cls_short:<20} {row["Baseline_Semantic"]:>10.2f} {row["Cleaned_Semantic"]:>10.2f} '
          f'{delta:>+9.2f} {status:>8}')

print('-'*80)
avg_baseline = baseline['semantic_i_roc'].mean()
avg_cleaned = cleaned['semantic_i_roc'].mean()
avg_label = 'Average'
print(f'{avg_label:<20} {avg_baseline:>10.2f} {avg_cleaned:>10.2f} {avg_cleaned-avg_baseline:>+9.2f}')
print('='*80)
print()

# 统计
improved = (comparison['Delta_Semantic'] > 0).sum()
degraded = (comparison['Delta_Semantic'] < 0).sum()
unchanged = (comparison['Delta_Semantic'] == 0).sum()

print('Summary:')
print(f'  Improved:  {improved} classes')
print(f'  Degraded:  {degraded} classes')
print(f'  Unchanged: {unchanged} classes')
print(f'  Average improvement: {avg_cleaned-avg_baseline:+.2f} points')
print()

# Top 5 改进
print('Top 5 Improved Classes:')
top5 = comparison.nlargest(5, 'Delta_Semantic')
for i, (cls, row) in enumerate(top5.iterrows(), 1):
    cls_short = cls.replace('mvtec-', '')
    print(f'  {i}. {cls_short:15s}: {row["Baseline_Semantic"]:.2f} → {row["Cleaned_Semantic"]:.2f} '
          f'({row["Delta_Semantic"]:+.2f})')
print()

# Top 5 下降
if degraded > 0:
    print('Top 5 Degraded Classes:')
    bottom5 = comparison.nsmallest(min(5, degraded), 'Delta_Semantic')
    for i, (cls, row) in enumerate(bottom5.iterrows(), 1):
        cls_short = cls.replace('mvtec-', '')
        print(f'  {i}. {cls_short:15s}: {row["Baseline_Semantic"]:.2f} → {row["Cleaned_Semantic"]:.2f} '
              f'({row["Delta_Semantic"]:+.2f})')
    print()

print('='*80)
