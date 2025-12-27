#!/usr/bin/env python3
"""
对比v1和v2在测试集上的最终AUROC性能
这才是真正的异常检测性能指标！
"""
import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 100)
print("测试集AUROC对比: v1 vs v2 (27类)")
print("=" * 100)

# 读取v1和v2的测试结果
v1_mvtec = pd.read_csv('result/v1_ema_rep05_margin01/mvtec/k_2/csv/Seed_111-results.csv')
v1_visa = pd.read_csv('result/v1_ema_rep05_margin01/visa/k_2/csv/Seed_111-results.csv')
v2_mvtec = pd.read_csv('result/v2_ema_rep10_nomargin/mvtec/k_2/csv/Seed_111-results.csv')
v2_visa = pd.read_csv('result/v2_ema_rep10_nomargin/visa/k_2/csv/Seed_111-results.csv')

# 合并数据
v1_all = pd.concat([v1_mvtec, v1_visa], ignore_index=True)
v2_all = pd.concat([v2_mvtec, v2_visa], ignore_index=True)

# 重命名列
v1_all.columns = ['class', 'i_roc_v1', 'p_roc_v1']
v2_all.columns = ['class', 'i_roc_v2', 'p_roc_v2']

# 合并v1和v2
df = pd.merge(v1_all, v2_all, on='class')

# 计算差异
df['i_roc_diff'] = df['i_roc_v2'] - df['i_roc_v1']
df['p_roc_diff'] = df['p_roc_v2'] - df['p_roc_v1']

print("\n【整体统计】")
print("=" * 100)
print(f"{'指标':<20} {'v1均值':<12} {'v2均值':<12} {'差异(v2-v1)':<15} {'v2优于v1类数':<15}")
print("-" * 100)

i_roc_v1_mean = df['i_roc_v1'].mean()
i_roc_v2_mean = df['i_roc_v2'].mean()
i_roc_diff = i_roc_v2_mean - i_roc_v1_mean
i_roc_better = (df['i_roc_diff'] > 0).sum()

p_roc_v1_mean = df['p_roc_v1'].mean()
p_roc_v2_mean = df['p_roc_v2'].mean()
p_roc_diff = p_roc_v2_mean - p_roc_v1_mean
p_roc_better = (df['p_roc_diff'] > 0).sum()

print(f"{'Image AUROC':<20} {i_roc_v1_mean:>7.2f}%     {i_roc_v2_mean:>7.2f}%     {i_roc_diff:>+7.2f}%         {i_roc_better:>3d}/{len(df)}")
print(f"{'Pixel AUROC':<20} {p_roc_v1_mean:>7.2f}%     {p_roc_v2_mean:>7.2f}%     {p_roc_diff:>+7.2f}%         {p_roc_better:>3d}/{len(df)}")

print("\n【详细对比 - 按Image AUROC差异排序】")
print("=" * 100)
print(f"{'类别':<22} {'v1 I-AUROC':<12} {'v2 I-AUROC':<12} {'差异':<10} {'v1 P-AUROC':<12} {'v2 P-AUROC':<12} {'差异':<10}")
print("-" * 100)

df_sorted = df.sort_values('i_roc_diff', ascending=False)
for _, row in df_sorted.iterrows():
    i_marker = "📈" if row['i_roc_diff'] > 0 else "📉" if row['i_roc_diff'] < 0 else "➡️"
    p_marker = "📈" if row['p_roc_diff'] > 0 else "📉" if row['p_roc_diff'] < 0 else "➡️"
    print(f"{row['class']:<22} "
          f"{row['i_roc_v1']:>7.2f}%     {row['i_roc_v2']:>7.2f}%     "
          f"{i_marker}{row['i_roc_diff']:>+6.2f}%   "
          f"{row['p_roc_v1']:>7.2f}%     {row['p_roc_v2']:>7.2f}%     "
          f"{p_marker}{row['p_roc_diff']:>+6.2f}%")

print("\n" + "=" * 100)
print("💡 关键结论")
print("=" * 100)

if abs(i_roc_diff) < 0.1:
    print(f"✅ Image AUROC: v1和v2性能几乎相同 (差异{i_roc_diff:+.2f}%)")
elif i_roc_diff > 0:
    print(f"📈 Image AUROC: v2优于v1 (+{i_roc_diff:.2f}%)")
else:
    print(f"📉 Image AUROC: v1优于v2 ({i_roc_diff:+.2f}%)")

if abs(p_roc_diff) < 0.1:
    print(f"✅ Pixel AUROC: v1和v2性能几乎相同 (差异{p_roc_diff:+.2f}%)")
elif p_roc_diff > 0:
    print(f"📈 Pixel AUROC: v2优于v1 (+{p_roc_diff:.2f}%)")
else:
    print(f"📉 Pixel AUROC: v1优于v2 ({p_roc_diff:+.2f}%)")

print(f"\nv2在{i_roc_better}/{len(df)}类上Image AUROC更优")
print(f"v2在{p_roc_better}/{len(df)}类上Pixel AUROC更优")

# 保存结果
df.to_csv('analysis/test_auroc_comparison.csv', index=False)
print(f"\n✅ 详细数据已保存: analysis/test_auroc_comparison.csv")

print("=" * 100)
