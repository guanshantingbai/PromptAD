import pandas as pd

print('='*80)
print('检查6个关键类别的数据一致性')
print('='*80)

key_classes = ['screw', 'toothbrush', 'hazelnut', 'capsule', 'pill', 'metal_nut']

# 1. Baseline数据
print('\n1. Baseline (result/baseline/mvtec/k_2/)')
df_bl = pd.read_csv('result/baseline/mvtec/k_2/csv/Seed_111-results.csv')
bl_scores = {}
for cls in key_classes:
    score = df_bl[df_bl.iloc[:, 0] == f'mvtec-{cls}']['i_roc'].values[0]
    bl_scores[cls] = score
    print(f'   {cls:<15} {score:.2f}')
bl_avg = sum(bl_scores.values()) / len(bl_scores)
print(f'   {"Average":<15} {bl_avg:.2f}')

# 2. 旧的CSV（双重融合，可能已被覆盖）
print('\n2. 旧CSV (result/prompt1_fixed/mvtec/k_2/csv/) - 可能是双重融合的结果')
df_old_csv = pd.read_csv('result/prompt1_fixed/mvtec/k_2/csv/Seed_111-results.csv')
old_csv_scores = {}
for cls in key_classes:
    score = df_old_csv[df_old_csv.iloc[:, 0] == f'mvtec-{cls}']['i_roc'].values[0]
    old_csv_scores[cls] = score
    print(f'   {cls:<15} {score:.2f}')
old_csv_avg = sum(old_csv_scores.values()) / len(old_csv_scores)
print(f'   {"Average":<15} {old_csv_avg:.2f}')
print(f'   vs Baseline: {old_csv_avg - bl_avg:+.2f}')

# 3. 刚才的重新测试结果（修复后）
print('\n3. 重新测试结果 (retest_all_cls_k2_results.csv) - 修复双重融合后')
df_retest = pd.read_csv('retest_all_cls_k2_results.csv')
retest_scores = {}
for cls in key_classes:
    score = df_retest[df_retest['class'] == cls]['i_roc'].values[0]
    retest_scores[cls] = score
    print(f'   {cls:<15} {score:.2f}')
retest_avg = sum(retest_scores.values()) / len(retest_scores)
print(f'   {"Average":<15} {retest_avg:.2f}')
print(f'   vs Baseline: {retest_avg - bl_avg:+.2f}')

# 4. 对比差异
print('\n' + '='*80)
print('逐类对比')
print('='*80)
print(f'{"类别":<15} {"Baseline":<12} {"旧CSV":<12} {"重新测试":<12} {"差异":<12}')
print('-'*80)
for cls in key_classes:
    bl = bl_scores[cls]
    old = old_csv_scores[cls]
    new = retest_scores[cls]
    diff = new - old
    status = '✅' if diff > 0 else ('=' if diff == 0 else '❌')
    print(f'{cls:<15} {bl:<12.2f} {old:<12.2f} {new:<12.2f} {diff:+11.2f} {status}')
print('-'*80)
print(f'{"Average":<15} {bl_avg:<12.2f} {old_csv_avg:<12.2f} {retest_avg:<12.2f} {retest_avg-old_csv_avg:+11.2f}')

print('\n' + '='*80)
print('关键发现:')
print('='*80)
print(f'旧CSV平均值: {old_csv_avg:.2f}% (vs baseline {old_csv_avg-bl_avg:+.2f}%)')
print(f'重新测试平均值: {retest_avg:.2f}% (vs baseline {retest_avg-bl_avg:+.2f}%)')
print(f'修复改善: {retest_avg-old_csv_avg:+.2f} 个百分点')

# 5. 检查我之前的错误数据来源
print('\n' + '='*80)
print('检查之前提到的86.05%和80.02%数据:')
print('='*80)
print(f'\n之前我说的"6类平均86.05%"应该是错误的！')
print(f'实际旧CSV的6类平均是: {old_csv_avg:.2f}%')
print(f'\n之前我说的"screw 80.02"也是错误的！')
print(f'实际screw的分数是: {old_csv_scores["screw"]:.2f}% (旧CSV), {retest_scores["screw"]:.2f}% (重测)')
