import pandas as pd

df = pd.read_csv('retest_all_cls_k2_results.csv')

key_classes = ['screw', 'toothbrush', 'hazelnut', 'capsule', 'pill', 'metal_nut']
key_results = df[df['class'].isin(key_classes)]

print('6个关键类别修复前后对比:')
print('='*80)
print(f"{'类别':<15} {'修复前(双重)':<15} {'修复后(正确)':<15} {'改善幅度':<15}")
print('-'*80)

# 修复前的数据（从之前的CSV）
old_data = {
    'screw': 69.66,
    'toothbrush': 89.44,
    'hazelnut': 90.71,
    'capsule': 80.89,
    'pill': 86.35,
    'metal_nut': 88.86
}

for cls in key_classes:
    old = old_data[cls]
    new = key_results[key_results['class'] == cls]['i_roc'].values[0]
    improvement = new - old
    status = '✅' if improvement > 0 else '❌'
    print(f'{cls:<15} {old:<15.2f} {new:<15.2f} {improvement:+14.2f} {status}')

old_avg = sum(old_data.values()) / len(old_data)
new_avg = key_results['i_roc'].mean()
print('-'*80)
print(f"{'Average':<15} {old_avg:<15.2f} {new_avg:<15.2f} {new_avg-old_avg:+14.2f}")
print('='*80)
print(f'\n修复带来的改善: {new_avg-old_avg:+.2f} 个百分点')
print(f'从 86.05% (双重融合) → {new_avg:.2f}% (正确融合)')
