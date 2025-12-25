#!/usr/bin/env python
"""
测试修复后的6个关键类别（k=2）
验证修复双重融合BUG后的结果
"""
import subprocess
import pandas as pd
import time

key_classes = ['screw', 'toothbrush', 'hazelnut', 'capsule', 'pill', 'metal_nut']

results = []

print('='*80)
print('测试修复后的6个关键类别 (k=2)')
print('='*80)

for cls in key_classes:
    print(f'\n测试 {cls}...')
    start = time.time()
    
    cmd = [
        'python', 'test_cls.py',
        '--dataset', 'mvtec',
        '--class_name', cls,
        '--k-shot', '2',
        '--n_pro', '3',
        '--n_pro_ab', '45',
        '--root-dir', 'result/prompt1_fixed'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # 提取结果
    for line in result.stdout.split('\n'):
        if 'Pixel-AUROC:' in line:
            score = float(line.split('Pixel-AUROC:')[1].strip())
            results.append({
                'class': cls,
                'score': score,
                'time': time.time() - start
            })
            print(f'  结果: {score:.2f}% ({time.time()-start:.1f}s)')
            break

# 保存结果
df = pd.DataFrame(results)
df.to_csv('test_key_6_fixed_results.csv', index=False)

# 对比baseline
df_bl = pd.read_csv('result/baseline/mvtec/k_2/csv/Seed_111-results.csv')

print('\n' + '='*80)
print('对比Baseline')
print('='*80)
print(f'{"Class":<15} {"Baseline":>10} {"Fixed":>10} {"Delta":>10}')
print('-'*50)

total_bl = 0
total_ours = 0

for _, row in df.iterrows():
    cls = row['class']
    ours = row['score']
    baseline = df_bl[df_bl.iloc[:, 0] == f'mvtec-{cls}']['i_roc'].values[0]
    delta = ours - baseline
    
    total_bl += baseline
    total_ours += ours
    
    status = '✅' if delta > 0 else '❌'
    print(f'{cls:<15} {baseline:>10.2f} {ours:>10.2f} {delta:>9.2f} {status}')

avg_bl = total_bl / len(df)
avg_ours = total_ours / len(df)
avg_delta = avg_ours - avg_bl

print('-'*50)
print(f'{"Average":<15} {avg_bl:>10.2f} {avg_ours:>10.2f} {avg_delta:>9.2f}')
print('='*80)
