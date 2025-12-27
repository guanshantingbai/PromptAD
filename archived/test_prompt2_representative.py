#!/usr/bin/env python3
"""
测试Prompt2代表类别的分类性能
对比baseline和prompt2在k=1,2,4下的表现
"""

import subprocess
import os
import time
import csv
import pandas as pd
from datetime import datetime

# 配置
SEED = 111
N_PRO = 1
N_PRO_AB = 4

# 代表类别
MVTEC_CLASSES = ['bottle', 'toothbrush', 'screw', 'capsule']
VISA_CLASSES = ['candle', 'pcb4', 'macaroni2']

# 目录配置
BASELINE_DIR = 'result/baseline'
PROMPT2_DIR = 'result/prompt2'


def test_class(dataset, cls, k_shot, method='baseline'):
    """测试单个类别"""
    result_dir = BASELINE_DIR if method == 'baseline' else PROMPT2_DIR
    
    print(f"\n测试: {method} - {dataset}-{cls} (k={k_shot})")
    
    cmd = [
        'python', 'test_cls.py',
        '--dataset', dataset,
        '--class_name', cls,
        '--k-shot', str(k_shot),
        '--seed', str(SEED),
        '--n_pro', str(N_PRO),
        '--n_pro_ab', str(N_PRO_AB),
        '--root-dir', result_dir
    ]
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        elapsed = time.time() - start_time
        
        # 从输出中提取i_roc
        output = result.stdout
        i_roc = None
        for line in output.split('\n'):
            if 'i_roc' in line.lower():
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'i_roc' in part.lower() and i+1 < len(parts):
                        try:
                            i_roc = float(parts[i+1].strip('%,'))
                            break
                        except:
                            pass
        
        print(f"  ✅ 完成! i_roc: {i_roc:.2f}% ({elapsed:.1f}s)")
        return True, i_roc, elapsed
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"  ❌ 失败! ({elapsed:.1f}s)")
        return False, None, elapsed


def load_result_from_csv(dataset, cls, k_shot, method='baseline'):
    """从CSV文件中读取结果"""
    result_dir = BASELINE_DIR if method == 'baseline' else PROMPT2_DIR
    csv_path = os.path.join(result_dir, dataset, f'k_{k_shot}', 'csv', f'Seed_{SEED}-results.csv')
    
    if not os.path.exists(csv_path):
        return None
    
    try:
        df = pd.read_csv(csv_path, index_col=0)
        row_key = f'{dataset}-{cls}'
        if row_key in df.index and 'i_roc' in df.columns:
            return df.loc[row_key, 'i_roc']
    except:
        pass
    
    return None


def main():
    print(f"\n{'='*80}")
    print(f"Prompt2 代表类别测试")
    print(f"配置: n_pro={N_PRO}, n_pro_ab={N_PRO_AB}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    all_classes = [(d, c) for d in ['mvtec'] for c in MVTEC_CLASSES] + \
                  [(d, c) for d in ['visa'] for c in VISA_CLASSES]
    k_values = [1, 2, 4]
    
    results = []
    
    for dataset, cls in all_classes:
        for k in k_values:
            print(f"\n{'='*60}")
            print(f"处理: {dataset}-{cls} k={k}")
            print(f"{'='*60}")
            
            # 获取baseline结果（直接从CSV读取）
            baseline_score = load_result_from_csv(dataset, cls, k, 'baseline')
            if baseline_score is None:
                print(f"  ⚠️  Baseline结果不存在，跳过")
                continue
            
            # 测试prompt2
            success, prompt2_score, elapsed = test_class(dataset, cls, k, 'prompt2')
            
            if success and prompt2_score is not None:
                diff = prompt2_score - baseline_score
                print(f"\n  📊 对比:")
                print(f"    Baseline: {baseline_score:.2f}%")
                print(f"    Prompt2:  {prompt2_score:.2f}%")
                print(f"    差异:     {diff:+.2f}%")
                
                results.append({
                    'dataset': dataset,
                    'class': cls,
                    'k': k,
                    'baseline': baseline_score,
                    'prompt2': prompt2_score,
                    'diff': diff,
                    'success': True
                })
            else:
                results.append({
                    'dataset': dataset,
                    'class': cls,
                    'k': k,
                    'baseline': baseline_score,
                    'prompt2': None,
                    'diff': None,
                    'success': False
                })
    
    # 保存结果
    csv_path = 'test_prompt2_representative_results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'class', 'k', 'baseline', 'prompt2', 'diff', 'success'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n{'='*80}")
    print(f"测试完成!")
    print(f"{'='*80}")
    print(f"结果已保存: {csv_path}")
    
    # 打印汇总表
    print(f"\n汇总表:")
    print(f"{'='*80}")
    
    success_results = [r for r in results if r['success']]
    if success_results:
        df = pd.DataFrame(success_results)
        
        # 按k值分组统计
        for k in k_values:
            k_results = df[df['k'] == k]
            if len(k_results) > 0:
                avg_baseline = k_results['baseline'].mean()
                avg_prompt2 = k_results['prompt2'].mean()
                avg_diff = k_results['diff'].mean()
                print(f"\nk={k}:")
                print(f"  Baseline平均: {avg_baseline:.2f}%")
                print(f"  Prompt2平均:  {avg_prompt2:.2f}%")
                print(f"  平均差异:     {avg_diff:+.2f}%")
        
        # 总体统计
        print(f"\n总体:")
        print(f"  Baseline平均: {df['baseline'].mean():.2f}%")
        print(f"  Prompt2平均:  {df['prompt2'].mean():.2f}%")
        print(f"  平均差异:     {df['diff'].mean():+.2f}%")
    
    print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
