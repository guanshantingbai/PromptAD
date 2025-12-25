#!/usr/bin/env python
"""
测试所有类别的k=4 CLS任务
使用现有checkpoint进行推理
"""
import subprocess
import pandas as pd
import time
from datetime import datetime
import sys

# MVTec 15类
mvtec_classes = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]

# VisA 12类
visa_classes = [
    'candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1',
    'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'
]

def test_class(dataset, class_name):
    """测试单个类别"""
    cmd = [
        'python', 'test_cls.py',
        '--dataset', dataset,
        '--class_name', class_name,
        '--k-shot', '4',
        '--n_pro', '3',
        '--n_pro_ab', '45',
        '--root-dir', 'result/prompt1_fixed'
    ]
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    duration = time.time() - start
    
    # 提取i_roc结果
    for line in result.stdout.split('\n'):
        if 'Pixel-AUROC:' in line:
            score = float(line.split('Pixel-AUROC:')[1].strip())
            return score, duration, 'success'
    
    # 如果没找到结果，检查是否有错误
    if result.returncode != 0:
        return None, duration, f'failed: {result.stderr[:100]}'
    
    return None, duration, 'no_result'

def main():
    start_time = datetime.now()
    print('='*80)
    print(f'测试k=4全类别CLS任务')
    print(f'开始时间: {start_time.strftime("%Y-%m-%d %H:%M:%S")}')
    print('='*80)
    
    results = []
    total_tasks = len(mvtec_classes) + len(visa_classes)
    completed = 0
    
    # 测试MVTec
    print('\n📦 测试MVTec (15类)')
    print('-'*80)
    for cls in mvtec_classes:
        completed += 1
        print(f'[{completed}/{total_tasks}] 测试 mvtec-{cls}...', end=' ', flush=True)
        
        try:
            score, duration, status = test_class('mvtec', cls)
            if status == 'success':
                print(f'✅ {score:.2f}% ({duration:.1f}s)')
                results.append({
                    'dataset': 'mvtec',
                    'class': cls,
                    'i_roc': score,
                    'duration': duration,
                    'status': 'success'
                })
            else:
                print(f'❌ {status}')
                results.append({
                    'dataset': 'mvtec',
                    'class': cls,
                    'i_roc': None,
                    'duration': duration,
                    'status': status
                })
        except Exception as e:
            print(f'❌ Exception: {str(e)[:50]}')
            results.append({
                'dataset': 'mvtec',
                'class': cls,
                'i_roc': None,
                'duration': 0,
                'status': f'exception: {str(e)[:50]}'
            })
    
    # 测试VisA
    print('\n📦 测试VisA (12类)')
    print('-'*80)
    for cls in visa_classes:
        completed += 1
        print(f'[{completed}/{total_tasks}] 测试 visa-{cls}...', end=' ', flush=True)
        
        try:
            score, duration, status = test_class('visa', cls)
            if status == 'success':
                print(f'✅ {score:.2f}% ({duration:.1f}s)')
                results.append({
                    'dataset': 'visa',
                    'class': cls,
                    'i_roc': score,
                    'duration': duration,
                    'status': 'success'
                })
            else:
                print(f'❌ {status}')
                results.append({
                    'dataset': 'visa',
                    'class': cls,
                    'i_roc': None,
                    'duration': duration,
                    'status': status
                })
        except Exception as e:
            print(f'❌ Exception: {str(e)[:50]}')
            results.append({
                'dataset': 'visa',
                'class': cls,
                'i_roc': None,
                'duration': 0,
                'status': f'exception: {str(e)[:50]}'
            })
    
    # 保存结果
    df = pd.DataFrame(results)
    output_csv = 'retest_all_cls_k4_results.csv'
    df.to_csv(output_csv, index=False)
    
    # 统计
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds() / 60
    
    success_df = df[df['status'] == 'success']
    n_success = len(success_df)
    n_failed = len(df) - n_success
    
    print('\n' + '='*80)
    print('测试完成！')
    print('='*80)
    print(f'总用时: {total_duration:.1f} 分钟')
    print(f'成功: {n_success}/{total_tasks}')
    print(f'失败: {n_failed}/{total_tasks}')
    
    if n_success > 0:
        avg_duration = success_df['duration'].mean()
        print(f'平均每个类别: {avg_duration:.1f} 秒')
    
    print(f'\n结果已保存到: {output_csv}')
    
    # 对比baseline
    if n_success > 0:
        print('\n' + '='*80)
        print('与Baseline对比')
        print('='*80)
        
        # 读取baseline
        df_bl_mvtec = pd.read_csv('result/baseline/mvtec/k_4/csv/Seed_111-results.csv')
        df_bl_visa = pd.read_csv('result/baseline/visa/k_4/csv/Seed_111-results.csv')
        
        mvtec_results = success_df[success_df['dataset'] == 'mvtec']
        visa_results = success_df[success_df['dataset'] == 'visa']
        
        print('\nMVTec:')
        mvtec_baseline_avg = 0
        mvtec_ours_avg = 0
        mvtec_count = 0
        
        for _, row in mvtec_results.iterrows():
            cls = row['class']
            ours = row['i_roc']
            bl_row = df_bl_mvtec[df_bl_mvtec.iloc[:, 0] == f'mvtec-{cls}']
            if len(bl_row) > 0:
                baseline = bl_row['i_roc'].values[0]
                delta = ours - baseline
                mvtec_baseline_avg += baseline
                mvtec_ours_avg += ours
                mvtec_count += 1
                status = '✅' if delta > 0 else '❌'
                print(f'  {cls:<15} {baseline:6.2f} → {ours:6.2f} ({delta:+6.2f}) {status}')
        
        if mvtec_count > 0:
            mvtec_baseline_avg /= mvtec_count
            mvtec_ours_avg /= mvtec_count
            print(f'  {"Average":<15} {mvtec_baseline_avg:6.2f} → {mvtec_ours_avg:6.2f} ({mvtec_ours_avg-mvtec_baseline_avg:+6.2f})')
        
        print('\nVisA:')
        visa_baseline_avg = 0
        visa_ours_avg = 0
        visa_count = 0
        
        for _, row in visa_results.iterrows():
            cls = row['class']
            ours = row['i_roc']
            bl_row = df_bl_visa[df_bl_visa.iloc[:, 0] == f'visa-{cls}']
            if len(bl_row) > 0:
                baseline = bl_row['i_roc'].values[0]
                delta = ours - baseline
                visa_baseline_avg += baseline
                visa_ours_avg += ours
                visa_count += 1
                status = '✅' if delta > 0 else '❌'
                print(f'  {cls:<15} {baseline:6.2f} → {ours:6.2f} ({delta:+6.2f}) {status}')
        
        if visa_count > 0:
            visa_baseline_avg /= visa_count
            visa_ours_avg /= visa_count
            print(f'  {"Average":<15} {visa_baseline_avg:6.2f} → {visa_ours_avg:6.2f} ({visa_ours_avg-visa_baseline_avg:+6.2f})')
        
        # 总体
        if mvtec_count + visa_count > 0:
            total_baseline_avg = (mvtec_baseline_avg * mvtec_count + visa_baseline_avg * visa_count) / (mvtec_count + visa_count)
            total_ours_avg = (mvtec_ours_avg * mvtec_count + visa_ours_avg * visa_count) / (mvtec_count + visa_count)
            print(f'\n总体平均: {total_baseline_avg:.2f} → {total_ours_avg:.2f} ({total_ours_avg-total_baseline_avg:+.2f})')

if __name__ == '__main__':
    main()
