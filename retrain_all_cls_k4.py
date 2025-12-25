#!/usr/bin/env python
"""
训练所有类别的k=4 CLS任务
"""
import subprocess
import pandas as pd
import time
from datetime import datetime
import concurrent.futures
import os

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

MAX_WORKERS = 2  # 并行任务数

def train_class(dataset, class_name):
    """训练单个类别"""
    task_name = f'{dataset}-{class_name}'
    
    cmd = [
        'python', 'train_cls.py',
        '--dataset', dataset,
        '--class_name', class_name,
        '--k-shot', '4',
        '--n_pro', '3',
        '--n_pro_ab', '45',
        '--Epoch', '100',
        '--lr', '0.002',
        '--root-dir', 'result/prompt1_fixed'
    ]
    
    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        duration = time.time() - start
        
        if result.returncode == 0:
            # 检查checkpoint是否存在
            ckpt_path = f'result/prompt1_fixed/{dataset}/k_4/checkpoint/CLS-Seed_111-{class_name}-check_point.pt'
            if os.path.exists(ckpt_path):
                return {
                    'dataset': dataset,
                    'class': class_name,
                    'duration': duration,
                    'status': 'success'
                }
            else:
                return {
                    'dataset': dataset,
                    'class': class_name,
                    'duration': duration,
                    'status': 'no_checkpoint'
                }
        else:
            return {
                'dataset': dataset,
                'class': class_name,
                'duration': duration,
                'status': f'failed: returncode={result.returncode}'
            }
    except subprocess.TimeoutExpired:
        return {
            'dataset': dataset,
            'class': class_name,
            'duration': 600,
            'status': 'timeout'
        }
    except Exception as e:
        return {
            'dataset': dataset,
            'class': class_name,
            'duration': 0,
            'status': f'exception: {str(e)[:50]}'
        }

def main():
    start_time = datetime.now()
    print('='*80)
    print(f'训练k=4全类别CLS任务 (并行={MAX_WORKERS})')
    print(f'开始时间: {start_time.strftime("%Y-%m-%d %H:%M:%S")}')
    print('='*80)
    
    # 准备所有任务
    tasks = []
    for cls in mvtec_classes:
        tasks.append(('mvtec', cls))
    for cls in visa_classes:
        tasks.append(('visa', cls))
    
    results = []
    completed = 0
    total_tasks = len(tasks)
    
    # 并行执行
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {executor.submit(train_class, dataset, cls): (dataset, cls) 
                         for dataset, cls in tasks}
        
        for future in concurrent.futures.as_completed(future_to_task):
            dataset, cls = future_to_task[future]
            completed += 1
            
            try:
                result = future.result()
                results.append(result)
                
                status = result['status']
                duration = result['duration']
                
                if status == 'success':
                    print(f'[{completed}/{total_tasks}] ✅ {dataset}-{cls} ({duration:.1f}s)')
                else:
                    print(f'[{completed}/{total_tasks}] ❌ {dataset}-{cls} - {status}')
                    
            except Exception as e:
                print(f'[{completed}/{total_tasks}] ❌ {dataset}-{cls} - Exception: {str(e)[:50]}')
                results.append({
                    'dataset': dataset,
                    'class': cls,
                    'duration': 0,
                    'status': f'exception: {str(e)[:50]}'
                })
    
    # 保存结果
    df = pd.DataFrame(results)
    output_csv = 'retrain_all_cls_k4_execution.csv'
    df.to_csv(output_csv, index=False)
    
    # 统计
    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds() / 60
    
    success_df = df[df['status'] == 'success']
    n_success = len(success_df)
    n_failed = len(df) - n_success
    
    print('\n' + '='*80)
    print('训练完成！')
    print('='*80)
    print(f'总用时: {total_duration:.1f} 分钟')
    print(f'成功: {n_success}/{total_tasks}')
    print(f'失败: {n_failed}/{total_tasks}')
    
    if n_success > 0:
        avg_duration = success_df['duration'].mean()
        print(f'平均每个类别: {avg_duration:.1f} 秒')
    
    print(f'\n执行日志已保存到: {output_csv}')
    
    if n_failed > 0:
        print(f'\n失败的类别:')
        failed_df = df[df['status'] != 'success']
        for _, row in failed_df.iterrows():
            print(f"  {row['dataset']}-{row['class']}: {row['status']}")

if __name__ == '__main__':
    main()
