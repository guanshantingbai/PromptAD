#!/usr/bin/env python3
"""
Prompt2探索: n_pro=1配置下并行训练k=1代表类别
使用2进程并行加速训练
"""

import subprocess
import os
import time
import sys
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# 配置
K_SHOT = 1
N_PRO = 1
N_PRO_AB = 4
SEED = 111
EPOCH = 100
LR = 0.002
MAX_WORKERS = 2  # 并行进程数

# 代表类别
MVTEC_CLASSES = ['bottle', 'toothbrush', 'screw', 'capsule']
VISA_CLASSES = ['candle', 'pcb4', 'macaroni2']

# 结果目录
RESULT_DIR = 'result/prompt2'

def train_class(dataset, cls, k_shot):
    """训练单个类别"""
    print(f"\n{'='*80}", flush=True)
    print(f"[进程启动] {dataset}-{cls} (k={k_shot})", flush=True)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"{'='*80}\n", flush=True)
    
    cmd = [
        'python', 'train_cls.py',
        '--dataset', dataset,
        '--class_name', cls,
        '--k-shot', str(k_shot),
        '--seed', str(SEED),
        '--n_pro', str(N_PRO),
        '--n_pro_ab', str(N_PRO_AB),
        '--Epoch', str(EPOCH),
        '--lr', str(LR),
        '--root-dir', RESULT_DIR
    ]
    
    start_time = time.time()
    
    try:
        # 使用unbuffered输出
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True,
            env={**os.environ, 'PYTHONUNBUFFERED': '1'}
        )
        elapsed = time.time() - start_time
        
        print(f"\n✅ [{dataset}-{cls}] 训练完成!", flush=True)
        print(f"⏱️  耗时: {elapsed/60:.1f}分钟", flush=True)
        return {
            'dataset': dataset,
            'class': cls,
            'k': k_shot,
            'success': True,
            'time': elapsed
        }
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ [{dataset}-{cls}] 训练失败!", flush=True)
        print(f"错误信息: {e.stderr[-500:]}", flush=True)
        return {
            'dataset': dataset,
            'class': cls,
            'k': k_shot,
            'success': False,
            'time': elapsed
        }


def main():
    print(f"\n{'='*80}", flush=True)
    print(f"Prompt2 并行训练 - k={K_SHOT} 代表类别", flush=True)
    print(f"配置: n_pro={N_PRO}, n_pro_ab={N_PRO_AB}", flush=True)
    print(f"并行度: {MAX_WORKERS} 进程", flush=True)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"{'='*80}\n", flush=True)
    sys.stdout.flush()
    
    # 创建结果目录
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    # 构建任务列表
    tasks = []
    for cls in MVTEC_CLASSES:
        tasks.append(('mvtec', cls, K_SHOT))
    for cls in VISA_CLASSES:
        tasks.append(('visa', cls, K_SHOT))
    
    results = []
    total_start = time.time()
    
    # 并行执行
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_task = {
            executor.submit(train_class, dataset, cls, k): (dataset, cls)
            for dataset, cls, k in tasks
        }
        
        # 收集结果
        for future in as_completed(future_to_task):
            dataset, cls = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
                print(f"\n[完成] {dataset}-{cls}: {'✅' if result['success'] else '❌'}", flush=True)
            except Exception as exc:
                print(f"\n[异常] {dataset}-{cls}: {exc}", flush=True)
                results.append({
                    'dataset': dataset,
                    'class': cls,
                    'k': K_SHOT,
                    'success': False,
                    'time': 0
                })
    
    # 统计结果
    total_time = time.time() - total_start
    success_count = sum(1 for r in results if r['success'])
    total_count = len(results)
    
    print(f"\n{'='*80}", flush=True)
    print(f"训练完成!", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"成功: {success_count}/{total_count}", flush=True)
    print(f"总耗时: {total_time/60:.1f}分钟", flush=True)
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    
    # 打印详细结果
    print(f"\n详细结果:", flush=True)
    for r in sorted(results, key=lambda x: (x['dataset'], x['class'])):
        status = "✅" if r['success'] else "❌"
        print(f"  {status} {r['dataset']}-{r['class']}: {r['time']/60:.1f}分钟", flush=True)
    
    # 保存执行记录
    import csv
    csv_path = f'train_prompt2_k{K_SHOT}_parallel_execution.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'class', 'k', 'success', 'time'])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n执行记录已保存: {csv_path}", flush=True)
    sys.stdout.flush()


if __name__ == '__main__':
    main()
