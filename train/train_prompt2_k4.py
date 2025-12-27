#!/usr/bin/env python3
"""
Prompt2探索: n_pro=1配置下训练k=4代表类别
代表类别: bottle, toothbrush, screw, capsule (MVTec) + candle, pcb4, macaroni2 (VisA)
"""

import subprocess
import os
import time
from datetime import datetime

# 配置
K_SHOT = 4
N_PRO = 1
N_PRO_AB = 4
SEED = 111
EPOCH = 100
LR = 0.002

# 代表类别
MVTEC_CLASSES = ['bottle', 'toothbrush', 'screw', 'capsule']
VISA_CLASSES = ['candle', 'pcb4', 'macaroni2']

# 结果目录
RESULT_DIR = 'result/prompt2'

def train_class(dataset, cls, k_shot):
    """训练单个类别"""
    print(f"\n{'='*80}")
    print(f"开始训练: {dataset}-{cls} (k={k_shot})")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
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
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        elapsed = time.time() - start_time
        
        print(f"✅ {dataset}-{cls} 训练完成!")
        print(f"⏱️  耗时: {elapsed/60:.1f}分钟")
        return True, elapsed
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ {dataset}-{cls} 训练失败!")
        print(f"错误信息: {e.stderr[-500:]}")
        return False, elapsed


def main():
    print(f"\n{'='*80}")
    print(f"Prompt2 训练 - k={K_SHOT} 代表类别")
    print(f"配置: n_pro={N_PRO}, n_pro_ab={N_PRO_AB}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # 创建结果目录
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    results = []
    total_start = time.time()
    
    # 训练MVTec类别
    for cls in MVTEC_CLASSES:
        success, elapsed = train_class('mvtec', cls, K_SHOT)
        results.append({
            'dataset': 'mvtec',
            'class': cls,
            'k': K_SHOT,
            'success': success,
            'time': elapsed
        })
    
    # 训练VisA类别
    for cls in VISA_CLASSES:
        success, elapsed = train_class('visa', cls, K_SHOT)
        results.append({
            'dataset': 'visa',
            'class': cls,
            'k': K_SHOT,
            'success': success,
            'time': elapsed
        })
    
    # 统计结果
    total_time = time.time() - total_start
    success_count = sum(1 for r in results if r['success'])
    total_count = len(results)
    
    print(f"\n{'='*80}")
    print(f"训练完成!")
    print(f"{'='*80}")
    print(f"成功: {success_count}/{total_count}")
    print(f"总耗时: {total_time/60:.1f}分钟")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 打印详细结果
    print(f"\n详细结果:")
    for r in results:
        status = "✅" if r['success'] else "❌"
        print(f"  {status} {r['dataset']}-{r['class']}: {r['time']/60:.1f}分钟")
    
    # 保存执行记录
    import csv
    csv_path = f'train_prompt2_k{K_SHOT}_execution.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'class', 'k', 'success', 'time'])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n执行记录已保存: {csv_path}")


if __name__ == '__main__':
    main()
