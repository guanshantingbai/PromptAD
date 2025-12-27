#!/usr/bin/env python3
"""
Prompt2通用训练程序: n_pro=1配置下的全类别并行训练
支持参数化k值和数据集选择
"""

import subprocess
import os
import time
import sys
import argparse
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# MVTec所有类别
MVTEC_ALL_CLASSES = [
    'carpet', 'grid', 'leather', 'tile', 'wood',
    'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
    'pill', 'screw', 'toothbrush', 'transistor', 'zipper'
]

# VisA所有类别
VISA_ALL_CLASSES = [
    'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
    'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3',
    'pcb4', 'pipe_fryum'
]

# 默认配置
DEFAULT_N_PRO = 1
DEFAULT_N_PRO_AB = 4
DEFAULT_SEED = 111
DEFAULT_EPOCH = 100
DEFAULT_LR = 0.002
DEFAULT_MAX_WORKERS = 2
DEFAULT_RESULT_DIR = 'result/prompt2'


def train_class(dataset, cls, k_shot, args):
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
        '--seed', str(args.seed),
        '--n_pro', str(args.n_pro),
        '--n_pro_ab', str(args.n_pro_ab),
        '--Epoch', str(args.epoch),
        '--lr', str(args.lr),
        '--root-dir', args.output
    ]
    
    start_time = time.time()
    
    try:
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
    parser = argparse.ArgumentParser(description='Prompt2全类别并行训练')
    
    # 必需参数
    parser.add_argument('--k-shot', type=int, required=True,
                        help='K-shot数量 (1, 2, 4)')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['mvtec', 'visa', 'all'],
                        help='数据集选择: mvtec, visa, 或 all')
    
    # 可选参数
    parser.add_argument('--n_pro', type=int, default=DEFAULT_N_PRO,
                        help=f'正常原型数量 (默认: {DEFAULT_N_PRO})')
    parser.add_argument('--n_pro_ab', type=int, default=DEFAULT_N_PRO_AB,
                        help=f'异常原型数量 (默认: {DEFAULT_N_PRO_AB})')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help=f'随机种子 (默认: {DEFAULT_SEED})')
    parser.add_argument('--epoch', type=int, default=DEFAULT_EPOCH,
                        help=f'训练轮数 (默认: {DEFAULT_EPOCH})')
    parser.add_argument('--lr', type=float, default=DEFAULT_LR,
                        help=f'学习率 (默认: {DEFAULT_LR})')
    parser.add_argument('--workers', type=int, default=DEFAULT_MAX_WORKERS,
                        help=f'并行进程数 (默认: {DEFAULT_MAX_WORKERS})')
    parser.add_argument('--output', type=str, default=DEFAULT_RESULT_DIR,
                        help=f'输出目录 (默认: {DEFAULT_RESULT_DIR})')
    
    args = parser.parse_args()
    
    # 确定要训练的类别
    if args.dataset == 'mvtec':
        classes = [(args.dataset, cls) for cls in MVTEC_ALL_CLASSES]
        dataset_name = 'MVTec-AD'
    elif args.dataset == 'visa':
        classes = [(args.dataset, cls) for cls in VISA_ALL_CLASSES]
        dataset_name = 'VisA'
    else:  # all
        classes = [(d, c) for d in ['mvtec', 'visa'] 
                   for c in (MVTEC_ALL_CLASSES if d == 'mvtec' else VISA_ALL_CLASSES)]
        dataset_name = 'MVTec-AD + VisA'
    
    print(f"\n{'='*80}", flush=True)
    print(f"Prompt2 全类别并行训练", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"数据集: {dataset_name}", flush=True)
    print(f"类别数: {len(classes)}", flush=True)
    print(f"K-shot: {args.k_shot}", flush=True)
    print(f"配置: n_pro={args.n_pro}, n_pro_ab={args.n_pro_ab}", flush=True)
    print(f"并行度: {args.workers} 进程", flush=True)
    print(f"训练轮数: {args.epoch} epochs", flush=True)
    print(f"学习率: {args.lr}", flush=True)
    print(f"输出目录: {args.output}", flush=True)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"{'='*80}\n", flush=True)
    sys.stdout.flush()
    
    # 创建结果目录
    os.makedirs(args.output, exist_ok=True)
    
    results = []
    total_start = time.time()
    
    # 并行执行
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # 提交所有任务
        future_to_task = {
            executor.submit(train_class, dataset, cls, args.k_shot, args): (dataset, cls)
            for dataset, cls in classes
        }
        
        # 收集结果
        completed = 0
        total = len(classes)
        for future in as_completed(future_to_task):
            dataset, cls = future_to_task[future]
            completed += 1
            try:
                result = future.result()
                results.append(result)
                status = '✅' if result['success'] else '❌'
                print(f"\n[{completed}/{total}] {dataset}-{cls}: {status}", flush=True)
            except Exception as exc:
                print(f"\n[{completed}/{total}] {dataset}-{cls}: ❌ 异常 - {exc}", flush=True)
                results.append({
                    'dataset': dataset,
                    'class': cls,
                    'k': args.k_shot,
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
    print(f"总耗时: {total_time/60:.1f}分钟 (平均: {total_time/total_count/60:.1f}分钟/类别)", flush=True)
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    
    # 打印详细结果
    print(f"\n详细结果:", flush=True)
    
    # 按数据集分组
    mvtec_results = [r for r in results if r['dataset'] == 'mvtec']
    visa_results = [r for r in results if r['dataset'] == 'visa']
    
    if mvtec_results:
        print(f"\nMVTec-AD ({len(mvtec_results)}类):", flush=True)
        mvtec_success = sum(1 for r in mvtec_results if r['success'])
        print(f"  成功率: {mvtec_success}/{len(mvtec_results)}", flush=True)
        for r in sorted(mvtec_results, key=lambda x: x['class']):
            status = "✅" if r['success'] else "❌"
            print(f"    {status} {r['class']}: {r['time']/60:.1f}分钟", flush=True)
    
    if visa_results:
        print(f"\nVisA ({len(visa_results)}类):", flush=True)
        visa_success = sum(1 for r in visa_results if r['success'])
        print(f"  成功率: {visa_success}/{len(visa_results)}", flush=True)
        for r in sorted(visa_results, key=lambda x: x['class']):
            status = "✅" if r['success'] else "❌"
            print(f"    {status} {r['class']}: {r['time']/60:.1f}分钟", flush=True)
    
    # 保存执行记录
    import csv
    dataset_suffix = args.dataset if args.dataset != 'all' else 'all'
    csv_path = f'train_prompt2_{dataset_suffix}_k{args.k_shot}_execution.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'class', 'k', 'success', 'time'])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n执行记录已保存: {csv_path}", flush=True)
    sys.stdout.flush()


if __name__ == '__main__':
    main()
