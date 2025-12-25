#!/usr/bin/env python3
"""
阶段2：MVTec全量训练 - 15个类别 × CLS+SEG × k=1,2,4
"""

import subprocess
import time
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置
DATASET = 'mvtec'
K_SHOTS = [1, 2, 4]
TASKS = ['cls', 'seg']
MAX_WORKERS = 2
OMP_NUM_THREADS = 3
ROOT_DIR = 'result/prompt1_fixed_full'

# MVTec全部15个类别
MVTEC_CLASSES = [
    'carpet', 'grid', 'leather', 'tile', 'wood',
    'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
    'pill', 'screw', 'toothbrush', 'transistor', 'zipper'
]

def train_single_task(cls_name, k_shot, task_type, task_id, total_tasks):
    """训练单个任务"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始任务 {task_id}/{total_tasks}: {cls_name} k={k_shot} ({task_type.upper()})")
    
    task_start = time.time()
    
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(OMP_NUM_THREADS)
    env['MKL_NUM_THREADS'] = str(OMP_NUM_THREADS)
    env['NUMEXPR_NUM_THREADS'] = str(OMP_NUM_THREADS)
    
    script = f"train_{task_type}.py"
    cmd = [
        "python", script,
        "--dataset", DATASET,
        "--class_name", cls_name,
        "--k-shot", str(k_shot),
        "--n_pro", "3",
        "--n_pro_ab", "4",
        "--Epoch", "100",
        "--lr", "0.002",
        "--root-dir", ROOT_DIR,
        "--vis", "False",
    ]
    
    try:
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True,
            env=env,
            timeout=600  # 10分钟超时
        )
        
        task_end = time.time()
        duration = (task_end - task_start) / 60
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 完成: {cls_name} k={k_shot} ({task_type.upper()}) (耗时: {duration:.1f}分钟)")
        
        return {
            'class': cls_name,
            'k': k_shot,
            'task': task_type,
            'status': 'success',
            'duration': duration
        }
        
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        task_end = time.time()
        duration = (task_end - task_start) / 60
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ 失败: {cls_name} k={k_shot} ({task_type.upper()}) (耗时: {duration:.1f}分钟)")
        
        error_msg = ""
        if isinstance(e, subprocess.CalledProcessError):
            error_msg = e.stderr[:200] if e.stderr else 'Unknown error'
        else:
            error_msg = "Timeout (>10min)"
        
        print(f"    错误: {error_msg}")
        
        return {
            'class': cls_name,
            'k': k_shot,
            'task': task_type,
            'status': 'failed',
            'duration': duration,
            'error': error_msg
        }

def main():
    print("="*80)
    print("阶段2：MVTec全量训练")
    print("="*80)
    
    print(f"\n训练配置：")
    print(f"  数据集: {DATASET}")
    print(f"  类别数: {len(MVTEC_CLASSES)}")
    print(f"  任务类型: {TASKS}")
    print(f"  K值: {K_SHOTS}")
    print(f"  总任务数: {len(MVTEC_CLASSES) * len(TASKS) * len(K_SHOTS)}")
    print(f"  并行数: {MAX_WORKERS}")
    print(f"  结果目录: {ROOT_DIR}")
    
    print(f"\nMVTec类别列表：")
    for i, cls_name in enumerate(MVTEC_CLASSES, 1):
        print(f"  {i:2d}. {cls_name}")
    
    total_tasks = len(MVTEC_CLASSES) * len(TASKS) * len(K_SHOTS)
    estimated_time = total_tasks * 2.5 / MAX_WORKERS
    print(f"\n即将开始训练 {total_tasks} 个任务...")
    print(f"预计时间: 约 {estimated_time:.0f} 分钟 (~{estimated_time / 60:.1f} 小时)")
    print()
    
    response = input("确认开始训练? (y/n): ")
    if response.lower() != 'y':
        print("已取消")
        return
    
    os.makedirs(ROOT_DIR, exist_ok=True)
    
    # 生成任务列表
    tasks = []
    task_id = 1
    for cls_name in MVTEC_CLASSES:
        for task_type in TASKS:
            for k_shot in K_SHOTS:
                tasks.append((cls_name, k_shot, task_type, task_id, total_tasks))
                task_id += 1
    
    print("="*80)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始并行训练...")
    print("="*80)
    print()
    
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {
            executor.submit(train_single_task, *task): task 
            for task in tasks
        }
        
        completed = 0
        for future in as_completed(future_to_task):
            result = future.result()
            results.append(result)
            completed += 1
            
            elapsed = (time.time() - start_time) / 60
            remaining = (elapsed / completed) * (total_tasks - completed) if completed > 0 else 0
            print(f"    进度: {completed}/{total_tasks} | 已用时: {elapsed:.1f}分钟 | 预计剩余: {remaining:.1f}分钟")
    
    end_time = time.time()
    total_duration = (end_time - start_time) / 60
    
    # 统计
    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = sum(1 for r in results if r['status'] == 'failed')
    
    print()
    print("="*80)
    print("MVTec全量训练完成！")
    print("="*80)
    print()
    print(f"总任务数: {total_tasks}")
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")
    print(f"成功率: {success_count/total_tasks*100:.1f}%")
    print(f"总耗时: {total_duration:.1f}分钟 ({total_duration/60:.2f}小时)")
    print(f"结果目录: {ROOT_DIR}")
    
    # 按类别汇总
    print(f"\n按类别汇总：")
    print(f"{'类别':<15} {'CLS成功':<10} {'SEG成功':<10} {'总计':<10}")
    print("-"*50)
    
    for cls_name in MVTEC_CLASSES:
        cls_results = [r for r in results if r['class'] == cls_name]
        cls_success = sum(1 for r in cls_results if r['status'] == 'success')
        seg_success = sum(1 for r in cls_results if r['status'] == 'success' and r['task'] == 'seg')
        total_cls = len(cls_results)
        
        print(f"{cls_name:<15} {cls_success - seg_success}/{len(K_SHOTS):<10} {seg_success}/{len(K_SHOTS):<10} {cls_success}/{total_cls}")
    
    if failed_count > 0:
        print(f"\n⚠️  有 {failed_count} 个任务失败")
        print(f"\n失败任务列表：")
        for result in results:
            if result['status'] == 'failed':
                print(f"  • {result['class']} k={result['k']} ({result['task'].upper()})")
    
    print("\n" + "="*80)
    print("下一步：")
    print("  1. 检查结果: ls -lh " + ROOT_DIR)
    print("  2. 对比性能: python compare_with_baseline.py --root-dir " + ROOT_DIR)
    print("="*80)

if __name__ == '__main__':
    main()
