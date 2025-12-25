#!/usr/bin/env python3
"""
阶段1：SEG任务验证 - 重训6个关键类别的分割任务
验证修复后的代码在SEG任务上是否同样有效
"""

import subprocess
import time
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置
DATASET = 'mvtec'
K_SHOTS = [1, 2, 4]
MAX_WORKERS = 2  # 并行任务数
OMP_NUM_THREADS = 3  # 每个任务的线程数
ROOT_DIR = 'result/prompt1_fixed_seg'

# 6个关键类别（与CLS相同）
KEY_CLASSES = [
    'screw',       # 语义+13.15% → 融合-5.66%
    'toothbrush',  # 语义+19.86% → 融合-8.62%
    'hazelnut',    # 语义+11.03% → 融合-8.52%
    'capsule',     # 语义+6.96% → 融合+4.59%
    'pill',        # 语义+0.62% → 融合-7.42%
    'metal_nut',   # 语义+3.15% → 融合-5.47%
]

def train_single_task(cls_name, k_shot, task_id, total_tasks):
    """训练单个SEG任务"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始任务 {task_id}/{total_tasks}: {cls_name} k={k_shot} (SEG)")
    
    task_start = time.time()
    
    # 设置环境变量限制线程数
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(OMP_NUM_THREADS)
    env['MKL_NUM_THREADS'] = str(OMP_NUM_THREADS)
    env['NUMEXPR_NUM_THREADS'] = str(OMP_NUM_THREADS)
    
    cmd = [
        "python", "train_seg.py",
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
            env=env
        )
        
        task_end = time.time()
        duration = (task_end - task_start) / 60
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 完成: {cls_name} k={k_shot} (SEG) (耗时: {duration:.1f}分钟)")
        
        return {
            'class': cls_name,
            'k': k_shot,
            'status': 'success',
            'duration': duration,
            'output': result.stdout[-500:] if result.stdout else ''
        }
        
    except subprocess.CalledProcessError as e:
        task_end = time.time()
        duration = (task_end - task_start) / 60
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ 失败: {cls_name} k={k_shot} (SEG) (耗时: {duration:.1f}分钟)")
        print(f"    错误: {e.stderr[:200] if e.stderr else 'Unknown error'}")
        
        return {
            'class': cls_name,
            'k': k_shot,
            'status': 'failed',
            'duration': duration,
            'error': e.stderr[:500] if e.stderr else str(e)
        }

def main():
    print("="*80)
    print("阶段1：SEG任务验证 - 重训6个关键类别")
    print("="*80)
    
    print(f"\n修复内容：train_seg.py 使用纯语义分支选择最佳模型")
    print(f"验证目的：确认修复在SEG任务上同样有效")
    
    print(f"\n重训配置：")
    print(f"  数据集: {DATASET}")
    print(f"  任务: 分割 (SEG)")
    print(f"  类别数: {len(KEY_CLASSES)}")
    print(f"  K值: {K_SHOTS}")
    print(f"  总任务数: {len(KEY_CLASSES) * len(K_SHOTS)}")
    print(f"  并行数: {MAX_WORKERS}")
    print(f"  每任务线程数: {OMP_NUM_THREADS}")
    print(f"  结果目录: {ROOT_DIR}")
    
    print(f"\n关键类别:")
    for cls_name in KEY_CLASSES:
        print(f"  - {cls_name}")
    
    total_tasks = len(KEY_CLASSES) * len(K_SHOTS)
    estimated_time = total_tasks * 2.5 / MAX_WORKERS  # SEG稍慢，每任务约2.5分钟
    print(f"\n即将开始训练 {total_tasks} 个SEG任务...")
    print(f"预计时间: 约 {estimated_time:.0f} 分钟 (~{estimated_time / 60:.1f} 小时)")
    print(f"（{MAX_WORKERS}个任务并行，相比串行节省约{(1 - 1/MAX_WORKERS)*100:.0f}%时间）")
    print()
    
    response = input("确认开始训练? (y/n): ")
    if response.lower() != 'y':
        print("已取消")
        return
    
    # 创建结果目录
    os.makedirs(ROOT_DIR, exist_ok=True)
    
    # 生成任务列表
    tasks = []
    task_id = 1
    for cls_name in KEY_CLASSES:
        for k_shot in K_SHOTS:
            tasks.append((cls_name, k_shot, task_id, total_tasks))
            task_id += 1
    
    # 并行执行
    print("="*80)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始并行重训...")
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
    
    # 统计结果
    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = sum(1 for r in results if r['status'] == 'failed')
    
    print()
    print("="*80)
    print("SEG重训完成！")
    print("="*80)
    print()
    print(f"总任务数: {total_tasks}")
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")
    print(f"总耗时: {total_duration:.1f}分钟 ({total_duration/60:.2f}小时)")
    print(f"结果目录: {ROOT_DIR}")
    
    # 详细结果
    print(f"\n详细结果：")
    print(f"{'类别':<15} {'K':<5} {'状态':<10} {'耗时(分钟)':<12}")
    print("-"*50)
    
    for result in sorted(results, key=lambda x: (x['class'], x['k'])):
        status_symbol = "✅" if result['status'] == 'success' else "❌"
        print(f"{result['class']:<15} {result['k']:<5} {status_symbol:<10} {result['duration']:<12.1f}")
    
    if failed_count > 0:
        print(f"\n⚠️  有 {failed_count} 个任务失败，请检查日志")
        print(f"\n失败任务详情：")
        for result in results:
            if result['status'] == 'failed':
                print(f"\n{result['class']} k={result['k']}:")
                print(f"  {result['error'][:200]}")
    
    print("\n" + "="*80)
    print("下一步：运行 python compare_seg_results.py 查看对比结果")
    print("="*80)

if __name__ == '__main__':
    main()
