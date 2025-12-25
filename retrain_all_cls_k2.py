#!/usr/bin/env python3
"""
CLS任务 k=2 全类别验证
验证多原型语义分支改进在k=2设置下的泛化能力（已在6类上验证+1.57%提升）
"""

import subprocess
import time
import os
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置
K_SHOT = 2  # 验证k=2
MAX_WORKERS = 2  # 并行任务数
OMP_NUM_THREADS = 3  # 每个任务的线程数
ROOT_DIR = 'result/prompt1_fixed'

# MVTec 15类
MVTEC_CLASSES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]

# VisA 12类
VISA_CLASSES = [
    'candle', 'capsules', 'cashew', 'chewinggum',
    'fryum', 'macaroni1', 'macaroni2',
    'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'
]

def train_single_task(dataset, cls_name, task_id, total_tasks):
    """训练单个CLS k=2任务"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始任务 {task_id}/{total_tasks}: {dataset}/{cls_name} k=2 (CLS)")
    
    task_start = time.time()
    
    # 设置环境变量限制线程数
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(OMP_NUM_THREADS)
    env['MKL_NUM_THREADS'] = str(OMP_NUM_THREADS)
    env['NUMEXPR_NUM_THREADS'] = str(OMP_NUM_THREADS)
    
    cmd = [
        "python", "train_cls.py",
        "--dataset", dataset,
        "--class_name", cls_name,
        "--k-shot", str(K_SHOT),
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
            timeout=900  # 15分钟超时
        )
        
        task_end = time.time()
        duration = (task_end - task_start) / 60
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 完成: {dataset}/{cls_name} k=2 (CLS) (耗时: {duration:.1f}分钟)")
        
        return {
            'dataset': dataset,
            'class': cls_name,
            'k': K_SHOT,
            'task': 'CLS',
            'status': 'success',
            'duration': duration,
            'output': result.stdout[-500:] if result.stdout else ''
        }
        
    except subprocess.TimeoutExpired:
        task_end = time.time()
        duration = (task_end - task_start) / 60
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⏱️ 超时: {dataset}/{cls_name} k=2 (CLS) (耗时: {duration:.1f}分钟)")
        return {
            'dataset': dataset,
            'class': cls_name,
            'k': K_SHOT,
            'task': 'CLS',
            'status': 'timeout',
            'duration': duration,
            'error': 'Timeout after 15 minutes'
        }
        
    except subprocess.CalledProcessError as e:
        task_end = time.time()
        duration = (task_end - task_start) / 60
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ 失败: {dataset}/{cls_name} k=2 (CLS) (耗时: {duration:.1f}分钟)")
        print(f"    错误: {e.stderr[:200] if e.stderr else 'Unknown error'}")
        
        return {
            'dataset': dataset,
            'class': cls_name,
            'k': K_SHOT,
            'task': 'CLS',
            'status': 'failed',
            'duration': duration,
            'error': e.stderr[:500] if e.stderr else str(e)
        }

def main():
    print("="*80)
    print("CLS任务 k=2 全类别验证")
    print("="*80)
    
    print(f"\n验证目标：")
    print(f"  基于6类验证结果(+1.57%提升)，验证k=2在全类别上的泛化能力")
    print(f"  k=2提供足够样本，多原型方法可有效建模正常/异常分布")
    
    print(f"\n训练配置：")
    print(f"  数据集: MVTec (15类) + VisA (12类)")
    print(f"  任务: 仅CLS (分类)")
    print(f"  K值: 2 (已验证有效的few-shot设置)")
    print(f"  总任务数: {len(MVTEC_CLASSES) + len(VISA_CLASSES)}")
    print(f"  并行数: {MAX_WORKERS}")
    print(f"  每任务线程数: {OMP_NUM_THREADS}")
    print(f"  结果目录: {ROOT_DIR}")
    
    print(f"\nMVTec 15类:")
    for cls_name in MVTEC_CLASSES:
        print(f"  - {cls_name}")
    
    print(f"\nVisA 12类:")
    for cls_name in VISA_CLASSES:
        print(f"  - {cls_name}")
    
    # 构建任务列表
    tasks = []
    task_id = 1
    total_tasks = len(MVTEC_CLASSES) + len(VISA_CLASSES)
    
    for cls_name in MVTEC_CLASSES:
        tasks.append(('mvtec', cls_name, task_id, total_tasks))
        task_id += 1
    
    for cls_name in VISA_CLASSES:
        tasks.append(('visa', cls_name, task_id, total_tasks))
        task_id += 1
    
    # 估算时间
    avg_time_per_task = 2.5  # 每个任务约2.5分钟
    estimated_time = (total_tasks / MAX_WORKERS) * avg_time_per_task
    
    print(f"\n即将开始训练 {total_tasks} 个CLS k=2任务...")
    print(f"预计时间: 约 {estimated_time:.0f} 分钟 (~{estimated_time/60:.1f} 小时)")
    print(f"（{MAX_WORKERS}个任务并行，相比串行节省约{(1-1/MAX_WORKERS)*100:.0f}%时间）")
    print("")
    print("="*80)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始并行训练...")
    print("="*80)
    
    start_time = time.time()
    results = []
    completed = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {executor.submit(train_single_task, *task): task for task in tasks}
        
        for future in as_completed(future_to_task):
            result = future.result()
            results.append(result)
            completed += 1
            
            # 显示进度
            elapsed = (time.time() - start_time) / 60
            if completed < total_tasks:
                estimated_remaining = (elapsed / completed) * (total_tasks - completed)
                print(f"    进度: {completed}/{total_tasks} | 已用时: {elapsed:.1f}分钟 | 预计剩余: {estimated_remaining:.1f}分钟")
    
    end_time = time.time()
    total_duration = (end_time - start_time) / 60
    
    # 统计结果
    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = sum(1 for r in results if r['status'] == 'failed')
    timeout_count = sum(1 for r in results if r['status'] == 'timeout')
    
    print("\n" + "="*80)
    print("CLS k=2 全类别验证完成！")
    print("="*80)
    
    print(f"\n总任务数: {total_tasks}")
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")
    print(f"超时: {timeout_count}")
    print(f"总耗时: {total_duration:.1f}分钟 ({total_duration/60:.2f}小时)")
    print(f"结果目录: {ROOT_DIR}")
    
    # 保存结果到CSV
    df = pd.DataFrame(results)
    df = df.sort_values(['dataset', 'class'])
    
    csv_path = f"{ROOT_DIR}/all_cls_k2_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n结果已保存到: {csv_path}")
    
    # 显示详细结果表格
    print(f"\n详细结果：")
    print(f"{'数据集':<10} {'类别':<15} {'状态':<10} {'耗时(分钟)':<12}")
    print("-" * 50)
    for r in results:
        status_icon = '✅' if r['status'] == 'success' else ('⏱️' if r['status'] == 'timeout' else '❌')
        print(f"{r['dataset']:<10} {r['class']:<15} {status_icon:<10} {r['duration']:<12.1f}")
    
    # 显示失败任务详情
    if failed_count > 0 or timeout_count > 0:
        print(f"\n{'⚠️  有 ' + str(failed_count + timeout_count) + ' 个任务未成功，请检查日志'}")
        
        print(f"\n失败/超时任务详情：")
        for r in results:
            if r['status'] != 'success':
                print(f"\n{r['dataset']}/{r['class']} k=2:")
                print(f"  {r.get('error', 'Unknown error')[:300]}")
    
    print("\n" + "="*80)
    print(f"下一步：运行对比脚本查看与baseline和k=1的对比")
    print("="*80)

if __name__ == '__main__':
    main()
