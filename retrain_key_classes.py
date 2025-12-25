#!/usr/bin/env python3
"""
策略2：快速重训关键类别（并行版本）

支持多任务并行训练，加速重训过程
"""

import subprocess
import time
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置
DATASET = "mvtec"
K_SHOTS = [1, 2, 4]
ROOT_DIR = "result/prompt1_fixed"
MAX_WORKERS = 2  # 并行任务数
OMP_NUM_THREADS = 3  # 每个任务的线程数

# 关键类别：语义提升大但融合下降
KEY_CLASSES = [
    ("screw", "语义+13.15% → 融合-5.66%"),
    ("toothbrush", "语义+19.86% → 融合-8.62%"),
    ("hazelnut", "语义+11.03% → 融合-8.52%"),
    ("capsule", "语义+6.96% → 融合+4.59%"),
    ("pill", "语义+0.62% → 融合-7.42%"),
    ("metal_nut", "语义+3.15% → 融合-5.47%"),
]

print("=" * 80)
print("策略2：重训关键类别（修复后 - 并行版本）")
print("=" * 80)
print()
print("修复内容：train_cls.py 使用纯语义分支选择最佳模型")
print("重训原因：之前用融合分数选择模型，导致保存的不是语义最佳checkpoint")
print()
print("重训配置：")
print(f"  数据集: {DATASET}")
print(f"  类别数: {len(KEY_CLASSES)}")
print(f"  K值: {K_SHOTS}")
print(f"  总任务数: {len(KEY_CLASSES) * len(K_SHOTS)}")
print(f"  并行数: {MAX_WORKERS}")
print(f"  每任务线程数: {OMP_NUM_THREADS}")
print(f"  结果目录: {ROOT_DIR}")
print()

print("关键类别及原因：")
for cls_name, reason in KEY_CLASSES:
    print(f"  - {cls_name:12s}: {reason}")
print()

total_tasks = len(KEY_CLASSES) * len(K_SHOTS)
# 并行后预计时间减少
estimated_time = total_tasks * 5 / MAX_WORKERS
print(f"即将开始训练 {total_tasks} 个任务...")
print(f"预计时间: 约 {estimated_time:.0f} 分钟 (~{estimated_time / 60:.1f} 小时)")
print(f"（{MAX_WORKERS}个任务并行，相比串行节省约{(1 - 1/MAX_WORKERS)*100:.0f}%时间）")
print()

response = input("确认开始训练? (y/n): ")
if response.lower() != 'y':
    print("已取消")
    exit(0)

# 创建结果目录
os.makedirs(ROOT_DIR, exist_ok=True)

def train_single_task(cls_name, k_shot, task_id, total_tasks):
    """训练单个任务"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始任务 {task_id}/{total_tasks}: {cls_name} k={k_shot}")
    
    task_start = time.time()
    
    # 设置环境变量限制线程数
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(OMP_NUM_THREADS)
    env['MKL_NUM_THREADS'] = str(OMP_NUM_THREADS)
    env['NUMEXPR_NUM_THREADS'] = str(OMP_NUM_THREADS)
    
    cmd = [
        "python", "train_cls.py",
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
            env=env  # 使用修改后的环境变量
        )
        task_time = time.time() - task_start
        status = "✅"
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {status} 完成: {cls_name} k={k_shot} (耗时: {task_time/60:.1f}分钟)")
        return (cls_name, k_shot, "成功", task_time)
    except subprocess.CalledProcessError as e:
        task_time = time.time() - task_start
        status = "❌"
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {status} 失败: {cls_name} k={k_shot}")
        # 保存错误日志
        error_log = f"{ROOT_DIR}/error_{cls_name}_k{k_shot}.log"
        with open(error_log, 'w') as f:
            f.write(e.stderr)
        print(f"    错误日志已保存: {error_log}")
        return (cls_name, k_shot, "失败", task_time)

# 开始训练
start_time = time.time()
print()
print("=" * 80)
print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始并行重训...")
print("=" * 80)
print()

# 准备所有任务
tasks = []
task_id = 0
for cls_name, reason in KEY_CLASSES:
    for k_shot in K_SHOTS:
        task_id += 1
        tasks.append((cls_name, k_shot, task_id))

results = []
completed = 0

# 使用线程池并行执行
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # 提交所有任务
    future_to_task = {
        executor.submit(train_single_task, cls_name, k_shot, task_id, total_tasks): (cls_name, k_shot)
        for cls_name, k_shot, task_id in tasks
    }
    
    # 收集结果
    for future in as_completed(future_to_task):
        cls_name, k_shot = future_to_task[future]
        try:
            result = future.result()
            results.append(result)
            completed += 1
            
            # 显示进度
            elapsed = time.time() - start_time
            avg_time = elapsed / completed
            remaining_tasks = total_tasks - completed
            remaining_time = remaining_tasks * avg_time / MAX_WORKERS  # 考虑并行
            
            print(f"    进度: {completed}/{total_tasks} | 已用时: {elapsed/60:.1f}分钟 | 预计剩余: {remaining_time/60:.1f}分钟")
        except Exception as e:
            print(f"    任务异常: {cls_name} k={k_shot}: {e}")
            results.append((cls_name, k_shot, "异常", 0))

# 总结
end_time = time.time()
total_time = end_time - start_time

print()
print("=" * 80)
print("重训完成！")
print("=" * 80)
print()
print(f"总任务数: {total_tasks}")
print(f"成功: {sum(1 for r in results if r[2] == '成功')}")
print(f"失败: {sum(1 for r in results if r[2] == '失败')}")
print(f"总耗时: {total_time/60:.1f}分钟 ({total_time/3600:.2f}小时)")
print(f"结果目录: {ROOT_DIR}")
print()

# 按类别排序显示结果
results_sorted = sorted(results, key=lambda x: (x[0], x[1]))

print("详细结果：")
print(f"{'类别':<12s} {'K':>3s} {'状态':>6s} {'耗时(分钟)':>12s}")
print("-" * 40)
for cls_name, k_shot, status, task_time in results_sorted:
    status_mark = "✅" if status == "成功" else "❌"
    print(f"{cls_name:<12s} {k_shot:>3d} {status_mark:>6s} {task_time/60:>11.1f}")

print()
print("下一步：")
print("1. 验证重训后的性能:")
print(f"   # 修改 verify_checkpoint_semantic.py 中 RESULT_DIR='{ROOT_DIR}'")
print("   python verify_checkpoint_semantic.py")
print()
print("2. 测试单个类别:")
print(f"   python test_cls.py --dataset mvtec --class_name screw --k-shot 2 \\")
print(f"       --root-dir {ROOT_DIR} --semantic-only True --vis False")
print()
