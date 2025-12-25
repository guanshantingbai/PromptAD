#!/usr/bin/env python3
"""
策略2：快速重训关键类别

自动训练受bug影响最严重的类别
"""

import subprocess
import time
from datetime import datetime

# 配置
DATASET = "mvtec"
K_SHOTS = [1, 2, 4]
ROOT_DIR = "result/prompt1_fixed"

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
print("策略2：重训关键类别（修复后）")
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
print(f"  结果目录: {ROOT_DIR}")
print()

print("关键类别及原因：")
for cls_name, reason in KEY_CLASSES:
    print(f"  - {cls_name:12s}: {reason}")
print()

total_tasks = len(KEY_CLASSES) * len(K_SHOTS)
print(f"即将开始训练 {total_tasks} 个任务...")
print(f"预计时间: 约 {total_tasks * 5} 分钟 (~{total_tasks * 5 / 60:.1f} 小时)")
print()

response = input("确认开始训练? (y/n): ")
if response.lower() != 'y':
    print("已取消")
    exit(0)

# 开始训练
start_time = time.time()
print()
print("=" * 80)
print(f"[{datetime.now().strftime('%H:%M:%S')}] 开始重训...")
print("=" * 80)
print()

results = []
task_num = 0

for cls_name, reason in KEY_CLASSES:
    for k_shot in K_SHOTS:
        task_num += 1
        print()
        print("=" * 80)
        print(f"任务 {task_num}/{total_tasks}: {cls_name} k={k_shot}")
        print("=" * 80)
        
        task_start = time.time()
        
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
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            task_time = time.time() - task_start
            status = "✅ 完成"
            print(f"{status} - 耗时: {task_time/60:.1f}分钟")
            results.append((cls_name, k_shot, "成功", task_time))
        except subprocess.CalledProcessError as e:
            task_time = time.time() - task_start
            status = "❌ 失败"
            print(f"{status}")
            print(f"错误: {e.stderr[-500:]}")  # 只显示最后500字符
            results.append((cls_name, k_shot, "失败", task_time))
        
        # 显示进度
        elapsed = time.time() - start_time
        avg_time = elapsed / task_num
        remaining = (total_tasks - task_num) * avg_time
        print(f"进度: {task_num}/{total_tasks} | 已用时: {elapsed/60:.1f}分钟 | 预计剩余: {remaining/60:.1f}分钟")

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
print(f"总耗时: {total_time/3600:.2f}小时 ({total_time/60:.1f}分钟)")
print(f"结果目录: {ROOT_DIR}")
print()

print("详细结果：")
print(f"{'类别':<12s} {'K':>3s} {'状态':>6s} {'耗时(分钟)':>12s}")
print("-" * 40)
for cls_name, k_shot, status, task_time in results:
    print(f"{cls_name:<12s} {k_shot:>3d} {status:>6s} {task_time/60:>11.1f}")

print()
print("下一步：")
print("1. 验证重训后的性能:")
print(f"   # 修改 RESULT_DIR='{ROOT_DIR}'")
print("   python verify_checkpoint_semantic.py")
print()
print("2. 对比重训前后:")
print("   python compare_retraining_results.py")
print()
