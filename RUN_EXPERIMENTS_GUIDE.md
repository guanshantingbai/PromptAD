# PromptAD 批量实验运行指南

## 脚本概述

`run_all_experiments.sh` 是一个全自动批量训练脚本，用于运行所有实验配置。

### 特性

✅ **7个实验配置**：original + 6个变体（qq/kk/vv × residual/no_residual）  
✅ **2个数据集**：MVTec (15类) + VisA (12类)  
✅ **2个任务**：分类 (cls) + 分割 (seg)  
✅ **3个 k-shot**：1, 2, 4  
✅ **3 GPU 固定分配**：GPU 1, 2, 3 轮询分配任务  
✅ **智能并行控制**：  
   - MVTec: k=1→6并行, k=2→4并行, k=4→3并行  
   - VisA: 一律2并行  
✅ **防重复执行**：检查 checkpoint，跳过已完成任务  
✅ **文件锁保护**：防止竞态条件和死锁  
✅ **nohup 安全**：支持长时间离线运行  

---

## 快速开始

### 1. 基本用法

```bash
# 直接运行（前台）
./run_all_experiments.sh

# 后台运行（推荐）
nohup ./run_all_experiments.sh > run_all_experiments_output.log 2>&1 &

# 查看进度
tail -f result/backbone1/logs/run_all_experiments.log

# 查看后台输出
tail -f run_all_experiments_output.log
```

### 2. 查看 GPU 使用情况

```bash
# 实时监控
watch -n 1 nvidia-smi

# 查看各 GPU 上的任务数
for gpu in 1 2 3; do
    echo "GPU $gpu: $(pgrep -f "train_.*\.py.*--gpu-id ${gpu}" | wc -l) 个任务"
done
```

### 3. 查看特定任务日志

```bash
# 进入日志目录
cd result/backbone1/logs

# 列出所有日志文件
ls -lht

# 查看特定任务
tail -f cls_mvtec_bottle_k2_qq_residual_gpu1.log
```

---

## 总任务量统计

| 项目 | 数量 | 说明 |
|------|------|------|
| 实验配置 | 7 | original + 6变体 |
| 数据集 | 2 | MVTec + VisA |
| 任务类型 | 2 | cls + seg |
| MVTec 类别 | 15 | bottle, cable, capsule, etc. |
| VisA 类别 | 12 | candle, cashew, pcb1, etc. |
| k-shot | 3 | 1, 2, 4 |
| **总任务数** | **7 × 2 × 2 × (15+12) × 3 = 2,268** | |

---

## 目录结构

```
result/backbone1/
├── logs/                              # 日志目录
│   ├── run_all_experiments.log       # 主日志
│   ├── cls_mvtec_bottle_k1_*.log     # 各任务日志
│   └── seg_visa_candle_k2_*.log
├── locks/                             # 锁文件目录
│   ├── gpu1_parallel.lock
│   ├── gpu2_parallel.lock
│   └── gpu3_parallel.lock
├── mvtec/                            # MVTec 结果
│   ├── k_1/
│   ├── k_2/
│   └── k_4/
└── visa/                             # VisA 结果
    ├── k_1/
    ├── k_2/
    └── k_4/
```

---

## 配置说明

### GPU 分配策略

脚本使用**轮询分配**策略：
- 任务按顺序分配到 GPU 1 → 2 → 3 → 1 → 2 → ...
- 每个 GPU 独立控制并行数
- 不依赖动态检测，避免负载不均

### 并行数配置

| 数据集 | k-shot | 并行数 | 说明 |
|--------|--------|--------|------|
| MVTec  | 1      | 6      | 训练数据少，速度快 |
| MVTec  | 2      | 4      | 中等并行 |
| MVTec  | 4      | 3      | 训练数据多，速度慢 |
| VisA   | 1/2/4  | 2      | 统一并行数 |

### Checkpoint 检查

路径格式：`result/backbone1/{dataset}/k_{k}/checkpoint/{task}_{exp_config}_{class}_check_point.pt`

如果 checkpoint 存在，任务会被跳过，输出 `[SKIP]`。

---

## 监控与调试

### 1. 实时监控整体进度

```bash
# 主日志
tail -f result/backbone1/logs/run_all_experiments.log

# 统计已完成/失败/运行中任务
grep "\[SUCCESS\]" result/backbone1/logs/run_all_experiments.log | wc -l
grep "\[FAILED\]" result/backbone1/logs/run_all_experiments.log | wc -l
grep "\[START\]" result/backbone1/logs/run_all_experiments.log | wc -l
```

### 2. 查看各 GPU 负载

```bash
#!/bin/bash
# 保存为 check_gpu_status.sh
for gpu in 1 2 3; do
    count=$(pgrep -f "train_.*\.py.*--gpu-id ${gpu}" | wc -l)
    echo "GPU $gpu: $count 个任务运行中"
    pgrep -af "train_.*\.py.*--gpu-id ${gpu}" | head -3
    echo "---"
done
```

### 3. 手动终止所有任务

```bash
# 谨慎操作！
pkill -f "train_cls.py"
pkill -f "train_seg.py"

# 清理锁文件
rm -rf result/backbone1/locks/*.lock
```

### 4. 重新运行失败任务

脚本会自动跳过已完成任务（checkpoint 存在）。直接重新运行即可：

```bash
./run_all_experiments.sh
```

---

## 预估时间

假设：
- 每个任务平均 30 分钟
- 3个GPU并行，平均每GPU运行4个任务

**总耗时** ≈ 2268 / (3 × 4) × 0.5h ≈ **95小时** ≈ **4天**

实际时间取决于：
- GPU 性能（RTX 3090）
- k-shot 大小（k=1 快，k=4 慢）
- 数据集大小
- 网络 I/O

---

## 故障排查

### 问题1：任务无法启动

**现象**：脚本运行但没有任务启动

**解决**：
```bash
# 检查锁文件
ls -l result/backbone1/locks/

# 删除僵尸锁
rm -f result/backbone1/locks/*.lock

# 检查 Python 环境
conda activate prompt_ad
which python
```

### 问题2：GPU 内存不足

**现象**：日志显示 CUDA OOM

**解决**：
```bash
# 减少并行数（编辑脚本）
# 将 MVTEC_CLS_PARALLEL 中的值减半

# 或减少 batch size
# 修改脚本中的 BATCH_SIZE=8 → BATCH_SIZE=4
```

### 问题3：checkpoint 路径错误

**现象**：任务重复运行

**解决**：
检查 `check_checkpoint` 函数中的路径格式是否与实际 `train_cls.py` 和 `train_seg.py` 保存的路径一致。

---

## 自定义配置

### 修改 GPU 列表

编辑脚本中的：
```bash
GPU_IDS=(1 2 3)  # 改为你的 GPU 编号
```

### 修改训练参数

编辑脚本中的：
```bash
EPOCH=100        # 训练轮数
BATCH_SIZE=8     # 批次大小
LR=0.002         # 学习率
N_CTX=4          # Prompt 上下文长度
```

### 只运行特定配置

编辑脚本中的：
```bash
EXP_CONFIGS=(
    "qq_residual"    # 只运行这一个配置
)
```

### 只运行特定数据集

注释掉不需要的部分：
```bash
# 注释掉 VisA CLS 部分
# echo "[VisA CLS]" | tee -a "$MAIN_LOG"
# ...
```

---

## 注意事项

⚠️ **重要提醒**：

1. **首次运行前**：确保 conda 环境 `prompt_ad` 已激活
2. **磁盘空间**：确保有足够空间存储 checkpoint 和日志（预计 ~50GB）
3. **网络连接**：如需下载预训练模型，确保网络畅通
4. **长时间运行**：建议在 tmux 或 screen 会话中运行
5. **定期检查**：每天查看一次日志，确保任务正常进行

---

## 示例输出

```
========================================
PromptAD 实验配置批量训练 (3 GPU)
开始时间: 2025-12-16 04:00:00
ROOT_DIR: result/backbone1
GPU_IDS: 1 2 3
实验配置数: 7
========================================

================================================
实验配置: original
================================================
[MVTec CLS]
  k=1, parallel=6
[START] cls mvtec/bottle k=1 config=original GPU=1
[START] cls mvtec/cable k=1 config=original GPU=2
[START] cls mvtec/capsule k=1 config=original GPU=3
...
[SUCCESS] cls mvtec/bottle k=1 config=original GPU=1
...
```

---

## 联系与支持

如有问题，请检查：
1. 主日志：`result/backbone1/logs/run_all_experiments.log`
2. 任务日志：`result/backbone1/logs/*_*.log`
3. GPU 状态：`nvidia-smi`
