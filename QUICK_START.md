# PromptAD 批量实验快速指南

## 🚀 快速开始

### 1. 运行完整实验（推荐在后台运行）

```bash
# 激活环境
conda activate prompt_ad

# 后台运行
nohup ./run_all_experiments.sh > run_output.log 2>&1 &

# 记录进程 ID
echo $! > run_experiments.pid
```

### 2. 运行测试版（验证脚本）

```bash
# 先运行测试版，验证逻辑正确
./run_test_experiments.sh

# 查看测试日志
tail -f result/backbone1/logs/run_test_experiments.log
```

### 3. 监控进度

```bash
# 实时监控（每10秒刷新）
watch -n 10 ./check_progress.sh

# 或手动查看
./check_progress.sh
```

---

## 📊 重要命令

### 查看日志

```bash
# 主日志
tail -f result/backbone1/logs/run_all_experiments.log

# 特定任务
tail -f result/backbone1/logs/cls_mvtec_bottle_k2_qq_residual_gpu1.log

# 后台输出
tail -f run_output.log
```

### GPU 监控

```bash
# 实时监控
watch -n 1 nvidia-smi

# 查看各 GPU 任务数
for gpu in 1 2 3; do
    echo "GPU $gpu: $(pgrep -f "train_.*\.py.*--gpu-id ${gpu}" | wc -l) 个任务"
done
```

### 停止实验

```bash
# 优雅停止（完成当前任务后停止）
kill $(cat run_experiments.pid)

# 强制停止所有训练
pkill -f "train_cls.py"
pkill -f "train_seg.py"

# 清理锁文件
rm -rf result/backbone1/locks/*.lock
```

---

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| `run_all_experiments.sh` | **主脚本**：运行所有实验（2268个任务） |
| `run_test_experiments.sh` | **测试脚本**：运行少量任务验证逻辑 |
| `check_progress.sh` | **监控脚本**：查看实验进度 |
| `RUN_EXPERIMENTS_GUIDE.md` | **详细文档**：完整使用说明 |

---

## ⚙️ 配置概览

### 实验配置（7个）
- `original`：原始 V2V 架构
- `qq_residual`：QQ attention + 残差
- `kk_residual`：KK attention + 残差
- `vv_residual`：VV attention + 残差
- `qq_no_residual`：QQ attention 无残差
- `kk_no_residual`：KK attention 无残差
- `vv_no_residual`：VV attention 无残差

### 数据集
- **MVTec**：15个类别，3个 k-shot (1,2,4)
- **VisA**：12个类别，3个 k-shot (1,2,4)

### 任务
- **CLS**：图像级分类
- **SEG**：像素级分割

### 并行控制
- MVTec k=1: 6 并行
- MVTec k=2: 4 并行
- MVTec k=4: 3 并行
- VisA: 2 并行

---

## 📈 总任务量

```
7个配置 × 2个数据集 × 2个任务 × (15+12)个类别 × 3个k-shot = 2,268 个任务
```

预计耗时：**~4天**（3个 RTX 3090 并行）

---

## 🔧 常见问题

### Q: 如何查看某个配置的所有结果？

```bash
grep "qq_residual" result/backbone1/logs/run_all_experiments.log | grep SUCCESS
```

### Q: 如何重新运行失败的任务？

失败的任务不会生成 checkpoint，脚本会自动重新运行。直接再次执行：
```bash
./run_all_experiments.sh
```

### Q: GPU 内存不足怎么办？

编辑 `run_all_experiments.sh`：
```bash
BATCH_SIZE=4  # 从 8 改为 4
```

### Q: 只想运行某个数据集？

注释掉不需要的部分：
```bash
# 在 run_all_experiments.sh 中注释掉 VisA 部分
# # ---- VisA CLS ----
# # echo "[VisA CLS]" | tee -a "$MAIN_LOG"
# # ...
```

---

## 🎯 结果目录结构

```
result/backbone1/
├── logs/
│   ├── run_all_experiments.log       # 主日志
│   └── *_*_*_*.log                   # 各任务日志
├── mvtec/
│   ├── k_1/
│   │   ├── checkpoint/               # 模型检查点
│   │   └── csv_results/              # CSV 结果
│   ├── k_2/
│   └── k_4/
└── visa/
    ├── k_1/
    ├── k_2/
    └── k_4/
```

---

## 📞 获取帮助

详细文档请查看：`RUN_EXPERIMENTS_GUIDE.md`

---

**祝实验顺利！** 🎉
