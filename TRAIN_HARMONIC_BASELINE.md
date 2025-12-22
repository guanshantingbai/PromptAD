# 训练Harmonic Baseline的完整指南

## 目标
重新训练使用**harmonic融合**的PromptAD模型，确保训练和推理一致。

---

## 步骤1: 修改训练代码支持harmonic融合

当前训练代码（`train_cls.py`和`train_seg.py`）使用的是原始`PromptAD.model`，它硬编码了max融合。需要修改为支持可配置的融合策略。

### 选项A：修改原始model.py（推荐用于理解baseline）

**文件**: `PromptAD/model.py`

在`forward()`方法中，将硬编码的`torch.maximum`改为可配置：

```python
# 在__init__中添加融合模式参数
def __init__(self, ..., fusion_mode='max'):
    ...
    self.fusion_mode = fusion_mode

# 在forward()中使用配置的融合模式
def forward(self, images, task):
    visual_features = self.encode_image(images)
    
    if task == 'seg':
        textual_anomaly_map = self.calculate_textual_anomaly_score(visual_features, 'seg')
        visual_anomaly_map = self.calculate_visual_anomaly_score(visual_features)
        
        # 根据融合模式选择策略
        if self.fusion_mode == 'max':
            anomaly_map = torch.maximum(textual_anomaly_map, visual_anomaly_map)
        elif self.fusion_mode == 'harmonic':
            eps = 1e-8
            anomaly_map = 1.0 / (1.0/(textual_anomaly_map+eps) + 1.0/(visual_anomaly_map+eps))
        elif self.fusion_mode == 'mean':
            anomaly_map = 0.5 * (textual_anomaly_map + visual_anomaly_map)
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")
        
        # ... 后续处理
```

**需要修改的文件**:
1. `PromptAD/model.py` - 添加fusion_mode参数
2. `train_cls.py` - 传入fusion_mode='harmonic'
3. `train_seg.py` - 传入fusion_mode='harmonic'

### 选项B：创建新的训练脚本（推荐用于快速实验）

直接复用modular scoring框架，创建`train_cls_harmonic.py`：

```bash
cp train_cls.py train_cls_harmonic.py
cp train_seg.py train_seg_harmonic.py
```

在新文件中，将评估部分改为使用harmonic scorer。

---

## 步骤2: 创建训练脚本

### 方案1: 修改原始代码（如果选择选项A）

```bash
# 1. 备份原始代码
cp PromptAD/model.py PromptAD/model_original.py

# 2. 手动修改model.py添加fusion_mode参数（见上文）

# 3. 修改train_cls.py和train_seg.py
# 在模型初始化处添加 fusion_mode='harmonic'
```

### 方案2: 使用modular框架（如果选择选项B）

创建 `train_harmonic.sh`:

```bash
#!/bin/bash

DATASETS=("mvtec" "visa")
K_SHOTS=(1 2 4)
FUSION_MODE="harmonic"

# 训练所有配置
for DATASET in "${DATASETS[@]}"; do
    if [ "$DATASET" == "mvtec" ]; then
        CLASSES=(bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper)
    else
        CLASSES=(candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum)
    fi
    
    for CLASS in "${CLASSES[@]}"; do
        for K in "${K_SHOTS[@]}"; do
            echo "Training $DATASET/$CLASS k=$K with $FUSION_MODE fusion"
            
            # CLS任务
            OUTPUT_DIR="result/harmonic_baseline/${DATASET}/k_${K}/${CLASS}"
            mkdir -p $OUTPUT_DIR
            
            # 运行训练（这里需要修改train_cls.py支持fusion_mode参数）
            python train_cls.py \
                --dataset $DATASET \
                --data-path /path/to/data \
                --obj $CLASS \
                --shot $K \
                --epoch 100 \
                --fusion-mode $FUSION_MODE \
                --save-path "${OUTPUT_DIR}/cls_checkpoint.pth" \
                > "${OUTPUT_DIR}/cls_train.log" 2>&1
            
            # SEG任务
            python train_seg.py \
                --dataset $DATASET \
                --data-path /path/to/data \
                --obj $CLASS \
                --shot $K \
                --epoch 100 \
                --fusion-mode $FUSION_MODE \
                --save-path "${OUTPUT_DIR}/seg_checkpoint.pth" \
                > "${OUTPUT_DIR}/seg_train.log" 2>&1
        done
    done
done
```

---

## 步骤3: 修改代码支持fusion_mode参数

### 修改 `train_cls.py`

在argparse部分添加：

```python
parser.add_argument('--fusion-mode', type=str, default='max', 
                    choices=['max', 'harmonic', 'mean'],
                    help='Fusion strategy for training')
```

在模型初始化部分（约第150行）：

```python
# 原来：
model = PromptAD(...)

# 改为：
if args.fusion_mode == 'max':
    model = PromptAD(...)  # 使用原始model
else:
    # 如果有修改过的model支持fusion_mode
    model = PromptAD(..., fusion_mode=args.fusion_mode)
```

### 修改 `train_seg.py`

同样的修改应用到SEG训练代码。

---

## 步骤4: 运行训练

### 快速测试（单个类别）

```bash
# 测试MVTec bottle k=1
python train_cls.py \
    --dataset mvtec \
    --data-path /path/to/MVTec \
    --obj bottle \
    --shot 1 \
    --epoch 100 \
    --fusion-mode harmonic \
    --save-path result/harmonic_baseline/mvtec/k_1/bottle/cls_checkpoint.pth
```

### 完整训练（所有162个任务）

```bash
chmod +x train_harmonic.sh
./train_harmonic.sh
```

**预计时间**: 
- 每个任务约15-30分钟（100 epoch）
- 162个任务 × 2（cls+seg）× 20分钟 ≈ **108小时** ≈ 4.5天

---

## 步骤5: 评估harmonic baseline

训练完成后，运行评估：

```bash
# 使用现有的run_gate_experiment.py，但指定harmonic检查点
python run_gate_experiment.py \
    --checkpoint-dir result/harmonic_baseline \
    --branch gate_harmonic \
    --dataset mvtec \
    --data-path /path/to/data
```

或者批量评估：

```bash
# 创建 run_harmonic_eval.sh
./run_gate_all.sh result/harmonic_baseline gate_harmonic
```

---

## 步骤6: 对比分析

评估完成后，对比max-baseline vs harmonic-baseline：

```python
import pandas as pd

# 加载两个baseline的结果
max_results = pd.read_csv('result/gate/analysis/full_results.csv')
harmonic_results = pd.read_csv('result/gate_harmonic/analysis/full_results.csv')

# 对比harmonic模式的性能
max_harmonic = max_results[max_results['mode'] == 'harmonic']
true_harmonic = harmonic_results[harmonic_results['mode'] == 'harmonic']

# 计算差异
merged = pd.merge(max_harmonic, true_harmonic, 
                  on=['dataset', 'class', 'k_shot', 'task'],
                  suffixes=('_max_trained', '_harmonic_trained'))

merged['improvement'] = merged['auroc_harmonic_trained'] - merged['auroc_max_trained']
print(f"平均改进: {merged['improvement'].mean():.2f}%")
print(f"最大改进: {merged['improvement'].max():.2f}%")
```

---

## 简化方案：只训练关键类别

如果时间有限，可以只训练**弱分支类别**验证harmonic的改进效果：

```bash
# MVTec关键类别
CLASSES=("toothbrush" "hazelnut" "metal_nut")

# VisA关键类别  
CLASSES=("pipe_fryum" "capsules" "pcb2")

# 只训练这6个类别 × 3 k-shot × 2 task = 36个任务
# 预计时间: 36 × 20分钟 ≈ 12小时
```

---

## 预期结果

基于train-test mismatch理论，harmonic-trained应该比max-trained的harmonic模式**提升约5%**：

| 配置 | Max-trained + Harmonic推理 | Harmonic-trained + Harmonic推理 | 预期改进 |
|------|---------------------------|--------------------------------|---------|
| MVTec CLS | 91.75% | ~96.5% | +4.75% |
| VisA CLS | 84.20% | ~88.5% | +4.30% |

这将验证：
1. **Harmonic融合的真实性能**（消除train-test mismatch）
2. **Semantic分支的改进空间**（如果仍然弱，说明是分支本身的问题）
3. **融合策略的影响**（对比max vs harmonic）

---

## 注意事项

1. **数据路径**: 确保`--data-path`指向正确的数据集位置
2. **GPU显存**: 训练需要约8-12GB显存
3. **检查点保存**: 确保`result/harmonic_baseline`目录有足够空间
4. **日志监控**: 定期检查训练日志确保没有错误
5. **可重复性**: 使用相同的seed和数据划分（读取`datasets/seeds_*/`）

---

## 快速开始命令

```bash
# 1. 创建输出目录
mkdir -p result/harmonic_baseline

# 2. 测试单个任务
python train_cls.py \
    --dataset mvtec \
    --data-path /path/to/MVTec \
    --obj bottle \
    --shot 1 \
    --epoch 100 \
    --fusion-mode harmonic

# 3. 如果成功，运行完整训练
./train_harmonic.sh
```

需要我帮你生成具体的修改代码吗？
