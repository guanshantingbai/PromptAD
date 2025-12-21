# prompt1_memory 分支说明

## 概述

这个分支基于 `prompt1` 分支，添加了**记忆库（Memory Bank）**功能，实现完整的双分支异常检测系统。

## 架构特点

### 1. 语义分支（Semantic Branch）
- **多原型方法**：使用 K 个正常原型 + M 个异常原型
- **对比prompt1纯语义的改进**：+3.37% (91.47% vs 88.10%)
- **实现**：`calculate_textual_anomaly_score()`

### 2. 视觉分支（Visual Branch / Memory Bank）
- **1-NN方法**：存储训练样本特征，计算测试样本与最近邻的距离
- **特征库**：
  - `feature_gallery1`: 全局CLS token特征 [M, dim]
  - `feature_gallery2`: 局部patch token特征 [M, num_patches, dim]
- **实现**：`calculate_visual_anomaly_score()`

### 3. 融合策略
- **调和平均（分子为1）**：
  ```
  score = 1 / (1/semantic_score + 1/visual_score)
  ```
- 相比标准调和平均（分子为2），这种方式更保守
- 实现位置：`forward()` 方法

## 关键改动

### model.py 的变化

1. **添加记忆库缓冲区**：
```python
self.register_buffer("feature_gallery1", ...)  # 全局特征
self.register_buffer("feature_gallery2", ...)  # 局部特征
```

2. **恢复 `build_image_feature_gallery()` 方法**：
```python
def build_image_feature_gallery(self, images):
    """从训练样本构建记忆库"""
    visual_features = self.encode_image(images)
    # 累积全局和局部特征
```

3. **恢复 `calculate_visual_anomaly_score()` 方法**：
```python
def calculate_visual_anomaly_score(self, visual_features, task):
    """基于1-NN计算视觉异常分数"""
    # 计算与记忆库的最大相似度
    # 转换为异常分数 (1 - similarity/T)
```

4. **修改 `forward()` 方法**：
```python
def forward(self, images, task):
    """双分支融合"""
    # 计算语义分数
    textual_anomaly = self.calculate_textual_anomaly_score(...)
    # 计算视觉分数
    visual_anomaly = self.calculate_visual_anomaly_score(...)
    # 调和平均融合
    fused = 1.0 / (1.0/textual + 1.0/visual)
```

## 理论预测

基于baseline的融合效果和prompt1的语义改进：

| 指标 | Baseline完整 | Prompt1纯语义 | 预测：Prompt1+MB |
|------|-------------|--------------|-----------------|
| MVTec k=2 CLS | 94.30% | 91.47% | **96.20%** |
| 改进 | - | +3.37% (vs baseline语义) | **+1.90%** (vs baseline完整) |

**改进保持率**：约 56%（3.37% → 1.90%）

原因：调和平均会"稀释"单个分支的改进。

## 使用方法

### 训练
训练过程与prompt1分支相同，但会自动构建记忆库：

```bash
# 使用现有的训练脚本
python train_cls.py --dataset mvtec --class_name bottle --k_shot 2 --n_pro 3 --n_pro_ab 6
```

训练时，`build_image_feature_gallery()` 会在每个epoch后被调用，累积训练样本特征。

### 测试
测试时会自动使用双分支融合：

```bash
# 使用现有的测试脚本
python test_cls.py --dataset mvtec --k_shot 2
```

### Checkpoint内容
保存的checkpoint现在包含：
- `normal_prototypes`: [K, dim] - 多个正常原型
- `abnormal_prototypes`: [M, dim] - 多个异常原型
- `feature_gallery1`: [N, dim] - 全局特征记忆库
- `feature_gallery2`: [N, num_patches, dim] - 局部特征记忆库

## 验证计划

1. **在MVTec k=2上测试**，验证是否达到预期的96.20%
2. **对比三个系统**：
   - Baseline完整（单原型+记忆库）：94.30%
   - Prompt1纯语义（多原型）：91.47%
   - **Prompt1+记忆库（多原型+记忆库）**：预期96.20%
3. **分析逐类效果**，确认多原型改进在融合后的保持情况

## 下一步

如果验证成功（≥96.20%），可以探索：
1. **优化融合策略**：尝试可学习的权重
2. **分析协同效应**：哪些类受益于多原型+记忆库组合
3. **更多原型数**：K=5, M=10 是否能进一步提升

## 分支对比

| 特性 | prompt1 | prompt1_memory |
|------|---------|----------------|
| 语义分支 | ✓ 多原型 | ✓ 多原型 |
| 视觉分支 | ✗ 无 | ✓ 记忆库1-NN |
| 融合方式 | - | ✓ 调和平均（分子=1） |
| 预期性能 | 91.47% | 96.20% |
| 用途 | 纯语义研究 | 完整系统部署 |

---

**创建日期**：2025年12月21日  
**基于分支**：prompt1  
**提交哈希**：9ec7aab
