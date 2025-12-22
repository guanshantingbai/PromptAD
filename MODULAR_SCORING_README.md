# PromptAD 的模块化评分框架

一个极简、模块化的框架，用于支持 PromptAD 风格的小样本异常检测中的多种评分策略与 oracle 分析。

## 特性

### 1. 独立的评分函数
所有函数具有完全一致的输入 / 输出接口：
- **`semantic_score`**：仅基于 CLIP 的文本-图像相似度
- **`memory_score`**：仅基于支持集的最近邻（NN）距离  
- **`max_score`**：逐元素取最大值的融合
- **`harmonic_score`**：调和均值融合（PromptAD 默认）

### 2. Oracle 门控（理论上界）
- 针对每个样本选择表现更优的分支（需要 GT）
- 为可学习门控提供性能上限
- 仅用于分析，不参与训练

### 3. 可靠性信号（基础设施）
为未来基于可靠性的门控机制预留接口：
- **语义分支**：逐 prompt 分数、prompt 方差
- **记忆分支**：top-k 相似度、最近邻间隔（d1-d2）

## 快速开始

### 基本用法

```python
from PromptAD import PromptAD
from PromptAD.model_modular import PromptADModular

# 创建基础模型（与原来一致）
base_model = PromptAD(...)
base_model.setup_text_features(class_name)
base_model.setup_memory_bank(support_features)

# 使用模块化评分器进行包装
model = PromptADModular(base_model, score_mode='max')

# 使用方式完全一致
scores = model(images, task='cls')
