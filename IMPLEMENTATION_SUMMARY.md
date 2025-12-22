# 实现总结：模块化评分框架

## 已实现内容

实现了一个极简、模块化的 PromptAD 风格异常检测框架，支持：

### ✅ Step 1：评分模块重构（已完成）

**新增：`PromptAD/scoring.py`**

提供了多个接口一致的评分函数：
- `semantic_score(visual_features, task)` → 仅基于 CLIP 的语义评分
- `memory_score(visual_features, task)` → 仅基于最近邻（NN）距离的记忆评分  
- `max_score(visual_features, task)` → 逐元素最大值融合
- `harmonic_score(visual_features, task)` → 调和均值融合
- `oracle_score(visual_features, task, gt_labels)` → 按样本选择最优分支

所有分数均归一化到 [0, 1] 区间，保证可公平比较。

### ✅ Step 2：Oracle 门控（已完成）

**实现位置：`PromptAD/scoring.py`**

- 对每个样本选择更优分支（语义 vs 记忆）
- 评估阶段需要使用 GT 标签
- 提供 oracle 性能作为理论上界
- 统计分支选择比例（语义 vs 记忆）
- 仅用于评估分析，不影响训练过程

### ✅ Step 3：可靠性信号（基础设施已就绪）

**为未来门控机制预留的接口：**

**语义分支：**
- `per_prompt_scores`：各个 prompt 模板的得分
- `prompt_variance`：prompt 间的方差（越小表示置信度越高）
- `confidence`：softmax 概率

**记忆分支：**
- `nn_margin`：距离间隔 d1 - d2（越大表示置信度越高）
- `top_k_similarities`：k 近邻距离
- `min_distance`：最近支持样本距离

这些信号目前已计算，但尚未用于门控决策。

## 文件结构

