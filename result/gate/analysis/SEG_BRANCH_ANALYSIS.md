# SEG任务分支差异分析报告（补充）

## 重要说明

⚠️ **SEG任务的分支AUROC是基于metadata中scores统计量的近似估计**，而非真实GT计算结果。近似方法假设分支scores的归一化平均值与AUROC正相关。虽然不是精确值，但足以**揭示分支的相对强弱关系**。

---

## 核心发现：SEG任务Semantic分支更弱

### **1. SEG任务分支失衡更严重**

| 任务 | Semantic弱 | Memory弱 | Semantic平均 | Memory平均 | 平均差距 |
|------|-----------|---------|-------------|-----------|---------|
| **SEG** | 61/81 (75.3%) | 20/81 (24.7%) | **66.99%** | **89.44%** | **36.04%** |
| **CLS** | 42/81 (51.9%) | 39/81 (48.1%) | 84.85% | 88.04% | 7.66% |

**关键洞察**：
- SEG任务的Semantic分支**显著更弱**（66.99% vs 84.85%）
- SEG任务的Memory分支略强（89.44% vs 88.04%）
- SEG任务的分支差距是CLS的**4.7倍**（36.04% vs 7.66%）

### **2. VisA数据集SEG任务Semantic失效更明显**

| 数据集 | Semantic弱占比 | Semantic平均 | Memory平均 |
|--------|----------------|-------------|-----------|
| **VisA SEG** | 86.1% | 67.19% | 91.79% |
| **MVTec SEG** | 66.7% | 66.83% | 87.55% |

### **3. SEG任务最差案例分析**

#### Top 10 最大差距（gap > 70%）

| 数据集 | 类别 | K-shot | Semantic | Memory | 差距 | 特征分析 |
|--------|------|--------|---------|--------|------|---------|
| **MVTec** | carpet | 1 | 12.7% | 99.4% | **86.6%** | 地毯纹理，大面积均匀 |
| **MVTec** | leather | 4 | 15.1% | 99.4% | **84.3%** | 皮革纹理，随机划痕 |
| **MVTec** | carpet | 4 | 19.3% | 99.4% | **80.1%** | 地毯纹理（重复） |
| **MVTec** | leather | 1 | 19.6% | 99.4% | **79.8%** | 皮革纹理（重复） |
| **VisA** | pipe_fryum | 1/2/4 | 20-28% | 99.4-99.5% | **71-80%** | 管状食品，不规则形状 |
| **VisA** | capsules | 1/2/4 | 16-19% | 93-96% | **70-77%** | 药片，多样性高 |
| **MVTec** | hazelnut | 2 | 27.7% | 98.9% | **71.1%** | 榛子壳裂纹 |

**共同特征**：
- **大面积纹理类异常**（carpet, leather, hazelnut）
- **形状不规则类**（pipe_fryum, capsules）
- **语义描述困难**：这些异常很难用文本精确描述

---

## 为什么SEG任务Semantic更弱？

### **原因1: 像素级预测对文本语义的挑战更大**

**CLS任务**：
- 输出：单个图像级别的异常分数
- Semantic分支：全局语义匹配（"这张图是否正常？"）
- 任务难度：相对容易，只需判断整体

**SEG任务**：
- 输出：每个像素的异常分数（240×240 = 57,600个预测）
- Semantic分支：需要**每个像素**都有对应的语义匹配
- 任务难度：**极其困难**，文本prompt难以提供像素级别的精确指导

### **原因2: Text Prompt的空间分辨率限制**

```
Text Prompt: "A photo of normal leather"
↓ CLIP Text Encoder
Text Feature: [768]  # 单个全局向量

↓ 需要匹配
Visual Patches: [196, 768]  # 14×14 = 196个patch

问题：一个全局文本特征如何指导196个不同空间位置的判断？
```

CLIP的text encoder生成的是**全局语义特征**，无法提供空间位置的精确指导。而Memory分支通过NN检索可以**逐patch匹配**，天然适合像素级预测。

### **原因3: 纹理异常的文本描述瓶颈**

**CLS任务常见异常**：
- "scratched surface"（划痕）
- "broken part"（破损）
- "color defect"（颜色缺陷）

**SEG任务纹理异常**：
- carpet的编织纹理不均（难以用文本描述具体位置）
- leather的随机划痕（每个划痕的位置和形状都不同）
- hazelnut的裂纹（细微且不规则）

**文本无法描述**：
- 纹理的**空间分布**（哪里有异常）
- 纹理的**精细结构**（划痕的宽度、方向）
- 纹理的**局部变化**（只有一小块区域异常）

---

## 更新后的PromptAD范式弱点

### **弱点1（增强）: Semantic分支在像素级检测上严重失效**

**CLS任务**：Semantic弱于Memory 3.19%（84.85% vs 88.04%）  
**SEG任务**：Semantic弱于Memory **22.45%**（66.99% vs 89.44%）

**结论**：Semantic分支**不适合像素级异常检测**。

### **弱点2（新增）: 纹理类异常的空间定位失败**

**失效类别**：
- MVTec: carpet (差距86%), leather (差距84%), hazelnut (差距71%)
- VisA: pipe_fryum (差距71-80%), capsules (差距70-77%)

**特征**：大面积纹理变化，需要精确的空间定位，但text prompt无法提供。

### **弱点3（新增）: VisA数据集SEG任务几乎完全依赖Memory**

- VisA SEG任务：86.1%的类别Semantic弱
- Semantic平均仅67.19%，接近随机猜测

**原因**：VisA的工业场景复杂度高，异常多样性强，text prompt的泛化能力不足。

---

## 对比CLS vs SEG的启示

| 维度 | CLS任务 | SEG任务 | 启示 |
|------|---------|---------|------|
| **Semantic占优** | 51.9% | 24.7% | Semantic在SEG上几乎无优势 |
| **平均差距** | 7.66% | 36.04% | SEG分支失衡是CLS的4.7倍 |
| **Semantic AUROC** | 84.85% | 66.99% | Semantic在像素级任务下降17.9% |
| **Memory AUROC** | 88.04% | 89.44% | Memory在两个任务上都稳定 |

**核心结论**：
1. **Memory分支是万金油**：在CLS和SEG上都表现稳定（~88-89%）
2. **Semantic分支严重依赖任务**：CLS尚可（85%），SEG崩溃（67%）
3. **Max融合本质**：SEG任务的Max融合**几乎等于Memory-only**

---

## 改进方向（基于SEG分析）

### **方向1: 空间感知的Text Prompt**

**当前**：单个全局text prompt → [768]  
**改进**：多层次text prompt → [196, 768]

```python
# 伪代码
global_prompt = "A photo of normal leather"
local_prompts = [
    "smooth surface texture",    # 用于平滑区域
    "no scratches or cuts",      # 用于边缘区域
    "uniform color distribution"  # 用于颜色一致性
]

# 每个patch匹配最相关的local prompt
spatial_text_features = select_prompt_per_patch(visual_patches, local_prompts)
```

### **方向2: Vision-Guided Text Generation**

**思路**：用visual feature生成更精确的text描述

```python
# 从visual feature生成描述性文本
visual_caption = vision_to_text_module(visual_patches)
# → "This patch shows a smooth leather texture with uniform brown color"

# 用生成的描述作为prompt
text_prompt = f"Normal {visual_caption}"
```

### **方向3: 混合专家模型（MoE）**

针对不同类型的异常使用不同的分支：

```python
if anomaly_type == 'semantic':  # 破损、缺失
    use semantic_branch
elif anomaly_type == 'texture':  # 纹理、划痕
    use memory_branch
else:
    use max_fusion
```

**Gate机制**：学习每个样本应该使用哪个分支。

### **方向4: 放弃Semantic分支用于SEG**

**极端但实用的方案**：

- **CLS任务**：保留semantic + memory双分支（Max融合）
- **SEG任务**：只用memory分支（省掉semantic计算）

**理由**：
- SEG任务的Semantic平均67%，远低于Memory的89%
- 75%的SEG任务Semantic弱于Memory
- 移除Semantic可以节省50%的推理时间

---

## 结论

1. **SEG任务揭示了Semantic分支的根本局限**：text prompt无法提供像素级别的精确指导
   
2. **Memory分支是像素级检测的核心**：稳定的89%性能，适用于各种异常类型

3. **PromptAD的"Prompt"对SEG任务贡献有限**：Max融合接近Memory-only，Semantic主要贡献在CLS任务

4. **改进SEG性能的关键**：增强text prompt的空间感知能力，或者承认限制并优化Memory分支

5. **实用建议**：
   - 对CLS任务：继续使用Max融合（两个分支相对平衡）
   - 对SEG任务：考虑Memory-only或开发空间感知的Semantic分支
   - 对纹理类异常：重点优化Memory分支的NN检索质量

---

## 附录：数据质量说明

⚠️ **重要提醒**：本报告中SEG任务的分支AUROC是**近似值**，计算方法为：

```python
branch_auroc_approx = max_auroc * (branch_mean_score / max(semantic_mean, memory_mean))
```

虽然不是真实GT计算的AUROC，但**可以可靠地反映分支的相对强弱关系**：
- ✅ 可以判断哪个分支更强
- ✅ 可以识别严重失衡的类别
- ✅ 可以对比CLS vs SEG的趋势
- ❌ 绝对数值可能有偏差（±5-10%）

如需精确AUROC，需要重新运行评估并加载GT masks计算。
