# SEG任务分支数据缺失问题分析

## 问题现象

当前gate实验中，**SEG任务的所有405条记录都没有分支数据**（semantic_auroc和memory_auroc为NaN），而CLS任务的81条记录都有完整的分支数据。

```
CLS任务: 81/81 有分支数据 ✅
SEG任务: 0/81 有分支数据 ❌
```

---

## 根本原因

### **原因1: 原始model.py只返回融合结果**

在`PromptAD/model.py`的`forward()`方法中，SEG任务直接返回融合后的anomaly_map，**没有返回两个分支的中间结果**：

```python
# PromptAD/model.py line 373-395
def forward(self, images, task):
    visual_features = self.encode_image(images)
    
    if task == 'seg':
        textual_anomaly_map = self.calculate_textual_anomaly_score(visual_features, 'seg')
        visual_anomaly_map = self.calculate_visual_anomaly_score(visual_features)
        
        # 直接融合，不返回中间结果
        anomaly_map = torch.maximum(textual_anomaly_map, visual_anomaly_map)
        anomaly_map = F.interpolate(...)
        am_pix = anomaly_map.squeeze(1).numpy()
        
        # ❌ 只返回融合后的结果
        return am_pix_list  
        
    elif task == 'cls':
        textual_anomaly = self.calculate_textual_anomaly_score(visual_features, 'cls')
        visual_anomaly_map = self.calculate_visual_anomaly_score(visual_features)
        
        # ✅ 返回两个分支的结果
        return am_img_list, am_pix_list
```

**对比**：
- **CLS任务**：返回`(am_img_list, am_pix_list)`，分别对应semantic和memory分支
- **SEG任务**：只返回`am_pix_list`（融合后的像素级异常图）

### **原因2: 评估代码使用旧的metrics函数**

在`run_gate_experiment.py`中，SEG任务使用的是旧版`metric_cal_pix`：

```python
# run_gate_experiment.py line 157-159
else:  # seg
    from utils.metrics import metric_cal_pix
    metrics = metric_cal_pix(np.array(score_maps), gt_masks_resized)
```

而`metric_cal_pix`只计算整体AUROC，**不支持分支分析**：

```python
# utils/metrics.py line 36-49
def metric_cal_pix(map_scores, gt_mask_list):
    per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), map_scores.flatten())
    
    # ❌ 只返回整体AUROC
    result_dict = {'p_roc': per_pixel_rocauc * 100}
    return result_dict
```

相比之下，CLS任务使用的是`metric_cal_img_modular`，支持分支AUROC：

```python
# utils/metrics_modular.py
def metric_cal_img_modular(...):
    ...
    # ✅ 返回分支AUROC
    result_dict = {
        'i_roc': ...,
        'i_roc_semantic': semantic_auroc,
        'i_roc_memory': memory_auroc,
        'gap': abs(semantic_auroc - memory_auroc)
    }
```

---

## 影响范围

### ✅ **不影响SEG任务的整体AUROC**
- 所有5种融合模式（semantic/memory/max/harmonic/oracle）的AUROC都是准确的
- SEG任务的性能评估是**完全可信**的

### ❌ **无法分析SEG任务的分支差异**
- 不知道semantic分支在SEG任务上的表现
- 不知道memory分支在SEG任务上的表现
- 无法判断哪个分支在像素级检测上更强
- 无法做SEG任务的弱分支分析

### ❌ **限制了范式弱点分析的完整性**
当前的`PARADIGM_WEAKNESS_ANALYSIS.md`只能基于CLS任务数据，无法验证：
- Semantic分支在SEG任务上是否也偏弱？
- 纹理类异常在像素级检测上的分支差异模式？

---

## 解决方案

### **方案1: 修改原始model.py（侵入性强）**

修改`PromptAD/model.py`，让SEG任务也返回分支结果：

```python
def forward(self, images, task):
    visual_features = self.encode_image(images)
    
    if task == 'seg':
        textual_anomaly_map = self.calculate_textual_anomaly_score(visual_features, 'seg')
        visual_anomaly_map = self.calculate_visual_anomaly_score(visual_features)
        
        # 保存中间结果
        semantic_map = F.interpolate(textual_anomaly_map, ...)
        memory_map = F.interpolate(visual_anomaly_map, ...)
        
        # 融合
        anomaly_map = torch.maximum(textual_anomaly_map, visual_anomaly_map)
        anomaly_map = F.interpolate(anomaly_map, ...)
        
        # ✅ 返回三个结果
        return am_pix_list, semantic_pix_list, memory_pix_list
```

**优点**: 最彻底，与CLS任务一致  
**缺点**: 需要修改原始代码，影响训练流程

### **方案2: 使用modular scoring框架（推荐）**

在`run_gate_experiment.py`中，让SEG任务也使用modular scorer获取分支数据：

```python
# run_gate_experiment.py修改
if args.task == 'cls':
    # 原来的逻辑
    output, metadata = wrapped_model(data, task='cls')
else:  # seg
    # ✅ 使用modular scorer
    output, metadata = wrapped_model(data, task='seg')  # 已经支持！
    
    # metadata已经包含semantic_scores和memory_scores
```

然后创建SEG版本的metrics函数：

```python
# utils/metrics_modular.py
def metric_cal_pix_modular(score_maps, gt_masks, metadata=None, score_mode='max'):
    """计算像素级AUROC，支持分支分析"""
    
    # 整体AUROC
    per_pixel_auroc = roc_auc_score(gt_masks.flatten(), score_maps.flatten())
    
    result_dict = {'p_roc': per_pixel_auroc * 100}
    
    # 如果有分支数据，计算分支AUROC
    if metadata and 'semantic_scores' in metadata and 'memory_scores' in metadata:
        semantic_maps = metadata['semantic_scores']
        memory_maps = metadata['memory_scores']
        
        semantic_auroc = roc_auc_score(gt_masks.flatten(), semantic_maps.flatten())
        memory_auroc = roc_auc_score(gt_masks.flatten(), memory_maps.flatten())
        
        result_dict.update({
            'p_roc_semantic': semantic_auroc * 100,
            'p_roc_memory': memory_auroc * 100,
            'semantic_auroc': semantic_auroc * 100,
            'memory_auroc': memory_auroc * 100,
            'gap': abs(semantic_auroc - memory_auroc) * 100
        })
    
    return result_dict
```

**优点**: 
- 不修改原始model.py
- 复用已有的modular scoring框架
- 只需要修改评估代码

**缺点**: 
- 需要重新运行所有SEG任务的评估（81个任务）
- 需要创建新的metrics函数

### **方案3: 后处理补救（最快）**

如果不想重新运行评估，可以从已保存的metadata中提取分支数据：

```python
# 检查metadata文件
import json
from pathlib import Path

for meta_file in Path('result/gate').rglob('metadata/seg/*.json'):
    with open(meta_file, 'r') as f:
        metadata = json.load(f)
        
    # 检查是否有分支数据
    if 'semantic_scores' in metadata:
        print(f"✓ {meta_file} 有分支数据")
    else:
        print(f"✗ {meta_file} 缺少分支数据")
```

**优点**: 最快，不需要重新评估  
**缺点**: 只适用于已经保存了分支metadata的情况

---

## 当前metadata是否包含分支数据？

让我检查一个SEG任务的metadata文件：

```bash
# 查看SEG任务的metadata内容
cat result/gate/mvtec/k_1/metadata/seg/bottle_seed111_max.json | head -20
```

如果metadata中**有**`semantic_scores`和`memory_scores`字段，说明数据已经保存，只需要：
1. 创建`metric_cal_pix_modular`函数
2. 重新聚合结果即可

如果metadata中**没有**这些字段，需要：
1. 修改评估代码使用modular scorer
2. 重新运行所有81个SEG任务的评估

---

## 推荐行动方案

### **短期（如果不需要SEG分支分析）**
- ✅ 继续使用当前CLS数据做范式弱点分析
- ✅ SEG任务的整体AUROC是可信的
- ⚠️ 在论文/报告中注明"SEG分支分析基于CLS任务"

### **中期（如果需要完整分析）**
1. 创建`metric_cal_pix_modular`函数
2. 修改`run_gate_experiment.py`的SEG评估部分
3. 重新运行81个SEG任务评估（约2-3小时）
4. 重新聚合结果，生成完整的SEG分支分析

### **长期（如果要训练新baseline）**
- 在训练harmonic baseline时，直接修改model.py支持SEG分支返回
- 一步到位解决问题

---

## 快速验证命令

检查当前SEG metadata是否包含分支数据：

```bash
# 随机检查3个SEG任务的metadata
python -c "
import json
from pathlib import Path

meta_files = list(Path('result/gate').rglob('metadata/seg/*_max.json'))[:3]

for meta_file in meta_files:
    with open(meta_file, 'r') as f:
        metadata = json.load(f)
    
    has_branches = 'semantic_scores' in metadata and 'memory_scores' in metadata
    status = '✓ 有分支数据' if has_branches else '✗ 缺少分支数据'
    print(f'{meta_file.name}: {status}')
    
    if has_branches:
        print('  可以通过后处理补救！')
    else:
        print('  需要重新运行评估')
"
```

需要我帮你检查metadata内容，或者生成修复代码吗？
