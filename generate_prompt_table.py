"""
生成完整的静态Prompt表格
可用于人工标注、打分和清洗
"""

import pandas as pd
from PromptAD.ad_prompts import get_all_classes_manual_prompts


def generate_full_prompt_table(output_path='prompts/manual_prompts_master_table.csv'):
    """
    生成主Prompt表格，包含所有类别的完整prompt
    """
    
    all_prompts_data = get_all_classes_manual_prompts()
    
    rows = []
    prompt_global_id = 0
    
    for classname in sorted(all_prompts_data.keys()):
        data = all_prompts_data[classname]
        
        for item in data['info']:
            prompt_global_id += 1
            
            row = {
                # 基本信息
                'prompt_id': prompt_global_id,
                'class': classname,
                'display_name': item['display_name'],
                'index_in_class': item['index'],
                'type': item['type'],
                
                # Prompt内容
                'template': item['template'],
                'full_text': item['text'],
                
                # 控制字段
                'enabled': True,  # 是否启用此prompt
                
                # 评分字段（人工标注）
                'manual_score': '',  # 人工打分 (0-10)
                'relevance': '',  # 相关性评分 (high/medium/low)
                'specificity': '',  # 特异性评分 (high/medium/low)
                
                # 分析字段（自动填充）
                'mean_similarity': '',  # 平均相似度
                'max_similarity': '',  # 最大相似度
                'contribution_rank': '',  # 贡献度排名
                
                # 备注
                'notes': '',  # 备注说明
                'action': '',  # 操作标记 (keep/remove/modify)
            }
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # 保存
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✓ Generated master prompt table: {output_path}")
    print(f"  Total prompts: {len(df)}")
    print(f"  Total classes: {len(all_prompts_data)}")
    print(f"\nColumns:")
    for col in df.columns:
        print(f"  - {col}")
    
    # 生成统计信息
    print(f"\nStatistics by class:")
    class_stats = df.groupby('class').agg({
        'prompt_id': 'count',
        'type': lambda x: (x == 'generic').sum()
    }).rename(columns={'prompt_id': 'total', 'type': 'generic'})
    class_stats['specific'] = class_stats['total'] - class_stats['generic']
    print(class_stats)
    
    return df


if __name__ == '__main__':
    df = generate_full_prompt_table()
    
    # 也生成一个示例说明文件
    with open('prompts/README.md', 'w') as f:
        f.write("""# Manual Prompts Master Table

## 文件说明

`manual_prompts_master_table.csv` 是所有静态prompt的主表，包含所有类别的完整异常描述。

## 表格结构

### 基本信息列
- `prompt_id`: 全局唯一ID
- `class`: 类别名称 (如 bottle, carpet)
- `display_name`: 显示名称 (如 printed circuit board)
- `index_in_class`: 在该类别中的索引 (0开始)
- `type`: 类型 (generic=通用, specific=特定)

### Prompt内容列
- `template`: 模板字符串 (如 "damaged {}")
- `full_text`: 完整文本 (如 "damaged bottle")

### 控制字段
- `enabled`: 是否启用 (True/False) - **修改此列可控制模型使用哪些prompts**

### 人工标注字段
- `manual_score`: 人工打分 (0-10分) - **可在此打分**
- `relevance`: 相关性 (high/medium/low) - **评估prompt与实际缺陷的相关性**
- `specificity`: 特异性 (high/medium/low) - **评估prompt的具体程度**

### 自动分析字段（运行分析后填充）
- `mean_similarity`: 平均相似度
- `max_similarity`: 最大相似度
- `contribution_rank`: 贡献度排名

### 备注字段
- `notes`: 备注说明 - **可添加任何说明**
- `action`: 操作标记 (keep/remove/modify) - **标记清洗决策**

## 使用流程

### 1. 查看和标注
```bash
# 在Excel或文本编辑器中打开CSV
open manual_prompts_master_table.csv
```

### 2. 人工评估和打分
- 填写 `manual_score` 列 (0-10分)
- 填写 `relevance` 列 (high/medium/low)
- 填写 `specificity` 列 (high/medium/low)
- 添加 `notes` 备注

### 3. 运行模型分析
```bash
# 运行贡献度分析，自动填充分析字段
python update_prompt_table_with_contribution.py --dataset mvtec --class bottle
```

### 4. 清洗决策
- 根据人工评分和自动分析，填写 `action` 列
- 设置 `enabled=False` 禁用低质量的prompts

### 5. 模型使用更新后的表格
```python
# model.py会自动读取这个表格，只使用enabled=True的prompts
model = PromptAD(class_name='bottle', ...)
```

## 示例：标注bottle类别

| prompt_id | class  | full_text                  | enabled | manual_score | relevance | action |
|-----------|--------|----------------------------|---------|--------------|-----------|--------|
| 1         | bottle | damaged bottle             | True    | 8            | high      | keep   |
| 2         | bottle | flawed bottle              | True    | 6            | medium    | keep   |
| 3         | bottle | abnormal bottle            | True    | 7            | high      | keep   |
| 8         | bottle | bottle with large breakage | True    | 9            | high      | keep   |
| 9         | bottle | bottle with small breakage | True    | 9            | high      | keep   |
| 10        | bottle | bottle with contamination  | True    | 8            | high      | keep   |

## 注意事项

1. **不要删除行** - 使用 `enabled=False` 来禁用prompt
2. **保持prompt_id唯一** - 不要修改此列
3. **修改后重新训练** - 更改表格后需要重新训练模型
4. **备份原表** - 修改前先备份
""")
    
    print(f"\n✓ Generated README: prompts/README.md")
