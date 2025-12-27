#!/usr/bin/env python3
"""
分析各类别的Collapse程度，为类别自适应Repulsion策略提供依据
"""
import pandas as pd
import numpy as np

# 从controlled experiment data读取semantic_std (反映Collapse程度)
df = pd.read_csv('analysis/controlled_comparison/controlled_experiment_data.csv')

# 计算Collapse指标 (semantic_std越小 = Collapse越严重)
# 使用Prompt2的baseline值
collapse_scores = {}
for _, row in df.iterrows():
    class_name = row['class']
    # Nearest Normal距离的std作为Collapse代理指标(暂时)
    # 理想情况应该直接从prototypes计算，但我们用nm_p2作为baseline
    collapse_scores[class_name] = {
        'nm_baseline': row['nm_p2'],  # Nearest normal距离
        'sep_baseline': row['sep_p2'],  # Separation
        'train_p2': row['train_p2'],
        'group': row['group']
    }

print("=" * 80)
print("类别Collapse程度分析 (基于Prompt2 baseline)")
print("=" * 80)
print(f"{'类别':<20} {'组别':<10} {'训练AUROC':<12} {'Separation':<12} {'建议λ_rep':<10}")
print("-" * 80)

# 根据训练难度和Separation分配Repulsion强度
recommendations = {}
for class_name, metrics in sorted(collapse_scores.items(), key=lambda x: x[1]['train_p2']):
    train_auroc = metrics['train_p2']
    separation = metrics['sep_baseline']
    group = metrics['group']
    
    # 决策规则:
    # 1. 训练AUROC < 70%: Severe Collapse → λ_rep=0.10
    # 2. 70% <= AUROC < 90% AND Separation < 0.15: Moderate → λ_rep=0.05
    # 3. AUROC >= 90% OR Separation >= 0.3: Stable → λ_rep=0.02
    
    if train_auroc < 70:
        lambda_rep = 0.10
        reason = "Severe (低AUROC)"
    elif train_auroc >= 99:
        lambda_rep = 0.02
        reason = "Stable (高AUROC)"
    elif separation >= 0.3:
        lambda_rep = 0.02
        reason = "Stable (高Separation)"
    elif separation < 0.05:
        lambda_rep = 0.10
        reason = "Severe (低Separation)"
    else:
        lambda_rep = 0.05
        reason = "Moderate"
    
    recommendations[class_name] = lambda_rep
    print(f"{class_name:<20} {group:<10} {train_auroc:>8.2f}%   {separation:>8.4f}     {lambda_rep:>5.2f}  ({reason})")

print("=" * 80)

# 统计分组
lambda_groups = {}
for class_name, lambda_rep in recommendations.items():
    if lambda_rep not in lambda_groups:
        lambda_groups[lambda_rep] = []
    lambda_groups[lambda_rep].append(class_name)

print("\n分组统计:")
for lambda_val in sorted(lambda_groups.keys(), reverse=True):
    classes = lambda_groups[lambda_val]
    print(f"  λ_rep={lambda_val:.2f}: {len(classes)}个类别")
    for cls in classes:
        print(f"    - {cls}")

print("\n" + "=" * 80)
print("💡 建议:")
print("=" * 80)
print("1. 使用类别自适应Repulsion策略")
print("2. 为每个类别设置专属λ_rep值")
print("3. 6类验证后扩展到27类")
print("\n实施方式:")
print("  方式A: 在训练脚本中硬编码 (快速)")
print("  方式B: 通过配置文件传递 (灵活)")
print("  方式C: 训练时自动检测并分配 (智能)")
print("=" * 80)

# 保存推荐配置
import json
config = {
    "adaptive_repulsion": True,
    "class_lambda_rep": recommendations,
    "rationale": "Based on Prompt2 baseline AUROC and Separation analysis"
}

with open('analysis/controlled_comparison/adaptive_repulsion_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("\n✅ 配置已保存到: analysis/controlled_comparison/adaptive_repulsion_config.json")

# 对比实验结果预测
print("\n" + "=" * 80)
print("🔮 效果预测")
print("=" * 80)
print("基于v2实验结果的推断:\n")

predictions = {
    'mvtec-toothbrush': {
        'v2_lambda': 0.10,
        'v2_result': -5.55,
        'new_lambda': recommendations['mvtec-toothbrush'],
        'predicted_change': '+3~5%',
        'reason': '降低Repulsion避免过度分散'
    },
    'visa-pcb2': {
        'v2_lambda': 0.10,
        'v2_result': +6.24,
        'new_lambda': recommendations['visa-pcb2'],
        'predicted_change': '+5~7%',
        'reason': '保持强Repulsion，Collapse严重类受益'
    },
    'mvtec-carpet': {
        'v2_lambda': 0.10,
        'v2_result': -0.04,
        'new_lambda': recommendations['mvtec-carpet'],
        'predicted_change': '±0.1%',
        'reason': '极低Repulsion减少对Stable类的干扰'
    }
}

for cls, pred in predictions.items():
    print(f"{cls}:")
    print(f"  v2 (λ={pred['v2_lambda']:.2f}): {pred['v2_result']:+.2f}%")
    print(f"  v3 (λ={pred['new_lambda']:.2f}): 预测{pred['predicted_change']}")
    print(f"  理由: {pred['reason']}\n")

print("整体预测: 训练AUROC +1.5~2.0% (优于v1的+1.10%)")
print("=" * 80)
