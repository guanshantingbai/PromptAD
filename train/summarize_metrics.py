#!/usr/bin/env python3
"""
汇总所有类别的诊断指标（独立运行）
"""

import os
import json
import pandas as pd
from pathlib import Path

def main():
    print("="*80)
    print("汇总诊断指标")
    print("="*80)
    
    # 读取性能数据
    performance_data = pd.read_csv('analysis/full_performance_comparison_k2.csv')
    classes_to_diagnose = performance_data['class'].tolist()
    
    # 汇总所有类别的诊断结果
    all_metrics = []
    missing_classes = []
    
    for class_name in classes_to_diagnose:
        dataset = class_name.split('-')[0]
        cls_short = class_name.split('-')[1]
        summary_path = f'diagnostics/{dataset}_k2_{cls_short}/summary.json'
        
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summary = json.load(f)
                
                # 提取关键指标
                metrics = {
                    'class': class_name,
                    'dataset': dataset,
                    # Metric A: 异常max偶然命中
                    'A_hit_mean': summary['metric_A']['mean'],
                    'A_hit_p95': summary['metric_A']['p95'],
                    'A_hit_rate_30': summary['metric_A']['hit_rate_0.3'],
                    # Metric B: 判别裕度
                    'B_normal_margin': summary['metric_B']['normal_mean'],
                    'B_abnormal_margin': summary['metric_B']['abnormal_mean'],
                    'B_separation': summary['metric_B']['separation'],
                    'B_overlap': summary['metric_B']['overlap_ratio'],
                    # Metric C: 原型塌缩
                    'C_collapse_score': summary['metric_C']['collapse_score'],
                    'C_similarity_mean': summary['metric_C']['mean'],
                    # Metric D: 坏原型归因
                    'D_max_proto_count': max(summary['metric_D']['proto_counts']),
                    'D_high_score_max': max(summary['metric_D']['high_score_proto_counts']),
                    # Metric E: 融合敏感性
                    'E_normal_semantic': summary['metric_E']['normal_semantic_mean'],
                    'E_abnormal_semantic': summary['metric_E']['abnormal_semantic_mean'],
                }
                all_metrics.append(metrics)
        else:
            missing_classes.append(class_name)
    
    # 转为DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    
    # 合并性能数据
    full_data = performance_data.merge(metrics_df, on='class', how='left')
    
    # 添加baseline强度分类
    full_data['baseline_strength'] = pd.cut(
        full_data['baseline_acc'],
        bins=[0, 85, 95, 100],
        labels=['Weak', 'Medium', 'Strong'],
        include_lowest=True
    )
    
    # 保存完整数据
    full_data.to_csv('analysis/full_metrics_k2.csv', index=False)
    
    print(f"✅ 成功汇总: {len(all_metrics)}/{len(classes_to_diagnose)} 个类别")
    if missing_classes:
        print(f"⚠️  缺失诊断数据的类别: {len(missing_classes)}")
        for cls in missing_classes:
            print(f"    - {cls}")
    
    print()
    print("="*80)
    print(f"✅ 完整指标矩阵已保存到: analysis/full_metrics_k2.csv")
    print("="*80)
    print()
    
    # 显示汇总统计
    print("📊 指标汇总统计:")
    print("-"*80)
    
    metric_summary = {
        'A_hit_mean': '异常max偶然命中(均值)',
        'A_hit_p95': '异常max偶然命中(P95)',
        'B_separation': '判别分离度',
        'B_overlap': '裕度重叠率',
        'C_collapse_score': '原型塌缩分数',
    }
    
    for metric, label in metric_summary.items():
        if metric in full_data.columns:
            mean_val = full_data[metric].mean()
            std_val = full_data[metric].std()
            min_val = full_data[metric].min()
            max_val = full_data[metric].max()
            print(f"{label:<25} 均值={mean_val:.4f}, 标准差={std_val:.4f}, 范围=[{min_val:.4f}, {max_val:.4f}]")
    
    print()
    print("="*80)
    print("✅ 汇总完成！")
    print("="*80)

if __name__ == '__main__':
    main()
