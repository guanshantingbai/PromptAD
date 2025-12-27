#!/usr/bin/env python3
"""
分析诊断结果，对比退化类 vs 改进类
"""

import os
import json
import pandas as pd
import numpy as np

# 定义类别类型
degraded_classes = {
    'mvtec-toothbrush': {'baseline': 98.89, 'prompt2': 86.94, 'delta': -11.95},
    'mvtec-capsule': {'baseline': 79.94, 'prompt2': 69.12, 'delta': -10.82},
    'mvtec-cable': {'baseline': 97.42, 'prompt2': 96.08, 'delta': -1.34},
    'visa-pcb2': {'baseline': 98.20, 'prompt2': 93.54, 'delta': -4.66},
    'visa-pipe_fryum': {'baseline': 97.37, 'prompt2': 88.84, 'delta': -8.53}
}

improved_classes = {
    'mvtec-screw': {'baseline': 58.66, 'prompt2': 73.36, 'delta': +14.70}
}

def load_class_summary(class_name):
    """加载某个类别的诊断摘要"""
    dataset = class_name.split('-')[0]
    class_short = class_name.split('-')[1]
    
    summary_path = f'diagnostics/{dataset}_k2_{class_short}/summary.json'
    if not os.path.exists(summary_path):
        return None
    
    with open(summary_path, 'r') as f:
        return json.load(f)

def main():
    print("="*80)
    print("多原型退化诊断分析报告")
    print("="*80)
    print()
    
    # 收集所有类别的指标
    all_metrics = []
    
    print("1️⃣  退化类别 (Degraded Classes)")
    print("-"*80)
    for class_name, perf in degraded_classes.items():
        summary = load_class_summary(class_name)
        if summary is None:
            print(f"⚠️  {class_name}: 未找到诊断数据")
            continue
        
        metrics = {
            'class': class_name,
            'type': '退化类',
            'baseline': perf['baseline'],
            'prompt2': perf['prompt2'],
            'delta': perf['delta'],
            'A_ab_max_mean': summary['metric_A']['mean'],
            'A_ab_max_p95': summary['metric_A']['p95'],
            'A_hit_rate_30': summary['metric_A']['hit_rate_0.3'],
            'B_normal_margin': summary['metric_B']['normal_mean'],
            'B_abnormal_margin': summary['metric_B']['abnormal_mean'],
            'B_separation': summary['metric_B']['separation'],
            'B_overlap': summary['metric_B']['overlap_ratio'],
            'C_collapse_score': summary['metric_C']['collapse_score'],
            'D_top1_count': summary['metric_D']['proto_counts'][np.argmax(summary['metric_D']['proto_counts'])]
        }
        all_metrics.append(metrics)
        
        print(f"\n📊 {class_name} (Baseline: {perf['baseline']:.2f}% → Prompt2: {perf['prompt2']:.2f}%, Δ={perf['delta']:.2f}%)")
        print(f"   [A] 异常max偶然命中: 均值={metrics['A_ab_max_mean']:.3f}, P95={metrics['A_ab_max_p95']:.3f}, >0.3命中率={metrics['A_hit_rate_30']:.2%}")
        print(f"   [B] 判别裕度: Normal={metrics['B_normal_margin']:.4f}, Abnormal={metrics['B_abnormal_margin']:.4f}, 分离度={metrics['B_separation']:.4f}, 重叠={metrics['B_overlap']:.2%}")
        print(f"   [C] 原型塌缩: collapse_score={metrics['C_collapse_score']:.3f}")
        print(f"   [D] 坏原型: Top1原型命中{metrics['D_top1_count']}次")
    
    print()
    print("2️⃣  改进类别 (Improved Classes)")
    print("-"*80)
    for class_name, perf in improved_classes.items():
        summary = load_class_summary(class_name)
        if summary is None:
            print(f"⚠️  {class_name}: 未找到诊断数据")
            continue
        
        metrics = {
            'class': class_name,
            'type': '改进类',
            'baseline': perf['baseline'],
            'prompt2': perf['prompt2'],
            'delta': perf['delta'],
            'A_ab_max_mean': summary['metric_A']['mean'],
            'A_ab_max_p95': summary['metric_A']['p95'],
            'A_hit_rate_30': summary['metric_A']['hit_rate_0.3'],
            'B_normal_margin': summary['metric_B']['normal_mean'],
            'B_abnormal_margin': summary['metric_B']['abnormal_mean'],
            'B_separation': summary['metric_B']['separation'],
            'B_overlap': summary['metric_B']['overlap_ratio'],
            'C_collapse_score': summary['metric_C']['collapse_score'],
            'D_top1_count': summary['metric_D']['proto_counts'][np.argmax(summary['metric_D']['proto_counts'])]
        }
        all_metrics.append(metrics)
        
        print(f"\n📊 {class_name} (Baseline: {perf['baseline']:.2f}% → Prompt2: {perf['prompt2']:.2f}%, Δ={perf['delta']:.2f}%)")
        print(f"   [A] 异常max偶然命中: 均值={metrics['A_ab_max_mean']:.3f}, P95={metrics['A_ab_max_p95']:.3f}, >0.3命中率={metrics['A_hit_rate_30']:.2%}")
        print(f"   [B] 判别裕度: Normal={metrics['B_normal_margin']:.4f}, Abnormal={metrics['B_abnormal_margin']:.4f}, 分离度={metrics['B_separation']:.4f}, 重叠={metrics['B_overlap']:.2%}")
        print(f"   [C] 原型塌缩: collapse_score={metrics['C_collapse_score']:.3f}")
        print(f"   [D] 坏原型: Top1原型命中{metrics['D_top1_count']}次")
    
    # 统计对比
    print()
    print("="*80)
    print("3️⃣  退化类 vs 改进类 对比统计")
    print("="*80)
    
    df = pd.DataFrame(all_metrics)
    
    # 按类型分组
    degraded_df = df[df['type'] == '退化类']
    improved_df = df[df['type'] == '改进类']
    
    print()
    print("📈 指标均值对比:")
    print("-"*80)
    print(f"{'指标':<30} {'退化类均值':>15} {'改进类均值':>15} {'差异':>15}")
    print("-"*80)
    
    metrics_to_compare = [
        ('A_ab_max_mean', '异常max偶然命中(均值)'),
        ('A_ab_max_p95', '异常max偶然命中(P95)'),
        ('A_hit_rate_30', '偶然命中率>0.3'),
        ('B_normal_margin', 'Normal样本裕度'),
        ('B_separation', '判别分离度'),
        ('B_overlap', '裕度重叠率'),
        ('C_collapse_score', '原型塌缩分数'),
    ]
    
    for metric_name, metric_label in metrics_to_compare:
        degraded_mean = degraded_df[metric_name].mean()
        improved_mean = improved_df[metric_name].mean()
        diff = degraded_mean - improved_mean
        
        # 判断趋势
        if metric_name in ['A_ab_max_mean', 'A_ab_max_p95', 'A_hit_rate_30', 'B_overlap', 'C_collapse_score']:
            # 这些指标越高越坏
            trend = "⬆️ 退化类更高" if diff > 0 else "⬇️ 改进类更高"
        else:
            # 这些指标越高越好
            trend = "⬆️ 退化类更好" if diff > 0 else "⬇️ 改进类更好"
        
        print(f"{metric_label:<30} {degraded_mean:>15.4f} {improved_mean:>15.4f} {diff:>15.4f}  {trend}")
    
    # 关键发现
    print()
    print("="*80)
    print("4️⃣  关键发现 (Key Findings)")
    print("="*80)
    
    findings = []
    
    # 假设1: 异常max偶然命中
    degraded_hit_mean = degraded_df['A_ab_max_mean'].mean()
    improved_hit_mean = improved_df['A_ab_max_mean'].mean()
    if degraded_hit_mean > improved_hit_mean:
        findings.append({
            'hypothesis': '假设1: 异常max偶然命中',
            'evidence': f'退化类异常max均值({degraded_hit_mean:.3f}) > 改进类({improved_hit_mean:.3f})',
            'conclusion': '✅ 支持 - 退化类确实有更高的偶然命中',
            'severity': 'HIGH' if degraded_hit_mean > 0.25 else 'MEDIUM'
        })
    else:
        findings.append({
            'hypothesis': '假设1: 异常max偶然命中',
            'evidence': f'退化类异常max均值({degraded_hit_mean:.3f}) ≤ 改进类({improved_hit_mean:.3f})',
            'conclusion': '❌ 不支持',
            'severity': 'LOW'
        })
    
    # 假设2: 原型塌缩
    degraded_collapse = degraded_df['C_collapse_score'].mean()
    improved_collapse = improved_df['C_collapse_score'].mean()
    if degraded_collapse > improved_collapse:
        findings.append({
            'hypothesis': '假设2: 原型塌缩',
            'evidence': f'退化类塌缩分数({degraded_collapse:.3f}) > 改进类({improved_collapse:.3f})',
            'conclusion': '✅ 支持 - 退化类原型更冗余',
            'severity': 'HIGH' if degraded_collapse > 0.93 else 'MEDIUM'
        })
    else:
        findings.append({
            'hypothesis': '假设2: 原型塌缩',
            'evidence': f'退化类塌缩分数({degraded_collapse:.3f}) ≤ 改进类({improved_collapse:.3f})',
            'conclusion': '❌ 不支持',
            'severity': 'LOW'
        })
    
    # 假设3: 判别裕度不足
    degraded_sep = degraded_df['B_separation'].mean()
    improved_sep = improved_df['B_separation'].mean()
    if degraded_sep < improved_sep:
        findings.append({
            'hypothesis': '假设3: 判别裕度不足',
            'evidence': f'退化类分离度({degraded_sep:.4f}) < 改进类({improved_sep:.4f})',
            'conclusion': '✅ 支持 - 退化类判别能力更弱',
            'severity': 'HIGH' if abs(degraded_sep) < 0.005 else 'MEDIUM'
        })
    else:
        findings.append({
            'hypothesis': '假设3: 判别裕度不足',
            'evidence': f'退化类分离度({degraded_sep:.4f}) ≥ 改进类({improved_sep:.4f})',
            'conclusion': '❌ 不支持',
            'severity': 'LOW'
        })
    
    # 打印发现
    for i, finding in enumerate(findings, 1):
        print(f"\n{i}. {finding['hypothesis']}")
        print(f"   证据: {finding['evidence']}")
        print(f"   结论: {finding['conclusion']}")
        print(f"   严重性: {finding['severity']}")
    
    # 保存分析结果
    df.to_csv('diagnostics/analysis_summary.csv', index=False)
    print()
    print("="*80)
    print("✅ 分析完成！详细数据已保存到: diagnostics/analysis_summary.csv")
    print("="*80)

if __name__ == '__main__':
    main()
