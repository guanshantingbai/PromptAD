#!/usr/bin/env python3
"""
Step 2: 批量运行诊断脚本，汇总全部27个类别的指标
"""

import os
import subprocess
import json
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# 读取类别列表
performance_data = pd.read_csv('analysis/full_performance_comparison_k2.csv')
classes_to_diagnose = performance_data['class'].tolist()

def run_diagnosis(class_name):
    """运行单个类别的诊断"""
    try:
        cmd = [
            'python', 'diagnose_prototypes.py',
            '--k-shot', '2',
            '--classes', class_name
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )
        
        if result.returncode == 0:
            return {'class': class_name, 'status': 'success'}
        else:
            return {'class': class_name, 'status': 'failed', 'error': result.stderr[-200:]}
    except subprocess.TimeoutExpired:
        return {'class': class_name, 'status': 'timeout'}
    except Exception as e:
        return {'class': class_name, 'status': 'error', 'error': str(e)}

def main():
    print("="*80)
    print("Step 2: 批量诊断全部27个类别")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总类别数: {len(classes_to_diagnose)}")
    print(f"并行进程数: 4")
    print("="*80)
    print()
    
    # 检查已完成的类别
    existing_diagnostics = []
    for class_name in classes_to_diagnose:
        dataset = class_name.split('-')[0]
        cls_short = class_name.split('-')[1]
        summary_path = f'diagnostics/{dataset}_k2_{cls_short}/summary.json'
        if os.path.exists(summary_path):
            existing_diagnostics.append(class_name)
    
    print(f"✅ 已完成诊断: {len(existing_diagnostics)}/{len(classes_to_diagnose)} 个类别")
    
    # 需要运行的类别
    classes_to_run = [c for c in classes_to_diagnose if c not in existing_diagnostics]
    
    if len(classes_to_run) == 0:
        print("所有类别已完成诊断，跳过批量运行")
    else:
        print(f"⏳ 需要运行: {len(classes_to_run)} 个类别")
        print()
        
        # 并行运行诊断
        results = []
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(run_diagnosis, cls): cls for cls in classes_to_run}
            
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed += 1
                
                status_icon = "✅" if result['status'] == 'success' else "❌"
                print(f"[{completed}/{len(classes_to_run)}] {status_icon} {result['class']}: {result['status']}")
        
        print()
        print("="*80)
        print(f"批量诊断完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 统计结果
        success_count = sum(1 for r in results if r['status'] == 'success')
        failed_count = len(results) - success_count
        print(f"成功: {success_count}, 失败: {failed_count}")
        
        if failed_count > 0:
            print("\n失败的类别:")
            for r in results:
                if r['status'] != 'success':
                    print(f"  - {r['class']}: {r['status']}")
                    if 'error' in r:
                        print(f"    错误: {r['error']}")
    
    print()
    print("="*80)
    print("汇总诊断指标...")
    print("="*80)
    
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
    
    # 保存完整数据
    full_data.to_csv('analysis/full_metrics_k2.csv', index=False)
    
    print(f"✅ 成功汇总: {len(all_metrics)}/{len(classes_to_diagnose)} 个类别")
    if missing_classes:
        print(f"⚠️  缺失诊断数据: {len(missing_classes)} 个类别")
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
    print("✅ Step 2 完成！")
    print("="*80)

if __name__ == '__main__':
    main()
