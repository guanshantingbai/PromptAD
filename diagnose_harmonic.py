#!/usr/bin/env python3
"""
诊断Harmonic融合与原始实现的差异
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path


def analyze_harmonic_discrepancy():
    """分析harmonic与max的差异"""
    
    result_dir = Path('result/gate')
    
    print("="*80)
    print("Harmonic vs Max 差异分析")
    print("="*80)
    
    discrepancies = []
    
    for json_file in result_dir.rglob('gate_results/*.json'):
        parts = json_file.parts
        dataset = parts[-4]
        k_str = parts[-3]
        k_shot = int(k_str.split('_')[1])
        
        filename = json_file.stem
        name_parts = filename.rsplit('_', 2)
        class_name = name_parts[0]
        task = name_parts[-1]
        
        with open(json_file, 'r') as f:
            results = json.load(f)
        
        if 'max' in results and 'harmonic' in results:
            max_auroc = results['max'].get('i_roc' if task == 'cls' else 'p_roc', 0)
            harmonic_auroc = results['harmonic'].get('i_roc' if task == 'cls' else 'p_roc', 0)
            
            diff = max_auroc - harmonic_auroc
            
            # 获取分支分数
            semantic_auroc = results['max'].get('i_roc_semantic', None)
            memory_auroc = results['max'].get('i_roc_memory', None)
            
            if semantic_auroc and memory_auroc:
                branch_gap = abs(semantic_auroc - memory_auroc)
                weak_branch = 'semantic' if semantic_auroc < memory_auroc else 'memory'
            else:
                branch_gap = 0
                weak_branch = 'unknown'
            
            discrepancies.append({
                'dataset': dataset,
                'class': class_name,
                'k_shot': k_shot,
                'task': task,
                'max': max_auroc,
                'harmonic': harmonic_auroc,
                'diff': diff,
                'semantic': semantic_auroc,
                'memory': memory_auroc,
                'branch_gap': branch_gap,
                'weak_branch': weak_branch
            })
    
    df = pd.DataFrame(discrepancies)
    
    # 总体统计
    print("\n【总体统计】")
    print(f"平均差异: {df['diff'].mean():.2f}%")
    print(f"中位数差异: {df['diff'].median():.2f}%")
    print(f"最大差异: {df['diff'].max():.2f}%")
    print(f"标准差: {df['diff'].std():.2f}%")
    
    # 按任务分组
    print("\n【按任务分组】")
    for task in ['cls', 'seg']:
        task_df = df[df['task'] == task]
        print(f"\n{task.upper()}:")
        print(f"  平均差异: {task_df['diff'].mean():.2f}%")
        print(f"  Max平均: {task_df['max'].mean():.2f}%")
        print(f"  Harmonic平均: {task_df['harmonic'].mean():.2f}%")
    
    # 按数据集分组
    print("\n【按数据集分组】")
    for dataset in ['mvtec', 'visa']:
        for task in ['cls', 'seg']:
            subset = df[(df['dataset'] == dataset) & (df['task'] == task)]
            if len(subset) > 0:
                print(f"\n{dataset.upper()} {task.upper()}:")
                print(f"  平均差异: {subset['diff'].mean():.2f}%")
                print(f"  Max平均: {subset['max'].mean():.2f}%")
                print(f"  Harmonic平均: {subset['harmonic'].mean():.2f}%")
    
    # 差异最大的案例
    print("\n【差异最大的10个案例】")
    top_diff = df.nlargest(10, 'diff')
    print(top_diff[['dataset', 'class', 'k_shot', 'task', 'max', 'harmonic', 'diff', 'branch_gap']].to_string(index=False))
    
    # 分析差异与分支差距的关系
    print("\n【差异与分支差距的关系】")
    
    # 只分析有分支信息的记录
    with_branches = df[df['semantic'].notna()]
    
    if len(with_branches) > 0:
        # 计算相关系数
        corr = with_branches['diff'].corr(with_branches['branch_gap'])
        print(f"差异与分支差距的相关系数: {corr:.3f}")
        
        # 分组统计
        print("\n按分支差距分组:")
        with_branches['gap_group'] = pd.cut(with_branches['branch_gap'], 
                                            bins=[0, 5, 10, 20, 100],
                                            labels=['<5%', '5-10%', '10-20%', '>20%'])
        
        for group in ['<5%', '5-10%', '10-20%', '>20%']:
            group_df = with_branches[with_branches['gap_group'] == group]
            if len(group_df) > 0:
                print(f"  {group}: 平均差异 {group_df['diff'].mean():.2f}%, "
                      f"样本数 {len(group_df)}")
    
    # Harmonic公式验证
    print("\n【Harmonic公式验证】")
    print("检查harmonic mean计算是否正确...")
    
    with_branches = df[df['semantic'].notna()]
    if len(with_branches) > 0:
        # 计算理论harmonic mean (修正公式)
        eps = 1e-8
        theoretical_harmonic = 1.0 / (0.5 * (1.0 / (with_branches['semantic'] + eps) + 
                                             1.0 / (with_branches['memory'] + eps)))
        
        # 检查是否匹配
        match_ratio = (abs(theoretical_harmonic - with_branches['harmonic']) < 0.01).mean()
        print(f"  公式匹配率: {match_ratio*100:.1f}%")
        
        if match_ratio < 0.95:
            print("  ⚠️ 警告: Harmonic计算可能有问题!")
            # 显示一些样本进行调试
            print("\n  样本检查:")
            sample = with_branches.head(3)
            for idx, row in sample.iterrows():
                expected = 1.0 / (0.5 * (1.0/(row['semantic']+eps) + 1.0/(row['memory']+eps)))
                print(f"    {row['dataset']}/{row['class']}: semantic={row['semantic']:.2f}, "
                      f"memory={row['memory']:.2f}, harmonic={row['harmonic']:.2f}, "
                      f"期望={expected:.2f}")
        else:
            print("  ✓ Harmonic计算正确")
    
    # 保存详细结果
    output_file = Path('result/gate/analysis/harmonic_discrepancy.csv')
    df.to_csv(output_file, index=False)
    print(f"\n✓ 详细结果已保存: {output_file}")
    
    return df


def compare_with_max_score_baseline():
    """对比max_score实验的结果"""
    print("\n" + "="*80)
    print("对比max_score基线结果")
    print("="*80)
    
    gate_results = []
    max_score_results = []
    
    # 加载gate实验的max结果
    for json_file in Path('result/gate').rglob('gate_results/*.json'):
        parts = json_file.parts
        dataset = parts[-4]
        k_str = parts[-3]
        k_shot = int(k_str.split('_')[1])
        
        filename = json_file.stem
        name_parts = filename.rsplit('_', 2)
        class_name = name_parts[0]
        task = name_parts[-1]
        
        with open(json_file, 'r') as f:
            results = json.load(f)
        
        if 'max' in results:
            metric_key = 'i_roc' if task == 'cls' else 'p_roc'
            gate_results.append({
                'dataset': dataset,
                'class': class_name,
                'k_shot': k_shot,
                'task': task,
                'auroc': results['max'][metric_key]
            })
    
    # 尝试加载max_score实验结果
    max_score_csv = Path('result/max_score/aggregated_results.csv')
    if max_score_csv.exists():
        baseline_df = pd.read_csv(max_score_csv)
        print(f"\n找到max_score基线结果: {len(baseline_df)}条记录")
        
        # 检查baseline_df的列
        print(f"baseline列名: {baseline_df.columns.tolist()}")
        
        gate_df = pd.DataFrame(gate_results)
        print(f"gate列名: {gate_df.columns.tolist()}")
        
        # 根据实际列名进行merge
        if 'Dataset' in baseline_df.columns:
            # 重命名为小写
            baseline_df = baseline_df.rename(columns={
                'Dataset': 'dataset',
                'Class': 'class',
                'K-shot': 'k_shot',
                'Task': 'task',
                'AUROC': 'auroc_baseline'
            })
        
        # Merge
        merged = pd.merge(gate_df, baseline_df, 
                         on=['dataset', 'class', 'k_shot', 'task'],
                         how='inner')
        
        if len(merged) > 0:
            merged['diff'] = merged['auroc'] - merged['auroc_baseline']
            
            print(f"\n匹配到 {len(merged)} 条可对比记录")
            print(f"平均差异: {merged['diff'].mean():.3f}%")
            print(f"最大差异: {merged['diff'].max():.3f}%")
            print(f"最小差异: {merged['diff'].min():.3f}%")
            
            # 按任务统计
            for task in ['cls', 'seg']:
                task_diff = merged[merged['task'] == task]['diff'].mean()
                print(f"{task.upper()} 平均差异: {task_diff:.3f}%")
        else:
            print("⚠️ 没有找到匹配的记录")
    else:
        print(f"⚠️ 未找到max_score基线结果文件: {max_score_csv}")


if __name__ == '__main__':
    df = analyze_harmonic_discrepancy()
    compare_with_max_score_baseline()
    
    print("\n" + "="*80)
    print("分析完成!")
    print("="*80)
