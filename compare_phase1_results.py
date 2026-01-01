#!/usr/bin/env python3
"""
对比 Phase 1 清洗前后的结果
比较 baseline vs cleaned prompts 的 AUROC 变化
"""

import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path


def load_baseline_results(dataset, k_shot, seed=111):
    """加载 baseline 结果"""
    
    baseline_path = f"result/baseline/{dataset}/k_{k_shot}/csv/Seed_{seed}-results.csv"
    
    if not os.path.exists(baseline_path):
        print(f"⚠️  Warning: Baseline results not found at {baseline_path}")
        return None
    
    df = pd.read_csv(baseline_path, index_col=0)  # 第一列是索引
    df.index.name = 'Class'
    df = df.reset_index()  # 将索引变成普通列
    print(f"✓ Loaded baseline results: {baseline_path}")
    print(f"  Classes: {len(df)}")
    
    return df


def load_cleaned_results(dataset, k_shot, seed=111):
    """加载 cleaned prompts 的结果"""
    
    # 直接读取汇总的CSV文件
    cleaned_path = f"result/phase1_cleaned/{dataset}/k_{k_shot}/csv/Seed_{seed}-results.csv"
    
    if not os.path.exists(cleaned_path):
        print(f"⚠️  Warning: Cleaned results not found at {cleaned_path}")
        return None
    
    df = pd.read_csv(cleaned_path, index_col=0)  # 第一列是索引
    df.index.name = 'Class'
    df = df.reset_index()  # 将索引变成普通列
    print(f"✓ Loaded cleaned results: {cleaned_path}")
    print(f"  Classes: {len(df)}")
    
    return df


def compare_results(baseline_df, cleaned_df, dataset, k_shot):
    """对比两组结果"""
    
    if baseline_df is None or cleaned_df is None:
        print("❌ Error: Missing baseline or cleaned results")
        return None
    
    # 合并数据
    # 注意：CSV文件中列名是小写的 'semantic_i_roc' 等
    baseline_df = baseline_df.rename(columns={
        'semantic_i_roc': 'Semantic_AUROC_Baseline',
        'memory_i_roc': 'Memory_AUROC_Baseline',
        'i_roc': 'Fusion_AUROC_Baseline'
    })
    
    cleaned_df = cleaned_df.rename(columns={
        'semantic_i_roc': 'Semantic_AUROC_Cleaned',
        'memory_i_roc': 'Memory_AUROC_Cleaned',
        'i_roc': 'Fusion_AUROC_Cleaned'
    })
    
    # 按 Class 合并
    comparison = pd.merge(
        baseline_df[['Class', 'Semantic_AUROC_Baseline', 'Memory_AUROC_Baseline', 'Fusion_AUROC_Baseline']],
        cleaned_df[['Class', 'Semantic_AUROC_Cleaned', 'Memory_AUROC_Cleaned', 'Fusion_AUROC_Cleaned']],
        on='Class',
        how='outer'
    )
    
    # 计算变化
    comparison['Semantic_Delta'] = (
        comparison['Semantic_AUROC_Cleaned'] - comparison['Semantic_AUROC_Baseline']
    )
    comparison['Memory_Delta'] = (
        comparison['Memory_AUROC_Cleaned'] - comparison['Memory_AUROC_Baseline']
    )
    comparison['Fusion_Delta'] = (
        comparison['Fusion_AUROC_Cleaned'] - comparison['Fusion_AUROC_Baseline']
    )
    
    # 按 Semantic Delta 排序
    comparison = comparison.sort_values('Semantic_Delta', ascending=False)
    
    return comparison


def print_summary(comparison):
    """打印汇总统计"""
    
    print(f"\n{'='*70}")
    print("Phase 1 Cleaning Results Summary")
    print(f"{'='*70}\n")
    
    # 总体统计
    avg_semantic_delta = comparison['Semantic_Delta'].mean()
    avg_memory_delta = comparison['Memory_Delta'].mean()
    
    improved_classes = len(comparison[comparison['Semantic_Delta'] > 0])
    degraded_classes = len(comparison[comparison['Semantic_Delta'] < 0])
    unchanged_classes = len(comparison[comparison['Semantic_Delta'] == 0])
    
    print(f"Overall Performance Change:")
    print(f"  Semantic AUROC: {avg_semantic_delta:+.2f} (average)")
    print(f"  Memory AUROC:   {avg_memory_delta:+.2f} (average)")
    print()
    
    print(f"Class Distribution:")
    print(f"  Improved:  {improved_classes} classes")
    print(f"  Degraded:  {degraded_classes} classes")
    print(f"  Unchanged: {unchanged_classes} classes")
    print()
    
    # Top 改进
    print(f"{'='*70}")
    print("Top 5 Improved Classes (Semantic AUROC):")
    print(f"{'='*70}\n")
    
    top_improved = comparison.head(5)
    for _, row in top_improved.iterrows():
        print(f"  {row['Class']:15s}: "
              f"{row['Semantic_AUROC_Baseline']:.2f} → {row['Semantic_AUROC_Cleaned']:.2f} "
              f"({row['Semantic_Delta']:+.2f})")
    
    # Top 下降
    print(f"\n{'='*70}")
    print("Top 5 Degraded Classes (Semantic AUROC):")
    print(f"{'='*70}\n")
    
    top_degraded = comparison.tail(5)[::-1]
    for _, row in top_degraded.iterrows():
        print(f"  {row['Class']:15s}: "
              f"{row['Semantic_AUROC_Baseline']:.2f} → {row['Semantic_AUROC_Cleaned']:.2f} "
              f"({row['Semantic_Delta']:+.2f})")
    
    print(f"\n{'='*70}")


def save_comparison(comparison, output_path):
    """保存对比结果"""
    
    comparison.to_csv(output_path, index=False, float_format='%.4f')
    print(f"\n✓ Comparison saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare Phase 1 baseline vs cleaned results'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['mvtec', 'visa'],
        help='Dataset name'
    )
    parser.add_argument(
        '--k-shot',
        type=int,
        default=2,
        help='K-shot value'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=111,
        help='Random seed'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV path (default: result/phase1_cleaned/{dataset}/k_{k_shot}/comparison.csv)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("Phase 1: Baseline vs Cleaned Comparison")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"K-shot:  {args.k_shot}")
    print(f"Seed:    {args.seed}")
    print("="*70)
    
    # 加载数据
    baseline_df = load_baseline_results(args.dataset, args.k_shot, args.seed)
    cleaned_df = load_cleaned_results(args.dataset, args.k_shot, args.seed)
    
    # 对比
    comparison = compare_results(baseline_df, cleaned_df, args.dataset, args.k_shot)
    
    if comparison is not None:
        # 打印汇总
        print_summary(comparison)
        
        # 保存结果
        if args.output is None:
            args.output = f"result/phase1_cleaned/{args.dataset}/k_{args.k_shot}/comparison.csv"
        
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        save_comparison(comparison, args.output)
        
        # 详细表格
        print(f"\n{'='*70}")
        print("Detailed Comparison:")
        print(f"{'='*70}\n")
        print(comparison.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
        print()


if __name__ == '__main__':
    main()
