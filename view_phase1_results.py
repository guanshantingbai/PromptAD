#!/usr/bin/env python
"""
查看 Prompt Purging Phase 1 结果的可视化工具
"""

import os
import sys
import pandas as pd
import argparse


def view_results(csv_path, top_k=None, show_all=False):
    """查看Phase 1结果"""
    
    if not os.path.exists(csv_path):
        print(f"❌ 文件不存在: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    # 提取信息
    classname = df['class'].iloc[0]
    display_name = df['display_name'].iloc[0]
    total_prompts = len(df)
    
    print("="*80)
    print(f"Prompt Purging Phase 1 结果")
    print("="*80)
    print(f"类别: {classname} ({display_name})")
    print(f"总 prompts: {total_prompts}")
    print(f"  - Generic: {(df['type'] == 'generic').sum()}")
    print(f"  - Specific: {(df['type'] == 'specific').sum()}")
    print(f"\n样本数: {df['num_samples'].iloc[0]}")
    
    # 风险等级分布
    print(f"\n风险等级分布:")
    for level in ['high', 'medium', 'low']:
        count = (df['risk_level'] == level).sum()
        pct = count / total_prompts * 100
        print(f"  - {level.capitalize()}: {count} ({pct:.1f}%)")
    
    # 显示详细信息
    if show_all:
        display_df = df
        title = "所有 prompts"
    else:
        display_df = df.head(top_k) if top_k else df.head(10)
        title = f"Top {len(display_df)} 高风险 prompts"
    
    print(f"\n{title}:")
    print("-"*80)
    print(f"{'Idx':<4} {'R_eps':>6} {'R_0':>6} {'Median':>7} {'Q10':>7} {'Mean':>7} {'Std':>7} {'Risk':<6} {'Prompt'}")
    print("-"*80)
    
    for _, row in display_df.iterrows():
        idx = int(row['prompt_index'])
        r_eps = row['R_j_eps']
        r_0 = row['R_j_0']
        median = row['median_margin']
        q10 = row['q10_margin']
        mean = row['mean_margin']
        std = row['std_margin']
        risk = row['risk_level']
        text = row['full_text']
        
        # 根据风险等级着色（使用ANSI颜色）
        if risk == 'high':
            risk_str = f"\033[91m{risk:<6}\033[0m"  # 红色
        elif risk == 'medium':
            risk_str = f"\033[93m{risk:<6}\033[0m"  # 黄色
        else:
            risk_str = f"\033[92m{risk:<6}\033[0m"  # 绿色
        
        print(f"[{idx:2d}] {r_eps:6.3f} {r_0:6.3f} {median:7.2f} {q10:7.2f} {mean:7.2f} {std:7.2f} {risk_str} {text}")
    
    # 统计摘要
    print("\n" + "="*80)
    print("统计摘要")
    print("="*80)
    print(f"R_j_eps: mean={df['R_j_eps'].mean():.3f}, median={df['R_j_eps'].median():.3f}, max={df['R_j_eps'].max():.3f}")
    print(f"R_j_0:   mean={df['R_j_0'].mean():.3f}, median={df['R_j_0'].median():.3f}, max={df['R_j_0'].max():.3f}")
    print(f"Median margin: mean={df['median_margin'].mean():.3f}, min={df['median_margin'].min():.3f}, max={df['median_margin'].max():.3f}")
    
    # 高风险 prompt 分析
    high_risk = df[df['risk_level'] == 'high']
    if len(high_risk) > 0:
        print(f"\n高风险 prompts 分析:")
        print(f"  - 数量: {len(high_risk)}")
        print(f"  - 平均 R_eps: {high_risk['R_j_eps'].mean():.3f}")
        print(f"  - 平均 R_0: {high_risk['R_j_0'].mean():.3f}")
        print(f"  - 平均 median margin: {high_risk['median_margin'].mean():.3f}")
        
        # 按类型统计
        print(f"\n  按类型分布:")
        for ptype in high_risk['type'].unique():
            count = (high_risk['type'] == ptype).sum()
            print(f"    - {ptype}: {count}")
    
    print("\n" + "="*80)


def compare_classes(result_dir, dataset, k_shot, top_k=3):
    """比较多个类别的结果"""
    
    dir_path = os.path.join(result_dir, dataset, f'k_{k_shot}')
    
    if not os.path.exists(dir_path):
        print(f"❌ 目录不存在: {dir_path}")
        return
    
    csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"❌ 未找到CSV文件: {dir_path}")
        return
    
    print("="*80)
    print(f"Prompt Purging Phase 1 - 多类别比较")
    print("="*80)
    print(f"数据集: {dataset}")
    print(f"K-shot: {k_shot}")
    print(f"类别数: {len(csv_files)}")
    print()
    
    # 收集所有类别的摘要信息
    summaries = []
    
    for csv_file in sorted(csv_files):
        csv_path = os.path.join(dir_path, csv_file)
        df = pd.read_csv(csv_path)
        
        classname = df['class'].iloc[0]
        total_prompts = len(df)
        high_risk_count = (df['risk_level'] == 'high').sum()
        high_risk_pct = high_risk_count / total_prompts * 100
        mean_r_eps = df['R_j_eps'].mean()
        max_r_eps = df['R_j_eps'].max()
        
        summaries.append({
            'class': classname,
            'total': total_prompts,
            'high_risk': high_risk_count,
            'high_risk_pct': high_risk_pct,
            'mean_R_eps': mean_r_eps,
            'max_R_eps': max_r_eps
        })
    
    # 按高风险比例排序
    summaries = sorted(summaries, key=lambda x: x['high_risk_pct'], reverse=True)
    
    print(f"{'Class':<15} {'Total':>6} {'High':>5} {'High%':>6} {'Mean R_eps':>11} {'Max R_eps':>10}")
    print("-"*80)
    
    for s in summaries:
        print(f"{s['class']:<15} {s['total']:>6} {s['high_risk']:>5} {s['high_risk_pct']:>6.1f}% {s['mean_R_eps']:>11.3f} {s['max_R_eps']:>10.3f}")
    
    # 显示每个类别的top-k高风险prompts
    print("\n" + "="*80)
    print(f"每个类别的 Top-{top_k} 高风险 prompts")
    print("="*80)
    
    for csv_file in sorted(csv_files):
        csv_path = os.path.join(dir_path, csv_file)
        df = pd.read_csv(csv_path)
        
        classname = df['class'].iloc[0]
        print(f"\n{classname}:")
        print("-"*60)
        
        top_prompts = df.head(top_k)
        for _, row in top_prompts.iterrows():
            idx = int(row['prompt_index'])
            r_eps = row['R_j_eps']
            r_0 = row['R_j_0']
            median = row['median_margin']
            text = row['full_text']
            print(f"  [{idx:2d}] R_eps={r_eps:.3f} | R_0={r_0:.3f} | median={median:6.2f} | {text}")


def main():
    parser = argparse.ArgumentParser(description='View Prompt Purging Phase 1 Results')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # view 命令：查看单个类别
    view_parser = subparsers.add_parser('view', help='View single class results')
    view_parser.add_argument('csv_path', type=str, help='Path to CSV file')
    view_parser.add_argument('--top', type=int, default=None, help='Show top K prompts (default: 10)')
    view_parser.add_argument('--all', action='store_true', help='Show all prompts')
    
    # compare 命令：比较多个类别
    compare_parser = subparsers.add_parser('compare', help='Compare multiple classes')
    compare_parser.add_argument('--dataset', type=str, required=True, choices=['mvtec', 'visa'])
    compare_parser.add_argument('--k_shot', type=int, default=2)
    compare_parser.add_argument('--result_dir', type=str, default='result/prompt_purging/phase1')
    compare_parser.add_argument('--top', type=int, default=3, help='Top K prompts per class')
    
    args = parser.parse_args()
    
    if args.command == 'view':
        view_results(args.csv_path, top_k=args.top, show_all=args.all)
    elif args.command == 'compare':
        compare_classes(args.result_dir, args.dataset, args.k_shot, top_k=args.top)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
