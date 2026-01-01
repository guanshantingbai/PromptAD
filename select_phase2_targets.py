"""
Phase 2 类别级清洗目标选择工具
选择标准：
1. Semantic AUROC < 阈值（默认 90）
2. Dangerous-but-Useful 占比 > 阈值（默认 50%）
"""

import pandas as pd
import argparse

def select_targets(dataset='mvtec', k_shot=2, 
                   semantic_threshold=90, useful_threshold=50):
    """
    选择需要类别级清洗的目标类别
    
    Args:
        dataset: 数据集名称
        k_shot: K-shot 数量
        semantic_threshold: Semantic AUROC 阈值（低于此值才考虑）
        useful_threshold: Useful 占比阈值（高于此值才考虑）
    """
    
    # 加载 Phase 1.5 类别汇总
    phase15_path = f"result/prompt_purging/phase1_5_summary/{dataset}_k{k_shot}_class_summary.csv"
    try:
        df_phase15 = pd.read_csv(phase15_path)
    except FileNotFoundError:
        print(f"❌ Phase 1.5 数据不存在: {phase15_path}")
        print(f"   请先运行 Phase 1.5 分析生成该文件")
        return None
    
    # 加载 Baseline 性能
    baseline_path = f"result/baseline/{dataset}/k_{k_shot}/csv/Seed_111-results.csv"
    df_baseline = pd.read_csv(baseline_path, index_col=0)
    
    # 提取类别名和 Semantic AUROC
    df_baseline['class'] = df_baseline.index.str.replace(f'{dataset}-', '')
    df_baseline = df_baseline[['class', 'semantic_i_roc']].rename(
        columns={'semantic_i_roc': 'semantic_auroc'}
    )
    
    # 合并数据
    df = df_phase15.merge(df_baseline, on='class')
    
    # 计算 Useful 占比
    df['useful_percentage'] = (df['dangerous_useful'] / df['total_prompts'] * 100)
    
    # 筛选条件
    df['meets_semantic'] = df['semantic_auroc'] < semantic_threshold
    df['meets_useful'] = df['useful_percentage'] > useful_threshold
    df['priority'] = df['meets_semantic'] & df['meets_useful']
    
    # 排序：优先级 > Useful占比 > Semantic分数（越低越优先）
    df = df.sort_values(
        by=['priority', 'useful_percentage', 'semantic_auroc'],
        ascending=[False, False, True]
    )
    
    return df

def print_report(df, dataset, semantic_threshold, useful_threshold):
    """打印选择报告"""
    
    print("="*80)
    print(f"Phase 2 类别级清洗目标选择")
    print("="*80)
    print(f"数据集: {dataset.upper()}")
    print(f"选择标准:")
    print(f"  - Semantic AUROC < {semantic_threshold}")
    print(f"  - Dangerous-but-Useful 占比 > {useful_threshold}%")
    print("="*80)
    print()
    
    # 符合条件的类别
    priority_classes = df[df['priority']]
    
    if len(priority_classes) > 0:
        print(f"✅ 找到 {len(priority_classes)} 个符合条件的类别（优先级排序）:")
        print()
        print(f"{'类别':<15} {'Semantic':<10} {'Useful%':<10} {'Useless%':<10} {'Total':<8} 建议")
        print("-"*80)
        
        for _, row in priority_classes.iterrows():
            cls = row['class']
            sem = row['semantic_auroc']
            useful_pct = row['useful_percentage']
            useless_pct = row['useless_percentage']
            total = row['total_prompts']
            
            print(f"{cls:<15} {sem:<10.2f} {useful_pct:<10.1f} {useless_pct:<10.1f} {total:<8} 类别级清洗")
    else:
        print("⚠️  没有找到完全符合条件的类别")
    
    print()
    print("="*80)
    print("其他候选类别（接近阈值）:")
    print("="*80)
    print()
    
    # 接近阈值的类别
    near_threshold = df[~df['priority'] & (
        (df['semantic_auroc'] < semantic_threshold + 10) | 
        (df['useful_percentage'] > useful_threshold - 10)
    )]
    
    if len(near_threshold) > 0:
        print(f"{'类别':<15} {'Semantic':<10} {'Useful%':<10} {'Useless%':<10} {'符合条件'}")
        print("-"*80)
        
        for _, row in near_threshold.iterrows():
            cls = row['class']
            sem = row['semantic_auroc']
            useful_pct = row['useful_percentage']
            useless_pct = row['useless_percentage']
            
            # 标记哪些条件符合
            conditions = []
            if sem < semantic_threshold:
                conditions.append("✓Semantic")
            else:
                conditions.append(f"Semantic:{sem:.0f}")
            
            if useful_pct > useful_threshold:
                conditions.append("✓Useful")
            else:
                conditions.append(f"Useful:{useful_pct:.0f}%")
            
            condition_str = ", ".join(conditions)
            
            print(f"{cls:<15} {sem:<10.2f} {useful_pct:<10.1f} {useless_pct:<10.1f} {condition_str}")
    
    print()
    print("="*80)
    print("完整排名（按 Useful 占比降序）:")
    print("="*80)
    print()
    
    df_sorted = df.sort_values('useful_percentage', ascending=False)
    print(f"{'排名':<5} {'类别':<15} {'Semantic':<10} {'Useful%':<10} {'Useless%':<10} {'Safe':<6} {'Total':<6}")
    print("-"*80)
    
    for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
        cls = row['class']
        sem = row['semantic_auroc']
        useful_pct = row['useful_percentage']
        useless_pct = row['useless_percentage']
        safe = row['safe']
        total = row['total_prompts']
        
        priority_marker = "⭐" if row['priority'] else ""
        
        print(f"{i:<5} {cls:<15} {sem:<10.2f} {useful_pct:<10.1f} {useless_pct:<10.1f} {safe:<6} {total:<6} {priority_marker}")
    
    print()
    print("="*80)
    print("建议操作:")
    print("="*80)
    
    if len(priority_classes) > 0:
        print("\n1. 对以下类别进行类别级 Prompt 清洗（删除 Useless prompts）:")
        for cls in priority_classes['class'].values:
            print(f"   - {cls}")
        
        print(f"\n2. 清洗后预期:")
        print(f"   - Semantic AUROC 提升（Useful prompts 占主导）")
        print(f"   - 噪声减少（Useless prompts 被移除）")
    else:
        print("\n当前阈值下没有优先清洗的类别。")
        print("建议:")
        print("  - 降低 useful_threshold (如 --useful-threshold 40)")
        print("  - 或提高 semantic_threshold (如 --semantic-threshold 95)")
    
    print()

def main():
    parser = argparse.ArgumentParser(description='Phase 2 类别级清洗目标选择')
    parser.add_argument('--dataset', type=str, default='mvtec', 
                       choices=['mvtec', 'visa'],
                       help='数据集名称')
    parser.add_argument('--k-shot', type=int, default=2,
                       help='K-shot 数量')
    parser.add_argument('--semantic-threshold', type=float, default=90,
                       help='Semantic AUROC 阈值（低于此值才考虑）')
    parser.add_argument('--useful-threshold', type=float, default=50,
                       help='Dangerous-but-Useful 占比阈值（高于此值才考虑）')
    parser.add_argument('--output', type=str, default=None,
                       help='保存结果到 CSV 文件')
    
    args = parser.parse_args()
    
    df = select_targets(
        dataset=args.dataset,
        k_shot=args.k_shot,
        semantic_threshold=args.semantic_threshold,
        useful_threshold=args.useful_threshold
    )
    
    if df is not None:
        print_report(df, args.dataset, args.semantic_threshold, args.useful_threshold)
        
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"\n✓ 结果已保存到: {args.output}")

if __name__ == "__main__":
    main()
