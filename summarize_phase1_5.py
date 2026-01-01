"""
Phase 1.5 结果汇总分析
汇总所有类别的 Prompt Classification 结果
"""

import os
import pandas as pd
import numpy as np
import argparse


def summarize_phase1_5_results(dataset='mvtec', k_shot=2):
    """汇总 Phase 1.5 的分类结果"""
    
    print("="*80)
    print(f"Phase 1.5 结果汇总: {dataset} (k={k_shot})")
    print("="*80)
    
    input_dir = f"result/prompt_purging/phase1_5/{dataset}/k_{k_shot}"
    
    if not os.path.exists(input_dir):
        print(f"\n✗ 未找到结果目录: {input_dir}")
        return
    
    # 读取所有 CSV 文件
    all_results = []
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('_classification.csv')]
    
    print(f"\n找到 {len(csv_files)} 个类别的结果")
    
    for csv_file in sorted(csv_files):
        file_path = os.path.join(input_dir, csv_file)
        df = pd.read_csv(file_path)
        all_results.append(df)
    
    # 合并所有结果
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # 总体统计
    total_prompts = len(combined_df)
    safe_count = (combined_df['prompt_classification'] == 'safe').sum()
    useful_count = (combined_df['prompt_classification'] == 'dangerous_useful').sum()
    useless_count = (combined_df['prompt_classification'] == 'dangerous_useless').sum()
    
    print(f"\n" + "="*80)
    print(f"总体统计")
    print("="*80)
    print(f"总 prompts: {total_prompts}")
    print(f"  - Safe:                {safe_count:>4} ({safe_count/total_prompts*100:>5.1f}%)")
    print(f"  - Dangerous-but-Useful: {useful_count:>4} ({useful_count/total_prompts*100:>5.1f}%)")
    print(f"  - Dangerous-and-Useless:{useless_count:>4} ({useless_count/total_prompts*100:>5.1f}%)")
    
    # 按类别统计
    print(f"\n" + "="*80)
    print(f"各类别统计")
    print("="*80)
    print(f"{'Class':<15} {'Total':<7} {'Safe':<7} {'Useful':<7} {'Useless':<8} {'Useless%':<10}")
    print("-"*80)
    
    class_summary = []
    
    for class_name in sorted(combined_df['class'].unique()):
        class_df = combined_df[combined_df['class'] == class_name]
        total = len(class_df)
        safe = (class_df['prompt_classification'] == 'safe').sum()
        useful = (class_df['prompt_classification'] == 'dangerous_useful').sum()
        useless = (class_df['prompt_classification'] == 'dangerous_useless').sum()
        useless_pct = useless / total * 100 if total > 0 else 0
        
        print(f"{class_name:<15} {total:<7} {safe:<7} {useful:<7} {useless:<8} {useless_pct:>5.1f}%")
        
        class_summary.append({
            'class': class_name,
            'total_prompts': total,
            'safe': safe,
            'dangerous_useful': useful,
            'dangerous_useless': useless,
            'useless_percentage': useless_pct,
        })
    
    # Generic vs Specific
    print(f"\n" + "="*80)
    print(f"Generic vs Specific Prompts")
    print("="*80)
    
    for ptype in ['generic', 'specific']:
        type_df = combined_df[combined_df['type'] == ptype]
        if len(type_df) == 0:
            continue
        
        total = len(type_df)
        safe = (type_df['prompt_classification'] == 'safe').sum()
        useful = (type_df['prompt_classification'] == 'dangerous_useful').sum()
        useless = (type_df['prompt_classification'] == 'dangerous_useless').sum()
        
        print(f"\n{ptype.capitalize()} ({total} prompts):")
        print(f"  - Safe:                {safe:>4} ({safe/total*100:>5.1f}%)")
        print(f"  - Dangerous-but-Useful: {useful:>4} ({useful/total*100:>5.1f}%)")
        print(f"  - Dangerous-and-Useless:{useless:>4} ({useless/total*100:>5.1f}%)")
    
    # 高风险 Generic Prompts
    print(f"\n" + "="*80)
    print(f"Dangerous-and-Useless Generic Prompts (跨类别)")
    print("="*80)
    
    useless_generic = combined_df[
        (combined_df['prompt_classification'] == 'dangerous_useless') & 
        (combined_df['type'] == 'generic')
    ]
    
    if len(useless_generic) > 0:
        # 按 template 分组，计算出现次数
        template_counts = useless_generic.groupby('template').size().sort_values(ascending=False)
        
        print(f"\n出现次数最多的 Dangerous-and-Useless Generic Prompts:")
        print("-"*80)
        
        for template, count in template_counts.head(10).items():
            # 计算该 template 的平均 separation_gap
            template_df = useless_generic[useless_generic['template'] == template]
            avg_gap = template_df['separation_gap'].mean()
            avg_r_eps = template_df['R_j_eps'].mean()
            
            print(f"  {template:<30} | 出现{count:>2}次 | avg_gap={avg_gap:>6.2f} | avg_R_eps={avg_r_eps:.3f}")
    
    # 保存汇总
    output_dir = f"result/prompt_purging/phase1_5_summary"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存类别级汇总
    class_summary_df = pd.DataFrame(class_summary)
    class_file = f"{output_dir}/{dataset}_k{k_shot}_class_summary.csv"
    class_summary_df.to_csv(class_file, index=False)
    print(f"\n✓ 类别汇总已保存: {class_file}")
    
    # 保存完整详情
    detail_file = f"{output_dir}/{dataset}_k{k_shot}_all_prompts.csv"
    combined_df.to_csv(detail_file, index=False)
    print(f"✓ 完整详情已保存: {detail_file}")
    
    print("\n" + "="*80)
    print("Phase 1.5 汇总完成！")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa'])
    parser.add_argument('--k_shot', type=int, default=2)
    
    args = parser.parse_args()
    
    summarize_phase1_5_results(args.dataset, args.k_shot)
