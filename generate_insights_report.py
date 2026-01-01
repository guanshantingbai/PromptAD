#!/usr/bin/env python
"""
Phase 1 深度分析报告
识别关键insights和清洗建议
"""

import pandas as pd
import numpy as np


def load_data():
    """加载所有分析数据"""
    mvtec_summary = pd.read_csv('result/prompt_purging/analysis/mvtec_class_summary.csv')
    visa_summary = pd.read_csv('result/prompt_purging/analysis/visa_class_summary.csv')
    mvtec_detail = pd.read_csv('result/prompt_purging/analysis/mvtec_prompt_details.csv')
    visa_detail = pd.read_csv('result/prompt_purging/analysis/visa_prompt_details.csv')
    
    return mvtec_summary, visa_summary, mvtec_detail, visa_detail


def analyze_risk_performance_relationship(summary_df, dataset_name):
    """分析风险与性能的关系"""
    
    print(f"\n{'='*80}")
    print(f"深度分析: {dataset_name}")
    print(f"{'='*80}")
    
    # 1. 识别反直觉的案例
    print("\n🔍 反直觉案例分析")
    print("-"*80)
    
    # 高风险但高性能
    high_risk_high_perf = summary_df[
        (summary_df['high_risk_pct'] > summary_df['high_risk_pct'].median()) &
        (summary_df['semantic_auroc'] > summary_df['semantic_auroc'].median())
    ]
    
    if len(high_risk_high_perf) > 0:
        print("\n高风险但高性能类别（Prompt质量可能不是瓶颈）:")
        for _, row in high_risk_high_perf.iterrows():
            print(f"  {row['class']}: {row['high_risk_pct']:.1f}% 高风险, "
                  f"Semantic AUROC={row['semantic_auroc']:.2f}")
            print(f"    → Memory AUROC={row['memory_auroc']:.2f} "
                  f"(差值: {abs(row['semantic_auroc'] - row['memory_auroc']):.2f})")
    
    # 低风险但低性能
    low_risk_low_perf = summary_df[
        (summary_df['high_risk_pct'] < summary_df['high_risk_pct'].median()) &
        (summary_df['semantic_auroc'] < summary_df['semantic_auroc'].median())
    ]
    
    if len(low_risk_low_perf) > 0:
        print("\n低风险但低性能类别（问题可能不在Prompt质量）:")
        for _, row in low_risk_low_perf.iterrows():
            print(f"  {row['class']}: {row['high_risk_pct']:.1f}% 高风险, "
                  f"Semantic AUROC={row['semantic_auroc']:.2f}")
            print(f"    → Memory AUROC={row['memory_auroc']:.2f} "
                  f"(Memory更好: {row['memory_auroc'] - row['semantic_auroc']:.2f})")
    
    # 2. Semantic vs Memory性能差异分析
    print("\n📊 Semantic vs Memory性能差异")
    print("-"*80)
    
    summary_df['sem_mem_diff'] = summary_df['semantic_auroc'] - summary_df['memory_auroc']
    
    # Memory明显更好的类别
    memory_better = summary_df[summary_df['sem_mem_diff'] < -5].sort_values('sem_mem_diff')
    if len(memory_better) > 0:
        print("\nMemory显著优于Semantic的类别（Prompt清洗潜力大）:")
        for _, row in memory_better.iterrows():
            print(f"  {row['class']}: Semantic={row['semantic_auroc']:.2f}, "
                  f"Memory={row['memory_auroc']:.2f} (差距 {abs(row['sem_mem_diff']):.2f})")
            print(f"    风险指标: {row['high_risk_pct']:.1f}% 高风险, "
                  f"R_j_eps={row['mean_R_j_eps']:.3f}")
    
    # Semantic明显更好的类别
    semantic_better = summary_df[summary_df['sem_mem_diff'] > 5].sort_values('sem_mem_diff', ascending=False)
    if len(semantic_better) > 0:
        print("\nSemantic显著优于Memory的类别（Prompt质量好）:")
        for _, row in semantic_better.iterrows():
            print(f"  {row['class']}: Semantic={row['semantic_auroc']:.2f}, "
                  f"Memory={row['memory_auroc']:.2f} (领先 {row['sem_mem_diff']:.2f})")
            print(f"    风险指标: {row['high_risk_pct']:.1f}% 高风险, "
                  f"R_j_eps={row['mean_R_j_eps']:.3f}")
    
    # 3. Generic vs Specific prompt分析
    print("\n🏷️  Generic vs Specific Prompt质量")
    print("-"*80)
    
    avg_generic_ratio = summary_df['generic_prompts'].sum() / summary_df['total_prompts'].sum()
    print(f"平均Generic比例: {avg_generic_ratio*100:.1f}%")
    
    # 识别高依赖generic的类别
    summary_df['generic_ratio'] = summary_df['generic_prompts'] / summary_df['total_prompts']
    high_generic = summary_df[summary_df['generic_ratio'] > 0.6].sort_values('semantic_auroc')
    
    if len(high_generic) > 0:
        print("\n高Generic依赖类别（>60%）:")
        for _, row in high_generic.iterrows():
            print(f"  {row['class']}: {row['generic_ratio']*100:.1f}% generic, "
                  f"Semantic AUROC={row['semantic_auroc']:.2f}")


def identify_cleaning_priorities(detail_df, dataset_name):
    """识别清洗优先级"""
    
    print(f"\n{'='*80}")
    print(f"清洗优先级建议: {dataset_name}")
    print(f"{'='*80}")
    
    # 按类别分组
    for classname in detail_df['class'].unique():
        class_df = detail_df[detail_df['class'] == classname].copy()
        
        # 只看高风险prompts
        high_risk = class_df[class_df['risk_level'] == 'high'].sort_values('R_j_eps', ascending=False)
        
        if len(high_risk) == 0:
            continue
        
        print(f"\n{classname} (Semantic AUROC={class_df['semantic_auroc'].iloc[0]:.2f})")
        print("-"*80)
        
        if len(high_risk) >= 5:
            print(f"  ⚠️  {len(high_risk)} 个高风险prompts，优先清洗候选:")
        else:
            print(f"  ℹ️  {len(high_risk)} 个高风险prompts:")
        
        for idx, row in high_risk.head(5).iterrows():
            risk_type = "SEVERE" if row['R_j_0'] > 0.5 else "MODERATE"
            print(f"    [{risk_type}] {row['full_text']}")
            print(f"      R_j_eps={row['R_j_eps']:.3f}, R_j_0={row['R_j_0']:.3f}, "
                  f"median_margin={row['median_margin']:.2f}")
            
            # 给出建议
            if row['R_j_0'] > 0.8:
                print(f"      💡 建议: 强烈建议删除或重写（80%+样本误判）")
            elif row['R_j_eps'] > 0.8:
                print(f"      💡 建议: 考虑删除或优化（边界模糊）")
            elif row['type'] == 'generic':
                print(f"      💡 建议: Generic prompt，考虑替换为specific")


def generate_actionable_summary(mvtec_summary, visa_summary, mvtec_detail, visa_detail):
    """生成可执行的行动摘要"""
    
    print("\n" + "="*80)
    print("📋 可执行行动摘要")
    print("="*80)
    
    all_summary = pd.concat([mvtec_summary, visa_summary])
    all_detail = pd.concat([mvtec_detail, visa_detail])
    
    # 1. 全局统计
    print("\n全局统计:")
    print(f"  总类别数: {len(all_summary)}")
    print(f"  总prompts: {all_summary['total_prompts'].sum()}")
    print(f"  高风险prompts: {all_summary['high_risk_count'].sum()} "
          f"({all_summary['high_risk_count'].sum() / all_summary['total_prompts'].sum() * 100:.1f}%)")
    
    # 2. 清洗建议分级
    print("\n清洗建议分级:")
    
    # Priority 1: 高风险+低性能
    p1_classes = all_summary[
        (all_summary['high_risk_pct'] > 50) & 
        (all_summary['semantic_auroc'] < 85)
    ]
    print(f"\n  Priority 1 (高风险+低性能): {len(p1_classes)} 个类别")
    if len(p1_classes) > 0:
        for _, row in p1_classes.iterrows():
            print(f"    - {row['dataset']}/{row['class']}: "
                  f"{row['high_risk_pct']:.0f}% 高风险, AUROC={row['semantic_auroc']:.1f}")
    
    # Priority 2: Memory明显更好
    p2_classes = all_summary[
        (all_summary['semantic_auroc'] - all_summary['memory_auroc'] < -10)
    ]
    print(f"\n  Priority 2 (Memory领先>10): {len(p2_classes)} 个类别")
    if len(p2_classes) > 0:
        for _, row in p2_classes.iterrows():
            gap = row['memory_auroc'] - row['semantic_auroc']
            print(f"    - {row['dataset']}/{row['class']}: "
                  f"Memory领先{gap:.1f}, {row['high_risk_pct']:.0f}% 高风险")
    
    # Priority 3: 中等风险+中等性能
    p3_classes = all_summary[
        (all_summary['high_risk_pct'] > 30) & 
        (all_summary['high_risk_pct'] <= 50) &
        (all_summary['semantic_auroc'] >= 85) &
        (all_summary['semantic_auroc'] < 95)
    ]
    print(f"\n  Priority 3 (中等风险+中等性能): {len(p3_classes)} 个类别")
    
    # 3. Generic prompt清洗建议
    print("\n通用Prompt清洗建议:")
    
    # 统计每个generic prompt的总体风险
    generic_prompts = all_detail[all_detail['type'] == 'generic'].groupby('template').agg({
        'R_j_eps': 'mean',
        'R_j_0': 'mean',
        'median_margin': 'mean',
        'class': 'count'
    }).rename(columns={'class': 'occurrence_count'}).sort_values('R_j_eps', ascending=False)
    
    print(f"\n  跨类别Generic Prompt风险排名:")
    for template, row in generic_prompts.head(5).iterrows():
        print(f"    \"{template}\": 平均R_j_eps={row['R_j_eps']:.3f}, "
              f"出现{int(row['occurrence_count'])}次")
        if row['R_j_eps'] > 0.5:
            print(f"      💡 高风险，考虑全局移除或重写")
    
    # 4. 下一步建议
    print("\n下一步行动:")
    print("  1. 从Priority 1类别开始ablation测试")
    print("  2. 对于高风险generic prompts，考虑全局删除")
    print("  3. 重点测试Memory领先>10的类别（清洗潜力大）")
    print("  4. 生成清洗后的prompt表版本（设置enabled=False）")
    print("  5. 重新训练并评估AUROC变化")


def main():
    print("="*80)
    print("Phase 1 深度分析报告")
    print("="*80)
    
    # 加载数据
    print("\n加载数据...")
    mvtec_summary, visa_summary, mvtec_detail, visa_detail = load_data()
    print(f"✓ MVTec: {len(mvtec_summary)} 个类别, {len(mvtec_detail)} 条prompts")
    print(f"✓ VisA: {len(visa_summary)} 个类别, {len(visa_detail)} 条prompts")
    
    # 分析
    analyze_risk_performance_relationship(mvtec_summary, "MVTec-AD")
    analyze_risk_performance_relationship(visa_summary, "VisA")
    
    identify_cleaning_priorities(mvtec_detail, "MVTec-AD")
    identify_cleaning_priorities(visa_detail, "VisA")
    
    generate_actionable_summary(mvtec_summary, visa_summary, mvtec_detail, visa_detail)
    
    print("\n" + "="*80)
    print("✅ 深度分析完成！")
    print("="*80)


if __name__ == '__main__':
    main()
