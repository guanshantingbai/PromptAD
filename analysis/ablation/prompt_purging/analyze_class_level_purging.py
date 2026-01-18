"""
对比类别级 Prompt Purging 的效果
比较 baseline_reducedprompt2 (类别级清洗) vs baseline (完整prompts)
重点分析 6 个修改的类别
"""

import os
import pandas as pd
import numpy as np

def load_results(root_dir, dataset, k_shot):
    """加载指定路径的结果CSV"""
    csv_path = f"{root_dir}/{dataset}/k_{k_shot}/csv/Seed_111-results.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0)
        return df
    else:
        print(f"[WARNING] File not found: {csv_path}")
        return None

def analyze_class_level_purging():
    """分析类别级 prompt purging 的效果"""
    
    baseline_dir = './result/baseline'
    purged_dir = './result/baseline_reducedprompt2'
    k_shot = 2
    
    # 加载数据
    df_baseline = load_results(baseline_dir, 'mvtec', k_shot)
    df_purged = load_results(purged_dir, 'mvtec', k_shot)
    
    if df_baseline is None or df_purged is None:
        print("[ERROR] Cannot load data!")
        return
    
    # 修改的 6 个类别
    tier1_classes = ['mvtec-metal_nut', 'mvtec-pill', 'mvtec-cable']
    tier2_classes = ['mvtec-screw', 'mvtec-capsule', 'mvtec-transistor']
    purged_classes = tier1_classes + tier2_classes
    
    # 提取数据
    comparison = pd.DataFrame()
    
    for cls in purged_classes:
        if cls in df_baseline.index and cls in df_purged.index:
            comparison = pd.concat([comparison, pd.DataFrame({
                'class': [cls],
                'baseline_i_roc': [df_baseline.loc[cls, 'i_roc']],
                'baseline_semantic': [df_baseline.loc[cls, 'semantic_i_roc']],
                'baseline_memory': [df_baseline.loc[cls, 'memory_i_roc']],
                'purged_i_roc': [df_purged.loc[cls, 'i_roc']],
                'purged_semantic': [df_purged.loc[cls, 'semantic_i_roc']],
                'purged_memory': [df_purged.loc[cls, 'memory_i_roc']],
                'delta_i_roc': [df_purged.loc[cls, 'i_roc'] - df_baseline.loc[cls, 'i_roc']],
                'delta_semantic': [df_purged.loc[cls, 'semantic_i_roc'] - df_baseline.loc[cls, 'semantic_i_roc']],
                'delta_memory': [df_purged.loc[cls, 'memory_i_roc'] - df_baseline.loc[cls, 'memory_i_roc']]
            })], ignore_index=True)
    
    # 计算平均值
    avg_row = {
        'class': 'AVERAGE',
        'baseline_i_roc': comparison['baseline_i_roc'].mean(),
        'baseline_semantic': comparison['baseline_semantic'].mean(),
        'baseline_memory': comparison['baseline_memory'].mean(),
        'purged_i_roc': comparison['purged_i_roc'].mean(),
        'purged_semantic': comparison['purged_semantic'].mean(),
        'purged_memory': comparison['purged_memory'].mean(),
        'delta_i_roc': comparison['delta_i_roc'].mean(),
        'delta_semantic': comparison['delta_semantic'].mean(),
        'delta_memory': comparison['delta_memory'].mean()
    }
    comparison = pd.concat([comparison, pd.DataFrame([avg_row])], ignore_index=True)
    
    # 保存结果
    output_dir = './analysis/prompt_purging'
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/class_level_purging_analysis.csv"
    comparison.to_csv(output_path, index=False, float_format='%.2f')
    
    # 打印报告
    print("\n" + "="*100)
    print("📊 CLASS-LEVEL PROMPT PURGING ANALYSIS (MVTec k=2)")
    print("="*100)
    print(f"\n📁 Output: {output_path}\n")
    
    print("Strategy: Remove Dangerous-and-Useless prompts (separation_gap < 0)")
    print("  - Tier 1 (必做): metal_nut, pill, cable")
    print("  - Tier 2 (推荐): screw, capsule, transistor\n")
    
    print("-"*100)
    print(f"{'Class':<20} {'Base i_roc':<12} {'Purged i_roc':<12} {'Δ i_roc':<10} {'Δ Semantic':<12} {'Δ Memory':<10}")
    print("-"*100)
    
    for idx, row in comparison.iterrows():
        cls = row['class'].replace('mvtec-', '')
        baseline_iroc = row['baseline_i_roc']
        purged_iroc = row['purged_i_roc']
        delta_iroc = row['delta_i_roc']
        delta_semantic = row['delta_semantic']
        delta_memory = row['delta_memory']
        
        # 添加符号指示
        symbol_iroc = '✅' if delta_iroc >= 0 else '❌'
        symbol_semantic = '✅' if delta_semantic >= 0 else '❌'
        
        if cls == 'AVERAGE':
            print("-"*100)
            print(f"{'AVERAGE':<20} {baseline_iroc:<12.2f} {purged_iroc:<12.2f} {delta_iroc:+10.2f} {symbol_iroc} {delta_semantic:+10.2f} {symbol_semantic} {delta_memory:+10.2f}")
        else:
            # 标记 Tier
            tier = '(T1)' if 'mvtec-' + cls in tier1_classes else '(T2)'
            print(f"{cls:<15} {tier:<5} {baseline_iroc:<12.2f} {purged_iroc:<12.2f} {delta_iroc:+10.2f} {symbol_iroc} {delta_semantic:+10.2f} {symbol_semantic} {delta_memory:+10.2f}")
    
    print("-"*100)
    
    # 统计正负提升
    positive_count = (comparison['delta_i_roc'] > 0).sum() - 1  # 减去 AVERAGE 行
    negative_count = (comparison['delta_i_roc'] < 0).sum()
    total_count = len(comparison) - 1
    
    print(f"\n📈 Results Summary:")
    print(f"   - Positive Δi_roc: {positive_count}/{total_count} classes")
    print(f"   - Negative Δi_roc: {negative_count}/{total_count} classes")
    print(f"   - Average Δi_roc: {avg_row['delta_i_roc']:+.2f}")
    print(f"   - Average Δsemantic: {avg_row['delta_semantic']:+.2f}")
    
    # 最佳/最差类别
    best_class = comparison.iloc[:-1].loc[comparison.iloc[:-1]['delta_semantic'].idxmax()]
    worst_class = comparison.iloc[:-1].loc[comparison.iloc[:-1]['delta_semantic'].idxmin()]
    
    print(f"\n🏆 Best Improvement: {best_class['class']} (Δsemantic: {best_class['delta_semantic']:+.2f})")
    print(f"⚠️  Worst Change: {worst_class['class']} (Δsemantic: {worst_class['delta_semantic']:+.2f})")
    
    print("\n" + "="*100)
    print("✅ Analysis complete!")
    print("="*100 + "\n")
    
    return comparison

if __name__ == '__main__':
    analyze_class_level_purging()
