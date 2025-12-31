"""
Baseline vs Purge2 整体对比
对比 6 个修改类别的最终效果
"""

import os
import pandas as pd

def compare_baseline_vs_purge2():
    # 加载数据
    baseline_path = './result/baseline/mvtec/k_2/csv/Seed_111-results.csv'
    purge2_path = './result/baseline_reducedprompt2/mvtec/k_2/csv/Seed_111-results.csv'
    
    df_baseline = pd.read_csv(baseline_path, index_col=0)
    df_purge2 = pd.read_csv(purge2_path, index_col=0)
    
    # 修改的 6 个类别
    target_classes = ['mvtec-metal_nut', 'mvtec-pill', 'mvtec-cable', 
                      'mvtec-screw', 'mvtec-capsule', 'mvtec-transistor']
    
    # 提取数据
    comparison = []
    for cls in target_classes:
        if cls in df_baseline.index and cls in df_purge2.index:
            comparison.append({
                'class': cls.replace('mvtec-', ''),
                'baseline_i_roc': df_baseline.loc[cls, 'i_roc'],
                'baseline_semantic': df_baseline.loc[cls, 'semantic_i_roc'],
                'purge2_i_roc': df_purge2.loc[cls, 'i_roc'],
                'purge2_semantic': df_purge2.loc[cls, 'semantic_i_roc'],
                'delta_i_roc': df_purge2.loc[cls, 'i_roc'] - df_baseline.loc[cls, 'i_roc'],
                'delta_semantic': df_purge2.loc[cls, 'semantic_i_roc'] - df_baseline.loc[cls, 'semantic_i_roc']
            })
    
    df_comp = pd.DataFrame(comparison)
    
    # 计算平均值
    avg_row = {
        'class': 'AVERAGE',
        'baseline_i_roc': df_comp['baseline_i_roc'].mean(),
        'baseline_semantic': df_comp['baseline_semantic'].mean(),
        'purge2_i_roc': df_comp['purge2_i_roc'].mean(),
        'purge2_semantic': df_comp['purge2_semantic'].mean(),
        'delta_i_roc': df_comp['delta_i_roc'].mean(),
        'delta_semantic': df_comp['delta_semantic'].mean()
    }
    df_comp = pd.concat([df_comp, pd.DataFrame([avg_row])], ignore_index=True)
    
    # 保存结果
    output_path = './analysis/prompt_purging/baseline_vs_purge2_comparison.csv'
    df_comp.to_csv(output_path, index=False, float_format='%.2f')
    
    # 打印报告
    print("\n" + "="*95)
    print("📊 BASELINE vs PURGE2 FINAL COMPARISON (MVTec k=2)")
    print("="*95)
    print("\n🔍 Comparing:")
    print("   - Baseline: 完整 prompts (8个通用 + 所有类别prompts)")
    print("   - Purge2: 清洗后 prompts (2个通用 + 精简类别prompts)")
    print("   - 清洗总量: 6个通用prompts + 9个类别prompts = 15个prompts")
    print(f"\n📁 Output: {output_path}\n")
    
    print("-"*95)
    print(f"{'Class':<15} {'Baseline i_roc':<15} {'Purge2 i_roc':<15} {'Δ i_roc':<12} {'Δ Semantic':<12}")
    print("-"*95)
    
    for idx, row in df_comp.iterrows():
        cls = row['class']
        base_iroc = row['baseline_i_roc']
        purge2_iroc = row['purge2_i_roc']
        delta_iroc = row['delta_i_roc']
        delta_semantic = row['delta_semantic']
        
        symbol = '✅' if delta_semantic >= 0 else '❌'
        
        if cls == 'AVERAGE':
            print("-"*95)
            print(f"{'AVERAGE':<15} {base_iroc:<15.2f} {purge2_iroc:<15.2f} {delta_iroc:+12.2f} {delta_semantic:+10.2f} {symbol}")
        else:
            print(f"{cls:<15} {base_iroc:<15.2f} {purge2_iroc:<15.2f} {delta_iroc:+12.2f} {delta_semantic:+10.2f} {symbol}")
    
    print("-"*95)
    
    # 统计
    positive_count = (df_comp['delta_semantic'] > 0).sum() - 1
    negative_count = (df_comp['delta_semantic'] < 0).sum()
    
    print(f"\n📈 Summary:")
    print(f"   - Improved (Δsemantic > 0): {positive_count}/6 classes")
    print(f"   - Degraded (Δsemantic < 0): {negative_count}/6 classes")
    print(f"   - Average Δi_roc: {avg_row['delta_i_roc']:+.2f}")
    print(f"   - Average Δsemantic: {avg_row['delta_semantic']:+.2f}")
    
    # 最佳/最差
    best_idx = df_comp.iloc[:-1]['delta_semantic'].idxmax()
    worst_idx = df_comp.iloc[:-1]['delta_semantic'].idxmin()
    
    best = df_comp.iloc[best_idx]
    worst = df_comp.iloc[worst_idx]
    
    print(f"\n🏆 Best Improvement: {best['class']} (Δsemantic: {best['delta_semantic']:+.2f})")
    print(f"⚠️  Worst Change: {worst['class']} (Δsemantic: {worst['delta_semantic']:+.2f})")
    
    print("\n💡 Prompt Purging Effect:")
    if avg_row['delta_semantic'] > 0:
        print(f"   ✅ Overall POSITIVE: Removing useless prompts improved semantic by {avg_row['delta_semantic']:+.2f}")
    else:
        print(f"   ❌ Overall NEGATIVE: Need to refine purging strategy")
    
    print("\n" + "="*95)
    print("✅ Final comparison complete!")
    print("="*95 + "\n")

if __name__ == '__main__':
    compare_baseline_vs_purge2()
