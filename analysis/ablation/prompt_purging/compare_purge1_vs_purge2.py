"""
Purge1 vs Purge2 增量对比
只对比修改的 6 个类别：metal_nut, pill, cable, screw, capsule, transistor
"""

import os
import pandas as pd

def compare_purge1_vs_purge2():
    # 加载数据
    purge1_path = './result/baseline_reducedprompt/mvtec/k_2/csv/Seed_111-results.csv'
    purge2_path = './result/baseline_reducedprompt2/mvtec/k_2/csv/Seed_111-results.csv'
    
    df_purge1 = pd.read_csv(purge1_path, index_col=0)
    df_purge2 = pd.read_csv(purge2_path, index_col=0)
    
    # 修改的 6 个类别
    target_classes = ['mvtec-metal_nut', 'mvtec-pill', 'mvtec-cable', 
                      'mvtec-screw', 'mvtec-capsule', 'mvtec-transistor']
    
    # 提取数据
    comparison = []
    for cls in target_classes:
        if cls in df_purge1.index and cls in df_purge2.index:
            comparison.append({
                'class': cls.replace('mvtec-', ''),
                'purge1_i_roc': df_purge1.loc[cls, 'i_roc'],
                'purge1_semantic': df_purge1.loc[cls, 'semantic_i_roc'],
                'purge2_i_roc': df_purge2.loc[cls, 'i_roc'],
                'purge2_semantic': df_purge2.loc[cls, 'semantic_i_roc'],
                'delta_i_roc': df_purge2.loc[cls, 'i_roc'] - df_purge1.loc[cls, 'i_roc'],
                'delta_semantic': df_purge2.loc[cls, 'semantic_i_roc'] - df_purge1.loc[cls, 'semantic_i_roc']
            })
    
    df_comp = pd.DataFrame(comparison)
    
    # 计算平均值
    avg_row = {
        'class': 'AVERAGE',
        'purge1_i_roc': df_comp['purge1_i_roc'].mean(),
        'purge1_semantic': df_comp['purge1_semantic'].mean(),
        'purge2_i_roc': df_comp['purge2_i_roc'].mean(),
        'purge2_semantic': df_comp['purge2_semantic'].mean(),
        'delta_i_roc': df_comp['delta_i_roc'].mean(),
        'delta_semantic': df_comp['delta_semantic'].mean()
    }
    df_comp = pd.concat([df_comp, pd.DataFrame([avg_row])], ignore_index=True)
    
    # 保存结果
    output_path = './analysis/prompt_purging/purge1_vs_purge2_comparison.csv'
    df_comp.to_csv(output_path, index=False, float_format='%.2f')
    
    # 打印报告
    print("\n" + "="*90)
    print("📊 PURGE1 vs PURGE2 INCREMENTAL COMPARISON (MVTec k=2)")
    print("="*90)
    print("\n🔍 Comparing:")
    print("   - Purge1 (baseline_reducedprompt): 通用 prompt 清洗 (6个通用prompts)")
    print("   - Purge2 (baseline_reducedprompt2): + 类别级 prompt 清洗 (9个类别prompts)")
    print(f"\n📁 Output: {output_path}\n")
    
    print("-"*90)
    print(f"{'Class':<15} {'Purge1 i_roc':<13} {'Purge2 i_roc':<13} {'Δ i_roc':<10} {'Δ Semantic':<12}")
    print("-"*90)
    
    for idx, row in df_comp.iterrows():
        cls = row['class']
        p1_iroc = row['purge1_i_roc']
        p2_iroc = row['purge2_i_roc']
        delta_iroc = row['delta_i_roc']
        delta_semantic = row['delta_semantic']
        
        symbol = '✅' if delta_semantic >= 0 else '❌'
        
        if cls == 'AVERAGE':
            print("-"*90)
            print(f"{'AVERAGE':<15} {p1_iroc:<13.2f} {p2_iroc:<13.2f} {delta_iroc:+10.2f} {delta_semantic:+10.2f} {symbol}")
        else:
            print(f"{cls:<15} {p1_iroc:<13.2f} {p2_iroc:<13.2f} {delta_iroc:+10.2f} {delta_semantic:+10.2f} {symbol}")
    
    print("-"*90)
    
    # 统计
    positive = (df_comp['delta_semantic'] > 0).sum() - 1
    negative = (df_comp['delta_semantic'] < 0).sum()
    
    print(f"\n📈 Summary:")
    print(f"   - Improved (Δsemantic > 0): {positive}/6 classes")
    print(f"   - Degraded (Δsemantic < 0): {negative}/6 classes")
    print(f"   - Average Δi_roc: {avg_row['delta_i_roc']:+.2f}")
    print(f"   - Average Δsemantic: {avg_row['delta_semantic']:+.2f}")
    
    print("\n" + "="*90)
    print("✅ Incremental analysis complete!")
    print("="*90 + "\n")

if __name__ == '__main__':
    compare_purge1_vs_purge2()
