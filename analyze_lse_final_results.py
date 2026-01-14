#!/usr/bin/env python3
"""
LSE Softmax-weighted Aggregation 结果分析
对比 baseline (fusion_normal) 与 LSE (tau=0.05, 0.5, 5.0)
"""

import pandas as pd
import numpy as np
import os

def load_results(result_dir):
    """加载MVTec和ViSA结果"""
    results = {}
    for dataset in ['mvtec', 'visa']:
        csv_path = f'{result_dir}/{dataset}/k_2/csv/Seed_111-results.csv'
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, index_col=0)
            # 去除dataset前缀
            df.index = df.index.str.replace(f'{dataset}-', '')
            results[dataset] = df
    return results

def main():
    print("="*80)
    print("LSE Softmax-Weighted Aggregation 最终结果分析")
    print("="*80)
    print()
    
    # 加载所有结果
    baseline = load_results('result/fusion_normal')
    lse_005 = load_results('result/lse_tau0.05')
    lse_05 = load_results('result/lse_tau0.5')
    lse_50 = load_results('result/lse_tau5.0')
    
    # 汇总分析
    for dataset in ['mvtec', 'visa']:
        if dataset not in baseline:
            continue
            
        print(f"\n{'='*80}")
        print(f"{dataset.upper()} Dataset Results")
        print('='*80)
        
        df_base = baseline[dataset]
        classes = df_base.index.tolist()
        
        # 准备对比表格
        comparison = []
        for cls in classes:
            row = {'Class': cls}
            
            # Baseline
            if cls in df_base.index:
                row['Baseline'] = df_base.loc[cls, 'i_roc']
                row['B_Sem'] = df_base.loc[cls, 'semantic_i_roc']
            
            # LSE results
            for tau, lse_results in [('0.05', lse_005), ('0.5', lse_05), ('5.0', lse_50)]:
                if dataset in lse_results and cls in lse_results[dataset].index:
                    row[f'τ={tau}'] = lse_results[dataset].loc[cls, 'i_roc']
                    row[f'τ={tau}_Sem'] = lse_results[dataset].loc[cls, 'semantic_i_roc']
            
            comparison.append(row)
        
        df_comp = pd.DataFrame(comparison)
        
        # 计算improvement
        for tau in ['0.05', '0.5', '5.0']:
            if f'τ={tau}' in df_comp.columns:
                df_comp[f'Δ{tau}'] = df_comp[f'τ={tau}'] - df_comp['Baseline']
        
        # 显示完整结果
        print(f"\n完整结果对比 (Image-level AUC %):")
        print("-" * 80)
        
        # 格式化显示
        display_cols = ['Class', 'Baseline', 'τ=0.05', 'Δ0.05', 'τ=0.5', 'Δ0.5', 'τ=5.0', 'Δ5.0']
        df_display = df_comp[display_cols].copy()
        
        for col in display_cols:
            if col != 'Class':
                df_display[col] = df_display[col].apply(lambda x: f'{x:6.2f}' if pd.notna(x) else '  N/A ')
        
        print(df_display.to_string(index=False))
        
        # 统计摘要
        print(f"\n统计摘要:")
        print("-" * 80)
        
        for tau in ['0.05', '0.5', '5.0']:
            delta_col = f'Δ{tau}'
            if delta_col in df_comp.columns:
                improvements = df_comp[delta_col].dropna()
                mean_delta = improvements.mean()
                positive = (improvements > 0).sum()
                negative = (improvements < 0).sum()
                neutral = (improvements == 0).sum()
                max_gain = improvements.max()
                max_loss = improvements.min()
                best_class = df_comp.loc[improvements.idxmax(), 'Class'] if len(improvements) > 0 else 'N/A'
                worst_class = df_comp.loc[improvements.idxmin(), 'Class'] if len(improvements) > 0 else 'N/A'
                
                print(f"\nτ={tau}:")
                print(f"  平均变化: {mean_delta:+.3f}%")
                print(f"  改进类别: {positive}/{len(improvements)} ({100*positive/len(improvements):.1f}%)")
                print(f"  退化类别: {negative}/{len(improvements)} ({100*negative/len(improvements):.1f}%)")
                print(f"  不变类别: {neutral}/{len(improvements)}")
                print(f"  最大提升: {max_gain:+.3f}% ({best_class})")
                print(f"  最大退化: {max_loss:+.3f}% ({worst_class})")
        
        # 语义分支分析
        print(f"\n语义分支对比 (Semantic AUC %):")
        print("-" * 80)
        sem_display_cols = ['Class', 'B_Sem', 'τ=0.05_Sem', 'τ=0.5_Sem', 'τ=5.0_Sem']
        df_sem = df_comp[sem_display_cols].copy()
        
        for col in sem_display_cols:
            if col != 'Class':
                df_sem[col] = df_sem[col].apply(lambda x: f'{x:6.2f}' if pd.notna(x) else '  N/A ')
        
        print(df_sem.to_string(index=False))
        
        # 语义分支统计
        print(f"\n语义分支变化:")
        for tau in ['0.05', '0.5', '5.0']:
            sem_col = f'τ={tau}_Sem'
            if sem_col in df_comp.columns:
                sem_delta = df_comp[sem_col] - df_comp['B_Sem']
                print(f"  τ={tau}: 平均变化 {sem_delta.mean():+.3f}%")
    
    # 全局总结
    print(f"\n{'='*80}")
    print("全局总结")
    print('='*80)
    
    all_deltas = {}
    for tau in ['0.05', '0.5', '5.0']:
        deltas = []
        for dataset in ['mvtec', 'visa']:
            if dataset in baseline:
                df_base = baseline[dataset]
                if dataset in eval(f'lse_{tau.replace(".", "")}'):
                    df_lse = eval(f'lse_{tau.replace(".", "")}')[dataset]
                    for cls in df_base.index:
                        if cls in df_lse.index:
                            delta = df_lse.loc[cls, 'i_roc'] - df_base.loc[cls, 'i_roc']
                            deltas.append(delta)
        all_deltas[tau] = deltas
    
    print(f"\n跨数据集整体表现:")
    for tau, deltas in all_deltas.items():
        if len(deltas) > 0:
            deltas = np.array(deltas)
            print(f"\nτ={tau} (共{len(deltas)}个类别):")
            print(f"  平均变化: {deltas.mean():+.4f}%")
            print(f"  中位数变化: {np.median(deltas):+.4f}%")
            print(f"  标准差: {deltas.std():.4f}%")
            print(f"  改进率: {(deltas > 0).sum()}/{len(deltas)} ({100*(deltas > 0).sum()/len(deltas):.1f}%)")
            print(f"  退化率: {(deltas < 0).sum()}/{len(deltas)} ({100*(deltas < 0).sum()/len(deltas):.1f}%)")
    
    # 推荐结论
    print(f"\n{'='*80}")
    print("结论与建议")
    print('='*80)
    
    best_tau = max(all_deltas.keys(), key=lambda t: np.mean(all_deltas[t]) if len(all_deltas[t]) > 0 else -999)
    best_mean = np.mean(all_deltas[best_tau])
    
    print(f"\n最佳τ值: {best_tau} (平均提升 {best_mean:+.4f}%)")
    
    print(f"\nτ值特性:")
    print(f"  - τ=0.05: 激进聚合 (接近max)，适合有明显异常特征的类别")
    print(f"  - τ=0.5:  平衡聚合 (加权平均)，综合多原型信息")
    print(f"  - τ=5.0:  保守聚合 (接近mean)，接近baseline行为")

if __name__ == '__main__':
    main()
