"""
Phase 1 一致性验证总结 - Part 2
"""

import os
import pandas as pd
import sys

def continue_summary_report(dataset='mvtec', classname='bottle'):
    """继续生成报告 - Part 2"""
    
    base_dir = "result/prompt_purging/sanity_tests"
    
    # Test C: 修复后的验证
    print("\n" + "="*80)
    print("Test C: 修复后的 Phase 1 结果")
    print("="*80)
    
    fixed_file = f"{base_dir}/{dataset}_{classname}_phase1_FIXED.csv"
    old_file = f"result/prompt_purging/phase1/{dataset}/k_2/{classname}_phase1_normal_side_risk_eps0.05.csv"
    
    if os.path.exists(fixed_file) and os.path.exists(old_file):
        df_fixed = pd.read_csv(fixed_file)
        df_old = pd.read_csv(old_file)
        
        print(f"\n✓ 修复内容: s_n = MAX(sim) 而非 MEAN(prototypes)")
        print(f"\n  旧版平均 R_j_0: {df_old['R_j_0'].mean():.4f}")
        print(f"  修复版平均 R_j_0: {df_fixed['R_j_0'].mean():.4f}")
        print(f"  差异: {abs(df_old['R_j_0'].mean() - df_fixed['R_j_0'].mean()):.4f}")
        
        old_high = (df_old['R_j_0'] >= 0.999).sum()
        fixed_high = (df_fixed['R_j_0'] >= 0.999).sum()
        
        print(f"\n  旧版 R_j_0=1.0 的数量: {old_high}/{len(df_old)}")
        print(f"  修复版 R_j_0=1.0 的数量: {fixed_high}/{len(df_fixed)}")
    else:
        print(f"\n  ⚠ 缺少对比文件")
    
    # Deep Diagnosis
    print("\n" + "="*80)
    print("Deep Diagnosis: 正常 vs 异常样本的 Margin")
    print("="*80)
    
    diag_file = f"{base_dir}/{dataset}_{classname}_normal_vs_abnormal.csv"
    if os.path.exists(diag_file):
        df_diag = pd.read_csv(diag_file)
        
        print(f"\n✓ 关键发现:")
        print(f"\n  正常样本:")
        print(f"    - Median margin: {df_diag['normal_median'].mean():.2f}")
        print(f"    - Margin < 0 比例: {df_diag['R_normal_negative'].mean()*100:.1f}%")
        
        print(f"\n  异常样本:")
        print(f"    - Median margin: {df_diag['abnormal_median'].mean():.2f}")
        print(f"    - Margin < 0 比例: {df_diag['R_abnormal_negative'].mean()*100:.1f}%")
        
        print(f"\n  分离度:")
        print(f"    - 平均 gap: {df_diag['separation_gap'].mean():.2f}")
        print(f"    - Gap > 0 的比例: {(df_diag['separation_gap'] > 0).sum()}/{len(df_diag)}")
        print(f"    - Gap > 1.0 的比例: {(df_diag['separation_gap'] > 1.0).sum()}/{len(df_diag)}")
        
        # 问题 prompts
        problem_prompts = df_diag[df_diag['separation_gap'] <= 0]
        if len(problem_prompts) > 0:
            print(f"\n  ⚠ 发现 {len(problem_prompts)} 个真正有问题的 prompts (分离度 ≤ 0):")
            for _, row in problem_prompts.iterrows():
                print(f"    - {row['template']}: gap={row['separation_gap']:.2f}")
    else:
        print(f"\n  ✗ 未找到诊断文件: {diag_file}")


if __name__ == '__main__':
    import sys
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'mvtec'
    classname = sys.argv[2] if len(sys.argv) > 2 else 'bottle'
    
    continue_summary_report(dataset, classname)
