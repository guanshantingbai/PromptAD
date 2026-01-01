"""Phase 2 vs Phase 1 详细比较分析"""
import pandas as pd
import sys

def compare_phase2():
    # 读取数据
    phase1_df = pd.read_csv('result/phase1_cleaned/mvtec/k_2/csv/Seed_111-results.csv')
    phase2_df = pd.read_csv('result/phase2_cleaned/mvtec/k_2/csv/Seed_111-results.csv')
    baseline_df = pd.read_csv('result/baseline/mvtec/k_2/csv/Seed_111-results.csv')
    
    # 提取Phase 2的6个目标类别
    target_classes = ['metal_nut', 'pill', 'cable', 'screw', 'capsule', 'transistor']
    
    print("="*80)
    print("Phase 2 类别级清洗效果分析 (6个目标类别)")
    print("="*80)
    print()
    
    # 读取清洗信息
    cleaning_info = {
        'metal_nut': {'total': 3, 'new': 2, 'pre': 1},
        'pill': {'total': 4, 'new': 1, 'pre': 3},
        'cable': {'total': 4, 'new': 2, 'pre': 2},
        'screw': {'total': 2, 'new': 0, 'pre': 2},
        'capsule': {'total': 5, 'new': 2, 'pre': 3},
        'transistor': {'total': 6, 'new': 2, 'pre': 3}
    }
    
    # 创建比较表
    comparison_data = []
    
    for cls in target_classes:
        baseline_row = baseline_df[baseline_df['class'] == f'mvtec-{cls}']
        phase1_row = phase1_df[phase1_df['class'] == f'mvtec-{cls}']
        phase2_row = phase2_df[phase2_df['class'] == f'mvtec-{cls}']
        
        if not baseline_row.empty and not phase1_row.empty and not phase2_row.empty:
            baseline_sem = baseline_row['semantic_i_roc'].values[0]
            phase1_sem = phase1_row['semantic_i_roc'].values[0]
            phase2_sem = phase2_row['semantic_i_roc'].values[0]
            
            phase1_fusion = phase1_row['i_roc'].values[0]
            phase2_fusion = phase2_row['i_roc'].values[0]
            
            info = cleaning_info[cls]
            
            comparison_data.append({
                'Class': cls,
                'Baseline': f"{baseline_sem:.2f}",
                'Phase1': f"{phase1_sem:.2f}",
                'Phase2': f"{phase2_sem:.2f}",
                'P1提升': f"{phase1_sem - baseline_sem:+.2f}",
                'P2提升': f"{phase2_sem - phase1_sem:+.2f}",
                'Total清洗': info['total'],
                'New禁用': info['new'],
                'Pre禁用': info['pre'],
                'Fusion': f"{phase2_fusion:.2f}"
            })
    
    # 打印表格
    df_comp = pd.DataFrame(comparison_data)
    
    print("【Semantic AUROC 三阶段对比】")
    print()
    print(df_comp[['Class', 'Baseline', 'Phase1', 'Phase2', 'P1提升', 'P2提升']].to_string(index=False))
    print()
    
    print("【清洗详情与Phase1/Phase2重叠分析】")
    print()
    print(df_comp[['Class', 'Total清洗', 'New禁用', 'Pre禁用', 'P2提升']].to_string(index=False))
    print()
    
    # 统计分析
    total_cleaned = sum(info['total'] for info in cleaning_info.values())
    total_new = sum(info['new'] for info in cleaning_info.values())
    total_pre = sum(info['pre'] for info in cleaning_info.values())
    
    print("="*80)
    print("【Phase 2 总结】")
    print("="*80)
    print()
    print(f"目标类别数量: {len(target_classes)}")
    print(f"计划清洗prompt总数: {total_cleaned}")
    print(f"  - Phase 2新禁用: {total_new} ({total_new/total_cleaned*100:.1f}%)")
    print(f"  - Phase 1已禁用: {total_pre} ({total_pre/total_cleaned*100:.1f}%)")
    print()
    
    # 检查是否有改进
    improved = df_comp[df_comp['P2提升'].str.replace('+', '').astype(float) > 0.01]
    unchanged = df_comp[df_comp['P2提升'].str.replace('+', '').astype(float).abs() < 0.01]
    
    print(f"Semantic AUROC变化:")
    print(f"  - 提升的类别: {len(improved)} ({', '.join(improved['Class'].tolist()) if len(improved) > 0 else '无'})")
    print(f"  - 无变化类别: {len(unchanged)} ({', '.join(unchanged['Class'].tolist())})")
    print()
    
    # 关键发现
    print("【关键发现】")
    print()
    print("1. Phase 1/Phase 2 重叠率: {:.1f}% ({}/{} prompts)".format(
        total_pre/total_cleaned*100, total_pre, total_cleaned
    ))
    print()
    print("2. 重叠原因分析:")
    print("   Phase 1全局禁用的6个模板 (abnormal, imperfect, blemished, flawed, defect, flaw)")
    print("   覆盖了Phase 2目标中的大部分Useless prompts")
    print()
    print("3. Phase 2独特价值 (9个新禁用的prompts):")
    for cls, info in cleaning_info.items():
        if info['new'] > 0:
            print(f"   - {cls}: {info['new']}个类别特定的Useless prompts")
    print()
    print("4. 性能影响:")
    if len(improved) == 0:
        print("   所有6个类别的Semantic AUROC保持不变")
        print("   说明这9个新禁用的prompts在Phase 1清洗后权重已经很低")
        print("   或者它们的负面影响已被Phase 1的清洗间接消除")
    else:
        print(f"   {len(improved)}个类别有提升，{len(unchanged)}个无变化")
    print()
    
    # 保存结果
    output_file = 'result/phase2_cleaned/mvtec/k_2/comparison_phase1_vs_phase2.csv'
    df_comp.to_csv(output_file, index=False)
    print(f"详细比较已保存: {output_file}")
    print()

if __name__ == '__main__':
    compare_phase2()
