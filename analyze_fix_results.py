"""
结构性修复效果验证脚本
对比 baseline vs new_paradigm_1 (失败) vs fix_validation (修复后)
"""
import pandas as pd
import numpy as np
import os

def load_results(root_dir, dataset, k_shot):
    """加载指定目录的results.csv"""
    csv_path = f"{root_dir}/{dataset}/k_{k_shot}/csv/Seed_111-results.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, index_col=0)
    return None

def compare_results():
    print("="*90)
    print("🔧 Multi-Abnormal Prototypes 结构性修复效果验证")
    print("="*90)
    print("\n对比三个版本:")
    print("  [Baseline]   - result/fusion_normal (原始PromptAD)")
    print("  [Failed]     - result/new_paradigm_1 (CE塌陷版本)")
    print("  [Fixed]      - result/fix_validation (t_train=10 + top-2 margin + label smoothing)")
    print("="*90)
    
    # 加载结果
    baseline = load_results('./result/fusion_normal', 'mvtec', 2)
    failed = load_results('./result/new_paradigm_1', 'mvtec', 2)
    fixed = load_results('./result/fix_validation', 'mvtec', 2)
    
    if fixed is None:
        print("\n⚠️  修复版本结果尚未生成，请等待训练完成")
        return
    
    # 测试类别
    test_classes_info = {
        'zipper': '严重退化(-30.92%)',
        'pill': '严重退化(-25.80%)',
        'cable': '严重退化(-21.18%)',
        'transistor': '中等退化(-12.31%)',
        'metal_nut': '中等退化(-10.02%)',
        'grid': '中等退化(-10.57%)',
        'toothbrush': '原有提升(+19.58%)',
        'bottle': '原稳定(+0.36%)',
        'screw': '低基线(-11.93%)',
    }
    
    print(f"\n{'Class':<15} {'类型':<20} {'Baseline':<10} {'Failed':<10} {'Fixed':<10} {'Δ(Fix-Base)':<15} {'修复效果':<10}")
    print("-"*90)
    
    total_delta_failed = []
    total_delta_fixed = []
    
    for cls, desc in test_classes_info.items():
        idx = f'mvtec-{cls}'
        
        if idx not in baseline.index or idx not in fixed.index:
            print(f"{cls:<15} {desc:<20} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<15} {'等待中...':<10}")
            continue
        
        base_sem = baseline.loc[idx, 'semantic_i_roc']
        
        if idx in failed.index:
            fail_sem = failed.loc[idx, 'semantic_i_roc']
            delta_failed = fail_sem - base_sem
        else:
            fail_sem = 0
            delta_failed = 0
        
        fix_sem = fixed.loc[idx, 'semantic_i_roc']
        delta_fixed = fix_sem - base_sem
        
        total_delta_failed.append(delta_failed)
        total_delta_fixed.append(delta_fixed)
        
        # 判断修复效果
        if delta_fixed > -2:  # 基本恢复
            status = "✅ 成功"
        elif delta_fixed > delta_failed:  # 有改善
            status = "✅ 改善"
        else:
            status = "❌ 仍差"
        
        print(f"{cls:<15} {desc:<20} {base_sem:>6.2f}%  {fail_sem:>6.2f}%  {fix_sem:>6.2f}%  {delta_fixed:>+7.2f}%      {status:<10}")
    
    print("="*90)
    print(f"{'Summary':<15} {'Mean Δ':<20}")
    print(f"{'Failed版本':<15} {np.mean(total_delta_failed):>+8.2f}%  (CE塌陷导致的退化)")
    print(f"{'Fixed版本':<15} {np.mean(total_delta_fixed):>+8.2f}%  (修复后)")
    print(f"{'改善幅度':<15} {np.mean(total_delta_fixed) - np.mean(total_delta_failed):>+8.2f}%")
    print("="*90)
    
    # 结论
    if np.mean(total_delta_fixed) > -3:
        print("\n✅ 结论: 结构性修复有效！CE不再早期塌陷，性能基本恢复到baseline水平")
        print("   建议: 可进行全类别训练验证")
    elif np.mean(total_delta_fixed) > np.mean(total_delta_failed) + 3:
        print("\n✅ 结论: 结构性修复显著改善！虽未完全恢复，但证明修复方向正确")
        print("   建议: 可微调t_train或margin参数后全量训练")
    else:
        print("\n⚠️  结论: 修复效果有限，需要重新审视训练策略")
        print("   建议: 考虑更根本的架构调整")

if __name__ == '__main__':
    compare_results()
