"""
Multi-Abnormal Prototype 假阳性假设验证 (简化版)
===========================================

通过对比 Baseline 和 Multi-Abnormal 模型在测试集上的分数分布，
验证 Multi-Abnormal 是否通过抬高正常样本假阳性导致性能退化。

策略：使用已有的训练结果，直接运行测试并收集分数分布。
"""

import os
import subprocess
import pandas as pd
import numpy as np

# 9个有数据的类别
TEST_CLASSES = ['grid', 'bottle', 'cable', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']

BASELINE_DIR = './result/fusion_normal'
MULTI_ABN_DIR = './result/fix_validation'

def run_test_and_collect_scores(class_name, root_dir, is_baseline=False):
    """
    运行test_cls.py并收集分数
    
    这个函数需要你手动实现，因为需要修改test_cls.py来输出分数分布
    """
    # 构造命令
    cmd = [
        'python', 'test_cls.py',
        '--dataset', 'mvtec',
        '--class_name', class_name,
        '--k-shot', '2',
        '--root-dir', root_dir,
        '--topk-abnormal', 'None' if is_baseline else '2',
        '--gpu-id', '0'
    ]
    
    # 运行并捕获输出
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    return result.stdout, result.stderr

def analyze_results():
    """分析已有的AUROC结果"""
    
    # 读取baseline和multi-abnormal的结果
    baseline_csv = f'{BASELINE_DIR}/mvtec/k_2/csv/Seed_111-results.csv'
    multi_abn_csv = f'{MULTI_ABN_DIR}/mvtec/k_2/csv/Seed_111-results.csv'
    
    # 检查文件是否存在
    if not os.path.exists(baseline_csv):
        print(f"❌ Baseline结果不存在: {baseline_csv}")
        print("\n需要先运行baseline训练:")
        print("  python run_seg.py  # 或使用适当的训练脚本")
        return
    
    if not os.path.exists(multi_abn_csv):
        print(f"❌ Multi-Abnormal结果不存在: {multi_abn_csv}")
        return
    
    # 读取CSV
    df_baseline = pd.read_csv(baseline_csv, index_col=0)
    df_multi_abn = pd.read_csv(multi_abn_csv, index_col=0)
    
    print(f"\n{'='*80}")
    print(f"基于已有AUROC的初步分析")
    print(f"{'='*80}\n")
    
    print(f"{'Class':<15} {'Baseline':>12} {'Multi-Abn':>12} {'Delta':>10} {'Status':>15}")
    print(f"{'-'*70}")
    
    for cls in TEST_CLASSES:
        cls_key = f'mvtec-{cls}'
        
        if cls_key in df_baseline.index and cls_key in df_multi_abn.index:
            base_val = df_baseline.loc[cls_key, 'semantic_i_roc']
            ma_val = df_multi_abn.loc[cls_key, 'semantic_i_roc']
            delta = ma_val - base_val
            
            if delta < -5:
                status = "❌ 退化"
            elif delta > 5:
                status = "✅ 提升"
            else:
                status = "➖ 稳定"
            
            print(f"{cls:<15} {base_val:>12.2f} {ma_val:>12.2f} {delta:>10.2f} {status:>15}")
        else:
            print(f"{cls:<15} {'N/A':>12} {'N/A':>12} {'N/A':>10} {'N/A':>15}")
    
    print(f"\n{'='*80}\n")
    
    print("\n⚠️  当前分析局限性:")
    print("  - 只能看到AUROC结果，无法验证假阳性假设")
    print("  - 需要分数分布数据（P95/P99）才能判断退化原因\n")
    
    print("💡 建议的验证方法:")
    print("  1. 修改test_cls.py，在inference时保存所有样本的anomaly score")
    print("  2. 对正常/异常样本分别统计分位数")
    print("  3. 对比baseline vs multi-abnormal的分布差异\n")

def main():
    print(f"\n{'='*80}")
    print(f"Multi-Abnormal Prototype 假阳性假设验证")
    print(f"{'='*80}\n")
    
    print("测试类别:", TEST_CLASSES)
    print(f"Baseline目录: {BASELINE_DIR}")
    print(f"Multi-Abnormal目录: {MULTI_ABN_DIR}\n")
    
    # 检查baseline是否存在
    if not os.path.exists(f'{BASELINE_DIR}/mvtec/k_2/csv/Seed_111-results.csv'):
        print("❌ Baseline结果不存在！")
        print("\n需要先运行baseline版本（无Multi-Abnormal）:")
        print("  1. 在train_cls.py中确保n_pro_ab=1（或使用旧版本代码）")
        print("  2. python run_cls.py")
        print("  3. 结果保存到result/fusion_normal\n")
        return
    
    # 分析已有结果
    analyze_results()
    
    print("\n" + "="*80)
    print("完整验证需要的额外工作")
    print("="*80)
    print("""
为了严格验证假阳性假设，需要修改test_cls.py添加以下功能：

1. 在inference循环中，收集每个样本的：
   - img_type (normal/abnormal)
   - semantic_score
   - memory_score
   - fusion_score

2. 对正常样本计算：
   - median(semantic_score | normal)
   - P95(semantic_score | normal)
   - P99(semantic_score | normal)

3. 对异常样本计算：
   - median(semantic_score | abnormal)
   - P95(semantic_score | abnormal)

4. 使用support set的P99作为阈值：
   - thr = P99(semantic_score | support_normal)
   - FPR = P(semantic_score > thr | test_normal)
   - TPR = P(semantic_score > thr | test_abnormal)

5. 对比 Baseline vs Multi-Abnormal：
   - 如果P99(normal)显著上升 且 FPR上升 → 假阳性假设成立
   - 如果P95(abnormal)上升 但 FPR也上升 → 增强表达但伴随假阳性
""")

if __name__ == '__main__':
    main()
