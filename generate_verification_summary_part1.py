"""
Phase 1 一致性验证总结报告生成器
汇总所有 sanity test 的结果
"""

import os
import pandas as pd
import numpy as np

def generate_summary_report(dataset='mvtec', classname='bottle'):
    """生成验证总结报告"""
    
    print("="*80)
    print(f"Phase 1 一致性验证总结 - {dataset}/{classname}")
    print("="*80)
    
    base_dir = "result/prompt_purging/sanity_tests"
    
    # Test A: Embedding 一致性
    print("\n" + "="*80)
    print("Test A: Prompt Embedding 来源验证")
    print("="*80)
    
    embedding_file = f"{base_dir}/{dataset}_{classname}_embedding_comparison.csv"
    if os.path.exists(embedding_file):
        df_embed = pd.read_csv(embedding_file)
        
        # 按 prompt 分组
        grouped = df_embed.groupby('prompt_id')
        avg_cos = grouped['cosine_similarity'].mean()
        avg_l2 = grouped['l2_distance'].mean()
        
        print(f"\n✓ 对比 'build_text_feature_gallery (含ctx)' vs 'raw CLIP encoding (无ctx)'")
        print(f"\n  余弦相似度: {avg_cos.mean():.4f} (平均)")
        print(f"  L2 距离:     {avg_l2.mean():.4f} (平均)")
        
        if avg_cos.mean() < 0.90:
            print(f"\n  ✓ 结论: 相似度低 (<0.90)，说明 ctx 正在生效")
            print(f"           Phase 1 确实使用了包含 ctx 的 text features")
        else:
            print(f"\n  ⚠ 结论: 相似度高 (>0.90)，ctx 影响很小")
    else:
        print(f"\n  ✗ 未找到结果文件: {embedding_file}")
    
    # Test B: Margin 计算验证
    print("\n" + "="*80)
    print("Test B: Margin 计算方式验证")
    print("="*80)
    
    margin_file = f"{base_dir}/{dataset}_{classname}_margin_comparison.csv"
    if os.path.exists(margin_file):
        df_margin = pd.read_csv(margin_file)
        
        print(f"\n✓ 对比 Phase 1 vs Model Forward 的 margin 计算")
        print(f"\n  平均绝对差异: {df_margin['mean_diff'].mean():.4f}")
        
        if df_margin['mean_diff'].mean() > 1.0:
            print(f"\n  ✗ 问题发现: 差异较大 (>1.0)")
            print(f"     原因: Phase 1 使用 mean(normal_prototypes)")
            print(f"           Forward 使用 max(sim to each prototype)")
        else:
            print(f"\n  ✓ 结论: 差异很小，计算方式一致")
    else:
        print(f"\n  ✗ 未找到结果文件: {margin_file}")
