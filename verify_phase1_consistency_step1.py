"""
Sanity Test A: 验证 Phase 1 的 prompt embedding 来源
检查是否使用了包含 ctx 的 text features
"""

import os
import torch
import argparse
import numpy as np

from PromptAD.model import PromptAD
from datasets.mvtec import mvtec_classes
from datasets.visa import visa_classes


def test_embedding_consistency(args):
    """
    对比两种方式获取的 text features:
    1. Phase 1 当前方式: model.abnormal_prototypes (来自 build_text_feature_gallery)
    2. 原始方式: CLIP.encode_text(raw_text) - 不包含 ctx
    """
    
    print("="*80)
    print("Sanity Test A: Prompt Embedding 一致性验证")
    print("="*80)
    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    
    # 创建模型
    model = PromptAD(
        out_size_h=args.img_resize // args.img_cropsize,
        out_size_w=args.img_resize // args.img_cropsize,
        device=device,
        backbone=args.backbone,
        pretrained_dataset=args.pretrained_dataset,
        n_ctx=args.n_ctx,
        n_pro=args.n_pro,
        n_ctx_ab=args.n_ctx_ab,
        n_pro_ab=args.n_pro_ab,
        class_name=args.classname,
        precision='fp16',
        k_shot=args.k_shot,
        img_resize=args.img_resize,
        img_cropsize=args.img_cropsize
    )
    
    model.to(device)
    model.eval()
    
    print(f"\n测试类别: {args.classname}")
    
    # 方法1: 使用 build_text_feature_gallery (Phase 1 当前使用)
    print("\n[方法1] 使用 build_text_feature_gallery() + ctx...")
    model.build_text_feature_gallery()
    
    prompt_info = model.get_manual_prompt_info()
    n_pro = prompt_info['n_pro']
    num_prompts = prompt_info['num_manual_templates']
    
    print(f"  - 加载了 {num_prompts} 条 manual prompts")
    print(f"  - 每条重复 {n_pro} 次")
    
    # 获取 Phase 1 使用的 features
    features_with_ctx = model.abnormal_prototypes[:num_prompts * n_pro].clone()
    print(f"  - abnormal_prototypes shape: {features_with_ctx.shape}")
    
    # 方法2: 直接编码原始文本 (不含 ctx)
    print("\n[方法2] 直接 CLIP.encode_text(raw_text) - 不含 ctx...")
    
    from PromptAD import CLIPAD
    
    raw_texts = []
    for detail in prompt_info['prompt_details']:
        full_text = detail['text']
        # 重复 n_pro 次
        for _ in range(n_pro):
            raw_texts.append(full_text)
    
    print(f"  - 准备编码 {len(raw_texts)} 条文本")
    
    # 直接编码（不经过 PromptLearner）
    tokenized = CLIPAD.tokenize(raw_texts).to(device)
    
    with torch.no_grad():
        features_no_ctx = model.model.encode_text(tokenized)
        # 归一化
        features_no_ctx = features_no_ctx / features_no_ctx.norm(dim=-1, keepdim=True)
    
    print(f"  - 编码结果 shape: {features_no_ctx.shape}")
    
    # 对比分析
    print("\n" + "="*80)
    print("对比分析:")
    print("="*80)
    
    # 计算余弦相似度
    cosine_sim = (features_with_ctx * features_no_ctx).sum(dim=-1).cpu().numpy()
    
    # 计算 L2 距离
    l2_dist = torch.norm(features_with_ctx - features_no_ctx, dim=-1).cpu().numpy()
    
    # 按 prompt 分组统计
    print(f"\n逐 Prompt 统计 (每个 prompt 有 {n_pro} 个副本):")
    print("-"*80)
    print(f"{'Prompt ID':<12} {'Template':<30} {'Avg Cosine':<12} {'Avg L2':<12}")
    print("-"*80)
    
    for j in range(num_prompts):
        start_idx = j * n_pro
        end_idx = start_idx + n_pro
        
        avg_cos = cosine_sim[start_idx:end_idx].mean()
        avg_l2 = l2_dist[start_idx:end_idx].mean()
        
        template = prompt_info['prompt_details'][j]['template']
        if len(template) > 28:
            template = template[:25] + "..."
        
        print(f"{j:<12} {template:<30} {avg_cos:<12.4f} {avg_l2:<12.4f}")
    
    # 总体统计
    print("\n" + "="*80)
    print("总体统计:")
    print("="*80)
    print(f"Cosine Similarity:")
    print(f"  - Mean: {cosine_sim.mean():.4f}")
    print(f"  - Std:  {cosine_sim.std():.4f}")
    print(f"  - Min:  {cosine_sim.min():.4f}")
    print(f"  - Max:  {cosine_sim.max():.4f}")
    print(f"  - % > 0.99: {(cosine_sim > 0.99).mean() * 100:.1f}%")
    print(f"  - % > 0.95: {(cosine_sim > 0.95).mean() * 100:.1f}%")
    print(f"  - % < 0.90: {(cosine_sim < 0.90).mean() * 100:.1f}%")
    
    print(f"\nL2 Distance:")
    print(f"  - Mean: {l2_dist.mean():.4f}")
    print(f"  - Std:  {l2_dist.std():.4f}")
    print(f"  - Min:  {l2_dist.min():.4f}")
    print(f"  - Max:  {l2_dist.max():.4f}")
    
    # 判断结论
    print("\n" + "="*80)
    print("结论:")
    print("="*80)
    
    if cosine_sim.mean() > 0.99:
        print("✓ 相似度极高 (>0.99)")
        print("  → Phase 1 使用的 embedding 可能与 raw text encoding 相同")
        print("  → 这意味着 ctx 可能没有生效，或者初始化的 ctx 影响很小")
    elif cosine_sim.mean() > 0.95:
        print("✓ 相似度很高 (0.95-0.99)")
        print("  → ctx 有轻微影响，但不明显")
    elif cosine_sim.mean() > 0.90:
        print("⚠ 相似度较高 (0.90-0.95)")
        print("  → ctx 有一定影响")
    else:
        print("✗ 相似度较低 (<0.90)")
        print("  → Phase 1 的 embedding 与 raw text 有显著差异")
        print("  → 这是预期的！说明 ctx 正在发挥作用")
    
    # 保存详细结果
    output_dir = f"result/prompt_purging/sanity_tests"
    os.makedirs(output_dir, exist_ok=True)
    
    import pandas as pd
    
    results = []
    for j in range(num_prompts):
        start_idx = j * n_pro
        end_idx = start_idx + n_pro
        
        detail = prompt_info['prompt_details'][j]
        
        for i in range(n_pro):
            idx = start_idx + i
            results.append({
                'prompt_id': j,
                'replica': i,
                'template': detail['template'],
                'full_text': detail['text'],
                'type': detail['type'],
                'cosine_similarity': cosine_sim[idx],
                'l2_distance': l2_dist[idx],
            })
    
    df = pd.DataFrame(results)
    output_file = f"{output_dir}/{args.dataset}_{args.classname}_embedding_comparison.csv"
    df.to_csv(output_file, index=False)
    print(f"\n详细结果已保存: {output_file}")
    
    return cosine_sim.mean(), l2_dist.mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # 数据集参数
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa'])
    parser.add_argument('--classname', type=str, default='bottle')
    parser.add_argument('--k_shot', type=int, default=2)
    
    # 模型参数
    parser.add_argument('--backbone', type=str, default='ViT-B-16-plus-240')
    parser.add_argument('--pretrained_dataset', type=str, default='laion400m_e32')
    parser.add_argument('--device', type=int, default=0)
    
    # Prompt 参数
    parser.add_argument('--n_ctx', type=int, default=12)
    parser.add_argument('--n_pro', type=int, default=4)
    parser.add_argument('--n_ctx_ab', type=int, default=12)
    parser.add_argument('--n_pro_ab', type=int, default=1)
    
    # 图像参数
    parser.add_argument('--img_resize', type=int, default=256)
    parser.add_argument('--img_cropsize', type=int, default=240)
    
    args = parser.parse_args()
    
    test_embedding_consistency(args)
