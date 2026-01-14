#!/usr/bin/env python3
"""
数值自检：LSE与softmax尺度验证
只运行一次forward，打印所有关键数值
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
import torch
from PromptAD.model import PromptAD

def debug_numerical_check():
    """运行一次forward进行数值自检"""
    
    print("\n" + "="*80)
    print("LSE 数值自检 - 运行单次 forward")
    print("="*80)
    
    # 配置
    class Args:
        dataset = 'mvtec'
        class_name = 'metal_nut'
        k_shot = 2
        backbone = 'ViT-B-16-plus-240'
        pretrained_dataset = 'laion400m_e32'
        n_ctx = 12
        n_ctx_ab = 12
        n_pro = 1
        n_pro_ab = 1
        img_resize = 240
        img_cropsize = 240
        seed = 111
    
    args = Args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 创建随机输入
    print(f"\n使用随机输入进行数值自检...")
    batch_size = 8
    image = torch.randn(batch_size, 3, 240, 240)
    
    print(f"Batch size: {batch_size}")
    print(f"Image shape: {image.shape}")
    
    # 创建模型
    print("\n创建模型...")
    model = PromptAD(
        out_size_h=240,
        out_size_w=240,
        device=device,
        backbone=args.backbone,
        pretrained_dataset=args.pretrained_dataset,
        n_ctx=args.n_ctx,
        n_pro=args.n_pro,
        n_ctx_ab=args.n_ctx_ab,
        n_pro_ab=args.n_pro_ab,
        class_name=args.class_name,
        k_shot=args.k_shot,
        img_resize=args.img_resize,
        img_cropsize=args.img_cropsize
    )
    
    model.to(device)
    
    # 加载checkpoint（参照test_cls.py的方式）
    checkpoint_path = f'./result/fusion_normal/{args.dataset}/k_{args.k_shot}/checkpoint/CLS-Seed_{args.seed}-{args.class_name}-check_point.pt'
    
    if not os.path.exists(checkpoint_path):
        print(f"\n警告: checkpoint不存在: {checkpoint_path}")
        print("将使用随机初始化模型进行数值范围自检...")
    else:
        print(f"\n加载 checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # 过滤掉_all后缀的key（它们会在build_text_feature_gallery时重新计算）
        checkpoint = {k: v for k, v in checkpoint.items() if not k.endswith('_all')}
        model.load_state_dict(checkpoint, strict=False)
        # 重新计算_all features（从加载的prompts）
        model.build_text_feature_gallery()
        print("✓ checkpoint加载成功，已重建text feature gallery")
    
    model.eval()
    
    # 移动到设备
    image = image.to(device)
    
    # 设置调试标记
    model._debug_print_once = True
    model._debug_lse_cls = True
    
    print("\n" + "="*80)
    print("开始 Forward Pass (CLS task, aggregation=lse)")
    print("="*80)
    
    # 运行forward
    with torch.no_grad():
        # 提取视觉特征
        visual_features = model.encode_image(image)
        
        # 测试不同τ值
        for tau in [5, 10, 20]:
            print(f"\n{'='*80}")
            print(f"测试 τ = {tau}")
            print('='*80)
            
            # 重新设置标记（每个τ都打印一次）
            if tau == 10:
                model._debug_print_once = True
                model._debug_lse_cls = True
            
            # 运行计算
            anomaly_score, logits = model.calculate_textual_anomaly_score(
                visual_features,
                task='cls',
                return_logits=True,
                aggregation='lse',
                lse_tau=tau
            )
            
            if tau != 10:
                # 简化输出
                print(f"\nτ={tau}: anomaly_score range=[{anomaly_score.min():.6f}, {anomaly_score.max():.6f}], "
                      f"mean={anomaly_score.mean():.6f}")
    
    print("\n" + "="*80)
    print("数值自检完成")
    print("="*80)

if __name__ == '__main__':
    debug_numerical_check()
