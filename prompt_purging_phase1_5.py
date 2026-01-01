"""
Prompt Purging – Phase 1.5: Prompt Classification

目标：在 Phase 1 基础上，引入异常样本计算 separation_gap，将 prompts 分为三类：
1. Safe Prompt: R_j_eps < 0.2
2. Dangerous-but-Useful: R_j_eps ≥ 0.2 且 separation_gap > 0
3. Dangerous-and-Useless: R_j_eps ≥ 0.2 且 separation_gap ≤ 0

核心原则：
- 不改模型结构，不引入训练
- 只做 inference-level 分析
- 使用修复后的 s_n 计算（max over normal prototypes）
"""

import os
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset as TorchDataset
import cv2

from PromptAD.model import PromptAD
from datasets.mvtec import load_mvtec, mvtec_classes
from datasets.visa import load_visa, visa_classes


class MixedDataset(TorchDataset):
    """包含正常和异常样本的数据集"""
    def __init__(self, img_paths, labels, img_size=256):
        self.img_paths = img_paths
        self.labels = labels
        self.img_size = img_size
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        gt = torch.zeros(self.img_size, self.img_size)
        
        return img, gt, label, ""


def compute_margins_with_separation(model, data_loader, epsilon=0.05, task='cls'):
    """
    计算每条 prompt 的正常/异常样本 margin，并计算 separation_gap
    
    Returns:
        list of dict: 每条 prompt 的统计指标
    """
    model.eval()
    
    prompt_info = model.get_manual_prompt_info()
    num_prompts = prompt_info['num_manual_templates']
    n_pro = prompt_info['n_pro']
    
    print(f"\n分析 {num_prompts} 条 manual prompts (每条 {n_pro} 个副本)")
    print(f"使用 epsilon = {epsilon}")
    
    # 分别存储正常和异常样本的 margin
    margins_normal = {j: [] for j in range(num_prompts)}
    margins_abnormal = {j: [] for j in range(num_prompts)}
    
    with torch.no_grad():
        for images, _, labels, _ in tqdm(data_loader, desc="Computing margins"):
            images = images.to(model.device)
            
            # 编码图像
            image_features = model.encode_image(images)
            if task == 'seg':
                cls_feature, patch_features, _, _ = image_features
                features = patch_features.mean(dim=1)
            else:
                cls_feature, _, _, _ = image_features
                features = cls_feature
            
            features = features / features.norm(dim=-1, keepdim=True)
            
            t = 100
            
            # ✅ 修复版：与 model.forward() 一致
            normal_sim = t * features @ model.normal_prototypes.T  # [N, K]
            s_n = normal_sim.max(dim=-1)[0]  # [N]
            
            # 异常分支
            manual_prototypes = model.abnormal_prototypes[:num_prompts * n_pro]
            manual_prototypes_reshaped = manual_prototypes.reshape(num_prompts, n_pro, -1)
            
            for j in range(num_prompts):
                prompt_prototypes = manual_prototypes_reshaped[j]
                s_a_j = t * (features @ prompt_prototypes.T)
                s_a_j_max = s_a_j.max(dim=-1)[0]
                
                # margin: m_j(x) = s_n(x) - s_a_j(x)
                m_j = (s_n - s_a_j_max).cpu().numpy()
                
                # 根据 label 分别存储
                for i, label in enumerate(labels):
                    if label == 0:  # 正常
                        margins_normal[j].append(m_j[i])
                    else:  # 异常
                        margins_abnormal[j].append(m_j[i])
    
    # 计算统计指标
    print("\n计算统计指标...")
    results = []
    
    for j in range(num_prompts):
        normal_vals = np.array(margins_normal[j])
        abnormal_vals = np.array(margins_abnormal[j])
        
        if len(normal_vals) == 0:
            print(f"  警告: prompt {j} 没有正常样本数据")
            continue
        
        if len(abnormal_vals) == 0:
            print(f"  警告: prompt {j} 没有异常样本数据")
            continue
        
        # 正常样本指标
        R_j_0 = (normal_vals < 0).mean()
        R_j_eps = (normal_vals < epsilon).mean()
        median_margin_normal = np.median(normal_vals)
        
        # 异常样本指标
        median_margin_abnormal = np.median(abnormal_vals)
        
        # separation_gap (关键指标)
        separation_gap = median_margin_normal - median_margin_abnormal
        
        # Prompt 分类
        if R_j_eps < 0.2:
            prompt_type = 'safe'
        elif separation_gap > 0:
            prompt_type = 'dangerous_useful'
        else:
            prompt_type = 'dangerous_useless'
        
        # 获取 prompt 详细信息
        prompt_detail = prompt_info['prompt_details'][j]
        
        result = {
            'prompt_index': j,
            'prompt_id': prompt_detail.get('prompt_id', j),
            'class': prompt_info['classname'],
            'display_name': prompt_info['display_name'],
            'type': prompt_detail['type'],  # generic / specific
            'template': prompt_detail.get('template', ''),
            'full_text': prompt_detail['text'],
            
            # Phase 1 指标（正常样本）
            'R_j_0': R_j_0,
            'R_j_eps': R_j_eps,
            'median_margin_normal': median_margin_normal,
            'mean_margin_normal': normal_vals.mean(),
            'std_margin_normal': normal_vals.std(),
            'q10_margin_normal': np.percentile(normal_vals, 10),
            'q90_margin_normal': np.percentile(normal_vals, 90),
            'num_normal_samples': len(normal_vals),
            
            # Phase 1.5 新增指标（异常样本）
            'median_margin_abnormal': median_margin_abnormal,
            'mean_margin_abnormal': abnormal_vals.mean(),
            'std_margin_abnormal': abnormal_vals.std(),
            'q10_margin_abnormal': np.percentile(abnormal_vals, 10),
            'q90_margin_abnormal': np.percentile(abnormal_vals, 90),
            'num_abnormal_samples': len(abnormal_vals),
            
            # 关键分类指标
            'separation_gap': separation_gap,
            'prompt_classification': prompt_type,
        }
        
        results.append(result)
    
    return results
"""
Phase 1.5 主函数和命令行接口 - Part 2
"""

def analyze_class_with_classification(args):
    """对单个类别进行 Prompt Classification 分析"""
    
    print("="*80)
    print(f"Prompt Purging Phase 1.5: {args.dataset} - {args.classname}")
    print("Prompt Classification with Separation Gap")
    print("="*80)
    
    # 加载数据
    print(f"\n加载数据...")
    if args.dataset == 'mvtec':
        train_data, test_data = load_mvtec(args.classname, args.k_shot)
    else:
        train_data, test_data = load_visa(args.classname, args.k_shot)
    
    train_img_paths, _, train_labels, _ = train_data
    test_img_paths, _, test_labels, _ = test_data
    
    # 使用训练集+测试集的正常样本
    train_normal_idx = [i for i, label in enumerate(train_labels) if label == 0]
    test_normal_idx = [i for i, label in enumerate(test_labels) if label == 0]
    
    # 使用测试集的异常样本（全部）
    test_abnormal_idx = [i for i, label in enumerate(test_labels) if label == 1]
    
    print(f"✓ 训练集正常样本: {len(train_normal_idx)}")
    print(f"✓ 测试集正常样本: {len(test_normal_idx)}")
    print(f"✓ 测试集异常样本: {len(test_abnormal_idx)}")
    
    # 合并所有样本
    all_paths = [train_img_paths[i] for i in train_normal_idx] + \
                [test_img_paths[i] for i in test_normal_idx] + \
                [test_img_paths[i] for i in test_abnormal_idx]
    
    all_labels = [0] * (len(train_normal_idx) + len(test_normal_idx)) + \
                 [1] * len(test_abnormal_idx)
    
    dataset = MixedDataset(all_paths, all_labels, img_size=256)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"✓ 总样本数: {len(dataset)} (正常: {len(train_normal_idx)+len(test_normal_idx)}, 异常: {len(test_abnormal_idx)})")
    
    # 加载模型
    print(f"\n加载模型...")
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    
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
    
    print("\n构建文本特征...")
    model.build_text_feature_gallery()
    
    # 执行分析
    results = compute_margins_with_separation(
        model, 
        data_loader, 
        epsilon=args.epsilon,
        task=args.task
    )
    
    # 转换为 DataFrame
    df = pd.DataFrame(results)
    
    # 统计各类别数量
    safe_count = (df['prompt_classification'] == 'safe').sum()
    dangerous_useful_count = (df['prompt_classification'] == 'dangerous_useful').sum()
    dangerous_useless_count = (df['prompt_classification'] == 'dangerous_useless').sum()
    
    # 保存结果
    output_dir = os.path.join(args.output_dir, args.dataset, f'k_{args.k_shot}')
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(
        output_dir, 
        f'{args.classname}_phase1_5_classification.csv'
    )
    
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*80}")
    print(f"分析摘要")
    print(f"{'='*80}")
    print(f"类别: {args.classname}")
    print(f"总 prompts: {len(df)}")
    print(f"  - Generic: {(df['type'] == 'generic').sum()}")
    print(f"  - Specific: {(df['type'] == 'specific').sum()}")
    
    print(f"\nPrompt 分类结果:")
    print(f"  - Safe (R_j_eps < 0.2):              {safe_count}")
    print(f"  - Dangerous-but-Useful (gap > 0):    {dangerous_useful_count}")
    print(f"  - Dangerous-and-Useless (gap ≤ 0):   {dangerous_useless_count}")
    
    print(f"\n各类别详情:")
    print("-"*80)
    
    # Safe prompts
    if safe_count > 0:
        print(f"\n✓ Safe Prompts ({safe_count}):")
        safe_df = df[df['prompt_classification'] == 'safe'].sort_values('R_j_eps')
        for _, row in safe_df.head(10).iterrows():
            print(f"  [{row['prompt_index']:>2}] R_eps={row['R_j_eps']:.3f} | gap={row['separation_gap']:>6.2f} | {row['full_text']}")
    
    # Dangerous-but-Useful
    if dangerous_useful_count > 0:
        print(f"\n⚠ Dangerous-but-Useful ({dangerous_useful_count}):")
        useful_df = df[df['prompt_classification'] == 'dangerous_useful'].sort_values('separation_gap', ascending=False)
        for _, row in useful_df.head(10).iterrows():
            print(f"  [{row['prompt_index']:>2}] R_eps={row['R_j_eps']:.3f} | gap={row['separation_gap']:>6.2f} | {row['full_text']}")
    
    # Dangerous-and-Useless
    if dangerous_useless_count > 0:
        print(f"\n✗ Dangerous-and-Useless ({dangerous_useless_count}):")
        useless_df = df[df['prompt_classification'] == 'dangerous_useless'].sort_values('separation_gap')
        for _, row in useless_df.head(10).iterrows():
            print(f"  [{row['prompt_index']:>2}] R_eps={row['R_j_eps']:.3f} | gap={row['separation_gap']:>6.2f} | {row['full_text']}")
    
    print(f"\n{'='*80}")
    print(f"\n✓ 结果已保存: {output_file}")
    print(f"\n✓ Phase 1.5 分析完成！")
    print(f"  下一步: 根据分类结果设计 multi-prototype 策略")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prompt Purging Phase 1.5: Prompt Classification')
    
    # 数据集参数
    parser.add_argument('--dataset', type=str, required=True, choices=['mvtec', 'visa'])
    parser.add_argument('--class', dest='classname', type=str, required=True)
    parser.add_argument('--k_shot', type=int, default=2)
    
    # 模型参数
    parser.add_argument('--backbone', type=str, default='ViT-B-16-plus-240')
    parser.add_argument('--pretrained_dataset', type=str, default='laion400m_e32')
    parser.add_argument('--n_ctx', type=int, default=12)
    parser.add_argument('--n_pro', type=int, default=4)
    parser.add_argument('--n_ctx_ab', type=int, default=12)
    parser.add_argument('--n_pro_ab', type=int, default=1)
    
    # 图像参数
    parser.add_argument('--img_resize', type=int, default=256)
    parser.add_argument('--img_cropsize', type=int, default=240)
    
    # Phase 1.5 参数
    parser.add_argument('--epsilon', type=float, default=0.05, help='Margin threshold')
    parser.add_argument('--task', type=str, default='cls', choices=['cls', 'seg'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=int, default=0)
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='result/prompt_purging/phase1_5')
    
    args = parser.parse_args()
    
    analyze_class_with_classification(args)
