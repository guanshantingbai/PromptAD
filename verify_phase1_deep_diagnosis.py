"""
深度诊断：为什么 bottle 有高 AUROC 但 Phase 1 显示大量负 margin？

关键洞察：
- Phase 1 只看正常样本：m_j(x_normal) = s_n - s_a_j
- 实际推理用 softmax：prob_abnormal = softmax([s_n, s_a_max])[1]
- 即使 m_j < 0，只要对异常样本的 margin 更负，分类仍然正确！

验证假设：
1. 对于正常样本：m_j 可能是负的（-5）
2. 对于异常样本：m_j 应该更负（-20）
3. 只要异常样本的 margin 更负，ROC 曲线仍然正确
"""

import os
import torch
import argparse
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset as TorchDataset

from PromptAD.model import PromptAD
from datasets.mvtec import load_mvtec
from datasets.visa import load_visa


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


def analyze_normal_vs_abnormal_margins(args):
    """
    对比正常和异常样本的 margin 分布
    验证：即使正常样本 margin < 0，只要异常样本更负，分类仍然有效
    """
    
    print("="*80)
    print("深度诊断: 正常 vs 异常样本的 Margin 分布")
    print("="*80)
    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    
    # 加载数据
    print(f"\n加载数据...")
    if args.dataset == 'mvtec':
        train_data, test_data = load_mvtec(args.classname, args.k_shot)
    else:
        train_data, test_data = load_visa(args.classname, args.k_shot)
    
    test_img_paths, _, test_labels, _ = test_data
    
    # 分别取正常和异常样本
    normal_idx = [i for i, label in enumerate(test_labels) if label == 0][:args.num_normal]
    abnormal_idx = [i for i, label in enumerate(test_labels) if label == 1][:args.num_abnormal]
    
    print(f"✓ 正常样本: {len(normal_idx)}")
    print(f"✓ 异常样本: {len(abnormal_idx)}")
    
    all_paths = [test_img_paths[i] for i in normal_idx] + [test_img_paths[i] for i in abnormal_idx]
    all_labels = [0] * len(normal_idx) + [1] * len(abnormal_idx)
    
    dataset = MixedDataset(all_paths, all_labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # 创建模型
    print(f"\n创建模型...")
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
    model.build_text_feature_gallery()
    
    prompt_info = model.get_manual_prompt_info()
    n_pro = prompt_info['n_pro']
    num_prompts = prompt_info['num_manual_templates']
    
    print(f"✓ {num_prompts} 条 manual prompts")
    
    # 分别统计正常和异常样本的 margin
    margins_normal = {j: [] for j in range(num_prompts)}
    margins_abnormal = {j: [] for j in range(num_prompts)}
    
    print(f"\n计算 margins...")
    
    with torch.no_grad():
        for images, _, labels, _ in tqdm(loader, desc="Processing"):
            images = images.to(device)
            
            # 编码图像
            image_features = model.encode_image(images)
            if args.task == 'seg':
                cls_feature, patch_features, _, _ = image_features
                features = patch_features.mean(dim=1)
            else:
                cls_feature, _, _, _ = image_features
                features = cls_feature
            
            features = features / features.norm(dim=-1, keepdim=True)
            
            t = 100
            
            # 正常分支（修复版）
            normal_sim = t * features @ model.normal_prototypes.T
            s_n = normal_sim.max(dim=-1)[0]
            
            # 异常分支
            manual_prototypes = model.abnormal_prototypes[:num_prompts * n_pro]
            manual_prototypes_reshaped = manual_prototypes.reshape(num_prompts, n_pro, -1)
            
            for j in range(num_prompts):
                prompt_prototypes = manual_prototypes_reshaped[j]
                s_a_j = t * (features @ prompt_prototypes.T)
                s_a_j_max = s_a_j.max(dim=-1)[0]
                
                m_j = (s_n - s_a_j_max).cpu().numpy()
                
                # 分别存储正常和异常样本的 margin
                for i, label in enumerate(labels):
                    if label == 0:
                        margins_normal[j].append(m_j[i])
                    else:
                        margins_abnormal[j].append(m_j[i])
    
    # 分析结果
    print("\n" + "="*80)
    print("对比分析: 正常样本 vs 异常样本")
    print("="*80)
    
    print(f"\n{'ID':<4} {'Template':<25} {'Normal Med':<12} {'Abnormal Med':<12} {'Diff':<10} {'Separable?':<12}")
    print("-"*90)
    
    results = []
    
    for j in range(num_prompts):
        normal_vals = np.array(margins_normal[j])
        abnormal_vals = np.array(margins_abnormal[j])
        
        if len(normal_vals) == 0 or len(abnormal_vals) == 0:
            continue
        
        normal_median = np.median(normal_vals)
        abnormal_median = np.median(abnormal_vals)
        diff = normal_median - abnormal_median  # 正常应该 > 异常
        
        # 判断是否可分
        separable = "✓" if diff > 1.0 else ("⚠" if diff > 0 else "✗")
        
        template = prompt_info['prompt_details'][j]['template']
        if len(template) > 23:
            template = template[:20] + "..."
        
        print(f"{j:<4} {template:<25} {normal_median:<12.2f} {abnormal_median:<12.2f} {diff:<10.2f} {separable:<12}")
        
        results.append({
            'prompt_id': j,
            'template': prompt_info['prompt_details'][j]['template'],
            'full_text': prompt_info['prompt_details'][j]['text'],
            'type': prompt_info['prompt_details'][j]['type'],
            'normal_median': normal_median,
            'normal_mean': normal_vals.mean(),
            'normal_std': normal_vals.std(),
            'normal_q10': np.percentile(normal_vals, 10),
            'normal_q90': np.percentile(normal_vals, 90),
            'abnormal_median': abnormal_median,
            'abnormal_mean': abnormal_vals.mean(),
            'abnormal_std': abnormal_vals.std(),
            'abnormal_q10': np.percentile(abnormal_vals, 10),
            'abnormal_q90': np.percentile(abnormal_vals, 90),
            'separation_gap': diff,
            'R_normal_negative': (normal_vals < 0).mean(),
            'R_abnormal_negative': (abnormal_vals < 0).mean(),
        })
    
    # 总体统计
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("总体统计:")
    print("="*80)
    
    print(f"\n正常样本:")
    print(f"  - Median margin 平均: {df['normal_median'].mean():.2f}")
    print(f"  - Margin < 0 的比例: {df['R_normal_negative'].mean()*100:.1f}%")
    
    print(f"\n异常样本:")
    print(f"  - Median margin 平均: {df['abnormal_median'].mean():.2f}")
    print(f"  - Margin < 0 的比例: {df['R_abnormal_negative'].mean()*100:.1f}%")
    
    print(f"\n分离度:")
    print(f"  - 平均分离 gap: {df['separation_gap'].mean():.2f}")
    print(f"  - Gap > 0 的比例: {(df['separation_gap'] > 0).sum()}/{len(df)}")
    print(f"  - Gap > 1.0 的比例: {(df['separation_gap'] > 1.0).sum()}/{len(df)}")
    
    # 关键洞察
    print("\n" + "="*80)
    print("关键洞察:")
    print("="*80)
    
    both_negative = df[(df['R_normal_negative'] > 0.9) & (df['R_abnormal_negative'] > 0.9)]
    
    if len(both_negative) > 0:
        print(f"\n✓ 发现 {len(both_negative)} 个 prompt，正常和异常样本都是负 margin！")
        print(f"  → 但如果异常样本更负（gap > 0），分类仍然有效")
        print(f"\n  这些 prompt 的分离度:")
        for _, row in both_negative.iterrows():
            status = "✓可分" if row['separation_gap'] > 1.0 else ("⚠弱" if row['separation_gap'] > 0 else "✗不可分")
            print(f"    {row['template'][:30]:<32} gap={row['separation_gap']:>6.2f} {status}")
    
    print(f"\n结论:")
    print(f"  - Phase 1 的 R_j_0 (正常样本 margin < 0) 是一个**误导性指标**")
    print(f"  - 关键是 **分离度**：正常 margin - 异常 margin")
    print(f"  - 即使都是负值，只要异常更负，ROC 性能仍然好")
    
    # 保存结果
    output_dir = f"result/prompt_purging/sanity_tests"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = f"{output_dir}/{args.dataset}_{args.classname}_normal_vs_abnormal.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✓ 详细结果已保存: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa'])
    parser.add_argument('--classname', type=str, default='bottle')
    parser.add_argument('--k_shot', type=int, default=2)
    parser.add_argument('--task', type=str, default='cls', choices=['cls', 'seg'])
    
    parser.add_argument('--backbone', type=str, default='ViT-B-16-plus-240')
    parser.add_argument('--pretrained_dataset', type=str, default='laion400m_e32')
    parser.add_argument('--device', type=int, default=0)
    
    parser.add_argument('--n_ctx', type=int, default=12)
    parser.add_argument('--n_pro', type=int, default=4)
    parser.add_argument('--n_ctx_ab', type=int, default=12)
    parser.add_argument('--n_pro_ab', type=int, default=1)
    
    parser.add_argument('--img_resize', type=int, default=256)
    parser.add_argument('--img_cropsize', type=int, default=240)
    
    parser.add_argument('--num_normal', type=int, default=30, help='正常样本数量')
    parser.add_argument('--num_abnormal', type=int, default=30, help='异常样本数量')
    parser.add_argument('--batch_size', type=int, default=8)
    
    args = parser.parse_args()
    
    analyze_normal_vs_abnormal_margins(args)
