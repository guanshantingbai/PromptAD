"""
Sanity Test C: 修复后的 Phase 1 margin 计算
使用与 forward 完全一致的方式: max over all prototypes
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


class SimpleNormalDataset(TorchDataset):
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


def compute_margins_fixed(model, normal_loader, epsilon=0.05, task='cls'):
    """
    修复版：使用与 forward 完全一致的计算方式
    s_n = MAX(sim to all K normal prototypes)  ← 修复点
    s_a_j = MAX(sim to n_pro replicas of prompt j)
    """
    model.eval()
    
    prompt_info = model.get_manual_prompt_info()
    num_prompts = prompt_info['num_manual_templates']
    n_pro = prompt_info['n_pro']
    
    print(f"\n✓ 分析 {num_prompts} 条 manual prompts (每条 {n_pro} 个副本)")
    print(f"✓ 使用修复后的计算: s_n = MAX(sim to all K normal prototypes)")
    
    margins = {j: [] for j in range(num_prompts)}
    
    with torch.no_grad():
        for images, _, labels, _ in tqdm(normal_loader, desc="Computing margins (FIXED)"):
            images = images.to(model.device)
            normal_mask = (labels == 0)
            if normal_mask.sum() == 0:
                continue
            images = images[normal_mask]
            
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
            
            # ✅ 修复: 与 forward 一致 - 取 MAX over all K normal prototypes
            normal_sim = t * features @ model.normal_prototypes.T  # [N, K]
            s_n = normal_sim.max(dim=-1)[0]  # [N] - 取最大值
            
            # 异常分支（与之前一致）
            manual_prototypes = model.abnormal_prototypes[:num_prompts * n_pro]
            manual_prototypes_reshaped = manual_prototypes.reshape(num_prompts, n_pro, -1)
            
            for j in range(num_prompts):
                prompt_prototypes = manual_prototypes_reshaped[j]
                s_a_j = t * (features @ prompt_prototypes.T)
                s_a_j_max = s_a_j.max(dim=-1)[0]
                
                # margin
                m_j = (s_n - s_a_j_max).cpu().numpy()
                margins[j].extend(m_j.tolist())
    
    # 计算统计
    results = []
    for j in range(num_prompts):
        margin_values = np.array(margins[j])
        if len(margin_values) == 0:
            continue
        
        R_j_0 = (margin_values < 0).mean()
        R_j_eps = (margin_values < epsilon).mean()
        
        prompt_detail = prompt_info['prompt_details'][j]
        
        result = {
            'prompt_index': j,
            'prompt_id': prompt_detail.get('prompt_id', j),
            'class': prompt_info['classname'],
            'type': prompt_detail['type'],
            'template': prompt_detail.get('template', ''),
            'full_text': prompt_detail['text'],
            'R_j_0': R_j_0,
            'R_j_eps': R_j_eps,
            'median_margin': np.median(margin_values),
            'q10_margin': np.percentile(margin_values, 10),
            'mean_margin': np.mean(margin_values),
            'std_margin': np.std(margin_values),
            'num_samples': len(margin_values),
        }
        results.append(result)
    
    return results


def test_fixed_vs_old(args):
    """对比修复前后的差异"""
    
    print("="*80)
    print("Sanity Test C: 修复前后对比")
    print("="*80)
    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    
    # 加载数据
    print(f"\n加载数据...")
    if args.dataset == 'mvtec':
        train_data, test_data = load_mvtec(args.classname, args.k_shot)
    else:
        train_data, test_data = load_visa(args.classname, args.k_shot)
    
    train_img_paths, _, train_labels, _ = train_data
    test_img_paths, _, test_labels, _ = test_data
    
    train_normal_idx = [i for i, label in enumerate(train_labels) if label == 0][:args.num_samples//2]
    test_normal_idx = [i for i, label in enumerate(test_labels) if label == 0][:args.num_samples//2]
    
    all_normal_paths = [train_img_paths[i] for i in train_normal_idx] + \
                       [test_img_paths[i] for i in test_normal_idx]
    all_normal_labels = [0] * len(all_normal_paths)
    
    print(f"✓ 使用 {len(all_normal_paths)} 个正常样本")
    
    normal_dataset = SimpleNormalDataset(all_normal_paths, all_normal_labels)
    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=False)
    
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
    model.build_text_feature_gallery()
    
    # 运行修复版
    results_fixed = compute_margins_fixed(model, normal_loader, epsilon=args.epsilon, task=args.task)
    
    # 加载旧的 Phase 1 结果（如果存在）
    old_file = f"result/prompt_purging/phase1/{args.dataset}/k_{args.k_shot}/{args.classname}_phase1_normal_side_risk_eps{args.epsilon}.csv"
    
    print("\n" + "="*80)
    print("对比结果:")
    print("="*80)
    
    if os.path.exists(old_file):
        df_old = pd.read_csv(old_file)
        df_fixed = pd.DataFrame(results_fixed)
        
        print(f"\n{'Prompt':<6} {'Template':<25} {'Old R_j_0':<12} {'Fixed R_j_0':<12} {'Diff':<10}")
        print("-"*80)
        
        for j in range(len(results_fixed)):
            template = results_fixed[j]['template']
            if len(template) > 23:
                template = template[:20] + "..."
            
            old_r = df_old.iloc[j]['R_j_0'] if j < len(df_old) else np.nan
            fixed_r = results_fixed[j]['R_j_0']
            diff = abs(old_r - fixed_r) if not np.isnan(old_r) else np.nan
            
            print(f"{j:<6} {template:<25} {old_r:<12.4f} {fixed_r:<12.4f} {diff:<10.4f}")
        
        print("\n总体对比:")
        print(f"  旧版平均 R_j_0: {df_old['R_j_0'].mean():.4f}")
        print(f"  修复版平均 R_j_0: {df_fixed['R_j_0'].mean():.4f}")
        print(f"  差异: {abs(df_old['R_j_0'].mean() - df_fixed['R_j_0'].mean()):.4f}")
        
        print(f"\n  旧版 R_j_0=1.0 的比例: {(df_old['R_j_0'] >= 0.999).sum()}/{len(df_old)}")
        print(f"  修复版 R_j_0=1.0 的比例: {(df_fixed['R_j_0'] >= 0.999).sum()}/{len(df_fixed)}")
        
    else:
        print(f"\n⚠️  未找到旧版结果文件: {old_file}")
        print("只显示修复版结果:")
        
        df_fixed = pd.DataFrame(results_fixed)
        print(f"\n{df_fixed[['template', 'R_j_0', 'R_j_eps', 'median_margin']].to_string(index=False)}")
        
        print(f"\n修复版统计:")
        print(f"  平均 R_j_0: {df_fixed['R_j_0'].mean():.4f}")
        print(f"  平均 R_j_eps: {df_fixed['R_j_eps'].mean():.4f}")
        print(f"  R_j_0=1.0 的比例: {(df_fixed['R_j_0'] >= 0.999).sum()}/{len(df_fixed)}")
    
    # 保存修复版结果
    output_dir = f"result/prompt_purging/sanity_tests"
    os.makedirs(output_dir, exist_ok=True)
    
    df_fixed = pd.DataFrame(results_fixed)
    output_file = f"{output_dir}/{args.dataset}_{args.classname}_phase1_FIXED.csv"
    df_fixed.to_csv(output_file, index=False)
    print(f"\n✓ 修复版结果已保存: {output_file}")
    
    # 结论
    print("\n" + "="*80)
    print("结论:")
    print("="*80)
    print("✓ 修复点: s_n = MAX(sim to all K prototypes) 而非 mean(prototypes)")
    print("✓ 现在与 model.forward() 的 semantic 分支完全一致")
    print("✓ 如果修复后 R_j_0=1.0 显著减少，说明原 Phase 1 有问题")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa'])
    parser.add_argument('--classname', type=str, default='bottle')
    parser.add_argument('--k_shot', type=int, default=2)
    parser.add_argument('--task', type=str, default='cls', choices=['cls', 'seg'])
    parser.add_argument('--epsilon', type=float, default=0.05)
    
    parser.add_argument('--backbone', type=str, default='ViT-B-16-plus-240')
    parser.add_argument('--pretrained_dataset', type=str, default='laion400m_e32')
    parser.add_argument('--device', type=int, default=0)
    
    parser.add_argument('--n_ctx', type=int, default=12)
    parser.add_argument('--n_pro', type=int, default=4)
    parser.add_argument('--n_ctx_ab', type=int, default=12)
    parser.add_argument('--n_pro_ab', type=int, default=1)
    
    parser.add_argument('--img_resize', type=int, default=256)
    parser.add_argument('--img_cropsize', type=int, default=240)
    
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    
    args = parser.parse_args()
    
    test_fixed_vs_old(args)
