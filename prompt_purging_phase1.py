"""
Prompt Purging – Phase 1: 正常侧语义失配诊断

目标：识别在正常样本上系统性误伤、与正常语义失配的静态 anomaly prompts（MAP）

分析对象：CSV 表中 enabled=True 的每条 manual anomaly prompt
数据范围：仅使用正常样本（训练集 + 测试集）
输出：每条 prompt 的正常侧风险评估指标
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


class SimpleNormalDataset(TorchDataset):
    """简单的正常样本数据集，用于Phase 1分析"""
    def __init__(self, img_paths, labels, img_size=256):
        self.img_paths = img_paths
        self.labels = labels
        self.img_size = img_size
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        
        # 读取图像
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 转换为tensor [H, W, C] -> [C, H, W]
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # 创建虚拟 gt 和 type
        gt = torch.zeros(self.img_size, self.img_size)
        
        return img, gt, label, ""


def compute_normal_side_risk(model, normal_loader, epsilon=0.05, task='seg'):
    """
    计算每条 manual prompt 在正常样本上的风险指标
    
    Args:
        model: 已加载的 PromptAD 模型
        normal_loader: 只包含正常样本的 DataLoader
        epsilon: margin 阈值，默认 0.05
        task: 'cls' 或 'seg'
    
    Returns:
        dict: 每条 prompt 的统计指标
    """
    model.eval()
    
    # 获取 prompt 信息
    prompt_info = model.get_manual_prompt_info()
    num_prompts = prompt_info['num_manual_templates']
    n_pro = prompt_info['n_pro']
    
    print(f"\n分析 {num_prompts} 条 manual prompts (每条重复 {n_pro} 次)")
    print(f"使用 epsilon = {epsilon}")
    print(f"Task = {task}")
    
    # 存储每条 prompt 对每个样本的 margin
    # margins[j] = list of margin values for prompt j
    margins = {j: [] for j in range(num_prompts)}
    
    print(f"\n处理 {len(normal_loader)} 个 batch 的正常样本...")
    
    with torch.no_grad():
        for batch_idx, (images, _, labels, _) in enumerate(tqdm(normal_loader, desc="Computing margins")):
            images = images.to(model.device)
            
            # 只处理正常样本（label=0）
            normal_mask = (labels == 0)
            if normal_mask.sum() == 0:
                continue
            
            images = images[normal_mask]
            
            # 编码图像
            image_features = model.encode_image(images)
            
            if task == 'seg':
                cls_feature, patch_features, _, _ = image_features
                # 使用 patch 级别的平均特征
                features = patch_features.mean(dim=1)  # [N, D]
            else:
                cls_feature, _, _, _ = image_features
                features = cls_feature  # [N, D]
            
            # 归一化
            features = features / features.norm(dim=-1, keepdim=True)
            
            N = features.shape[0]
            
            # 获取正常原型
            normal_prototypes = model.normal_prototypes  # [K, D]
            
            # ✅ 修复: 与 model.forward() 一致 - 计算与每个原型的相似度，然后取 MAX
            # 旧版错误: 先 mean(prototypes) 再计算相似度
            # 正确方式: 分别计算相似度再取 max
            t = 100  # temperature
            normal_sim = t * features @ normal_prototypes.T  # [N, K]
            s_n = normal_sim.max(dim=-1)[0]  # [N] - 取最大相似度
            
            # 获取 manual prompt 原型（前 num_prompts * n_pro 个）
            manual_prototypes = model.abnormal_prototypes[:num_prompts * n_pro]  # [num_prompts * n_pro, D]
            
            # 计算与每条 prompt 的相似度
            # 每条 prompt 有 n_pro 个原型，取最大相似度
            manual_prototypes_reshaped = manual_prototypes.reshape(num_prompts, n_pro, -1)  # [num_prompts, n_pro, D]
            
            for j in range(num_prompts):
                prompt_prototypes = manual_prototypes_reshaped[j]  # [n_pro, D]
                s_a_j = t * (features @ prompt_prototypes.T)  # [N, n_pro]
                s_a_j_max = s_a_j.max(dim=-1)[0]  # [N]
                
                # 计算 margin: m_j(x) = s_n(x) - s_a_j(x)
                m_j = (s_n - s_a_j_max).cpu().numpy()
                
                margins[j].extend(m_j.tolist())
    
    # 计算统计指标
    print("\n计算统计指标...")
    results = []
    
    for j in range(num_prompts):
        margin_values = np.array(margins[j])
        
        if len(margin_values) == 0:
            continue
        
        # 计算各项指标
        R_j_0 = (margin_values < 0).mean()  # P(m_j < 0)
        R_j_eps = (margin_values < epsilon).mean()  # P(m_j < epsilon)
        
        median_margin = np.median(margin_values)
        q10_margin = np.percentile(margin_values, 10)
        mean_margin = np.mean(margin_values)
        std_margin = np.std(margin_values)
        
        # 获取 prompt 详细信息
        prompt_detail = prompt_info['prompt_details'][j]
        
        result = {
            'prompt_index': j,
            'prompt_id': prompt_detail.get('prompt_id', j),
            'class': prompt_info['classname'],
            'display_name': prompt_info['display_name'],
            'type': prompt_detail['type'],
            'template': prompt_detail.get('template', ''),
            'full_text': prompt_detail['text'],
            'R_j_0': R_j_0,
            'R_j_eps': R_j_eps,
            'median_margin': median_margin,
            'q10_margin': q10_margin,
            'mean_margin': mean_margin,
            'std_margin': std_margin,
            'num_samples': len(margin_values),
        }
        
        results.append(result)
    
    return results


def analyze_class(args):
    """分析单个类别"""
    
    print("="*80)
    print(f"Prompt Purging Phase 1: {args.dataset} - {args.classname}")
    print("="*80)
    
    # 加载数据
    print(f"\n加载数据...")
    if args.dataset == 'mvtec':
        train_data, test_data = load_mvtec(args.classname, args.k_shot)
    else:
        train_data, test_data = load_visa(args.classname, args.k_shot)
    
    # 创建数据集（只保留正常样本）
    train_img_paths, _, train_labels, train_types = train_data
    test_img_paths, _, test_labels, test_types = test_data
    
    # 过滤正常样本（label=0 表示正常）
    # ⚠️ 关键确认：同时使用训练集和测试集的正常样本
    train_normal_idx = [i for i, label in enumerate(train_labels) if label == 0]
    test_normal_idx = [i for i, label in enumerate(test_labels) if label == 0]
    
    train_normal_data = (
        [train_img_paths[i] for i in train_normal_idx],
        [0] * len(train_normal_idx),
        [train_labels[i] for i in train_normal_idx],
        [train_types[i] for i in train_normal_idx]
    )
    
    test_normal_data = (
        [test_img_paths[i] for i in test_normal_idx],
        [0] * len(test_normal_idx),
        [test_labels[i] for i in test_normal_idx],
        [test_types[i] for i in test_normal_idx]
    )
    
    print(f"✓ 训练集正常样本: {len(train_normal_idx)}")
    print(f"✓ 测试集正常样本: {len(test_normal_idx)}")
    print(f"  → 确认：两者都将用于风险分析")
    
    # 合并正常样本
    all_normal_img_paths = train_normal_data[0] + test_normal_data[0]
    all_normal_gt = train_normal_data[1] + test_normal_data[1]
    all_normal_labels = train_normal_data[2] + test_normal_data[2]
    all_normal_types = train_normal_data[3] + test_normal_data[3]
    
    all_normal_data = (all_normal_img_paths, all_normal_gt, all_normal_labels, all_normal_types)
    
    normal_dataset = SimpleNormalDataset(all_normal_img_paths, all_normal_labels, img_size=256)
    normal_loader = DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"✓ 总正常样本: {len(normal_dataset)}")
    
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
    
    # 注意：Phase 1 不需要加载checkpoint
    # 因为我们只需要text prototypes（通过build_text_feature_gallery()实时生成）
    # Checkpoint中的memory bank (feature_gallery1/2) 对Phase 1无用
    print(f"\n⚠️ Phase 1 说明：")
    print(f"  - 不需要加载checkpoint")
    print(f"  - Text prototypes通过prompts实时编码")
    print(f"  - Memory bank不参与正常侧风险分析")
    
    model.to(device)
    model.eval()
    
    # 构建文本特征
    print("\n构建文本特征...")
    model.build_text_feature_gallery()
    
    # 执行分析
    results = compute_normal_side_risk(
        model, 
        normal_loader, 
        epsilon=args.epsilon,
        task=args.task
    )
    
    # 转换为 DataFrame
    df = pd.DataFrame(results)
    
    # 按 R_j_eps 从高到低排序
    df = df.sort_values('R_j_eps', ascending=False)
    
    # 添加风险等级标记
    threshold_high = df['R_j_eps'].quantile(0.8)  # Top 20%
    threshold_medium = df['R_j_eps'].quantile(0.5)  # Top 50%
    
    df['risk_level'] = 'low'
    df.loc[df['R_j_eps'] >= threshold_medium, 'risk_level'] = 'medium'
    df.loc[df['R_j_eps'] >= threshold_high, 'risk_level'] = 'high'
    
    # 保存结果
    output_dir = os.path.join(args.output_dir, args.dataset, f'k_{args.k_shot}')
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(
        output_dir, 
        f'{args.classname}_phase1_normal_side_risk_eps{args.epsilon}.csv'
    )
    
    df.to_csv(output_file, index=False)
    print(f"\n✓ 结果已保存: {output_file}")
    
    # 打印摘要
    print("\n" + "="*80)
    print("分析摘要")
    print("="*80)
    print(f"类别: {args.classname}")
    print(f"总 prompts: {len(df)}")
    print(f"  - Generic: {(df['type'] == 'generic').sum()}")
    print(f"  - Specific: {(df['type'] == 'specific').sum()}")
    print(f"\n风险等级分布:")
    print(f"  - High risk (top 20%): {(df['risk_level'] == 'high').sum()}")
    print(f"  - Medium risk: {(df['risk_level'] == 'medium').sum()}")
    print(f"  - Low risk: {(df['risk_level'] == 'low').sum()}")
    
    print(f"\n高风险 prompts (Top 10):")
    print("-"*80)
    top_10 = df.head(10)
    for idx, row in top_10.iterrows():
        print(f"[{row['prompt_index']:2d}] R_eps={row['R_j_eps']:.3f} | "
              f"R_0={row['R_j_0']:.3f} | median={row['median_margin']:6.2f} | "
              f"{row['full_text']}")
    
    print("\n" + "="*80)
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Prompt Purging Phase 1: Normal Side Risk Analysis')
    
    # 数据集参数
    parser.add_argument('--dataset', type=str, required=True, choices=['mvtec', 'visa'])
    parser.add_argument('--class', type=str, dest='classname', required=True)
    parser.add_argument('--k_shot', type=int, default=2)
    
    # 模型参数
    parser.add_argument('--backbone', type=str, default='ViT-B-16-plus-240')
    parser.add_argument('--pretrained_dataset', type=str, default='laion400m_e32')
    parser.add_argument('--n_ctx', type=int, default=12)
    parser.add_argument('--n_pro', type=int, default=4)
    parser.add_argument('--n_ctx_ab', type=int, default=12)
    parser.add_argument('--n_pro_ab', type=int, default=4)
    parser.add_argument('--img_resize', type=int, default=240)
    parser.add_argument('--img_cropsize', type=int, default=240)
    
    # 分析参数
    parser.add_argument('--epsilon', type=float, default=0.05, 
                        help='Margin threshold for R_j_eps')
    parser.add_argument('--task', type=str, default='cls', choices=['cls', 'seg'], 
                        help='Task type: cls for classification, seg for segmentation')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=int, default=0)
    
    # 输出
    parser.add_argument('--output_dir', type=str, 
                        default='result/prompt_purging/phase1')
    
    args = parser.parse_args()
    
    # 执行分析
    df = analyze_class(args)
    
    print("\n✓ Phase 1 分析完成！")
    print(f"  下一步：查看结果并进行 prompt ablation 测试")


if __name__ == '__main__':
    main()
