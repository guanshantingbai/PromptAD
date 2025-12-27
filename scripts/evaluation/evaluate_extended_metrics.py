#!/usr/bin/env python3
"""
扩展评估脚本 - 补全诊断性指标
不修改任何训练代码，仅扩展评估指标

完成三项任务：
【1】拆分 AUROC：Normal-only 和 Abnormal-only AUROC
【2】样本级 margin 分布分析：margin = s_normal - s_ab_max
【3】semantic 分支贡献分析：semantic 与 fusion 的相关性

作者: 诊断分析扩展
日期: 2025-12-26
"""

import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import json

from datasets import dataset_classes, get_dataloader_from_args, denormalization
from utils.training_utils import setup_seed
from PromptAD import PromptAD
from sklearn.metrics import roc_auc_score, roc_curve


def get_dir_from_args(task, **kwargs):
    """从参数获取目录路径"""
    root_dir = kwargs['root_dir']
    dataset = kwargs['dataset']
    class_name = kwargs['class_name']
    k_shot = kwargs['k_shot']
    seed = kwargs['seed']
    
    # 直接使用root_dir，不添加version子目录
    exp_dir = Path(root_dir) / dataset / f'k_{k_shot}'
    
    # Checkpoints在统一目录
    checkpoint_dir = exp_dir / 'checkpoint'
    check_path = checkpoint_dir / f'{task}-Seed_{seed}-{class_name}-check_point.pt'
    
    # CSV和IMG在类别子目录
    class_dir = exp_dir / class_name
    img_dir = class_dir / 'imgs'
    csv_dir = class_dir / 'csv'
    
    img_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = csv_dir / f'{task}.csv'
    
    return str(img_dir), str(csv_path), str(check_path)


def extract_detailed_scores(model, data, device):
    """
    提取详细的分数信息
    
    返回字典:
    - normal_scores: [N, K_normal] 与所有normal原型的相似度
    - abnormal_scores: [N, M_abnormal] 与所有abnormal原型的相似度
    - max_normal: [N] 最大normal相似度
    - max_abnormal: [N] 最大abnormal相似度
    - semantic_score: [N] semantic分支的异常分数
    - visual_score: [N] visual分支的异常分数（如果有）
    - fusion_score: [N] 最终融合分数
    """
    model.eval()
    
    with torch.no_grad():
        # Encode image
        visual_features = model.encode_image(data)
        global_feature = visual_features[0]  # [N, dim]
        
        # Calculate textual scores (semantic branch)
        t = model.model.logit_scale
        
        # Normal similarities
        normal_sim = t * global_feature @ model.normal_prototypes.T  # [N, K_normal]
        max_normal_sim = normal_sim.max(dim=-1)[0]  # [N]
        
        # Abnormal similarities
        abnormal_sim = t * global_feature @ model.abnormal_prototypes.T  # [N, M_abnormal]
        max_abnormal_sim = abnormal_sim.max(dim=-1)[0]  # [N]
        
        # Semantic anomaly score (softmax)
        logits = torch.stack([max_normal_sim, max_abnormal_sim], dim=-1)  # [N, 2]
        prob = logits.softmax(dim=-1)
        semantic_score = prob[:, 1]  # [N]
        
        # Margin = max_normal - max_abnormal
        margin = max_normal_sim - max_abnormal_sim  # [N]
        
        # Calculate visual score (if fusion is used)
        try:
            # Get seg-level visual score and average to image-level
            textual_anomaly_map = model.calculate_textual_anomaly_score(visual_features, 'seg')
            visual_anomaly_map = model.calculate_visual_anomaly_score(visual_features)
            
            # Average over spatial dimensions for image-level score
            visual_score = visual_anomaly_map.mean(dim=(1, 2, 3))  # [N]
            
            # Fusion (same as model forward)
            fusion_map = (textual_anomaly_map + visual_anomaly_map) / 2
            fusion_score = fusion_map.mean(dim=(1, 2, 3))  # [N]
        except:
            visual_score = torch.zeros_like(semantic_score)
            fusion_score = semantic_score
    
    return {
        'normal_scores': normal_sim.cpu().numpy(),  # [N, K_normal]
        'abnormal_scores': abnormal_sim.cpu().numpy(),  # [N, M_abnormal]
        'max_normal': max_normal_sim.cpu().numpy(),  # [N]
        'max_abnormal': max_abnormal_sim.cpu().numpy(),  # [N]
        'margin': margin.cpu().numpy(),  # [N]
        'semantic_score': semantic_score.cpu().numpy(),  # [N]
        'visual_score': visual_score.cpu().numpy(),  # [N]
        'fusion_score': fusion_score.cpu().numpy(),  # [N]
    }


def evaluate_with_extended_metrics(model, args, test_dataloader, train_dataloader, device, check_path):
    """
    扩展评估：收集所有样本的详细分数
    """
    # Load checkpoint
    model.eval()
    checkpoint = torch.load(check_path)
    
    # Load prototypes
    if 'normal_prototypes' in checkpoint:
        model.normal_prototypes = checkpoint['normal_prototypes'].clone()
        if model.precision == 'fp16':
            model.normal_prototypes = model.normal_prototypes.half()
        model.normal_prototypes = model.normal_prototypes.to(device)
    
    if 'abnormal_prototypes' in checkpoint:
        model.abnormal_prototypes = checkpoint['abnormal_prototypes'].clone()
        if model.precision == 'fp16':
            model.abnormal_prototypes = model.abnormal_prototypes.half()
        model.abnormal_prototypes = model.abnormal_prototypes.to(device)
    
    # Build memory bank if needed
    if 'feature_gallery1' not in checkpoint or checkpoint['feature_gallery1'].sum() == 0:
        print("Building memory bank from training data...")
        features1, features2 = [], []
        with torch.no_grad():
            for (data, _, _, _, _) in tqdm(train_dataloader, desc="Building memory bank"):
                data = [model.transform(Image.fromarray(f.numpy())) for f in data]
                data = torch.stack(data, dim=0).to(device)
                _, _, feature_map1, feature_map2 = model.encode_image(data)
                features1.append(feature_map1)
                features2.append(feature_map2)
        
        features1 = torch.cat(features1, dim=0)
        features2 = torch.cat(features2, dim=0)
        model.build_image_feature_gallery(features1, features2)
        print(f"Memory bank built: {model.feature_gallery1.shape[0]} samples")
    
    # Collect all samples
    all_scores = []
    all_labels = []
    all_names = []
    
    print("Extracting detailed scores from test set...")
    for (data, mask, label, name, img_type) in tqdm(test_dataloader):
        data = [model.transform(Image.fromarray(f.numpy())) for f in data]
        data = torch.stack(data, dim=0).to(device)
        
        # Extract detailed scores
        scores_dict = extract_detailed_scores(model, data, device)
        
        # Store per-sample
        for i in range(data.shape[0]):
            sample_scores = {
                'name': name[i],
                'label': label[i].item() if torch.is_tensor(label[i]) else label[i],
                'normal_scores': scores_dict['normal_scores'][i],
                'abnormal_scores': scores_dict['abnormal_scores'][i],
                'max_normal': scores_dict['max_normal'][i],
                'max_abnormal': scores_dict['max_abnormal'][i],
                'margin': scores_dict['margin'][i],
                'semantic_score': scores_dict['semantic_score'][i],
                'visual_score': scores_dict['visual_score'][i],
                'fusion_score': scores_dict['fusion_score'][i],
            }
            all_scores.append(sample_scores)
            all_labels.append(sample_scores['label'])
            all_names.append(sample_scores['name'])
    
    return all_scores, np.array(all_labels), all_names


def calculate_split_auroc(scores_list, labels):
    """
    【任务1】拆分 AUROC
    
    计算：
    - Overall AUROC（标准）
    - Normal-only AUROC（只用正常样本，看能否区分"好"与"中等"）
    - Abnormal-only AUROC（只用异常样本，看能否区分不同异常严重程度）
    """
    # 提取 semantic 和 fusion scores
    semantic_scores = np.array([s['semantic_score'] for s in scores_list])
    fusion_scores = np.array([s['fusion_score'] for s in scores_list])
    
    # Overall AUROC
    try:
        overall_semantic_auroc = roc_auc_score(labels, semantic_scores)
    except:
        overall_semantic_auroc = np.nan
    
    try:
        overall_fusion_auroc = roc_auc_score(labels, fusion_scores)
    except:
        overall_fusion_auroc = np.nan
    
    # Normal-only AUROC (label=0)
    # 理念: 正常样本中，分数越低越"确信正常"，分数越高可能是"边界样本"
    # 如果所有正常样本分数相同，AUROC无法计算
    normal_mask = labels == 0
    normal_semantic = semantic_scores[normal_mask]
    normal_fusion = fusion_scores[normal_mask]
    
    # 创建伪标签：分数低于中位数为"真正常"(0)，高于中位数为"疑似"(1)
    if len(normal_semantic) > 1 and normal_semantic.std() > 1e-6:
        normal_pseudo_labels = (normal_semantic > np.median(normal_semantic)).astype(int)
        try:
            normal_semantic_auroc = roc_auc_score(normal_pseudo_labels, normal_semantic)
        except:
            normal_semantic_auroc = np.nan
    else:
        normal_semantic_auroc = np.nan
    
    if len(normal_fusion) > 1 and normal_fusion.std() > 1e-6:
        normal_pseudo_labels_fusion = (normal_fusion > np.median(normal_fusion)).astype(int)
        try:
            normal_fusion_auroc = roc_auc_score(normal_pseudo_labels_fusion, normal_fusion)
        except:
            normal_fusion_auroc = np.nan
    else:
        normal_fusion_auroc = np.nan
    
    # Abnormal-only AUROC (label=1)
    # 理念: 异常样本中，分数越高越"明显异常"，分数越低可能是"难检测异常"
    abnormal_mask = labels == 1
    abnormal_semantic = semantic_scores[abnormal_mask]
    abnormal_fusion = fusion_scores[abnormal_mask]
    
    # 创建伪标签：分数高于中位数为"明显异常"(1)，低于中位数为"难检测"(0)
    if len(abnormal_semantic) > 1 and abnormal_semantic.std() > 1e-6:
        abnormal_pseudo_labels = (abnormal_semantic > np.median(abnormal_semantic)).astype(int)
        try:
            abnormal_semantic_auroc = roc_auc_score(abnormal_pseudo_labels, abnormal_semantic)
        except:
            abnormal_semantic_auroc = np.nan
    else:
        abnormal_semantic_auroc = np.nan
    
    if len(abnormal_fusion) > 1 and abnormal_fusion.std() > 1e-6:
        abnormal_pseudo_labels_fusion = (abnormal_fusion > np.median(abnormal_fusion)).astype(int)
        try:
            abnormal_fusion_auroc = roc_auc_score(abnormal_pseudo_labels_fusion, abnormal_fusion)
        except:
            abnormal_fusion_auroc = np.nan
    else:
        abnormal_fusion_auroc = np.nan
    
    return {
        'overall_semantic_auroc': overall_semantic_auroc,
        'overall_fusion_auroc': overall_fusion_auroc,
        'normal_semantic_auroc': normal_semantic_auroc,
        'normal_fusion_auroc': normal_fusion_auroc,
        'abnormal_semantic_auroc': abnormal_semantic_auroc,
        'abnormal_fusion_auroc': abnormal_fusion_auroc,
        'n_normal': normal_mask.sum(),
        'n_abnormal': abnormal_mask.sum(),
    }


def analyze_margin_distribution(scores_list, labels):
    """
    【任务2】样本级 margin 分布分析
    
    Margin = max_normal - max_abnormal
    
    正值表示"更接近normal"，负值表示"更接近abnormal"
    """
    margins = np.array([s['margin'] for s in scores_list])
    
    # 按label分组统计
    normal_mask = labels == 0
    abnormal_mask = labels == 1
    
    normal_margins = margins[normal_mask]
    abnormal_margins = margins[abnormal_mask]
    
    def compute_stats(data):
        if len(data) == 0:
            return {
                'mean': np.nan, 'std': np.nan,
                'min': np.nan, 'p10': np.nan, 'p25': np.nan,
                'median': np.nan, 'p75': np.nan, 'p90': np.nan, 'max': np.nan,
                'n': 0
            }
        return {
            'mean': data.mean(),
            'std': data.std(),
            'min': data.min(),
            'p10': np.percentile(data, 10),
            'p25': np.percentile(data, 25),
            'median': np.median(data),
            'p75': np.percentile(data, 75),
            'p90': np.percentile(data, 90),
            'max': data.max(),
            'n': len(data)
        }
    
    return {
        'all': compute_stats(margins),
        'normal': compute_stats(normal_margins),
        'abnormal': compute_stats(abnormal_margins),
        'separation': normal_margins.mean() - abnormal_margins.mean() if len(normal_margins) > 0 and len(abnormal_margins) > 0 else np.nan,
    }


def analyze_semantic_contribution(scores_list, labels):
    """
    【任务3】semantic 分支贡献分析
    
    计算：
    - semantic 与 fusion 的相关性（整体 + 按label分组）
    - semantic 分支的相对贡献强度
    """
    semantic_scores = np.array([s['semantic_score'] for s in scores_list])
    fusion_scores = np.array([s['fusion_score'] for s in scores_list])
    visual_scores = np.array([s['visual_score'] for s in scores_list])
    
    # 计算相关性
    from scipy.stats import pearsonr, spearmanr
    
    # Overall correlation
    try:
        overall_pearson, overall_pearson_p = pearsonr(semantic_scores, fusion_scores)
        overall_spearman, overall_spearman_p = spearmanr(semantic_scores, fusion_scores)
    except:
        overall_pearson, overall_pearson_p = np.nan, np.nan
        overall_spearman, overall_spearman_p = np.nan, np.nan
    
    # By label
    normal_mask = labels == 0
    abnormal_mask = labels == 1
    
    try:
        normal_pearson, normal_pearson_p = pearsonr(semantic_scores[normal_mask], fusion_scores[normal_mask])
    except:
        normal_pearson, normal_pearson_p = np.nan, np.nan
    
    try:
        abnormal_pearson, abnormal_pearson_p = pearsonr(semantic_scores[abnormal_mask], fusion_scores[abnormal_mask])
    except:
        abnormal_pearson, abnormal_pearson_p = np.nan, np.nan
    
    # 相对贡献：semantic与visual的差异
    semantic_visual_diff = np.abs(semantic_scores - visual_scores).mean()
    
    # 分支差异统计
    diff_normal = np.abs(semantic_scores[normal_mask] - visual_scores[normal_mask]).mean() if normal_mask.sum() > 0 else np.nan
    diff_abnormal = np.abs(semantic_scores[abnormal_mask] - visual_scores[abnormal_mask]).mean() if abnormal_mask.sum() > 0 else np.nan
    
    return {
        'overall_pearson': overall_pearson,
        'overall_pearson_p': overall_pearson_p,
        'overall_spearman': overall_spearman,
        'overall_spearman_p': overall_spearman_p,
        'normal_pearson': normal_pearson,
        'normal_pearson_p': normal_pearson_p,
        'abnormal_pearson': abnormal_pearson,
        'abnormal_pearson_p': abnormal_pearson_p,
        'semantic_visual_diff_mean': semantic_visual_diff,
        'semantic_visual_diff_normal': diff_normal,
        'semantic_visual_diff_abnormal': diff_abnormal,
    }


def main(args):
    kwargs = vars(args)
    
    if kwargs['seed'] is None:
        kwargs['seed'] = 222
    
    setup_seed(kwargs['seed'])
    
    device = f"cuda:0" if kwargs['use_cpu'] == 0 else "cpu"
    kwargs['device'] = device
    
    # Get directories
    img_dir, csv_path, check_path = get_dir_from_args('CLS', **kwargs)
    
    # Force num_workers=0
    kwargs['num_workers'] = 0
    
    # Get dataloaders
    test_dataloader, _ = get_dataloader_from_args(phase='test', perturbed=False, **kwargs)
    train_dataloader, _ = get_dataloader_from_args(phase='train', perturbed=False, **kwargs)
    
    kwargs['out_size_h'] = kwargs['resolution']
    kwargs['out_size_w'] = kwargs['resolution']
    
    # Load model
    model = PromptAD(**kwargs)
    model = model.to(device)
    
    print(f"="*80)
    print(f"扩展评估: {kwargs['dataset']} - {kwargs['class_name']} (k={kwargs['k_shot']})")
    print(f"="*80)
    
    # Evaluate
    scores_list, labels, names = evaluate_with_extended_metrics(
        model, args, test_dataloader, train_dataloader, device, check_path
    )
    
    print(f"\n✅ 收集了 {len(scores_list)} 个样本的详细分数")
    print(f"   - Normal: {(labels==0).sum()}, Abnormal: {(labels==1).sum()}")
    
    # 【任务1】拆分 AUROC
    print(f"\n" + "="*80)
    print(f"【任务1】拆分 AUROC")
    print(f"="*80)
    split_auroc = calculate_split_auroc(scores_list, labels)
    
    print(f"\n整体 AUROC:")
    print(f"  Semantic: {split_auroc['overall_semantic_auroc']:.4f}")
    print(f"  Fusion:   {split_auroc['overall_fusion_auroc']:.4f}")
    
    print(f"\nNormal-only AUROC (区分正常样本内部差异):")
    print(f"  Semantic: {split_auroc['normal_semantic_auroc']:.4f} (n={split_auroc['n_normal']})")
    print(f"  Fusion:   {split_auroc['normal_fusion_auroc']:.4f}")
    
    print(f"\nAbnormal-only AUROC (区分异常样本严重程度):")
    print(f"  Semantic: {split_auroc['abnormal_semantic_auroc']:.4f} (n={split_auroc['n_abnormal']})")
    print(f"  Fusion:   {split_auroc['abnormal_fusion_auroc']:.4f}")
    
    # 【任务2】Margin 分布
    print(f"\n" + "="*80)
    print(f"【任务2】样本级 Margin 分布")
    print(f"="*80)
    margin_stats = analyze_margin_distribution(scores_list, labels)
    
    print(f"\nNormal 样本 Margin:")
    print(f"  均值: {margin_stats['normal']['mean']:.4f} ± {margin_stats['normal']['std']:.4f}")
    print(f"  分位数: P10={margin_stats['normal']['p10']:.4f}, P50={margin_stats['normal']['median']:.4f}, P90={margin_stats['normal']['p90']:.4f}")
    
    print(f"\nAbnormal 样本 Margin:")
    print(f"  均值: {margin_stats['abnormal']['mean']:.4f} ± {margin_stats['abnormal']['std']:.4f}")
    print(f"  分位数: P10={margin_stats['abnormal']['p10']:.4f}, P50={margin_stats['abnormal']['median']:.4f}, P90={margin_stats['abnormal']['p90']:.4f}")
    
    print(f"\nMargin Separation (Normal - Abnormal):")
    print(f"  {margin_stats['separation']:.4f}")
    
    # 【任务3】Semantic 贡献
    print(f"\n" + "="*80)
    print(f"【任务3】Semantic 分支贡献分析")
    print(f"="*80)
    semantic_contrib = analyze_semantic_contribution(scores_list, labels)
    
    print(f"\nSemantic 与 Fusion 相关性:")
    print(f"  Overall: Pearson r={semantic_contrib['overall_pearson']:.4f}, p={semantic_contrib['overall_pearson_p']:.4f}")
    print(f"  Normal:  Pearson r={semantic_contrib['normal_pearson']:.4f}, p={semantic_contrib['normal_pearson_p']:.4f}")
    print(f"  Abnormal: Pearson r={semantic_contrib['abnormal_pearson']:.4f}, p={semantic_contrib['abnormal_pearson_p']:.4f}")
    
    print(f"\nSemantic-Visual 分支差异:")
    print(f"  Overall: {semantic_contrib['semantic_visual_diff_mean']:.4f}")
    print(f"  Normal:  {semantic_contrib['semantic_visual_diff_normal']:.4f}")
    print(f"  Abnormal: {semantic_contrib['semantic_visual_diff_abnormal']:.4f}")
    
    # 保存结果
    output_dir = Path('analysis/extended_metrics')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 添加version_tag到文件名以区分不同版本
    version_tag = kwargs.get('version_tag', '')
    if version_tag:
        class_id = f"{kwargs['dataset']}_{kwargs['class_name']}_{version_tag}_k{kwargs['k_shot']}"
    else:
        class_id = f"{kwargs['dataset']}_{kwargs['class_name']}_k{kwargs['k_shot']}"
    
    # 1. 拆分AUROC
    split_auroc_df = pd.DataFrame([split_auroc])
    split_auroc_df['class'] = class_id
    split_auroc_path = output_dir / f'{class_id}_split_auroc.csv'
    split_auroc_df.to_csv(split_auroc_path, index=False)
    
    # 2. Margin统计
    margin_df = pd.DataFrame({
        'class': [class_id] * 3,
        'group': ['all', 'normal', 'abnormal'],
        'mean': [margin_stats['all']['mean'], margin_stats['normal']['mean'], margin_stats['abnormal']['mean']],
        'std': [margin_stats['all']['std'], margin_stats['normal']['std'], margin_stats['abnormal']['std']],
        'p10': [margin_stats['all']['p10'], margin_stats['normal']['p10'], margin_stats['abnormal']['p10']],
        'median': [margin_stats['all']['median'], margin_stats['normal']['median'], margin_stats['abnormal']['median']],
        'p90': [margin_stats['all']['p90'], margin_stats['normal']['p90'], margin_stats['abnormal']['p90']],
        'n': [margin_stats['all']['n'], margin_stats['normal']['n'], margin_stats['abnormal']['n']],
    })
    margin_path = output_dir / f'{class_id}_margin_stats.csv'
    margin_df.to_csv(margin_path, index=False)
    
    # 3. Semantic贡献
    semantic_contrib_df = pd.DataFrame([semantic_contrib])
    semantic_contrib_df['class'] = class_id
    semantic_path = output_dir / f'{class_id}_semantic_contrib.csv'
    semantic_contrib_df.to_csv(semantic_path, index=False)
    
    # 4. 样本级详细数据（可选，用于后续深入分析）
    samples_df = pd.DataFrame([
        {
            'class': class_id,
            'name': s['name'],
            'label': s['label'],
            'max_normal': s['max_normal'],
            'max_abnormal': s['max_abnormal'],
            'margin': s['margin'],
            'semantic_score': s['semantic_score'],
            'fusion_score': s['fusion_score'],
        }
        for s in scores_list
    ])
    samples_path = output_dir / f'{class_id}_sample_scores.csv'
    samples_df.to_csv(samples_path, index=False)
    
    print(f"\n" + "="*80)
    print(f"✅ 扩展指标已保存到:")
    print(f"   - {split_auroc_path}")
    print(f"   - {margin_path}")
    print(f"   - {semantic_path}")
    print(f"   - {samples_path}")
    print(f"="*80)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_args():
    parser = argparse.ArgumentParser(description='Extended Evaluation Metrics')
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa'])
    parser.add_argument('--class_name', type=str, default='carpet')
    
    parser.add_argument('--img-resize', type=int, default=240)
    parser.add_argument('--img-cropsize', type=int, default=240)
    parser.add_argument('--resolution', type=int, default=400)
    
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument("--root-dir", type=str, default="./result")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--gpu-id", type=int, default=0)
    
    # Method parameters
    parser.add_argument('--k-shot', type=int, default=2)
    parser.add_argument("--backbone", type=str, default="ViT-B-16-plus-240")
    parser.add_argument("--pretrained_dataset", type=str, default="laion400m_e32")
    parser.add_argument("--version-tag", type=str, default='', help='版本标识（如v1, v2），添加到输出文件名')
    
    # Prompt tuning parameters
    parser.add_argument("--n_ctx", type=int, default=4)
    parser.add_argument("--n_ctx_ab", type=int, default=1)
    parser.add_argument("--n_pro", type=int, default=1)
    parser.add_argument("--n_pro_ab", type=int, default=4)
    
    parser.add_argument("--use-cpu", type=int, default=0)
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import os
    
    args = get_args()
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu_id}"
    main(args)
