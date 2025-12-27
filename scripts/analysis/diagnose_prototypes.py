#!/usr/bin/env python3
"""
多原型退化诊断工具
分析Prompt2在强类上退化、难类上改进的机制
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets import *
from PromptAD import *
from utils.training_utils import *

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class PrototypeDiagnostics:
    """多原型诊断工具类"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
        # 提取原型
        self.model.eval()
        self.model.build_text_feature_gallery()
        
        # 正常原型 [n_pro, dim]
        self.normal_prototypes = model.normal_prototypes.cpu().numpy()
        # 异常原型 [n_ab, dim]
        self.abnormal_prototypes = model.abnormal_prototypes.cpu().numpy()
        
        self.n_normal = self.normal_prototypes.shape[0]
        self.n_abnormal = self.abnormal_prototypes.shape[0]
        
        print(f"提取原型: {self.n_normal}个正常, {self.n_abnormal}个异常")
    
    def extract_features(self, dataloader):
        """提取所有样本的特征和标签"""
        from PIL import Image
        features = []
        labels = []
        
        with torch.no_grad():
            for data, _, label, _, _ in tqdm(dataloader, desc="提取特征"):
                # 应用transform (data是numpy uint8格式)
                data = [self.model.transform(Image.fromarray(f.numpy())) for f in data]
                data = torch.stack(data, dim=0).to(self.device)
                
                # 只取cls token特征
                cls_feature, _, _, _ = self.model.encode_image(data)
                features.append(cls_feature.cpu().numpy())
                labels.append(label.numpy())
        
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        
        # 归一化
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
        
        return features, labels
    
    def compute_similarities(self, features):
        """
        计算与所有原型的相似度
        
        Returns:
            normal_sim: [N, n_normal]
            abnormal_sim: [N, n_abnormal]
            normal_max: [N]
            abnormal_max: [N]
        """
        # 计算相似度
        normal_sim = features @ self.normal_prototypes.T  # [N, n_normal]
        abnormal_sim = features @ self.abnormal_prototypes.T  # [N, n_abnormal]
        
        # Max pooling
        normal_max = normal_sim.max(axis=1)
        abnormal_max = abnormal_sim.max(axis=1)
        
        return normal_sim, abnormal_sim, normal_max, abnormal_max
    
    def metric_A_max_hit_rate(self, features, labels, output_dir, class_name):
        """
        A. Normal→Abnormal Max Hit Rate
        统计正常样本与异常原型的最大相似度分布
        """
        normal_mask = (labels == 0)
        abnormal_mask = (labels == 1)
        
        _, abnormal_sim, _, abnormal_max = self.compute_similarities(features)
        
        # 只关注正常样本
        normal_features = features[normal_mask]
        normal_ab_max = abnormal_max[normal_mask]
        
        # 统计量
        stats = {
            'mean': float(normal_ab_max.mean()),
            'std': float(normal_ab_max.std()),
            'p50': float(np.percentile(normal_ab_max, 50)),
            'p90': float(np.percentile(normal_ab_max, 90)),
            'p95': float(np.percentile(normal_ab_max, 95)),
            'p99': float(np.percentile(normal_ab_max, 99)),
            'max': float(normal_ab_max.max())
        }
        
        # 不同阈值的命中率
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        hit_rates = {f'hit_rate_{tau}': float((normal_ab_max > tau).mean()) 
                     for tau in thresholds}
        stats.update(hit_rates)
        
        # 保存统计
        with open(f'{output_dir}/{class_name}_metricA_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        # 可视化：直方图
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 左图：正常样本的abnormal_max分布
        axes[0].hist(normal_ab_max, bins=50, alpha=0.7, edgecolor='black')
        axes[0].axvline(stats['mean'], color='r', linestyle='--', label=f"Mean: {stats['mean']:.3f}")
        axes[0].axvline(stats['p95'], color='orange', linestyle='--', label=f"P95: {stats['p95']:.3f}")
        axes[0].set_xlabel('Max Abnormal Similarity')
        axes[0].set_ylabel('Count')
        axes[0].set_title(f'{class_name}: Normal Samples → Abnormal Max')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 右图：命中率条形图
        axes[1].bar(range(len(thresholds)), [hit_rates[f'hit_rate_{t}'] for t in thresholds])
        axes[1].set_xticks(range(len(thresholds)))
        axes[1].set_xticklabels([f'{t}' for t in thresholds])
        axes[1].set_xlabel('Threshold')
        axes[1].set_ylabel('Hit Rate')
        axes[1].set_title('Hit Rate at Different Thresholds')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{class_name}_metricA.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return stats
    
    def metric_B_margin_distribution(self, features, labels, output_dir, class_name):
        """
        B. Margin Distribution & Overlap
        统计判别裕度：margin = normal_sim - abnormal_max
        """
        normal_mask = (labels == 0)
        abnormal_mask = (labels == 1)
        
        normal_sim, abnormal_sim, normal_max, abnormal_max = self.compute_similarities(features)
        
        # 计算margin（假设单正常原型，直接用normal_max）
        margin = normal_max - abnormal_max
        
        margin_normal = margin[normal_mask]
        margin_abnormal = margin[abnormal_mask]
        
        # 统计量
        stats = {
            'normal_mean': float(margin_normal.mean()),
            'normal_std': float(margin_normal.std()),
            'abnormal_mean': float(margin_abnormal.mean()),
            'abnormal_std': float(margin_abnormal.std()),
            'separation': float(margin_normal.mean() - margin_abnormal.mean())
        }
        
        # 计算重叠度（简化版：重叠区间比例）
        min_normal = margin_normal.min()
        max_normal = margin_normal.max()
        min_abnormal = margin_abnormal.min()
        max_abnormal = margin_abnormal.max()
        
        overlap_start = max(min_normal, min_abnormal)
        overlap_end = min(max_normal, max_abnormal)
        overlap = max(0, overlap_end - overlap_start)
        total_range = max(max_normal, max_abnormal) - min(min_normal, min_abnormal)
        stats['overlap_ratio'] = float(overlap / total_range if total_range > 0 else 0)
        
        # KS距离（分布差异）
        from scipy.stats import ks_2samp
        ks_stat, ks_pval = ks_2samp(margin_normal, margin_abnormal)
        stats['ks_distance'] = float(ks_stat)
        stats['ks_pvalue'] = float(ks_pval)
        
        # 保存统计
        with open(f'{output_dir}/{class_name}_metricB_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        # 可视化：两类margin分布
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 左图：直方图对比
        axes[0].hist(margin_normal, bins=50, alpha=0.5, label='Normal', color='blue')
        axes[0].hist(margin_abnormal, bins=50, alpha=0.5, label='Abnormal', color='red')
        axes[0].axvline(stats['normal_mean'], color='blue', linestyle='--', 
                       label=f"Normal Mean: {stats['normal_mean']:.3f}")
        axes[0].axvline(stats['abnormal_mean'], color='red', linestyle='--',
                       label=f"Abnormal Mean: {stats['abnormal_mean']:.3f}")
        axes[0].set_xlabel('Margin (Normal_sim - Abnormal_max)')
        axes[0].set_ylabel('Count')
        axes[0].set_title(f'{class_name}: Margin Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 右图：箱线图
        axes[1].boxplot([margin_normal, margin_abnormal], labels=['Normal', 'Abnormal'])
        axes[1].set_ylabel('Margin')
        axes[1].set_title('Margin Comparison')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{class_name}_metricB.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return stats
    
    def metric_C_prototype_collapse(self, output_dir, class_name):
        """
        C. Prototype Collapse Score
        计算异常原型之间的相似度（塌缩程度）
        """
        # 计算异常原型两两相似度
        ab_sim_matrix = self.abnormal_prototypes @ self.abnormal_prototypes.T
        
        # 提取非对角线元素
        mask = ~np.eye(self.n_abnormal, dtype=bool)
        off_diag = ab_sim_matrix[mask]
        
        # 统计量
        stats = {
            'mean': float(off_diag.mean()),
            'std': float(off_diag.std()),
            'max': float(off_diag.max()),
            'p90': float(np.percentile(off_diag, 90)),
            'p95': float(np.percentile(off_diag, 95)),
            'top10_mean': float(np.sort(off_diag)[-10:].mean() if len(off_diag) >= 10 else off_diag.max())
        }
        
        # Collapse score定义
        stats['collapse_score'] = stats['top10_mean']
        
        # 保存统计
        with open(f'{output_dir}/{class_name}_metricC_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        # 可视化：相似度矩阵热力图
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 左图：热力图
        im = axes[0].imshow(ab_sim_matrix, cmap='RdYlBu_r', vmin=-0.2, vmax=1.0)
        axes[0].set_xlabel('Abnormal Prototype Index')
        axes[0].set_ylabel('Abnormal Prototype Index')
        axes[0].set_title(f'{class_name}: Abnormal Prototype Similarity')
        plt.colorbar(im, ax=axes[0])
        
        # 右图：非对角线分布
        axes[1].hist(off_diag, bins=50, alpha=0.7, edgecolor='black')
        axes[1].axvline(stats['mean'], color='r', linestyle='--', label=f"Mean: {stats['mean']:.3f}")
        axes[1].axvline(stats['p95'], color='orange', linestyle='--', label=f"P95: {stats['p95']:.3f}")
        axes[1].set_xlabel('Cosine Similarity')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Off-diagonal Similarity Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{class_name}_metricC.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return stats
    
    def metric_D_bad_prototype_responsibility(self, features, labels, output_dir, class_name):
        """
        D. Bad Prototype Responsibility
        统计每个异常原型对正常样本误报的责任
        """
        normal_mask = (labels == 0)
        
        _, abnormal_sim, _, _ = self.compute_similarities(features)
        
        # 正常样本
        normal_ab_sim = abnormal_sim[normal_mask]  # [N_normal, n_ab]
        
        # 每个正常样本的最大命中原型
        argmax_proto = normal_ab_sim.argmax(axis=1)  # [N_normal]
        max_scores = normal_ab_sim.max(axis=1)  # [N_normal]
        
        # 统计每个原型的责任
        proto_counts = np.bincount(argmax_proto, minlength=self.n_abnormal)
        proto_avg_scores = np.array([
            normal_ab_sim[argmax_proto == i, i].mean() if proto_counts[i] > 0 else 0
            for i in range(self.n_abnormal)
        ])
        
        # 高分误报（top-20%）
        threshold = np.percentile(max_scores, 80)
        high_score_mask = max_scores > threshold
        high_score_proto_counts = np.bincount(argmax_proto[high_score_mask], 
                                               minlength=self.n_abnormal)
        
        # 保存统计
        stats = {
            'proto_counts': proto_counts.tolist(),
            'proto_avg_scores': proto_avg_scores.tolist(),
            'high_score_proto_counts': high_score_proto_counts.tolist(),
            'high_score_threshold': float(threshold)
        }
        
        # 找出Top-5坏原型
        top5_indices = np.argsort(high_score_proto_counts)[-5:][::-1]
        stats['top5_bad_prototypes'] = [
            {
                'index': int(i),
                'count': int(proto_counts[i]),
                'high_score_count': int(high_score_proto_counts[i]),
                'avg_score': float(proto_avg_scores[i])
            }
            for i in top5_indices
        ]
        
        with open(f'{output_dir}/{class_name}_metricD_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        # 可视化
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 上图：总体责任
        x = np.arange(self.n_abnormal)
        axes[0].bar(x, proto_counts, alpha=0.7, label='All hits')
        axes[0].set_xlabel('Abnormal Prototype Index')
        axes[0].set_ylabel('Hit Count')
        axes[0].set_title(f'{class_name}: Prototype Hit Counts (Normal Samples)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 下图：高分误报责任
        axes[1].bar(x, high_score_proto_counts, alpha=0.7, color='red', 
                   label=f'High-score hits (>{threshold:.3f})')
        # 标注Top-5
        for i in top5_indices:
            axes[1].text(i, high_score_proto_counts[i], f'{i}', 
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        axes[1].set_xlabel('Abnormal Prototype Index')
        axes[1].set_ylabel('High-score Hit Count')
        axes[1].set_title('High-score Misclassification Responsibility')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{class_name}_metricD.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return stats
    
    def metric_E_fusion_sensitivity(self, features, labels, dataloader, output_dir, class_name):
        """
        E. Fusion Sensitivity (可选)
        分析融合对最终分数的影响
        """
        # 需要获取visual分支的分数
        # 这里简化处理，只分析semantic分支
        normal_mask = (labels == 0)
        abnormal_mask = (labels == 1)
        
        _, _, normal_max, abnormal_max = self.compute_similarities(features)
        
        # Semantic分数（简化：用softmax模拟）
        logits = np.stack([normal_max, abnormal_max], axis=1)
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        semantic_scores = exp_logits[:, 1] / exp_logits.sum(axis=1)
        
        # 统计
        stats = {
            'normal_semantic_mean': float(semantic_scores[normal_mask].mean()),
            'normal_semantic_std': float(semantic_scores[normal_mask].std()),
            'abnormal_semantic_mean': float(semantic_scores[abnormal_mask].mean()),
            'abnormal_semantic_std': float(semantic_scores[abnormal_mask].std())
        }
        
        with open(f'{output_dir}/{class_name}_metricE_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        # 可视化
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(semantic_scores[normal_mask], bins=50, alpha=0.5, label='Normal', color='blue')
        ax.hist(semantic_scores[abnormal_mask], bins=50, alpha=0.5, label='Abnormal', color='red')
        ax.set_xlabel('Semantic Anomaly Score')
        ax.set_ylabel('Count')
        ax.set_title(f'{class_name}: Semantic Score Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{class_name}_metricE.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return stats


def diagnose_class(args, class_name, dataset_name):
    """诊断单个类别"""
    
    print(f"\n{'='*80}")
    print(f"诊断类别: {dataset_name}-{class_name}")
    print(f"{'='*80}\n")
    
    # 设置设备
    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    
    # 加载数据 - 使用统一的CLIPDataset
    from datasets import load_function_dict
    test_dataset = CLIPDataset(
        load_function=load_function_dict[dataset_name],
        category=class_name,
        phase='test',
        k_shot=args.k_shot,
        transform=None
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)  # num_workers=0避免多进程问题
    
    # 加载模型
    model = PromptAD(
        dataset_name=dataset_name,
        class_name=class_name,
        k_shot=args.k_shot,
        device=device,
        n_ctx=4,
        n_ctx_ab=1,
        n_pro=args.n_pro,
        n_pro_ab=args.n_pro_ab,
        backbone='ViT-B-16-plus-240',
        pretrained_dataset='laion400m_e32',
        out_size_h=240,
        out_size_w=240,
        img_resize=240,
        img_cropsize=240
    )
    model = model.to(device)
    model.eval_mode()
    
    # 加载checkpoint - 直接加载到模型的prototypes
    checkpoint_path = f'{args.checkpoint_dir}/{dataset_name}/k_{args.k_shot}/checkpoint/CLS-Seed_{args.seed}-{class_name}-check_point.pt'
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  Checkpoint不存在: {checkpoint_path}")
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load prototypes (not prompt learner state_dict)
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
    
    print(f"✅ 加载checkpoint: {checkpoint_path}")
    print(f"   Normal prototypes: {model.normal_prototypes.shape}")
    print(f"   Abnormal prototypes: {model.abnormal_prototypes.shape}")
    
    # 创建输出目录
    output_dir = f'{args.output_dir}/{dataset_name}_k{args.k_shot}_{class_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化诊断工具
    diagnostics = PrototypeDiagnostics(model, device)
    
    # 提取特征
    print("提取测试集特征...")
    features, labels = diagnostics.extract_features(test_loader)
    print(f"特征形状: {features.shape}, 标签分布: Normal={(labels==0).sum()}, Abnormal={(labels==1).sum()}")
    
    # 运行所有指标
    results = {}
    
    print("\n[Metric A] 异常侧偶然命中分析...")
    results['metric_A'] = diagnostics.metric_A_max_hit_rate(features, labels, output_dir, class_name)
    
    print("[Metric B] 判别裕度分析...")
    results['metric_B'] = diagnostics.metric_B_margin_distribution(features, labels, output_dir, class_name)
    
    print("[Metric C] 原型塌缩分析...")
    results['metric_C'] = diagnostics.metric_C_prototype_collapse(output_dir, class_name)
    
    print("[Metric D] 坏原型归因分析...")
    results['metric_D'] = diagnostics.metric_D_bad_prototype_responsibility(features, labels, output_dir, class_name)
    
    print("[Metric E] 融合敏感性分析...")
    results['metric_E'] = diagnostics.metric_E_fusion_sensitivity(features, labels, test_loader, output_dir, class_name)
    
    # 保存综合结果
    with open(f'{output_dir}/summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ 诊断完成！结果保存在: {output_dir}/")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='多原型退化诊断工具')
    
    parser.add_argument('--k-shot', type=int, default=2, help='K-shot (默认2)')
    parser.add_argument('--n_pro', type=int, default=1, help='正常原型数量')
    parser.add_argument('--n_pro_ab', type=int, default=4, help='异常原型数量')
    parser.add_argument('--seed', type=int, default=111, help='随机种子')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID')
    parser.add_argument('--checkpoint-dir', type=str, default='result/prompt2',
                        help='Checkpoint目录')
    parser.add_argument('--output-dir', type=str, default='diagnostics',
                        help='输出目录')
    
    # 指定要诊断的类别
    parser.add_argument('--classes', type=str, nargs='+',
                        default=['mvtec-toothbrush', 'mvtec-capsule', 'mvtec-cable',
                                'visa-pcb2', 'visa-pipe_fryum', 'mvtec-screw'],
                        help='要诊断的类别列表 (格式: dataset-classname)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"多原型退化诊断")
    print(f"{'='*80}")
    print(f"K-shot: {args.k_shot}")
    print(f"配置: n_pro={args.n_pro}, n_pro_ab={args.n_pro_ab}")
    print(f"诊断类别: {len(args.classes)}个")
    print(f"输出目录: {args.output_dir}")
    print(f"{'='*80}\n")
    
    # 逐个诊断
    all_results = {}
    for class_spec in args.classes:
        dataset_name, class_name = class_spec.split('-', 1)
        try:
            results = diagnose_class(args, class_name, dataset_name)
            if results:
                all_results[class_spec] = results
        except Exception as e:
            print(f"❌ 诊断失败: {class_spec}")
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
    
    # 生成对比报告
    print(f"\n{'='*80}")
    print(f"生成对比报告")
    print(f"{'='*80}\n")
    
    # 创建对比表格
    comparison_data = []
    for class_spec, results in all_results.items():
        row = {
            'class': class_spec,
            'A_normal_ab_max_mean': results['metric_A']['mean'],
            'A_normal_ab_max_p95': results['metric_A']['p95'],
            'B_normal_margin_mean': results['metric_B']['normal_mean'],
            'B_margin_separation': results['metric_B']['separation'],
            'B_overlap_ratio': results['metric_B']['overlap_ratio'],
            'C_collapse_score': results['metric_C']['collapse_score'],
            'D_top1_bad_proto_count': results['metric_D']['top5_bad_prototypes'][0]['high_score_count']
        }
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    df.to_csv(f'{args.output_dir}/comparison_table.csv', index=False)
    print(f"对比表格已保存: {args.output_dir}/comparison_table.csv")
    print("\n对比结果:")
    print(df.to_string(index=False))
    
    print(f"\n✅ 全部诊断完成！")


if __name__ == '__main__':
    main()
