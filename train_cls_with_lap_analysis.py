"""
训练时监控LAP向量的多样性
基于train_cls.py，添加LAP相似度分析
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 导入原始训练代码
from train_cls import *

def analyze_lap_diversity(model, epoch, save_dir):
    """在训练过程中分析LAP向量的多样性"""
    # 获取LAP向量
    lap_vectors = model.prompt_learner.abnormal_ctx.data  # [n_pro_ab, n_ctx_ab, dim]
    
    n_pro_ab, n_ctx_ab, dim = lap_vectors.shape
    
    # 计算每个LAP的平均embedding
    lap_avg = lap_vectors.mean(dim=1)  # [n_pro_ab, dim]
    lap_avg_norm = lap_avg / lap_avg.norm(dim=1, keepdim=True)
    
    # 计算相似度矩阵
    similarity_matrix = lap_avg_norm @ lap_avg_norm.T
    similarity_matrix = similarity_matrix.cpu().numpy()
    
    # 提取上三角
    triu_indices = np.triu_indices(n_pro_ab, k=1)
    pairwise_sims = similarity_matrix[triu_indices]
    
    stats = {
        'epoch': epoch,
        'mean_sim': pairwise_sims.mean(),
        'std_sim': pairwise_sims.std(),
        'max_sim': pairwise_sims.max(),
        'min_sim': pairwise_sims.min(),
        'high_sim_ratio': (pairwise_sims > 0.9).sum() / len(pairwise_sims),
        'very_high_sim_ratio': (pairwise_sims > 0.95).sum() / len(pairwise_sims),
    }
    
    # 打印统计
    print(f"\n[Epoch {epoch}] LAP多样性分析:")
    print(f"  平均相似度: {stats['mean_sim']:.4f} ± {stats['std_sim']:.4f}")
    print(f"  范围: [{stats['min_sim']:.4f}, {stats['max_sim']:.4f}]")
    print(f"  高相似度对 (>0.9): {stats['high_sim_ratio']*100:.1f}%")
    print(f"  极高相似度对 (>0.95): {stats['very_high_sim_ratio']*100:.1f}%")
    
    # 可视化（每10个epoch保存一次）
    if epoch % 10 == 0 or epoch == 1:
        plt.figure(figsize=(8, 6))
        sns.heatmap(similarity_matrix, 
                    annot=True, 
                    fmt='.2f', 
                    cmap='RdYlGn_r',
                    vmin=0.0, 
                    vmax=1.0,
                    square=True,
                    cbar_kws={'label': 'Cosine Similarity'})
        plt.title(f'LAP Similarity Matrix (Epoch {epoch})')
        plt.xlabel('LAP Index')
        plt.ylabel('LAP Index')
        plt.tight_layout()
        
        save_path = save_dir / f"lap_similarity_epoch{epoch:03d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return stats

def fit_with_analysis(model, args, test_loader, device, check_path, train_loader, class_name, dataset_name, output_dir):
    """修改后的训练函数，添加LAP分析"""
    
    # 创建分析目录
    analysis_dir = Path(output_dir) / dataset_name / class_name / f"k_{args.k_shot}"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 输出目录: {analysis_dir.absolute()}\n")
    
    # 收集统计数据
    lap_stats_history = []
    
    # 构建特征库（与原始代码相同）
    model.eval()
    features1 = []
    features2 = []
    for (data, mask, label, name, img_type) in train_loader:
        data = data.to(device)
        _, _, feature1, feature2 = model.encode_image(data)
        features1.append(feature1.detach().cpu())
        features2.append(feature2.detach().cpu())
    features1 = torch.cat(features1, dim=0)
    features2 = torch.cat(features2, dim=0)
    model.build_image_feature_gallery(features1, features2)
    
    optimizer = torch.optim.SGD(model.prompt_learner.parameters(), lr=args.lr, 
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.Epoch, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss().to(device)
    criterion_tip = TripletLoss(margin=0.0)
    
    best_result_dict = None
    
    for epoch in range(args.Epoch):
        # 训练一个epoch（与原始代码相同）
        model.train()
        for (data, mask, label, name, img_type) in train_loader:
            data = data.to(device)
            
            normal_text_prompt, abnormal_text_prompt_handle, abnormal_text_prompt_learned = model.prompt_learner()
            
            optimizer.zero_grad()
            
            normal_text_features = model.encode_text_embedding(normal_text_prompt, model.tokenized_normal_prompts)
            abnormal_text_features_handle = model.encode_text_embedding(abnormal_text_prompt_handle, 
                                                                        model.tokenized_abnormal_prompts_handle)
            abnormal_text_features_learned = model.encode_text_embedding(abnormal_text_prompt_learned, 
                                                                         model.tokenized_abnormal_prompts_learned)
            abnormal_text_features = torch.cat([abnormal_text_features_handle, abnormal_text_features_learned], dim=0)
            
            mean_ad_handle = torch.mean(F.normalize(abnormal_text_features_handle, dim=-1), dim=0)
            mean_ad_learned = torch.mean(F.normalize(abnormal_text_features_learned, dim=-1), dim=0)
            loss_match_abnormal = (mean_ad_handle - mean_ad_learned).norm(dim=0) ** 2.0
            
            cls_feature, _, _, _ = model.encode_image(data)
            
            normal_text_features_ahchor = normal_text_features.mean(dim=0).unsqueeze(0)
            normal_text_features_ahchor = normal_text_features_ahchor / normal_text_features_ahchor.norm(dim=-1, keepdim=True)
            
            abnormal_text_features_ahchor = abnormal_text_features.mean(dim=0).unsqueeze(0)
            abnormal_text_features_ahchor = abnormal_text_features_ahchor / abnormal_text_features_ahchor.norm(dim=-1, keepdim=True)
            abnormal_text_features = abnormal_text_features / abnormal_text_features.norm(dim=-1, keepdim=True)
            
            l_pos = torch.einsum('nc,cm->nm', cls_feature, normal_text_features_ahchor.transpose(0, 1))
            l_neg_v2t = torch.einsum('nc,cm->nm', cls_feature, abnormal_text_features.transpose(0, 1))
            
            if model.precision == 'fp16':
                logit_scale = model.model.logit_scale.half()
            else:
                logit_scale = model.model.logit_scale
            
            logits_v2t = torch.cat([l_pos, l_neg_v2t], dim=-1) * logit_scale
            target_v2t = torch.zeros([logits_v2t.shape[0]], dtype=torch.long).to(device)
            
            loss_v2t = criterion(logits_v2t, target_v2t)
            trip_loss = criterion_tip(cls_feature, normal_text_features_ahchor, abnormal_text_features_ahchor)
            loss = loss_v2t + trip_loss + loss_match_abnormal * args.lambda1
            
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        model.build_text_feature_gallery()
        
        # 🔥 添加LAP分析
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == args.Epoch - 1:
            stats = analyze_lap_diversity(model, epoch + 1, analysis_dir)
            lap_stats_history.append(stats)
        
        # 评估（简化版，只在关键epoch）
        if (epoch + 1) % 10 == 0 or epoch == args.Epoch - 1:
            model.eval()
            scores_semantic = []
            for (data, mask, label, name, img_type) in test_loader:
                data = data.to(device)
                visual_features = model.encode_image(data)
                score_semantic = model.calculate_textual_anomaly_score(visual_features, 'cls')
                scores_semantic.append(score_semantic)
            
            scores_semantic = np.concatenate(scores_semantic)
            # 修复标签构建
            num_pos = len([l for (_, _, l, _, _) in test_loader.dataset if l == 0])
            num_neg = len([l for (_, _, l, _, _) in test_loader.dataset if l == 1])
            gt_list = [0] * num_pos + [1] * num_neg
            roc = round(roc_auc_score(gt_list, scores_semantic) * 100, 2)
            print(f"  Semantic i_roc: {roc:.2f}")
    
    # 保存统计历史
    import pandas as pd
    df = pd.DataFrame(lap_stats_history)
    csv_path = analysis_dir / "lap_diversity_history.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nLAP多样性历史已保存到: {csv_path}")
    
    # 绘制趋势图
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(df['epoch'], df['mean_sim'], marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Similarity')
    plt.title('LAP平均相似度')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(df['epoch'], df['max_sim'], marker='o', label='Max')
    plt.plot(df['epoch'], df['min_sim'], marker='o', label='Min')
    plt.xlabel('Epoch')
    plt.ylabel('Similarity')
    plt.title('LAP相似度范围')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(df['epoch'], df['high_sim_ratio'] * 100, marker='o', label='>0.9')
    plt.plot(df['epoch'], df['very_high_sim_ratio'] * 100, marker='o', label='>0.95')
    plt.xlabel('Epoch')
    plt.ylabel('Percentage (%)')
    plt.title('高相似度对比例')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    trend_path = analysis_dir / "lap_diversity_trend.png"
    plt.savefig(trend_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"趋势图已保存到: {trend_path}")
    
    # 保存模型
    save_model(model, check_path, ['feature_gallery1', 'feature_gallery2', 'text_features'])


if __name__ == '__main__':
    import argparse
    from datasets import get_dataloader_from_args
    
    parser = argparse.ArgumentParser("PromptAD LAP Analysis")
    parser.add_argument("--dataset", type=str, default="mvtec")
    parser.add_argument("--class_name", type=str, default="metal_nut")
    parser.add_argument("--k_shot", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="./analysis/lap_diversity_training",
                       help="输出目录，存放LAP分析结果")
    parser.add_argument("--Epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda1", type=float, default=1.0)
    parser.add_argument("--img_resize", type=int, default=240)
    parser.add_argument("--img_cropsize", type=int, default=240)
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"LAP多样性分析训练")
    print(f"  Dataset: {args.dataset}")
    print(f"  Class: {args.class_name}")
    print(f"  K-shot: {args.k_shot}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*80}\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 准备transform
    from torchvision import transforms
    from PIL import Image
    
    mean_train = [0.48145466, 0.4578275, 0.40821073]
    std_train = [0.26862954, 0.26130258, 0.27577711]
    
    def _convert_to_rgb(image):
        return image.convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((args.img_resize, args.img_resize), Image.BICUBIC),
        transforms.CenterCrop(args.img_cropsize),
        _convert_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train, std=std_train)
    ])
    
    # 加载数据（使用原始的get_dataloader_from_args）
    train_loader, train_dataset = get_dataloader_from_args(
        phase='train',
        dataset=args.dataset,
        class_name=args.class_name,
        img_size=args.img_cropsize,
        k_shot=args.k_shot,
        batch_size=args.k_shot,
        transform=transform
    )
    
    test_loader, test_dataset = get_dataloader_from_args(
        phase='test',
        dataset=args.dataset,
        class_name=args.class_name,
        img_size=args.img_cropsize,
        k_shot=0,  # test phase不需要k_shot
        batch_size=1,
        transform=transform
    )
    
    # 创建模型
    model = PromptAD(
        out_size_h=60, out_size_w=60,
        device=device,
        backbone='ViT-B-16-plus-240',
        pretrained_dataset='laion400m_e32',
        n_ctx=16, n_pro=1,
        n_ctx_ab=4, n_pro_ab=10,  # 10个LAP原型
        class_name=args.class_name,
        k_shot=args.k_shot,
        img_resize=args.img_resize,
        img_cropsize=args.img_cropsize
    ).to(device)
    
    # 训练并分析
    check_path = f"{args.output_dir}/{args.dataset}/{args.class_name}/k_{args.k_shot}/checkpoint.pt"
    fit_with_analysis(model, args, test_loader, device, check_path, 
                     train_loader, args.class_name, args.dataset, args.output_dir)
