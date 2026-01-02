"""
对比实验：Average vs MaxPooling 聚合方式

目标：
1. 使用相同的训练checkpoint
2. 分别用average和maxpooling评估
3. 对比AUROC差异
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from pathlib import Path

from PromptAD import PromptAD
from datasets import get_dataloader_from_args
from torchvision import transforms
from PIL import Image

def get_transform(img_size=240):
    mean_train = [0.48145466, 0.4578275, 0.40821073]
    std_train = [0.26862954, 0.26130258, 0.27577711]
    
    def _convert_to_rgb(image):
        return image.convert('RGB')
    
    return transforms.Compose([
        transforms.Resize((img_size, img_size), Image.BICUBIC),
        transforms.CenterCrop(img_size),
        _convert_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train, std=std_train)
    ])

def evaluate_with_aggregation(model, test_loader, device, aggregation='average'):
    """使用指定聚合方式评估模型"""
    model.eval()
    scores = []
    labels = []
    
    with torch.no_grad():
        for (data, mask, label, name, img_type) in test_loader:
            data = data.to(device)
            visual_features = model.encode_image(data)
            
            # 🎯 使用指定的聚合方式
            score = model.calculate_textual_anomaly_score(
                visual_features, 
                'cls', 
                aggregation=aggregation
            )
            
            scores.append(score)
            labels.extend(label.numpy())
    
    scores = np.concatenate(scores)
    labels = np.array(labels)
    
    # 计算AUROC
    auroc = roc_auc_score(labels, scores) * 100
    
    return auroc, scores, labels

def compare_aggregations(checkpoint_path, dataset, class_name, k_shot=2, img_size=240):
    """对比average和maxpooling"""
    
    print(f"\n{'='*80}")
    print(f"对比实验: {dataset}/{class_name} K={k_shot}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*80}\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载测试数据
    transform = get_transform(img_size)
    test_loader, _ = get_dataloader_from_args(
        phase='test',
        dataset=dataset,
        class_name=class_name,
        img_size=img_size,
        k_shot=0,
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
        n_ctx_ab=4, n_pro_ab=10,
        class_name=class_name,
        k_shot=k_shot,
        img_resize=img_size,
        img_cropsize=img_size
    ).to(device)
    
    # 加载checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        print(f"✅ Checkpoint loaded")
        
        # 检查是否有所有文本特征
        has_all_features = ('normal_text_features_all' in checkpoint and 
                           'abnormal_text_features_all' in checkpoint)
        
        if has_all_features:
            print(f"   Normal features: {checkpoint['normal_text_features_all'].shape}")
            print(f"   Abnormal features: {checkpoint['abnormal_text_features_all'].shape}")
        else:
            print(f"   ⚠️ Checkpoint只有平均锚点，需要重新训练以获得所有向量")
            return None
    else:
        print(f"❌ Checkpoint不存在: {checkpoint_path}")
        return None
    
    # 评估：Average
    print(f"\n📊 [方法1] Average聚合...")
    auroc_avg, scores_avg, labels = evaluate_with_aggregation(
        model, test_loader, device, aggregation='average'
    )
    print(f"   AUROC: {auroc_avg:.2f}")
    
    # 评估：MaxPooling
    print(f"\n📊 [方法2] MaxPooling聚合...")
    auroc_max, scores_max, _ = evaluate_with_aggregation(
        model, test_loader, device, aggregation='maxpooling'
    )
    print(f"   AUROC: {auroc_max:.2f}")
    
    # 对比结果
    print(f"\n{'='*80}")
    print(f"对比结果")
    print(f"{'='*80}")
    print(f"Average聚合:    {auroc_avg:.2f}")
    print(f"MaxPooling聚合: {auroc_max:.2f}")
    print(f"差异:          {auroc_max - auroc_avg:+.2f}")
    
    if auroc_max > auroc_avg:
        print(f"✅ MaxPooling表现更好 (+{auroc_max - auroc_avg:.2f})")
    elif auroc_max < auroc_avg:
        print(f"⚠️  Average表现更好 ({auroc_max - auroc_avg:.2f})")
    else:
        print(f"➡️  两者表现相同")
    
    print(f"\n分数分布对比:")
    print(f"  Average - 正常样本: {scores_avg[labels==0].mean():.4f} ± {scores_avg[labels==0].std():.4f}")
    print(f"           异常样本: {scores_avg[labels==1].mean():.4f} ± {scores_avg[labels==1].std():.4f}")
    print(f"  MaxPool - 正常样本: {scores_max[labels==0].mean():.4f} ± {scores_max[labels==0].std():.4f}")
    print(f"           异常样本: {scores_max[labels==1].mean():.4f} ± {scores_max[labels==1].std():.4f}")
    
    return {
        'dataset': dataset,
        'class': class_name,
        'k_shot': k_shot,
        'auroc_average': auroc_avg,
        'auroc_maxpooling': auroc_max,
        'diff': auroc_max - auroc_avg
    }

def batch_compare(result_dir='promptpurging', dataset='mvtec', classes=None, k_shots=[1, 2, 4]):
    """批量对比多个类别和K-shot"""
    
    if classes is None:
        if dataset == 'mvtec':
            classes = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 
                      'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                      'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
        else:
            classes = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
                      'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    
    results = []
    
    for k_shot in k_shots:
        for class_name in classes:
            checkpoint_path = f"./result/{result_dir}/{dataset}/k_{k_shot}/checkpoint/CLS-Seed_111-{class_name}-check_point.pt"
            
            if not os.path.exists(checkpoint_path):
                print(f"\n⚠️ Checkpoint不存在，跳过: {class_name} K={k_shot}")
                continue
            
            result = compare_aggregations(checkpoint_path, dataset, class_name, k_shot)
            
            if result is not None:
                results.append(result)
    
    if results:
        # 汇总结果
        df = pd.DataFrame(results)
        
        print(f"\n{'='*80}")
        print(f"汇总结果 ({dataset})")
        print(f"{'='*80}")
        print(df.to_string(index=False))
        
        # 统计
        print(f"\n统计:")
        print(f"  平均差异: {df['diff'].mean():+.2f}")
        print(f"  MaxPooling更好: {(df['diff'] > 0).sum()} / {len(df)}")
        print(f"  Average更好: {(df['diff'] < 0).sum()} / {len(df)}")
        
        # 保存结果
        output_dir = Path(f"./analysis/aggregation_comparison/{dataset}")
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"{result_dir}_comparison.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n结果已保存到: {csv_path}")
        
        return df
    
    return None

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser("聚合方式对比实验")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="单个checkpoint路径")
    parser.add_argument("--result_dir", type=str, default="promptpurging",
                       help="结果目录 (baseline/promptpurging)")
    parser.add_argument("--dataset", type=str, default="mvtec")
    parser.add_argument("--class_name", type=str, default="metal_nut")
    parser.add_argument("--k_shot", type=int, default=2)
    parser.add_argument("--batch", action="store_true",
                       help="批量对比所有类别")
    
    args = parser.parse_args()
    
    if args.batch:
        # 批量对比
        batch_compare(
            result_dir=args.result_dir,
            dataset=args.dataset,
            k_shots=[1, 2, 4]
        )
    else:
        # 单个对比
        if args.checkpoint is None:
            args.checkpoint = f"./result/{args.result_dir}/{args.dataset}/k_{args.k_shot}/checkpoint/CLS-Seed_111-{args.class_name}-check_point.pt"
        
        compare_aggregations(
            checkpoint_path=args.checkpoint,
            dataset=args.dataset,
            class_name=args.class_name,
            k_shot=args.k_shot
        )
