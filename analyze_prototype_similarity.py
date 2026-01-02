"""
分析正常原型与训练图像cls token的相似度

任务：
1. 加载baseline checkpoint，提取normal原型
2. 加载k=2的训练图像（2张正常样本）
3. 提取训练图像的cls token
4. 计算相似度
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
import numpy as np
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

def analyze_normal_prototype_similarity(
    checkpoint_path,
    dataset='mvtec',
    class_name='metal_nut',
    k_shot=2,
    img_size=240
):
    """
    分析正常原型与训练图像的相似度
    
    Args:
        checkpoint_path: baseline checkpoint路径
        dataset: 数据集名称
        class_name: 类别名称
        k_shot: few-shot数量
        img_size: 图像大小
    """
    
    print(f"\n{'='*80}")
    print(f"分析正常原型与训练图像cls token的相似度")
    print(f"{'='*80}")
    print(f"Dataset: {dataset}")
    print(f"Class: {class_name}")
    print(f"K-shot: {k_shot}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*80}\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1️⃣ 创建模型
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
    
    # 2️⃣ 加载checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint不存在: {checkpoint_path}")
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    print(f"✅ Checkpoint已加载\n")
    
    # 3️⃣ 提取正常原型
    # text_features[0]是normal原型，text_features[1]是abnormal原型
    normal_prototype = model.text_features[0]  # [640,]
    print(f"📊 正常原型:")
    print(f"   Shape: {normal_prototype.shape}")
    print(f"   Norm: {normal_prototype.norm():.4f}")
    print(f"   Device: {normal_prototype.device}")
    
    # 4️⃣ 加载训练数据
    transform = get_transform(img_size)
    train_loader, train_dataset = get_dataloader_from_args(
        phase='train',
        dataset=dataset,
        class_name=class_name,
        img_size=img_size,
        k_shot=k_shot,
        batch_size=k_shot,
        transform=transform
    )
    
    print(f"\n📊 训练数据:")
    print(f"   样本数: {len(train_dataset)}")
    print(f"   Batch数: {len(train_loader)}")
    
    # 5️⃣ 提取训练图像的cls token
    model.eval()
    train_cls_tokens = []
    train_img_names = []
    
    with torch.no_grad():
        for batch_idx, (data, mask, label, name, img_type) in enumerate(train_loader):
            data = data.to(device)
            
            # encode_image返回: [cls_token, patch_tokens, feature_map1, feature_map2]
            cls_token, _, _, _ = model.encode_image(data)  # [batch_size, 640]
            
            train_cls_tokens.append(cls_token)
            train_img_names.extend(name)
            
            print(f"\n   Batch {batch_idx + 1}:")
            print(f"   - Images: {name}")
            print(f"   - CLS token shape: {cls_token.shape}")
            print(f"   - CLS token norm: {[f'{tok.norm():.4f}' for tok in cls_token]}")
    
    train_cls_tokens = torch.cat(train_cls_tokens, dim=0)  # [k_shot, 640]
    print(f"\n   总CLS tokens: {train_cls_tokens.shape}")
    
    # 6️⃣ 计算相似度
    print(f"\n{'='*80}")
    print(f"相似度分析")
    print(f"{'='*80}\n")
    
    # 余弦相似度 = dot product (因为都已经归一化)
    similarities = []
    
    for i, (cls_token, img_name) in enumerate(zip(train_cls_tokens, train_img_names)):
        # 计算余弦相似度
        similarity = (cls_token @ normal_prototype).item()
        similarities.append(similarity)
        
        print(f"训练图像 {i+1}: {img_name}")
        print(f"  CLS token norm: {cls_token.norm():.4f}")
        print(f"  与正常原型的相似度: {similarity:.6f}")
        print()
    
    # 7️⃣ 统计
    similarities = np.array(similarities)
    
    print(f"{'='*80}")
    print(f"统计结果")
    print(f"{'='*80}")
    print(f"平均相似度: {similarities.mean():.6f}")
    print(f"标准差:     {similarities.std():.6f}")
    print(f"最小值:     {similarities.min():.6f}")
    print(f"最大值:     {similarities.max():.6f}")
    
    # 8️⃣ 解释
    print(f"\n{'='*80}")
    print(f"解释")
    print(f"{'='*80}")
    print(f"""
余弦相似度范围: [-1, 1]
- 1.0:  完全相同
- 0.0:  正交（无关）
- -1.0: 完全相反

观察:
- 如果相似度 > 0.9: 训练图像与正常原型高度一致
- 如果相似度 < 0.5: 训练图像与正常原型差异较大
- 相似度越高，说明正常原型越好地代表了训练样本

训练目标:
- loss_v2t: 拉近训练图像cls token与normal原型
- triplet loss: 推远abnormal原型
- 期望: 高相似度 (>0.8)
    """)
    
    return {
        'class_name': class_name,
        'k_shot': k_shot,
        'normal_prototype_norm': normal_prototype.norm().item(),
        'train_cls_norms': [tok.norm().item() for tok in train_cls_tokens],
        'similarities': similarities.tolist(),
        'mean_similarity': similarities.mean(),
        'std_similarity': similarities.std(),
        'min_similarity': similarities.min(),
        'max_similarity': similarities.max(),
    }

def batch_analyze(result_dir='baseline', dataset='mvtec', classes=None, k_shot=2):
    """批量分析多个类别"""
    
    if classes is None:
        if dataset == 'mvtec':
            classes = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 
                      'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                      'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
        else:
            classes = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
                      'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    
    results = []
    
    for class_name in classes:
        checkpoint_path = f"./result/{result_dir}/{dataset}/k_{k_shot}/checkpoint/CLS-Seed_111-{class_name}-check_point.pt"
        
        if not os.path.exists(checkpoint_path):
            print(f"\n⚠️ Checkpoint不存在，跳过: {class_name}")
            continue
        
        result = analyze_normal_prototype_similarity(
            checkpoint_path, dataset, class_name, k_shot
        )
        
        if result is not None:
            results.append(result)
    
    if results:
        # 汇总
        import pandas as pd
        
        summary_data = []
        for r in results:
            summary_data.append({
                'class': r['class_name'],
                'k_shot': r['k_shot'],
                'mean_sim': r['mean_similarity'],
                'std_sim': r['std_similarity'],
                'min_sim': r['min_similarity'],
                'max_sim': r['max_similarity'],
            })
        
        df = pd.DataFrame(summary_data)
        
        print(f"\n{'='*80}")
        print(f"批量分析汇总 ({dataset}, K={k_shot})")
        print(f"{'='*80}")
        print(df.to_string(index=False))
        
        print(f"\n全局统计:")
        print(f"  平均相似度: {df['mean_sim'].mean():.4f} ± {df['mean_sim'].std():.4f}")
        print(f"  最高相似度: {df['max_sim'].max():.4f}")
        print(f"  最低相似度: {df['min_sim'].min():.4f}")
        
        # 保存
        output_dir = Path(f"./analysis/prototype_similarity/{dataset}")
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f"{result_dir}_k{k_shot}_similarity.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n结果已保存到: {csv_path}")
        
        return df
    
    return None

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser("正常原型相似度分析")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="checkpoint路径")
    parser.add_argument("--result_dir", type=str, default="baseline",
                       help="结果目录")
    parser.add_argument("--dataset", type=str, default="mvtec")
    parser.add_argument("--class_name", type=str, default="metal_nut")
    parser.add_argument("--k_shot", type=int, default=2)
    parser.add_argument("--batch", action="store_true",
                       help="批量分析所有类别")
    
    args = parser.parse_args()
    
    if args.batch:
        # 批量分析
        batch_analyze(
            result_dir=args.result_dir,
            dataset=args.dataset,
            k_shot=args.k_shot
        )
    else:
        # 单个分析
        if args.checkpoint is None:
            args.checkpoint = f"./result/{args.result_dir}/{args.dataset}/k_{args.k_shot}/checkpoint/CLS-Seed_111-{args.class_name}-check_point.pt"
        
        analyze_normal_prototype_similarity(
            checkpoint_path=args.checkpoint,
            dataset=args.dataset,
            class_name=args.class_name,
            k_shot=args.k_shot
        )
