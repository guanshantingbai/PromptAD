"""
在已训练模型上运行Manual Prompt贡献度分析

用法:
python run_prompt_contribution_analysis.py --dataset mvtec --class bottle --k_shot 2 --seed 111 --ckpt path/to/checkpoint.pth
"""

import os
import sys
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from PromptAD.model import PromptAD
from datasets.dataset import Dataset
from datasets.mvtec import load_mvtec, mvtec_classes
from datasets.visa import load_visa, visa_classes


def load_model_from_checkpoint(args):
    """从checkpoint加载模型"""
    
    # 确定类别名称
    if args.dataset == 'mvtec':
        class_name = f'mvtec-{args.classname}'
    else:
        class_name = f'visa-{args.classname}'
    
    # 创建模型
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
    
    # 加载checkpoint
    if os.path.exists(args.ckpt):
        print(f"Loading checkpoint from: {args.ckpt}")
        checkpoint = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"✓ Model loaded successfully")
    else:
        print(f"Warning: Checkpoint not found at {args.ckpt}")
        print("Using randomly initialized model for analysis")
    
    model.to(device)
    model.eval()
    
    return model


def analyze_on_dataset(model, dataloader, task='seg'):
    """在数据集上分析prompt贡献度"""
    
    all_stats = []
    all_labels = []
    
    print(f"\nAnalyzing on {len(dataloader)} batches...")
    
    with torch.no_grad():
        for images, gt, labels, _ in tqdm(dataloader):
            images = images.to(model.device)
            
            # 编码图像
            image_features = model.encode_image(images)
            
            # 分析prompt贡献
            result = model.analyze_manual_prompt_contribution(
                image_features, 
                task=task, 
                return_details=False
            )
            
            all_stats.append(result['stats'])
            all_labels.extend(labels.cpu().numpy())
    
    # 聚合统计
    prompt_info = model.get_manual_prompt_info()
    num_prompts = prompt_info['num_manual_templates']
    
    print(f"\nAggregating statistics for {num_prompts} prompts...")
    
    aggregated = []
    for i in range(num_prompts):
        template_stats = [batch[i] for batch in all_stats]
        
        agg = {
            'index': i,
            'template': template_stats[0]['template'],
            'text': template_stats[0]['text'],
            'type': template_stats[0]['type'],
            'mean_similarity': np.mean([s['mean_similarity'] for s in template_stats]),
            'max_similarity': np.max([s['max_similarity'] for s in template_stats]),
            'min_similarity': np.min([s['min_similarity'] for s in template_stats]),
            'std_similarity': np.mean([s['std_similarity'] for s in template_stats]),
        }
        aggregated.append(agg)
    
    return aggregated, prompt_info


def print_contribution_report(aggregated, prompt_info, top_k=None):
    """打印贡献度报告"""
    
    # 按平均相似度排序
    sorted_by_mean = sorted(aggregated, key=lambda x: x['mean_similarity'], reverse=True)
    
    print(f"\n{'='*130}")
    print(f"Manual Prompt Contribution Analysis")
    print(f"Class: {prompt_info['classname']} | Display: {prompt_info['display_name']}")
    print(f"Total Templates: {prompt_info['num_manual_templates']} | Total Prototypes: {prompt_info['num_manual_prototypes']}")
    print(f"{'='*130}")
    print(f"{'Rank':<6} {'ID':<5} {'Type':<10} {'Mean Sim':<12} {'Max Sim':<12} {'Min Sim':<12} {'Std':<10} {'Text':<50}")
    print(f"{'-'*130}")
    
    display_list = sorted_by_mean[:top_k] if top_k else sorted_by_mean
    
    for rank, stat in enumerate(display_list, 1):
        type_tag = 'Generic' if stat['type'] == 'generic' else 'Specific'
        print(f"{rank:<6} {stat['index']:<5} {type_tag:<10} "
              f"{stat['mean_similarity']:<12.4f} {stat['max_similarity']:<12.4f} "
              f"{stat['min_similarity']:<12.4f} {stat['std_similarity']:<10.4f} "
              f"{stat['text']:<50}")
    
    if top_k and top_k < len(sorted_by_mean):
        print(f"\n... (showing top {top_k} of {len(sorted_by_mean)})")
    
    print(f"{'='*130}\n")
    
    # 统计信息
    generic_prompts = [p for p in aggregated if p['type'] == 'generic']
    specific_prompts = [p for p in aggregated if p['type'] == 'specific']
    
    print("Summary Statistics:")
    print(f"  Generic Prompts: {len(generic_prompts)}")
    if generic_prompts:
        print(f"    Mean Similarity: {np.mean([p['mean_similarity'] for p in generic_prompts]):.4f}")
        print(f"    Max Similarity:  {np.max([p['mean_similarity'] for p in generic_prompts]):.4f}")
    
    print(f"\n  Specific Prompts: {len(specific_prompts)}")
    if specific_prompts:
        print(f"    Mean Similarity: {np.mean([p['mean_similarity'] for p in specific_prompts]):.4f}")
        print(f"    Max Similarity:  {np.max([p['mean_similarity'] for p in specific_prompts]):.4f}")
    print()


def save_results(aggregated, prompt_info, args, output_path):
    """保存结果到CSV"""
    
    # 按平均相似度排序
    sorted_results = sorted(aggregated, key=lambda x: x['mean_similarity'], reverse=True)
    
    # 添加rank和元信息
    for rank, item in enumerate(sorted_results, 1):
        item['rank'] = rank
        item['classname'] = prompt_info['classname']
        item['dataset'] = args.dataset
        item['k_shot'] = args.k_shot
        item['seed'] = args.seed
    
    df = pd.DataFrame(sorted_results)
    
    # 调整列顺序
    columns = ['rank', 'dataset', 'classname', 'k_shot', 'seed', 'index', 'type', 
               'mean_similarity', 'max_similarity', 'min_similarity', 'std_similarity',
               'template', 'text']
    df = df[columns]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✓ Results saved to: {output_path}\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze Manual Prompt Contribution')
    
    # 数据集参数
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa'])
    parser.add_argument('--class', type=str, dest='classname', required=True,
                        help='Class name (e.g., bottle, carpet)')
    parser.add_argument('--k_shot', type=int, default=2)
    parser.add_argument('--seed', type=int, default=111)
    
    # 模型参数
    parser.add_argument('--backbone', type=str, default='ViT-B-16-plus-240')
    parser.add_argument('--pretrained_dataset', type=str, default='laion400m_e32')
    parser.add_argument('--n_ctx', type=int, default=12)
    parser.add_argument('--n_pro', type=int, default=4)
    parser.add_argument('--n_ctx_ab', type=int, default=12)
    parser.add_argument('--n_pro_ab', type=int, default=4)
    parser.add_argument('--img_resize', type=int, default=240)
    parser.add_argument('--img_cropsize', type=int, default=240)
    
    # Checkpoint和设备
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=int, default=0)
    
    # 分析参数
    parser.add_argument('--task', type=str, default='seg', choices=['cls', 'seg'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--top_k', type=int, default=None,
                        help='Show top K prompts (None for all)')
    
    # 输出
    parser.add_argument('--output_dir', type=str, 
                        default='result/prompt_contribution_analysis')
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"\nLoading {args.dataset} dataset - {args.classname} (k={args.k_shot}, seed={args.seed})...")
    
    if args.dataset == 'mvtec':
        train_data, test_data = load_mvtec(args.classname, args.k_shot)
    else:
        train_data, test_data = load_visa(args.classname, args.k_shot)
    
    test_dataset = Dataset(test_data, preproc=None, test=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"✓ Test samples: {len(test_dataset)}")
    
    # 加载模型
    model = load_model_from_checkpoint(args)
    
    # 构建文本特征库
    print("\nBuilding text feature gallery...")
    model.build_text_feature_gallery()
    print("✓ Text features built")
    
    # 分析
    aggregated, prompt_info = analyze_on_dataset(model, test_loader, task=args.task)
    
    # 打印报告
    print_contribution_report(aggregated, prompt_info, top_k=args.top_k)
    
    # 保存结果
    output_filename = f"{args.dataset}_{args.classname}_k{args.k_shot}_seed{args.seed}_contribution.csv"
    output_path = os.path.join(args.output_dir, args.dataset, f"k_{args.k_shot}", output_filename)
    save_results(aggregated, prompt_info, args, output_path)


if __name__ == '__main__':
    main()
