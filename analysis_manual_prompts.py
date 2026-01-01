"""
分析和可视化Manual Prompts的工具脚本

功能:
1. 展示所有类别的静态prompt表格
2. 分析特定类别的prompt构成
3. 评估prompt在测试集上的贡献度
"""

import os
import sys
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from PromptAD.ad_prompts import (
    get_full_manual_prompts, 
    get_all_classes_manual_prompts,
    print_manual_prompts_table
)
from PromptAD.model import PromptAD
from datasets.mvtec import load_mvtec, mvtec_classes
from datasets.visa import load_visa, visa_classes


def show_all_prompts_summary():
    """展示所有类别的prompt概览"""
    all_prompts = get_all_classes_manual_prompts()
    
    print(f"\n{'='*100}")
    print(f"{'All Classes Manual Prompts Summary':^100}")
    print(f"{'='*100}")
    print(f"{'Class':<20} {'Generic':<10} {'Specific':<10} {'Total':<10} {'Dataset':<15}")
    print(f"{'-'*100}")
    
    # MVTec classes
    for classname in mvtec_classes:
        if classname in all_prompts:
            data = all_prompts[classname]
            print(f"{classname:<20} {data['num_generic']:<10} {data['num_specific']:<10} {data['num_total']:<10} {'MVTec':<15}")
    
    print(f"{'-'*100}")
    
    # VisA classes  
    for classname in visa_classes:
        if classname in all_prompts:
            data = all_prompts[classname]
            print(f"{classname:<20} {data['num_generic']:<10} {data['num_specific']:<10} {data['num_total']:<10} {'VisA':<15}")
    
    print(f"{'='*100}\n")


def show_class_prompts_detail(classname):
    """展示特定类别的详细prompt信息"""
    prompts, info = get_full_manual_prompts(classname)
    
    print(f"\n{'='*100}")
    print(f"Class: {classname} (Display: {info[0]['display_name']})")
    print(f"Total: {len(prompts)} prompts (Generic: 8, Specific: {len(prompts) - 8})")
    print(f"{'='*100}")
    print(f"{'ID':<5} {'Type':<10} {'Template':<45} {'Full Text':<40}")
    print(f"{'-'*100}")
    
    for item in info:
        type_tag = 'Generic' if item['type'] == 'generic' else 'Specific'
        print(f"{item['index']:<5} {type_tag:<10} {item['template']:<45} {item['text']:<40}")
    
    print(f"{'='*100}\n")


def export_prompts_to_csv(output_path='result/manual_prompts_table.csv'):
    """导出所有prompt到CSV文件"""
    all_prompts = get_all_classes_manual_prompts()
    
    rows = []
    for classname in sorted(all_prompts.keys()):
        data = all_prompts[classname]
        for item in data['info']:
            row = {
                'class': classname,
                'display_name': item['display_name'],
                'index': item['index'],
                'type': item['type'],
                'template': item['template'],
                'text': item['text'],
                'dataset': 'MVTec' if classname in mvtec_classes else 'VisA'
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Exported {len(rows)} prompts to: {output_path}")
    print(f"  Total classes: {len(all_prompts)}")
    print(f"  Columns: {', '.join(df.columns)}\n")
    
    return df


def analyze_prompt_contribution_on_model(model, dataloader, task='seg', save_path=None):
    """
    在已加载的模型上分析prompt贡献度
    
    Args:
        model: 已加载的PromptAD模型
        dataloader: 数据加载器
        task: 'cls' 或 'seg'
        save_path: 保存结果的路径
    """
    model.eval()
    
    all_stats = []
    
    print(f"\nAnalyzing manual prompt contribution on {len(dataloader)} samples...")
    
    for batch_idx, (images, _, labels, _) in enumerate(tqdm(dataloader)):
        images = images.to(model.device)
        
        # 编码图像
        image_features = model.encode_image(images)
        
        # 分析prompt贡献
        result = model.analyze_manual_prompt_contribution(image_features, task=task, return_details=False)
        all_stats.append(result['stats'])
    
    # 聚合统计
    prompt_info = model.get_manual_prompt_info()
    num_prompts = prompt_info['num_manual_templates']
    
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
            'std_similarity': np.mean([s['std_similarity'] for s in template_stats]),
        }
        aggregated.append(agg)
    
    # 按平均相似度排序
    aggregated_sorted = sorted(aggregated, key=lambda x: x['mean_similarity'], reverse=True)
    
    # 打印结果
    print(f"\n{'='*120}")
    print(f"Manual Prompt Contribution Analysis - {prompt_info['classname']}")
    print(f"{'='*120}")
    print(f"{'Rank':<6} {'ID':<5} {'Type':<10} {'Mean Sim':<12} {'Max Sim':<12} {'Std':<12} {'Text':<50}")
    print(f"{'-'*120}")
    
    for rank, stat in enumerate(aggregated_sorted, 1):
        type_tag = 'Generic' if stat['type'] == 'generic' else 'Specific'
        print(f"{rank:<6} {stat['index']:<5} {type_tag:<10} {stat['mean_similarity']:<12.4f} "
              f"{stat['max_similarity']:<12.4f} {stat['std_similarity']:<12.4f} {stat['text']:<50}")
    
    print(f"{'='*120}\n")
    
    # 保存到CSV
    if save_path:
        df = pd.DataFrame(aggregated_sorted)
        df.insert(0, 'rank', range(1, len(df) + 1))
        df.insert(1, 'classname', prompt_info['classname'])
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"✓ Saved contribution analysis to: {save_path}\n")
    
    return aggregated_sorted


def main():
    parser = argparse.ArgumentParser(description='Manual Prompts Analysis Tool')
    parser.add_argument('--mode', type=str, default='summary', 
                        choices=['summary', 'detail', 'export'],
                        help='Operation mode')
    parser.add_argument('--class', type=str, dest='classname', default=None,
                        help='Class name for detail mode')
    parser.add_argument('--output', type=str, default='result/manual_prompts_table.csv',
                        help='Output path for export mode')
    
    args = parser.parse_args()
    
    if args.mode == 'summary':
        show_all_prompts_summary()
        
    elif args.mode == 'detail':
        if args.classname is None:
            print("Error: --class is required for detail mode")
            return
        show_class_prompts_detail(args.classname)
        
    elif args.mode == 'export':
        export_prompts_to_csv(args.output)


if __name__ == '__main__':
    main()
