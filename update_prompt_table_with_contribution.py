"""
更新Prompt表格中的贡献度分析字段
在训练好的模型上运行分析，并将结果写回表格
"""

import os
import pandas as pd
import torch
import argparse
from torch.utils.data import DataLoader

from PromptAD.model import PromptAD
from datasets.dataset import Dataset
from datasets.mvtec import load_mvtec, mvtec_classes
from datasets.visa import load_visa, visa_classes


def analyze_and_update_table(args):
    """运行分析并更新表格"""
    
    # 读取表格
    df = pd.read_csv(args.table_path)
    print(f"✓ Loaded prompt table: {args.table_path}")
    print(f"  Total prompts: {len(df)}")
    
    # 加载数据
    print(f"\nLoading {args.dataset} dataset - {args.classname} (k={args.k_shot})...")
    if args.dataset == 'mvtec':
        train_data, test_data = load_mvtec(args.classname, args.k_shot)
    else:
        train_data, test_data = load_visa(args.classname, args.k_shot)
    
    test_dataset = Dataset(test_data, preproc=None, test=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"✓ Test samples: {len(test_dataset)}")
    
    # 加载模型
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
        print(f"\nLoading checkpoint: {args.ckpt}")
        checkpoint = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("✓ Model loaded")
    else:
        print(f"Warning: Checkpoint not found, using random initialization")
    
    model.to(device)
    model.eval()
    
    # 构建文本特征
    print("\nBuilding text features...")
    model.build_text_feature_gallery()
    
    # 运行贡献度分析
    print(f"\nAnalyzing prompt contribution on test set...")
    from run_prompt_contribution_analysis import analyze_on_dataset
    aggregated, prompt_info = analyze_on_dataset(model, test_loader, task=args.task)
    
    # 按平均相似度排序，计算rank
    sorted_by_sim = sorted(aggregated, key=lambda x: x['mean_similarity'], reverse=True)
    rank_map = {item['index']: rank+1 for rank, item in enumerate(sorted_by_sim)}
    
    # 更新表格中该类别的行
    print(f"\nUpdating table for class '{args.classname}'...")
    updated_count = 0
    
    for idx, row in df.iterrows():
        if row['class'] == args.classname and row['enabled']:
            # 找到对应的分析结果
            index_in_class = row['index_in_class']
            
            # 在aggregated中查找
            matching = [item for item in aggregated if item['index'] == index_in_class]
            if matching:
                result = matching[0]
                df.at[idx, 'mean_similarity'] = round(result['mean_similarity'], 4)
                df.at[idx, 'max_similarity'] = round(result['max_similarity'], 4)
                df.at[idx, 'contribution_rank'] = rank_map.get(index_in_class, '')
                updated_count += 1
    
    print(f"✓ Updated {updated_count} prompts for class '{args.classname}'")
    
    # 保存表格
    backup_path = args.table_path.replace('.csv', '_backup.csv')
    if not os.path.exists(backup_path):
        df_original = pd.read_csv(args.table_path)
        df_original.to_csv(backup_path, index=False)
        print(f"✓ Created backup: {backup_path}")
    
    df.to_csv(args.table_path, index=False)
    print(f"✓ Saved updated table: {args.table_path}")
    
    # 打印该类别的结果
    print(f"\n{'='*120}")
    print(f"Contribution Analysis Results - {args.classname}")
    print(f"{'='*120}")
    print(f"{'Rank':<6} {'ID':<5} {'Type':<10} {'Mean Sim':<12} {'Text':<60}")
    print(f"{'-'*120}")
    
    for rank, item in enumerate(sorted_by_sim[:20], 1):  # 只显示前20
        type_tag = 'Generic' if item['type'] == 'generic' else 'Specific'
        print(f"{rank:<6} {item['index']:<5} {type_tag:<10} {item['mean_similarity']:<12.4f} {item['text'][:60]}")
    
    if len(sorted_by_sim) > 20:
        print(f"... (showing top 20 of {len(sorted_by_sim)})")
    
    print(f"{'='*120}\n")


def main():
    parser = argparse.ArgumentParser(description='Update Prompt Table with Contribution Analysis')
    
    # 表格路径
    parser.add_argument('--table_path', type=str, 
                        default='prompts/manual_prompts_master_table.csv',
                        help='Path to prompt master table')
    
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
    
    # Checkpoint和设备
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--device', type=int, default=0)
    
    # 分析参数
    parser.add_argument('--task', type=str, default='seg', choices=['cls', 'seg'])
    parser.add_argument('--batch_size', type=int, default=8)
    
    args = parser.parse_args()
    
    analyze_and_update_table(args)


if __name__ == '__main__':
    main()
