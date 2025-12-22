"""
Gate实验框架 - 支持checkpoint检查和metadata保存

特性：
1. 自动检测checkpoint，避免重复训练
2. 保存所有评分模式的结果到单个CSV
3. 保存可靠性信号（nn_margin等）到独立文件
4. 支持消融实验（semantic/memory/max/harmonic/oracle）
"""

import argparse
import torch
import numpy as np
import os
import json
from pathlib import Path
from datasets import get_dataloader_from_args, dataset_classes
from PromptAD import PromptAD
from PromptAD.model_modular import PromptADModular
from utils.metrics_modular import metric_cal_img_modular, analyze_reliability_signals
from utils.training_utils import setup_seed
from utils.csv_utils import save_metric
from tqdm import tqdm


class NumpyEncoder(json.JSONEncoder):
    """Convert numpy types to native Python types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def check_checkpoint_exists(output_dir, dataset, k_shot, class_name, task, seed=111):
    """检查checkpoint是否已存在"""
    checkpoint_dir = Path(output_dir) / dataset / f'k_{k_shot}' / 'checkpoint'
    task_upper = 'CLS' if task == 'cls' else 'SEG'
    checkpoint_file = checkpoint_dir / f'{task_upper}-Seed_{seed}-{class_name}-check_point.pt'
    
    return checkpoint_file.exists(), checkpoint_file


def save_metadata(metadata, output_dir, dataset, k_shot, class_name, task, score_mode, seed=111):
    """保存可靠性信号metadata到JSON文件"""
    meta_dir = Path(output_dir) / dataset / f'k_{k_shot}' / 'metadata' / task
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    meta_file = meta_dir / f'{class_name}_seed{seed}_{score_mode}.json'
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_meta = convert_to_serializable(metadata)
    
    with open(meta_file, 'w') as f:
        json.dump(serializable_meta, f, indent=2, cls=NumpyEncoder)
    
    return meta_file


def evaluate_all_modes(base_model, test_dataloader, device, args, output_dir):
    """评估所有评分模式并保存结果"""
    
    modes_to_test = ['semantic', 'memory', 'max', 'harmonic', 'oracle']
    all_results = {}
    
    print(f"\n{'='*60}")
    print(f"评估所有评分模式 - {args.class_name}")
    print(f"{'='*60}\n")
    
    for mode in modes_to_test:
        print(f"\n[{mode.upper()}] 评估中...")
        print("-" * 40)
        
        # Wrap model
        wrapped_model = PromptADModular(base_model, score_mode=mode, return_metadata=True)
        wrapped_model.eval_mode()
        
        scores_img = []
        score_maps = []
        gt_list = []
        gt_masks = []  # For seg task
        all_metadata = []
        
        # Evaluate
        for data, mask, label, name, img_type in tqdm(test_dataloader, desc=f'{mode}'):
            data = data.to(device)
            
            with torch.no_grad():
                if args.task == 'cls':
                    if mode == 'oracle':
                        output, metadata = wrapped_model(data, task='cls', gt_labels=label.numpy())
                    else:
                        output, metadata = wrapped_model(data, task='cls')
                    
                    score_img, score_map = output
                    scores_img.extend(score_img)
                    score_maps.extend(score_map)
                    gt_list.extend(label.numpy())
                else:  # seg
                    if mode == 'oracle':
                        output, metadata = wrapped_model(data, task='seg', gt_labels=label.numpy())
                    else:
                        output, metadata = wrapped_model(data, task='seg')
                    
                    score_map = output  # Already a list of arrays
                    score_maps += score_map  # Use += to extend list
                    scores_img.extend([s.max() for s in score_map])
                    
                    # For seg, collect masks not labels
                    for m in mask:
                        m = m.cpu().numpy() if torch.is_tensor(m) else m
                        m[m > 0] = 1
                        gt_masks.append(m)
                    
                    gt_list.extend(label.numpy())
                
                # Collect metadata from each batch
                if metadata:
                    all_metadata.append(metadata)
        
        # Merge metadata from all batches
        merged_metadata = None
        if all_metadata:
            merged_metadata = {}
            # Concatenate arrays from all batches
            for key in all_metadata[0].keys():
                if isinstance(all_metadata[0][key], np.ndarray):
                    merged_metadata[key] = np.concatenate([m[key] for m in all_metadata])
                else:
                    # For non-array values, take the first one
                    merged_metadata[key] = all_metadata[0][key]
        
        # Compute metrics
        if args.task == 'cls':
            metrics = metric_cal_img_modular(
                np.array(scores_img),
                np.array(gt_list),
                np.array(score_maps),
                metadata=merged_metadata,
                score_mode=mode
            )
        else:  # seg
            from utils.metrics import metric_cal_pix
            # Resize masks to match score map resolution
            import cv2
            gt_masks_resized = [cv2.resize(m, (args.resolution, args.resolution), 
                                          interpolation=cv2.INTER_NEAREST) for m in gt_masks]
            metrics = metric_cal_pix(np.array(score_maps), gt_masks_resized)
        
        # Add reliability analysis
        if merged_metadata:
            reliability = analyze_reliability_signals(merged_metadata, np.array(gt_list))
            metrics.update(reliability)
        
        # Save metadata
        if merged_metadata:
            meta_file = save_metadata(
                merged_metadata,
                output_dir,
                args.dataset,
                args.k_shot,
                args.class_name,
                args.task,
                mode,
                args.seed
            )
            print(f"   → Metadata saved: {meta_file}")
        
        all_results[mode] = metrics
        
        # Print results
        if args.task == 'cls':
            print(f"   Image AUROC: {metrics.get('i_roc', 0.0):.2f}%")
            if 'i_roc_semantic' in metrics:
                print(f"     ├─ Semantic: {metrics['i_roc_semantic']:.2f}%")
                print(f"     ├─ Memory:   {metrics['i_roc_memory']:.2f}%")
                print(f"     └─ Gap:      {metrics['gap']:.2f}%")
        else:
            print(f"   Pixel AUROC: {metrics.get('p_roc', 0.0):.2f}%")
        
        if mode == 'oracle' and 'oracle_semantic_ratio' in metrics:
            print(f"   Oracle选择:")
            print(f"     ├─ Semantic: {metrics['oracle_semantic_ratio']:.1f}%")
            print(f"     └─ Memory:   {metrics['oracle_memory_ratio']:.1f}%")
    
    # Summary table
    print(f"\n{'='*60}")
    print("汇总对比")
    print(f"{'='*60}")
    metric_key = 'i_roc' if args.task == 'cls' else 'p_roc'
    print(f"{'模式':<15} {'AUROC':<10} {'与Oracle差距':<15}")
    print("-" * 60)
    
    oracle_score = all_results.get('oracle', {}).get(metric_key, 0.0)
    
    for mode in modes_to_test:
        auroc = all_results[mode].get(metric_key, 0.0)
        gap = oracle_score - auroc
        print(f"{mode:<15} {auroc:>8.2f}%  {gap:>12.2f}%")
    
    print(f"{'='*60}\n")
    
    return all_results


def main(args):
    # Setup
    setup_seed(args.seed)
    device = f"cuda:{args.gpu_id}" if not args.use_cpu else "cpu"
    
    print(f"\n{'='*60}")
    print(f"Gate实验: {args.dataset}/{args.class_name} k={args.k_shot}")
    print(f"{'='*60}\n")
    
    # Determine checkpoint path
    if args.checkpoint_dir:
        # Use external checkpoint directory (e.g., result/max_score)
        external_checkpoint_dir = Path(args.checkpoint_dir) / args.dataset / f'k_{args.k_shot}' / 'checkpoint'
        task_upper = 'CLS' if args.task == 'cls' else 'SEG'
        external_checkpoint_path = external_checkpoint_dir / f'{task_upper}-Seed_{args.seed}-{args.class_name}-check_point.pt'
        
        if external_checkpoint_path.exists():
            print(f"✓ 找到外部检查点: {external_checkpoint_path}")
            print(f"  将直接加载，跳过训练\n")
            checkpoint_path = external_checkpoint_path
            skip_training = True
            exists = True
        else:
            print(f"✗ 外部检查点不存在: {external_checkpoint_path}")
            print(f"  回退到本地checkpoint检查\n")
            # Fallback to local checkpoint
            exists, checkpoint_path = check_checkpoint_exists(
                args.root_dir,
                args.dataset,
                args.k_shot,
                args.class_name,
                args.task,
                args.seed
            )
            skip_training = exists and not args.force_retrain
    else:
        # Check local checkpoint
        exists, checkpoint_path = check_checkpoint_exists(
            args.root_dir,
            args.dataset,
            args.k_shot,
            args.class_name,
            args.task,
            args.seed
        )
        
        if exists and not args.force_retrain:
            print(f"✓ 检查点已存在: {checkpoint_path}")
            print(f"  跳过训练，直接加载模型进行评估\n")
            skip_training = True
        else:
            if exists:
                print(f"⚠️  检查点已存在但设置了 --force-retrain")
            else:
                print(f"✓ 检查点不存在: {checkpoint_path}")
            print(f"  将进行训练\n")
            skip_training = False
    
    # Create model
    kwargs = vars(args)
    kwargs['device'] = device
    kwargs['out_size_h'] = kwargs['resolution']
    kwargs['out_size_w'] = kwargs['resolution']
    
    base_model = PromptAD(**kwargs).to(device)
    
    # Get dataloaders
    train_dataloader, _ = get_dataloader_from_args(
        phase='train',
        perturbed=False,
        transform=base_model.transform,
        **kwargs
    )
    
    test_dataloader, _ = get_dataloader_from_args(
        phase='test',
        perturbed=False,
        transform=base_model.transform,
        **kwargs
    )
    
    base_model.eval_mode()
    
    if not skip_training:
        # Setup memory bank (requires training)
        print("构建记忆库...")
        features1 = []
        features2 = []
        for data, mask, label, name, img_type in tqdm(train_dataloader, desc='提取特征'):
            data = data.to(device)
            with torch.no_grad():
                _, _, feature_map1, feature_map2 = base_model.encode_image(data)
                features1.append(feature_map1)
                features2.append(feature_map2)
        
        features1 = torch.cat(features1, dim=0)
        features2 = torch.cat(features2, dim=0)
        base_model.build_image_feature_gallery(features1, features2)
        
        # Save checkpoint
        print(f"保存检查点: {checkpoint_path}")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(base_model.state_dict(), checkpoint_path)
    else:
        # Load checkpoint
        print(f"加载检查点: {checkpoint_path}")
        base_model.load_state_dict(torch.load(checkpoint_path), strict=False)
        print(f"✓ 记忆库已从检查点恢复")
    
    # Evaluate all modes
    all_results = evaluate_all_modes(base_model, test_dataloader, device, args, args.root_dir)
    
    # Save results summary
    result_dir = Path(args.root_dir) / args.dataset / f'k_{args.k_shot}' / 'gate_results'
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f'{args.class_name}_seed{args.seed}_{args.task}.json'
    
    with open(result_file, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    
    print(f"\n✓ 结果已保存: {result_file}\n")
    
    return all_results


def get_args():
    parser = argparse.ArgumentParser(description='Gate实验框架')
    
    # Dataset args
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa'])
    parser.add_argument('--class_name', type=str, default='carpet')
    parser.add_argument('--k-shot', type=int, default=4)
    parser.add_argument('--task', type=str, default='cls', choices=['cls', 'seg'])
    
    # Model args
    parser.add_argument('--img-resize', type=int, default=240)
    parser.add_argument('--img-cropsize', type=int, default=240)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--backbone', type=str, default='ViT-B-16-plus-240')
    parser.add_argument('--pretrained_dataset', type=str, default='laion400m_e32')
    
    # Prompt args
    parser.add_argument('--n_ctx', type=int, default=4)
    parser.add_argument('--n_ctx_ab', type=int, default=1)
    parser.add_argument('--n_pro', type=int, default=3)
    parser.add_argument('--n_pro_ab', type=int, default=4)
    
    # System args
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--use-cpu', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=0)
    
    # Output args
    parser.add_argument('--root-dir', type=str, default='./result_gate')
    
    # Control args
    parser.add_argument('--force-retrain', action='store_true',
                        help='强制重新训练，即使checkpoint已存在')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='外部checkpoint目录(例如: result/max_score)。如果指定，将从该目录加载已有checkpoint')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    results = main(args)
