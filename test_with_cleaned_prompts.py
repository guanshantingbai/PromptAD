#!/usr/bin/env python3
"""
Phase 1: 使用清洗后的 prompts 重新测试（不重新训练）
加载 baseline 的 checkpoint，但使用当前的 prompt table（已禁用6个全局问题prompts）
"""

import argparse
import torch
from datasets import get_dataloader_from_args
from utils.csv_utils import *
from utils.metrics import *
from utils.training_utils import *
from PromptAD import PromptAD
from utils.eval_utils import *
from tqdm import tqdm
import os


def to_numpy(x):
    """统一转换为numpy array"""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    else:
        return x


def test_with_cleaned_prompts(args):
    """使用cleaned prompts测试已有模型"""
    
    kwargs = vars(args)
    setup_seed(kwargs['seed'])
    
    device = f"cuda:{args.gpu_id}" if not args.use_cpu else "cpu"
    kwargs['device'] = device
    
    # 设置输出路径
    kwargs['out_size_h'] = kwargs['resolution']
    kwargs['out_size_w'] = kwargs['resolution']
    
    # 创建模型（会自动使用当前的 prompt table）
    print(f"Creating model for {args.dataset}/{args.class_name}...")
    model = PromptAD(**kwargs)
    model = model.to(device)
    
    # 加载 baseline checkpoint
    checkpoint_path = f"result/baseline/{args.dataset}/k_{args.k_shot}/checkpoint/CLS-Seed_{args.seed}-{args.class_name}-check_point.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    print(f"✓ Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 兼容旧版本checkpoint格式
    if 'text_features' in checkpoint and 'normal_prototypes' not in checkpoint:
        # 旧版本：text_features = [normal, abnormal]
        print("  Detected old checkpoint format (text_features)")
        text_features = checkpoint['text_features']
        checkpoint['normal_prototypes'] = text_features[0:1]  # 第1个是normal
        checkpoint['abnormal_prototypes'] = text_features[1:2]  # 第2个是abnormal
    
    model.load_state_dict(checkpoint, strict=False)
    
    # 加载 memory bank
    if 'feature_gallery1' in checkpoint and 'feature_gallery2' in checkpoint:
        print("  Loading memory bank from checkpoint")
        model.feature_gallery1 = checkpoint['feature_gallery1'].to(device)
        model.feature_gallery2 = checkpoint['feature_gallery2'].to(device)
        # 不需要重新构建memory bank
        skip_memory_build = True
    else:
        skip_memory_build = False
    
    # 打印加载的 prompts 信息
    print(f"\n{'='*70}")
    print(f"Loaded Prompts Info:")
    print(f"{'='*70}")
    if hasattr(model, 'manual_prompts_templates'):
        print(f"Manual prompts: {len(model.manual_prompts_templates)} templates")
    print(f"Normal prototypes shape: {model.normal_prototypes.shape}")
    print(f"Abnormal prototypes shape: {model.abnormal_prototypes.shape}")
    print(f"{'='*70}\n")
    
    # 设置为评估模式
    model.eval_mode()
    
    # 构建 memory bank (如果checkpoint没有提供)
    if not skip_memory_build:
        # 获取训练数据（用于 memory bank）
        print("Loading training data for memory bank...")
        train_dataloader, _ = get_dataloader_from_args(
            phase='train', perturbed=False, 
            transform=model.transform, **kwargs
        )
        
        # 构建 memory bank
        print("Building memory bank...")
        features1 = []
        features2 = []
        with torch.no_grad():
            for data, mask, label, name, img_type in tqdm(train_dataloader, desc="Building memory bank"):
                data = data.to(device)
                _, _, feature_map1, feature_map2 = model.encode_image(data)
                features1.append(feature_map1)
                features2.append(feature_map2)
        
        features1 = torch.cat(features1, dim=0)
        features2 = torch.cat(features2, dim=0)
        model.memory_bank = [features1, features2]
    else:
        print("  Skipping memory bank building (loaded from checkpoint)")
    
    # 获取测试数据
    print("Loading test data...")
    test_dataloader, _ = get_dataloader_from_args(
        phase='test', perturbed=False,
        transform=model.transform, **kwargs
    )
    
    # 运行测试
    print("Running inference...")
    scores_img = []
    score_maps = []
    gt_list = []
    gt_mask_list = []
    
    import cv2
    for (data, mask, label, name, img_type) in tqdm(test_dataloader, desc="Testing"):
        # 收集ground truth
        for l, m in zip(label, mask):
            l = l.cpu().numpy() if torch.is_tensor(l) else l
            m = m.cpu().numpy() if torch.is_tensor(m) else m
            m[m > 0] = 1
            gt_list.append(l)
            gt_mask_list.append(m)
        
        data = data.to(device)
        
        with torch.no_grad():
            # Semantic分支
            visual_features = model.encode_image(data)
            textual_anomaly = model.calculate_textual_anomaly_score(visual_features, 'cls')
            textual_anomaly_map = model.calculate_textual_anomaly_score(visual_features, 'seg')
            
            # Memory分支  
            memory_anomaly_map = model.calculate_visual_anomaly_score(visual_features)
            
            # 统一转为numpy
            textual_anomaly_np = to_numpy(textual_anomaly)
            textual_anomaly_map_np = to_numpy(textual_anomaly_map)
            memory_anomaly_map_np = to_numpy(memory_anomaly_map)
            
            # Image-level memory score (取map的最大值)
            memory_anomaly_np = memory_anomaly_map_np.reshape(memory_anomaly_map_np.shape[0], -1).max(axis=1)
            
            # 调和平均融合: 1/fusion = 1/semantic + 1/memory
            # 避免除零：给极小值加一个小的epsilon
            eps = 1e-10
            fusion_anomaly_np = 2 / (1/(textual_anomaly_np + eps) + 1/(memory_anomaly_np + eps))
            fusion_anomaly_map_np = 2 / (1/(textual_anomaly_map_np + eps) + 1/(memory_anomaly_map_np + eps))
        
        scores_img.append({
            'semantic': textual_anomaly_np.tolist(),
            'memory': memory_anomaly_np.tolist(),
            'fusion': fusion_anomaly_np.tolist()
        })
        
        for i in range(textual_anomaly_map_np.shape[0]):
            score_maps.append({
                'semantic': textual_anomaly_map_np[i, 0],
                'memory': memory_anomaly_map_np[i, 0],
                'fusion': fusion_anomaly_map_np[i, 0]
            })
    
    # Resize gt_masks
    gt_mask_list = [cv2.resize(mask, (args.resolution, args.resolution), 
                               interpolation=cv2.INTER_NEAREST) for mask in gt_mask_list]
    
    # 计算metrics
    print("\nCalculating metrics...")
    
    # Flatten scores
    semantic_scores = []
    memory_scores = []
    fusion_scores = []
    semantic_maps = []
    memory_maps = []
    fusion_maps = []
    
    for s in scores_img:
        semantic_scores += s['semantic']
        memory_scores += s['memory']
        fusion_scores += s['fusion']
    
    for m in score_maps:
        semantic_maps.append(m['semantic'])
        memory_maps.append(m['memory'])
        fusion_maps.append(m['fusion'])
    
    # 计算每个分支的metrics (只计算image-level)
    semantic_result = metric_cal_img(np.array(semantic_scores), gt_list, np.array(semantic_maps))
    memory_result = metric_cal_img(np.array(memory_scores), gt_list, np.array(memory_maps))
    fusion_result = metric_cal_img(np.array(fusion_scores), gt_list, np.array(fusion_maps))
    
    # 暂时不计算pixel-level（因为需要resize），用0代替
    semantic_result['p_roc'] = 0.0
    memory_result['p_roc'] = 0.0
    fusion_result['p_roc'] = 0.0
    
    # 调试信息
    print(f"\nDebug Info:")
    print(f"  Semantic scores range: [{np.min(semantic_scores):.4f}, {np.max(semantic_scores):.4f}]")
    print(f"  Memory scores range:   [{np.min(memory_scores):.4f}, {np.max(memory_scores):.4f}]")
    print(f"  GT labels: {np.unique(gt_list, return_counts=True)}")
    
    # 打印结果
    print(f"\n{'='*70}")
    print(f"Results for {args.dataset}/{args.class_name}:")
    print(f"{'='*70}")
    print(f"Semantic AUROC: {semantic_result['i_roc']:.2f}")
    print(f"Memory AUROC:   {memory_result['i_roc']:.2f}")
    print(f"Fusion AUROC:   {fusion_result['i_roc']:.2f}")
    print(f"{'='*70}\n")
    
    return {
        'semantic': semantic_result,
        'memory': memory_result,
        'fusion': fusion_result
    }


def save_results_to_csv(results_dict, args):
    """保存结果到CSV"""
    
    # 使用自定义输出目录或默认的 phase1_cleaned
    base_dir = getattr(args, 'output_dir', 'result/phase1_cleaned')
    output_dir = f"{base_dir}/{args.dataset}/k_{args.k_shot}/csv"
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, f"Seed_{args.seed}-results.csv")
    
    # 构建行数据
    row_data = {
        'class': f"{args.dataset}-{args.class_name}",
        'i_roc': results_dict['fusion']['i_roc'],
        'p_roc': results_dict['fusion']['p_roc'],
        'semantic_i_roc': results_dict['semantic']['i_roc'],
        'memory_i_roc': results_dict['memory']['i_roc'],
    }
    
    # 追加或创建CSV
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0)
        # 更新或添加行
        df.loc[row_data['class']] = row_data
    else:
        df = pd.DataFrame([row_data])
        df.set_index('class', inplace=True)
    
    df.to_csv(csv_path)
    print(f"✓ Results saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Test with cleaned prompts (no retraining)')
    parser.add_argument('--dataset', type=str, required=True, choices=['mvtec', 'visa'])
    parser.add_argument('--class_name', type=str, required=True)
    parser.add_argument('--k-shot', type=int, default=2)
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--use-cpu', action='store_true')
    parser.add_argument('--output-dir', type=str, default='result/phase1_cleaned',
                       help='Output directory for results')
    
    # 模型参数
    parser.add_argument('--img-resize', type=int, default=240)
    parser.add_argument('--img-cropsize', type=int, default=240)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=400)
    parser.add_argument('--backbone', type=str, default='ViT-B-16-plus-240')
    parser.add_argument('--pretrained_dataset', type=str, default='laion400m_e32')
    
    # Prompt learner参数
    parser.add_argument('--n-ctx', type=int, default=12, help='Number of context tokens for normal prompts')
    parser.add_argument('--n-pro', type=int, default=4, help='Number of normal prototypes')
    parser.add_argument('--n-ctx-ab', type=int, default=12, help='Number of context tokens for abnormal prompts')
    parser.add_argument('--n-pro-ab', type=int, default=10, help='Number of abnormal prototypes')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Phase 1: Testing with Cleaned Prompts")
    print("="*70)
    print(f"Dataset:    {args.dataset}")
    print(f"Class:      {args.class_name}")
    print(f"K-shot:     {args.k_shot}")
    print(f"Seed:       {args.seed}")
    print(f"Note:       Using cleaned prompts (6 global prompts disabled)")
    print("="*70)
    print()
    
    # 运行测试
    results = test_with_cleaned_prompts(args)
    
    if results is not None:
        # 保存结果
        save_results_to_csv(results, args)
        print("\n✅ Testing complete!")
    else:
        print("\n❌ Testing failed!")
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
