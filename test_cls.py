import argparse

import torch
import torch.optim.lr_scheduler
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from datasets import *
from datasets import dataset_classes
from utils.csv_utils import *
from utils.metrics import *
from utils.training_utils import *
from PromptAD import *
from utils.eval_utils import *
from torchvision import transforms
from tqdm import tqdm

TASK = 'CLS'

def test(model,
        args,
        dataloader: DataLoader,
        device: str,
        img_dir: str,
        check_path: str,
        train_dataloader: DataLoader = None,
        ):

    # change the model into eval mode
    model.eval_mode()

    # Load checkpoint, filtering out incompatible keys
    checkpoint = torch.load(check_path)
    # Remove _all suffix keys - they will be recomputed from individual prompts
    # (Training and testing may have different numbers of prompts, causing shape mismatch)
    checkpoint = {k: v for k, v in checkpoint.items() if not k.endswith('_all')}
    model.load_state_dict(checkpoint, strict=False)

    # ===== PROTOTYPE FUSION: Mix learned prototype with training image features =====
    prototype_lambda = getattr(args, 'prototype_lambda', 0.0)
    
    # Initialize fusion metrics (will be populated if fusion is used)
    fusion_metrics = {
        'lambda': prototype_lambda,
        'sim_before': None,  # similarity(p_learned, p_img_mean) 
        'sim_after': None,   # similarity(p_fused, p_img_mean)
        'sim_to_learned': None,  # similarity(p_fused, p_learned)
        'sim_individual_mean': None,  # mean similarity to each training image
        'sim_individual_std': None,   # std similarity to each training image
    }
    
    # Compute training image features if train_dataloader is provided
    if train_dataloader is not None:
        with torch.no_grad():
            train_cls_tokens = []
            for data, _, _, _, _ in train_dataloader:
                data = [model.transform(Image.fromarray(f.numpy())) for f in data]
                data = torch.stack(data, dim=0).to(device)
                cls_features, _, _, _ = model.encode_image(data)
                train_cls_tokens.append(cls_features)
            train_cls_tokens = torch.cat(train_cls_tokens, dim=0)  # [K, 640]
    
    use_prototype_fusion = (prototype_lambda > 0.0 and train_dataloader is not None)
    
    if use_prototype_fusion:
        print(f"\n{'='*80}")
        print(f"[Prototype Fusion] Enabled with λ={prototype_lambda}")
        print(f"{'='*80}")
        
        # Get original learned prototype
        p_learned = model.text_features[0].clone()  # [640]
        print(f"📊 Original learned prototype: {p_learned.shape}, norm={p_learned.norm():.4f}")
        
        # Calculate p_img from collected training tokens
        p_img = train_cls_tokens.mean(dim=0)  # [640]
        print(f"📊 Training image prototype (mean): {p_img.shape}, norm={p_img.norm():.4f}")
        
        # Calculate similarity before fusion (p_learned vs p_img)
        sim_before = F.cosine_similarity(p_learned.unsqueeze(0), p_img.unsqueeze(0)).item()
        print(f"📊 Similarity(p_learned, p_img_mean) = {sim_before:.4f}")
        fusion_metrics['sim_before'] = sim_before
        
        # Fuse prototypes: p_final = normalize((1-λ) * p_learned + λ * p_img)
        p_fused = (1 - prototype_lambda) * p_learned + prototype_lambda * p_img
        p_fused = F.normalize(p_fused, dim=0)
        print(f"📊 Fused prototype: {p_fused.shape}, norm={p_fused.norm():.4f}")
        
        # Calculate similarity after fusion (p_fused vs p_img and p_learned)
        sim_after = F.cosine_similarity(p_fused.unsqueeze(0), p_img.unsqueeze(0)).item()
        sim_to_learned = F.cosine_similarity(p_fused.unsqueeze(0), p_learned.unsqueeze(0)).item()
        print(f"📊 Similarity(p_fused, p_img_mean) = {sim_after:.4f}")
        print(f"📊 Similarity(p_fused, p_learned) = {sim_to_learned:.4f}")
        fusion_metrics['sim_after'] = sim_after
        fusion_metrics['sim_to_learned'] = sim_to_learned
        
        # Calculate similarity to individual training images
        sims_individual = F.cosine_similarity(
            p_fused.unsqueeze(0).expand(train_cls_tokens.shape[0], -1),
            train_cls_tokens,
            dim=1
        )  # [K]
        sim_individual_mean = sims_individual.mean().item()
        sim_individual_std = sims_individual.std().item()
        print(f"📊 Similarity to individual images: mean={sim_individual_mean:.4f}, std={sim_individual_std:.4f}")
        fusion_metrics['sim_individual_mean'] = sim_individual_mean
        fusion_metrics['sim_individual_std'] = sim_individual_std
        
        # Update model's normal prototype
        model.text_features[0] = p_fused
        print(f"✅ Normal prototype updated!\n")
    elif train_dataloader is not None and prototype_lambda == 0.0:
        # λ=0: No fusion, but still compute baseline similarity
        p_learned = model.text_features[0].clone()  # [640]
        p_img = train_cls_tokens.mean(dim=0)  # [640]
        
        sim_baseline = F.cosine_similarity(p_learned.unsqueeze(0), p_img.unsqueeze(0)).item()
        fusion_metrics['sim_before'] = sim_baseline
        fusion_metrics['sim_after'] = sim_baseline  # No change when λ=0
        fusion_metrics['sim_to_learned'] = 1.0  # p_fused = p_learned
        
        # Individual similarities
        sims_individual = F.cosine_similarity(
            p_learned.unsqueeze(0).expand(train_cls_tokens.shape[0], -1),
            train_cls_tokens,
            dim=1
        )
        fusion_metrics['sim_individual_mean'] = sims_individual.mean().item()
        fusion_metrics['sim_individual_std'] = sims_individual.std().item()
    else:
        if prototype_lambda > 0:
            print(f"⚠️  Prototype fusion disabled: train_dataloader not provided")
    
    scores_semantic = []
    scores_memory = []
    scores_fusion = []
    score_maps = []
    test_imgs = []
    gt_list = []
    gt_mask_list = []
    names = []
    
    # Get semantic alpha (scaling factor for semantic scores)
    semantic_alpha = getattr(args, 'semantic_alpha', 1.0)
    use_alpha_scale = (semantic_alpha != 1.0)
    
    if use_alpha_scale:
        print(f"[INFO] Semantic alpha scaling enabled: alpha={semantic_alpha}")
    
    # ===== MAIN INFERENCE LOOP =====
    for (data, mask, label, name, img_type) in dataloader:

        data = [model.transform(Image.fromarray(f.numpy())) for f in data]
        data = torch.stack(data, dim=0)

        for d, n, l, m in zip(data, name, label, mask):
            test_imgs += [denormalization(d.cpu().numpy())]
            l = l.numpy()
            m = m.numpy()
            m[m > 0] = 1

            names += [n]
            gt_list += [l]
            gt_mask_list += [m]

        data = data.to(device)
        
        # Get visual features
        visual_features = model.encode_image(data)
        
        # Calculate semantic scores
        semantic_scores = model.calculate_textual_anomaly_score(
            visual_features, 'cls'
        )
        
        # Calculate memory scores
        memory_scores = model.calculate_memory_image_score(visual_features)
        
        # Calculate score maps (for pixel-level evaluation)
        textual_anomaly_map = model.calculate_textual_anomaly_score(
            visual_features, 'seg'
        )
        visual_anomaly_map = model.calculate_visual_anomaly_score(visual_features)
        anomaly_map = torch.maximum(textual_anomaly_map, visual_anomaly_map)
        anomaly_map = torch.nn.functional.interpolate(
            anomaly_map, size=(model.out_size_h, model.out_size_w), 
            mode='bilinear', align_corners=False
        )
        am_pix = anomaly_map.squeeze(1).cpu().numpy()
        
        from scipy.ndimage import gaussian_filter
        am_pix_list = []
        for i in range(am_pix.shape[0]):
            am_pix[i] = gaussian_filter(am_pix[i], sigma=4)
            am_pix_list.append(am_pix[i])
        
        score_maps += am_pix_list
        scores_semantic += semantic_scores.tolist()
        scores_memory += memory_scores.tolist()

    test_imgs, score_maps, gt_mask_list = specify_resolution(test_imgs, score_maps, gt_mask_list,
                                                             resolution=(args.resolution, args.resolution))
    
    # Convert to numpy arrays
    semantic_img_scores = np.array(scores_semantic)
    memory_img_scores = np.array(scores_memory)
    
    # ===== FUSION: Baseline vs Weighted Harmonic Mean =====
    # Baseline fusion: standard harmonic mean (alpha=1.0)
    # Formula: 1 / (1/memory + 1/semantic)
    eps = 1e-10
    fusion_baseline = 1.0 / (1.0 / (memory_img_scores + eps) + 1.0 / (semantic_img_scores + eps))
    
    if use_alpha_scale:
        # Weighted harmonic mean: 1 / (1/memory + alpha/semantic)
        # - alpha=0: fusion = memory (ignore semantic)
        # - alpha=1: fusion = baseline (equal weights)
        # - alpha>1: semantic has MORE weight
        # - alpha<1: memory has MORE weight
        fusion_weighted = 1.0 / (1.0 / (memory_img_scores + eps) + semantic_alpha / (semantic_img_scores + eps))
        
        print(f"[INFO] Weighted harmonic mean: 1/(1/memory + {semantic_alpha}/semantic)")
        print(f"[INFO] Interpretation: alpha={semantic_alpha} → semantic weight = {semantic_alpha:.2f}, memory weight = 1.00")
        
        # Use weighted fusion as final score
        fusion_img_scores = fusion_weighted
    else:
        # Use baseline fusion
        fusion_img_scores = fusion_baseline
    
    # Calculate metrics for each branch
    from utils.metrics import metric_cal_img_only
    result_semantic = metric_cal_img_only(semantic_img_scores, gt_list)
    result_memory = metric_cal_img_only(memory_img_scores, gt_list)
    result_fusion = metric_cal_img_only(fusion_img_scores, gt_list)
    
    # Classification task: only image-level metrics (no p_roc)
    result_dict = {
        'i_roc': result_fusion['i_roc'],
        'semantic_i_roc': result_semantic['i_roc'],
        'memory_i_roc': result_memory['i_roc'],
        'fusion_i_roc': result_fusion['i_roc'],
        # Add fusion metrics
        'prototype_lambda': fusion_metrics['lambda'],
        'sim_before': fusion_metrics['sim_before'],
        'sim_after': fusion_metrics['sim_after'],
        'sim_to_learned': fusion_metrics['sim_to_learned'],
        'sim_individual_mean': fusion_metrics['sim_individual_mean'],
        'sim_individual_std': fusion_metrics['sim_individual_std'],
    }
    
    # If alpha weighting enabled, also compute baseline metrics for comparison
    if use_alpha_scale:
        result_baseline = metric_cal_img_only(fusion_baseline, gt_list)
        result_dict['fusion_baseline_i_roc'] = result_baseline['i_roc']
        result_dict['fusion_weighted_i_roc'] = result_fusion['i_roc']
        delta_roc = result_fusion['i_roc'] - result_baseline['i_roc']
        result_dict['delta_i_roc'] = delta_roc
        
        print(f"\n[Weighted Harmonic Mean Results] (alpha={semantic_alpha})")
        print(f"  Baseline Fusion AUROC: {result_baseline['i_roc']:.4f}")
        print(f"  Weighted Fusion AUROC: {result_fusion['i_roc']:.4f}")
        print(f"  Δ AUROC (weighted - baseline): {delta_roc:+.4f}")
    
    # Visualization (optional - can be implemented if needed)
    # if args.vis:
    #     pass

    return result_dict


def main(args):
    kwargs = vars(args)

    if kwargs['seed'] is None:
        kwargs['seed'] = 222

    setup_seed(kwargs['seed'])

    if kwargs['use_cpu'] == 0:
        device = f"cuda:0"
    else:
        device = f"cpu"
    kwargs['device'] = device

    # Determine paths based on mode
    dataset = kwargs['dataset']
    k_shot = kwargs['k_shot']
    seed = kwargs['seed']
    class_name = kwargs['class_name']
    semantic_alpha = kwargs.get('semantic_alpha', 1.0)
    output_dir = kwargs.get('output_dir', None)
    
    # Load checkpoint from promptpurging (contains Purge3 trained weights)
    baseline_root = "./result/promptpurging"
    check_path = f"{baseline_root}/{dataset}/k_{k_shot}/checkpoint/CLS-Seed_{seed}-{class_name}-check_point.pt"
    
    # Determine output directory
    if output_dir:
        # User specified output directory
        custom_root = f"{output_dir}/{dataset}/k_{k_shot}"
        os.makedirs(f"{custom_root}/csv", exist_ok=True)
        os.makedirs(f"{custom_root}/images", exist_ok=True)
        img_dir = f"{custom_root}/images"
        csv_path = f"{custom_root}/csv/Seed_{seed}-results.csv"
        print(f"[INFO] Using custom output directory: {custom_root}")
        print(f"[INFO] Checkpoint loaded from: {check_path}")
    elif semantic_alpha != 1.0:
        # Alpha scaling mode
        alpha_root = f"./result/alpha_scale/{dataset}/k_{k_shot}"
        os.makedirs(f"{alpha_root}/csv", exist_ok=True)
        os.makedirs(f"{alpha_root}/images", exist_ok=True)
        img_dir = f"{alpha_root}/images"
        csv_path = f"{alpha_root}/csv/{dataset}.csv"
        print(f"[INFO] Alpha scaling mode: Results will be saved to {alpha_root}")
        print(f"[INFO] Checkpoint loaded from: {check_path}")
    else:
        # Default mode: use standard paths
        img_dir, csv_path, _ = get_dir_from_args(TASK, **kwargs)
    
    # Override checkpoint path if explicitly specified
    if kwargs.get('checkpoint') is not None:
        # Direct checkpoint file path provided
        check_path = kwargs['checkpoint']
        print(f"[INFO] Using checkpoint from: {check_path}")
    elif kwargs.get('checkpoint_dir') is not None:
        # Checkpoint directory provided
        check_path = f"{kwargs['checkpoint_dir']}/checkpoint/CLS-Seed_{kwargs['seed']}-{kwargs['class_name']}-check_point.pt"
        print(f"[INFO] Using checkpoint from: {check_path}")

    # get the test dataloader (force num_workers=0 for compatibility)
    kwargs_loader = kwargs.copy()
    kwargs_loader['num_workers'] = 0
    test_dataloader, test_dataset_inst = get_dataloader_from_args(phase='test', perturbed=False, **kwargs_loader)
    
    # Get training dataloader if:
    # 1. Prototype fusion is enabled (lambda > 0), OR
    # 2. Custom output directory specified (for λ sweep experiments, need baseline similarity)
    train_dataloader = None
    prototype_lambda = kwargs.get('prototype_lambda', 0.0)
    need_train_data = (prototype_lambda >= 0.0 and output_dir is not None) or prototype_lambda > 0.0
    
    if need_train_data:
        train_dataloader, _ = get_dataloader_from_args(phase='train', perturbed=False, **kwargs_loader)
        print(f"[INFO] Training dataloader loaded (λ={prototype_lambda})")

    kwargs['out_size_h'] = kwargs['resolution']
    kwargs['out_size_w'] = kwargs['resolution']

    # get the model
    model = PromptAD(**kwargs)
    model = model.to(device)

    # as the pro metric calculation is costly, we only calculate it in the last evaluation
    metrics = test(model, args, test_dataloader, device, img_dir=img_dir, check_path=check_path, 
                   train_dataloader=train_dataloader)

    fusion_roc = round(metrics['fusion_i_roc'], 2)
    semantic_roc = round(metrics['semantic_i_roc'], 2)
    memory_roc = round(metrics['memory_i_roc'], 2)
    object = kwargs['class_name']
    
    # Print results based on mode
    semantic_alpha = kwargs.get('semantic_alpha', 1.0)
    if semantic_alpha != 1.0:
        baseline_roc = round(metrics.get('fusion_baseline_i_roc', fusion_roc), 2)
        scaled_roc = round(metrics.get('fusion_scaled_i_roc', fusion_roc), 2)
        delta = round(metrics.get('delta_i_roc', 0), 4)
        print(f'Object:{object} === [Alpha={semantic_alpha}] Baseline:{baseline_roc}, Scaled:{scaled_roc}, Δ:{delta:+.4f} | Semantic:{semantic_roc}, Memory:{memory_roc}\n')
    else:
        print(f'Object:{object} =========================== Fusion-AUROC:{fusion_roc}, Semantic:{semantic_roc}, Memory:{memory_roc}\n')

    save_metric(metrics, dataset_classes[kwargs['dataset']], kwargs['class_name'],
                kwargs['dataset'], csv_path)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_args():
    parser = argparse.ArgumentParser(description='Anomaly detection')
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa'])
    parser.add_argument('--class_name', type=str, default='carpet')

    parser.add_argument('--img-resize', type=int, default=240)
    parser.add_argument('--img-cropsize', type=int, default=240)
    parser.add_argument('--resolution', type=int, default=400)

    parser.add_argument('--batch-size', type=int, default=400)
    parser.add_argument('--vis', type=str2bool, choices=[True, False], default=True)
    parser.add_argument("--root-dir", type=str, default="./result")
    parser.add_argument("--load-memory", type=str2bool, default=True)
    parser.add_argument("--cal-pro", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--gpu-id", type=int, default=0)

    # pure test
    parser.add_argument("--pure-test", type=str2bool, default=False)

    # method related parameters
    parser.add_argument('--k-shot', type=int, default=1)
    parser.add_argument("--backbone", type=str, default="ViT-B-16-plus-240",
                        choices=['ViT-B-16-plus-240', 'ViT-B-16'])
    parser.add_argument("--pretrained_dataset", type=str, default="laion400m_e32")
    parser.add_argument("--version", type=str, default='')

    parser.add_argument("--use-cpu", type=int, default=0)

    # prompt tuning hyper-parameter
    parser.add_argument("--n_ctx", type=int, default=4)
    parser.add_argument("--n_ctx_ab", type=int, default=1)
    parser.add_argument("--n_pro", type=int, default=1)
    parser.add_argument("--n_pro_ab", type=int, default=4)

    # semantic alpha scaling parameter
    parser.add_argument("--semantic-alpha", type=float, default=1.0,
                        help="Semantic score scaling factor (default: 1.0 = no scaling)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory for alpha scaling results (e.g., ./result/test_alpha)")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Override checkpoint directory (default: uses baseline checkpoint)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Direct path to checkpoint file (overrides --checkpoint-dir)")
    
    # prototype fusion parameter
    parser.add_argument("--prototype-lambda", type=float, default=0.0,
                        help="Prototype fusion weight: p_final = (1-λ)*p_learned + λ*p_img (default: 0.0 = no fusion)")
    


    args = parser.parse_args()

    return args


if __name__ == '__main__':
    import os

    args = get_args()
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu_id}"
    main(args)
