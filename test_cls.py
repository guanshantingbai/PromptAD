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
        ):

    # change the model into eval mode
    model.eval_mode()

    model.load_state_dict(torch.load(check_path), strict=False)

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
        semantic_scores = model.calculate_textual_anomaly_score(visual_features, 'cls')
        
        # Calculate memory scores
        memory_scores = model.calculate_memory_image_score(visual_features)
        
        # Calculate score maps (for pixel-level evaluation)
        textual_anomaly_map = model.calculate_textual_anomaly_score(visual_features, 'seg')
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
        'fusion_i_roc': result_fusion['i_roc']
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
    
    # Override checkpoint path if explicitly specified via checkpoint_dir argument
    if kwargs['checkpoint_dir'] is not None:
        check_path = f"{kwargs['checkpoint_dir']}/checkpoint/CLS-Seed_{kwargs['seed']}-{kwargs['class_name']}-check_point.pt"
        print(f"[INFO] Using checkpoint from: {check_path}")

    # get the test dataloader (force num_workers=0 for compatibility)
    kwargs_loader = kwargs.copy()
    kwargs_loader['num_workers'] = 0
    test_dataloader, test_dataset_inst = get_dataloader_from_args(phase='test', perturbed=False, **kwargs_loader)

    kwargs['out_size_h'] = kwargs['resolution']
    kwargs['out_size_w'] = kwargs['resolution']

    # get the model
    model = PromptAD(**kwargs)
    model = model.to(device)

    # as the pro metric calculation is costly, we only calculate it in the last evaluation
    metrics = test(model, args, test_dataloader, device, img_dir=img_dir, check_path=check_path)

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

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    import os

    args = get_args()
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu_id}"
    main(args)
