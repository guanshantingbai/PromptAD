import argparse
import torch
import torch.optim.lr_scheduler
from datasets import *
from datasets import dataset_classes
from utils.csv_utils import *
from utils.metrics import *
from utils.training_utils import *
from PromptAD import *
from utils.eval_utils import *

TASK = 'CLS'

def test_alpha_grid(model, args, dataloader, device, img_dir, check_path, alpha_values):
    """
    Test with multiple alpha values for semantic weight scaling.
    
    Args:
        alpha_values: List of alpha values to test, e.g., [0, 0.1, 0.2, ..., 2.0]
    
    Returns:
        result_dict: Dictionary containing metrics for each alpha
    """
    model.eval_mode()
    model.load_state_dict(torch.load(check_path), strict=False)

    scores_semantic = []
    scores_memory = []
    score_maps = []
    test_imgs = []
    gt_list = []
    gt_mask_list = []
    names = []

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
        semantic_scores, memory_scores, fusion_scores, score_map = model(data, 'cls')
        
        score_maps += score_map
        scores_semantic += semantic_scores
        scores_memory += memory_scores

    test_imgs, score_maps, gt_mask_list = specify_resolution(
        test_imgs, score_maps, gt_mask_list,
        resolution=(args.resolution, args.resolution)
    )
    
    # Convert to numpy arrays
    semantic_img_scores = np.array(scores_semantic)
    memory_img_scores = np.array(scores_memory)
    
    # Calculate base metrics
    from utils.metrics import metric_cal_img_only, metric_cal_img
    result_semantic = metric_cal_img_only(semantic_img_scores, gt_list)
    result_memory = metric_cal_img_only(memory_img_scores, gt_list)
    
    # Use first alpha for pixel-level metrics (doesn't matter which)
    baseline_fusion = 1.0 / (1.0 / semantic_img_scores + 1.0 / memory_img_scores)
    result_dict = metric_cal_img(baseline_fusion, gt_list, np.array(score_maps))
    
    # Store base metrics
    result_dict['semantic_i_roc'] = result_semantic['i_roc']
    result_dict['memory_i_roc'] = result_memory['i_roc']
    
    # Test each alpha value
    for alpha in alpha_values:
        # Apply alpha scaling to semantic scores
        semantic_scaled = alpha * semantic_img_scores
        
        # Compute fusion with scaled semantic
        # Add epsilon to avoid division by zero
        eps = 1e-10
        fusion_alpha = 1.0 / (1.0 / (semantic_scaled + eps) + 1.0 / (memory_img_scores + eps))
        
        # Calculate fusion AUROC
        result_fusion_alpha = metric_cal_img_only(fusion_alpha, gt_list)
        
        # Store with alpha-specific key
        alpha_key = f'alpha_{alpha:.1f}_fusion_i_roc'
        result_dict[alpha_key] = result_fusion_alpha['i_roc']
    
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

    # Define alpha values to test
    alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0]
    
    # Prepare experiment directories
    dataset = kwargs['dataset']
    k_shot = kwargs['k_shot']
    seed = kwargs['seed']
    class_name = kwargs['class_name']
    
    # Output to result/baseline/alpha
    output_root = f"./result/baseline/alpha/{dataset}/k_{k_shot}"
    os.makedirs(f"{output_root}/csv", exist_ok=True)
    os.makedirs(f"{output_root}/images", exist_ok=True)
    
    img_dir = f"{output_root}/images"
    csv_path = f"{output_root}/csv/{dataset}.csv"
    
    # Load checkpoint from baseline
    check_path = f"./result/baseline/{dataset}/k_{k_shot}/checkpoint/CLS-Seed_{seed}-{class_name}-check_point.pt"
    
    print(f"[INFO] Alpha Grid Search Mode")
    print(f"[INFO] Testing alpha values: {alpha_values}")
    print(f"[INFO] Results will be saved to {output_root}")
    print(f"[INFO] Checkpoint loaded from: {check_path}")

    # Get test dataloader
    kwargs_loader = kwargs.copy()
    kwargs_loader['num_workers'] = 0
    test_dataloader, test_dataset_inst = get_dataloader_from_args(
        phase='test', perturbed=False, **kwargs_loader
    )

    kwargs['out_size_h'] = kwargs['resolution']
    kwargs['out_size_w'] = kwargs['resolution']

    # Get model
    model = PromptAD(**kwargs)
    model = model.to(device)

    # Run test with alpha grid
    metrics = test_alpha_grid(
        model, args, test_dataloader, device, 
        img_dir=img_dir, check_path=check_path,
        alpha_values=alpha_values
    )

    # Print results
    semantic_roc = round(metrics['semantic_i_roc'], 2)
    memory_roc = round(metrics['memory_i_roc'], 2)
    
    print(f'\n[Alpha Grid Search Results] Class: {class_name}')
    print(f'  Semantic AUROC: {semantic_roc}')
    print(f'  Memory AUROC:   {memory_roc}')
    print(f'\n  Alpha-weighted Fusion AUROC:')
    
    for alpha in alpha_values:
        alpha_key = f'alpha_{alpha:.1f}_fusion_i_roc'
        auroc = round(metrics[alpha_key], 2)
        print(f'    α={alpha:.1f}: {auroc}')

    # Save metrics
    save_metric(metrics, dataset_classes[kwargs['dataset']], kwargs['class_name'],
                kwargs['dataset'], csv_path)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_args():
    parser = argparse.ArgumentParser(description='Alpha Grid Search for Semantic Weight')
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa'])
    parser.add_argument('--class_name', type=str, default='carpet')

    parser.add_argument('--img-resize', type=int, default=240)
    parser.add_argument('--img-cropsize', type=int, default=240)
    parser.add_argument('--resolution', type=int, default=400)

    parser.add_argument('--batch-size', type=int, default=400)
    parser.add_argument('--vis', type=str2bool, choices=[True, False], default=True)
    parser.add_argument("--load-memory", type=str2bool, default=True)
    parser.add_argument("--cal-pro", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--pure-test", type=str2bool, default=False)

    # Method related parameters
    parser.add_argument('--k-shot', type=int, default=2)
    parser.add_argument("--backbone", type=str, default="ViT-B-16-plus-240",
                        choices=['ViT-B-16-plus-240', 'ViT-B-16'])
    parser.add_argument("--pretrained_dataset", type=str, default="laion400m_e32")
    parser.add_argument("--version", type=str, default='')
    parser.add_argument("--use-cpu", type=int, default=0)

    # Prompt tuning hyper-parameters
    parser.add_argument("--n_ctx", type=int, default=4)
    parser.add_argument("--n_ctx_ab", type=int, default=1)
    parser.add_argument("--n_pro", type=int, default=1)
    parser.add_argument("--n_pro_ab", type=int, default=4)
    
    # MAP/LAP control
    parser.add_argument("--use-lap", type=str2bool, default=True,
                        help="Use LAP (Least Anomalous Patches). Set False for MAP-only mode.")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import os
    args = get_args()
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu_id}"
    main(args)
