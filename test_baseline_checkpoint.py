"""
Use baseline checkpoint for inference with diagnostics.
Handles checkpoint compatibility (baseline may have feature_gallery keys).
"""

import argparse
import torch
import json
from pathlib import Path

from datasets import *
from datasets import dataset_classes
from utils.csv_utils import *
from utils.metrics import *
from utils.training_utils import *
from PromptAD import *
from utils.eval_utils import *
from utils.prototype_diagnostics import PrototypeDiagnostics
from torchvision import transforms
from tqdm import tqdm

TASK = 'CLS'


def clean_checkpoint(checkpoint_path):
    """
    Load checkpoint and remove incompatible keys (feature_gallery).
    Returns cleaned state dict with only text prototypes.
    """
    checkpoint = torch.load(checkpoint_path)
    
    # Map old keys to new keys
    key_mapping = {
        'text_features': None,  # Remove this, will be rebuilt
        'feature_gallery1': None,  # Remove visual gallery
        'feature_gallery2': None,  # Remove visual gallery
    }
    
    cleaned = {}
    for k, v in checkpoint.items():
        if k in key_mapping:
            if key_mapping[k] is not None:
                cleaned[key_mapping[k]] = v
            # else: skip this key
        else:
            cleaned[k] = v
    
    print(f"Original checkpoint keys: {list(checkpoint.keys())}")
    print(f"Cleaned checkpoint keys: {list(cleaned.keys())}")
    
    return cleaned


def test_with_baseline_checkpoint(
    model,
    args,
    dataloader: DataLoader,
    device: str,
    checkpoint_path: str,
):
    """Test using baseline checkpoint."""
    
    # change the model into eval mode
    model.eval_mode()
    
    # Clean and load checkpoint
    print(f"\n{'='*80}")
    print(f"Loading baseline checkpoint: {checkpoint_path}")
    print(f"{'='*80}")
    
    if Path(checkpoint_path).exists():
        cleaned_checkpoint = clean_checkpoint(checkpoint_path)
        model.load_state_dict(cleaned_checkpoint, strict=False)
        print(f"✓ Checkpoint loaded (strict=False to skip missing keys)")
    else:
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        print(f"  Please check the path!")
        return None
    
    # Build text prototypes from current prompt learner
    print(f"\nRebuilding text prototypes from prompt learner...")
    model.build_text_feature_gallery()
    print(f"✓ Text prototypes rebuilt")
    print(f"  Normal prototypes shape: {model.normal_prototypes.shape}")
    print(f"  Abnormal prototypes shape: {model.abnormal_prototypes.shape}")
    
    # Initialize diagnostics
    diagnostics = PrototypeDiagnostics()
    
    scores_img = []
    score_maps = []
    test_imgs = []
    gt_list = []
    gt_mask_list = []
    names = []

    print(f"\n{'='*80}")
    print(f"Running inference with diagnostics...")
    print(f"{'='*80}\n")
    
    for (data, mask, label, name, img_type) in tqdm(dataloader, desc="Processing batches"):
        
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
        
        # ===== DIAGNOSTICS: Collect statistics during inference =====
        with torch.no_grad():
            visual_features = model.encode_image(data)
            diagnostics.update_from_inference(
                visual_features=visual_features,
                normal_prototypes=model.normal_prototypes,
                abnormal_prototypes=model.abnormal_prototypes,
                logit_scale=model.model.logit_scale
            )
        
        # Continue normal inference
        score_img, score_map = model(data, 'cls')
        score_maps += score_map
        scores_img += score_img

    test_imgs, score_maps, gt_mask_list = specify_resolution(
        test_imgs, score_maps, gt_mask_list,
        resolution=(args.resolution, args.resolution)
    )
    result_dict = metric_cal_img(np.array(scores_img), gt_list, np.array(score_maps))
    
    # ===== DIAGNOSTICS: Print summary =====
    diagnostics.print_summary(
        normal_prototypes=model.normal_prototypes,
        abnormal_prototypes=model.abnormal_prototypes
    )
    
    # ===== DIAGNOSTICS: Save statistics to file =====
    if args.save_diagnostics:
        stats = diagnostics.export_statistics(
            normal_prototypes=model.normal_prototypes,
            abnormal_prototypes=model.abnormal_prototypes
        )
        
        # Save to baseline diagnostics folder
        diag_dir = Path("result/baseline_diagnostics")
        diag_dir.mkdir(exist_ok=True)
        diag_file = diag_dir / f"{args.dataset}_{args.class_name}_k{args.k_shot}_diagnostics.json"
        
        with open(diag_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nDiagnostics saved to: {diag_file}")
    
    return result_dict


def main(args):
    kwargs = vars(args)

    if kwargs['seed'] is None:
        kwargs['seed'] = 111

    setup_seed(kwargs['seed'])

    if kwargs['use_cpu'] == 0:
        device = f"cuda:0"
    else:
        device = f"cpu"
    kwargs['device'] = device

    # Construct baseline checkpoint path
    checkpoint_path = (
        f"result/baseline/{args.dataset}/k_{args.k_shot}/checkpoint/"
        f"{TASK}-Seed_{args.seed}-{args.class_name}-check_point.pt"
    )

    # get the test dataloader (without transform, will apply in loop)
    test_dataloader, test_dataset_inst = get_dataloader_from_args(
        phase='test', perturbed=False, **kwargs
    )

    kwargs['out_size_h'] = kwargs['resolution']
    kwargs['out_size_w'] = kwargs['resolution']

    # get the model
    model = PromptAD(**kwargs)
    model = model.to(device)

    # Run test with baseline checkpoint
    metrics = test_with_baseline_checkpoint(
        model, args, test_dataloader, device, 
        checkpoint_path=checkpoint_path
    )
    
    if metrics is None:
        print("\n✗ Failed to run inference. Please check checkpoint path.")
        return

    p_roc = round(metrics['i_roc'], 2)
    object_name = kwargs['class_name']
    print(f'\n{"="*80}')
    print(f'Object: {object_name} | Image-AUROC: {p_roc}')
    print(f'{"="*80}\n')

    # Save results to baseline_results folder
    result_dir = Path("result/baseline_results")
    result_dir.mkdir(exist_ok=True)
    csv_path = result_dir / f"{args.dataset}_k{args.k_shot}_results.csv"
    
    save_metric(metrics, dataset_classes[kwargs['dataset']], kwargs['class_name'],
                kwargs['dataset'], str(csv_path))


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_args():
    parser = argparse.ArgumentParser(description='Test with baseline checkpoint')
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa'])
    parser.add_argument('--class_name', type=str, default='carpet')

    parser.add_argument('--img-resize', type=int, default=240)
    parser.add_argument('--img-cropsize', type=int, default=240)
    parser.add_argument('--resolution', type=int, default=400)

    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--vis', type=str2bool, choices=[True, False], default=True)
    parser.add_argument("--load-memory", type=str2bool, default=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cal-pro", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--gpu-id", type=int, default=0)

    # method related parameters
    parser.add_argument('--k-shot', type=int, default=2)
    parser.add_argument("--backbone", type=str, default="ViT-B-16-plus-240",
                        choices=['ViT-B-16-plus-240', 'ViT-B-16'])
    parser.add_argument("--pretrained_dataset", type=str, default="laion400m_e32")

    parser.add_argument("--use-cpu", type=int, default=0)

    # prompt tuning hyper-parameter (should match baseline training)
    parser.add_argument("--n_ctx", type=int, default=4)
    parser.add_argument("--n_ctx_ab", type=int, default=1)
    parser.add_argument("--n_pro", type=int, default=1)
    parser.add_argument("--n_pro_ab", type=int, default=4)

    # Diagnostics
    parser.add_argument("--save-diagnostics", type=str2bool, default=True,
                        help="Save diagnostic statistics to JSON file")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    import os

    args = get_args()
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu_id}"
    main(args)
