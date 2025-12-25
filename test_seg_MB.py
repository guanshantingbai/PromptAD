"""
test_seg_MB.py - Memory Bank Only Version for Segmentation
This version uses ONLY the visual branch (memory bank) without semantic branch.
"""
import argparse

import torch.optim.lr_scheduler

from datasets import *
from datasets import dataset_classes
from utils.csv_utils import *
from utils.metrics import *
from utils.training_utils import *
from PromptAD import *
from utils.eval_utils import *
from torchvision import transforms
from tqdm import tqdm

TASK = 'SEG_MB'  # Memory Bank only


def build_memory_bank(model, train_dataloader, device):
    """Build memory bank from training data (visual features only)"""
    model.eval_mode()
    
    features1 = []
    features2 = []
    
    print("Building memory bank from training data...")
    for (data, mask, label, name, img_type) in tqdm(train_dataloader, desc="Extracting features"):
        data = data.to(device)
        _, _, feature_map1, feature_map2 = model.encode_image(data)
        features1.append(feature_map1)
        features2.append(feature_map2)
    
    features1 = torch.cat(features1, dim=0)
    features2 = torch.cat(features2, dim=0)
    model.build_image_feature_gallery(features1, features2)
    
    print(f"Memory bank built: feature1 shape={features1.shape}, feature2 shape={features2.shape}")


def test_visual_only(model,
                     args,
                     test_dataloader: DataLoader,
                     device: str,
                     img_dir: str,
                     ):
    """Test using visual branch only (memory bank)"""
    
    model.eval_mode()
    
    score_maps = []
    test_imgs = []
    gt_mask_list = []
    names = []

    print("Testing with visual branch only...")
    for (data, mask, label, name, img_type) in tqdm(test_dataloader, desc="Testing"):
        data = data.to(device)
        
        for d, n, l, m in zip(data, name, label, mask):
            if args.vis:
                test_imgs.append(denormalization(d.cpu().numpy()))
            m = m.cpu().numpy() if torch.is_tensor(m) else m
            m[m > 0] = 1
            names.append(n)
            gt_mask_list.append(m)
        
        # Use only visual anomaly score
        visual_features = model.encode_image(data)
        visual_anomaly_map = model.calculate_visual_anomaly_score(visual_features)
        
        # Interpolate to target resolution
        anomaly_map = F.interpolate(visual_anomaly_map, 
                                    size=(model.out_size_h, model.out_size_w), 
                                    mode='bilinear', 
                                    align_corners=False)
        
        am_pix = anomaly_map.squeeze(1).cpu().numpy()
        
        # Apply Gaussian filter
        from scipy.ndimage import gaussian_filter
        for i in range(am_pix.shape[0]):
            am_pix[i] = gaussian_filter(am_pix[i], sigma=4)
            score_maps.append(am_pix[i])

    # Resize for evaluation
    import cv2
    gt_mask_list = [cv2.resize(mask, (args.resolution, args.resolution), 
                               interpolation=cv2.INTER_NEAREST) for mask in gt_mask_list]
    if args.vis:
        test_imgs = [cv2.resize(img, (args.resolution, args.resolution), 
                               interpolation=cv2.INTER_CUBIC) for img in test_imgs]
    
    result_dict = metric_cal_pix(np.array(score_maps), gt_mask_list)
    
    if args.vis:
        plot_sample_cv2(names, test_imgs, {'PromptAD_MB': score_maps}, 
                       gt_mask_list, save_folder=img_dir)
    
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

    # Prepare the experiment dir with MB suffix
    kwargs['version'] = kwargs.get('version', '') + '_MB'
    img_dir, csv_path, _ = get_dir_from_args(TASK, **kwargs)

    # Get the model
    kwargs['out_size_h'] = kwargs['resolution']
    kwargs['out_size_w'] = kwargs['resolution']
    
    model = PromptAD(**kwargs)
    model = model.to(device)

    # Get train dataloader to build memory bank
    train_dataloader, _ = get_dataloader_from_args(phase='train', 
                                                    perturbed=False, 
                                                    transform=model.transform, 
                                                    **kwargs)
    
    # Get test dataloader
    test_dataloader, _ = get_dataloader_from_args(phase='test', 
                                                   perturbed=False, 
                                                   transform=model.transform, 
                                                   **kwargs)

    # Build memory bank from training data
    build_memory_bank(model, train_dataloader, device)
    
    # Test using visual branch only
    metrics = test_visual_only(model, args, test_dataloader, device, img_dir=img_dir)

    p_roc = round(metrics['p_roc'], 2)
    object = kwargs['class_name']
    print(f'Object:{object} =========================== Pixel-AUROC:{p_roc} (Visual Branch Only)\n')

    save_metric(metrics, dataset_classes[kwargs['dataset']], kwargs['class_name'],
                kwargs['dataset'], csv_path)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_args():
    parser = argparse.ArgumentParser(description='Anomaly detection - Memory Bank Only')
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa'])
    parser.add_argument('--class_name', type=str, default='carpet')

    parser.add_argument('--img-resize', type=int, default=240)
    parser.add_argument('--img-cropsize', type=int, default=240)
    parser.add_argument('--resolution', type=int, default=240)

    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--vis', type=str2bool, choices=[True, False], default=False)
    parser.add_argument("--root-dir", type=str, default="./result_MB")
    parser.add_argument("--load-memory", type=str2bool, default=True)
    parser.add_argument("--cal-pro", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--gpu-id", type=int, default=0)

    # Method related parameters
    parser.add_argument('--k-shot', type=int, default=1)
    parser.add_argument("--backbone", type=str, default="ViT-B-16-plus-240",
                        choices=['ViT-B-16-plus-240', 'ViT-B-16'])
    parser.add_argument("--pretrained_dataset", type=str, default="laion400m_e32")
    parser.add_argument("--version", type=str, default='')

    parser.add_argument("--use-cpu", type=int, default=0)

    # Prompt tuning hyper-parameter (not used but needed for model initialization)
    parser.add_argument("--n_ctx", type=int, default=4)
    parser.add_argument("--n_ctx_ab", type=int, default=1)
    parser.add_argument("--n_pro", type=int, default=1)
    parser.add_argument("--n_pro_ab", type=int, default=4)
    
    # Dataloader configuration
    parser.add_argument("--num-workers", type=int, default=4,
                        help='Number of data loading workers')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    import os

    args = get_args()
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu_id}"
    main(args)
