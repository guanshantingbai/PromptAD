import argparse

import torch
import torch.optim.lr_scheduler
import torch.nn.functional as F

from datasets import *
from datasets import dataset_classes
from utils.csv_utils import *
from utils.metrics import *
from utils.training_utils import *
from PromptAD import *
from utils.eval_utils import *
from torchvision import transforms
from tqdm import tqdm

TASK = 'SEG'

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
        print(f"[Prototype Fusion - SEG] Enabled with λ={prototype_lambda}")
        print(f"{'='*80}")
        
        # Get original learned prototype
        p_learned = model.text_features[0].clone()  # [640]
        print(f"📊 Original learned prototype: {p_learned.shape}, norm={p_learned.norm():.4f}")
        
        # Calculate p_img from collected training tokens
        p_img = train_cls_tokens.mean(dim=0)  # [640]
        print(f"📊 Training image prototype (mean): {p_img.shape}, norm={p_img.norm():.4f}")
        
        # Calculate similarity before fusion
        sim_before = F.cosine_similarity(p_learned.unsqueeze(0), p_img.unsqueeze(0)).item()
        print(f"📊 Similarity(p_learned, p_img_mean) = {sim_before:.4f}")
        
        # Fuse prototypes: p_final = normalize((1-λ) * p_learned + λ * p_img)
        p_fused = (1 - prototype_lambda) * p_learned + prototype_lambda * p_img
        p_fused = F.normalize(p_fused, dim=0)
        print(f"📊 Fused prototype: {p_fused.shape}, norm={p_fused.norm():.4f}")
        
        # Calculate similarity after fusion
        sim_after = F.cosine_similarity(p_fused.unsqueeze(0), p_img.unsqueeze(0)).item()
        sim_to_learned = F.cosine_similarity(p_fused.unsqueeze(0), p_learned.unsqueeze(0)).item()
        print(f"📊 Similarity(p_fused, p_img_mean) = {sim_after:.4f}")
        print(f"📊 Similarity(p_fused, p_learned) = {sim_to_learned:.4f}")
        
        # Update model's normal prototype
        model.text_features[0] = p_fused
        print(f"✅ Normal prototype updated!\n")

    score_maps = []
    test_imgs = []
    gt_mask_list = []
    names = []

    for (data, mask, label, name, img_type) in dataloader:

        data = [model.transform(Image.fromarray(f.numpy())) for f in data]
        data = torch.stack(data, dim=0)

        for d, n, l, m in zip(data, name, label, mask):
            test_imgs += [denormalization(d.cpu().numpy())]
            m = m.numpy()
            m[m > 0] = 1

            names += [n]
            gt_mask_list += [m]

        data = data.to(device)
        score_map = model(data, 'seg')
        score_maps += score_map

    test_imgs, score_maps, gt_mask_list = specify_resolution(test_imgs, score_maps, gt_mask_list, resolution=(args.resolution, args.resolution))
    result_dict = metric_cal_pix(np.array(score_maps), gt_mask_list)

    torch.save(model.state_dict(), check_path)
    if args.vis:
        plot_sample_cv2(names, test_imgs, {'PromptAD': score_maps}, gt_mask_list, save_folder=img_dir)

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

    # prepare the experiment dir
    img_dir, csv_path, check_path = get_dir_from_args(TASK, **kwargs)

    # get the test dataloader (force num_workers=0 for compatibility)
    kwargs_loader = kwargs.copy()
    kwargs_loader['num_workers'] = 0
    test_dataloader, test_dataset_inst = get_dataloader_from_args(phase='test', perturbed=False, **kwargs_loader)
    
    # Get training dataloader if prototype fusion is enabled
    train_dataloader = None
    prototype_lambda = kwargs.get('prototype_lambda', 0.0)
    if prototype_lambda > 0.0:
        train_dataloader, _ = get_dataloader_from_args(phase='train', perturbed=False, **kwargs_loader)
        print(f"[INFO] Training dataloader loaded for prototype fusion (λ={prototype_lambda})")

    kwargs['out_size_h'] = kwargs['resolution']
    kwargs['out_size_w'] = kwargs['resolution']

    # get the model
    model = PromptAD(**kwargs)
    model = model.to(device)

    # as the pro metric calculation is costly, we only calculate it in the last evaluation
    metrics = test(model, args, test_dataloader, device, img_dir=img_dir, check_path=check_path, 
                   train_dataloader=train_dataloader)

    p_roc = round(metrics['p_roc'], 2)
    object = kwargs['class_name']
    print(f'Object:{object} =========================== Pixel-AUROC:{p_roc}\n')

    save_metric(metrics, dataset_classes[kwargs['dataset']], kwargs['class_name'],
                kwargs['dataset'], csv_path)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_args():
    parser = argparse.ArgumentParser(description='Anomaly detection')
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa'])
    parser.add_argument('--class_name', type=str, default='transistor')

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
    
    # Multi-Abnormal Prototypes inference parameters
    parser.add_argument("--topk-abnormal", type=int, default=None,
                        help="Top-k aggregation for abnormal prototypes (k=1: max, k>1: top-k mean, None: mean baseline)")
    
    # Prototype Fusion parameters
    parser.add_argument("--prototype-lambda", type=float, default=0.0,
                        help="Fusion weight λ for mixing learned prototype with training images (0.0=learned only, 1.0=images only)")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    import os

    args = get_args()
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu_id}"
    main(args)
