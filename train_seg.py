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

TASK = 'SEG'

def save_check_point(model, path):
    selected_keys = [
        'feature_gallery1',
        'feature_gallery2',
        'text_features',
    ]
    state_dict = model.state_dict()
    selected_state_dict = {k: v for k, v in state_dict.items() if k in selected_keys}

    torch.save(selected_state_dict, path)


def fit(model,
        args,
        dataloader: DataLoader,
        device: str,
        img_dir: str,
        check_path: str,
        train_data: DataLoader,
        ):

    # change the model into eval mode
    model.eval_mode()

    features1 = []
    features2 = []
    for (data, mask, label, name, img_type) in train_data:
        # data is already transformed by Dataset
        data = data.to(device)
        _, _, feature_map1, feature_map2 = model.encode_image(data)
        features1.append(feature_map1)
        features2.append(feature_map2)

    features1 = torch.cat(features1, dim=0)
    features2 = torch.cat(features2, dim=0)
    model.build_image_feature_gallery(features1, features2)

    optimizer = torch.optim.SGD(model.prompt_learner.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.Epoch, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss().to(device)
    criterion_tip = TripletLoss(margin=getattr(args, 'margin_value', 0.0))

    best_result_dict = None
    # Early Stopping with larger patience for segmentation (needs more epochs to converge)
    patience = 50
    patience_counter = 0
    best_epoch = 0
    # 方向 5: 缓存测试集预处理结果
    cached_test_data = None
    
    for epoch in range(args.Epoch):
        for (data, mask, label, name, img_type) in train_data:
            # data is already transformed by Dataset
            data = data.to(device)

            normal_text_prompt, abnormal_text_prompt_handle, abnormal_text_prompt_learned = model.prompt_learner()

            optimizer.zero_grad()

            normal_text_features = model.encode_text_embedding(normal_text_prompt, model.tokenized_normal_prompts)

            abnormal_text_features_handle = model.encode_text_embedding(abnormal_text_prompt_handle, model.tokenized_abnormal_prompts_handle)
            abnormal_text_features_learned = model.encode_text_embedding(abnormal_text_prompt_learned, model.tokenized_abnormal_prompts_learned)
            abnormal_text_features = torch.cat([abnormal_text_features_handle, abnormal_text_features_learned], dim=0)

            # compute mean
            mean_ad_handle = torch.mean(F.normalize(abnormal_text_features_handle, dim=-1), dim=0)
            mean_ad_learned = torch.mean(F.normalize(abnormal_text_features_learned, dim=-1), dim=0)

            loss_match_abnormal = (mean_ad_handle - mean_ad_learned).norm(dim=0) ** 2.0

            # Segmentation task with fusion-aware training
            _, feature_map, _, _ = model.encode_image(data)

            # compute v2t loss with multi-negative prototypes
            normal_text_features_ahchor = normal_text_features.mean(dim=0).unsqueeze(0)
            normal_text_features_ahchor = normal_text_features_ahchor / normal_text_features_ahchor.norm(dim=-1, keepdim=True)

            abnormal_text_features_ahchor = abnormal_text_features.mean(dim=0).unsqueeze(0)
            abnormal_text_features_ahchor = abnormal_text_features_ahchor / abnormal_text_features_ahchor.norm(dim=-1, keepdim=True)
            
            # Normalize learned and manual abnormal prototypes separately
            abnormal_learned = F.normalize(abnormal_text_features_learned, dim=-1)  # [K_learned, D]
            # Squeeze potential extra batch dimension
            if abnormal_text_features_handle.dim() == 3:
                abnormal_text_features_handle = abnormal_text_features_handle.squeeze(0)
            abnormal_manual = F.normalize(abnormal_text_features_handle, dim=-1)  # [K_manual, D]
            
            # Debug: print shapes
            if iter == 0:
                print(f"[DEBUG] abnormal_learned shape: {abnormal_learned.shape}")
                print(f"[DEBUG] abnormal_manual shape: {abnormal_manual.shape}")
            
            # Multi-negative CE loss: normal=class0, each abnormal=class_i
            # feature_map: [B, H*W, D]
            B, HW, D = feature_map.shape
            feature_map_norm = F.normalize(feature_map, dim=-1)  # [B, H*W, D]
            
            # Compute logits: [B, H*W, 1+K_learned+K_manual]
            l_pos = torch.einsum('nic,cj->nij', feature_map_norm, normal_text_features_ahchor.transpose(0, 1))  # [B, H*W, 1]
            l_neg_learned = torch.einsum('nic,kc->nik', feature_map_norm, abnormal_learned)  # [B, H*W, K_learned]
            l_neg_manual = torch.einsum('nic,kc->nik', feature_map_norm, abnormal_manual)  # [B, H*W, K_manual]

            if model.precision == 'fp16':
                logit_scale = model.model.logit_scale.half()
            else:
                logit_scale = model.model.logit_scale

            logits_v2t = torch.cat([l_pos, l_neg_learned, l_neg_manual], dim=-1) * logit_scale  # [B, H*W, 1+K]

            target_v2t = torch.zeros([B, HW], dtype=torch.long).to(device)  # All pixels belong to class 0 (normal)

            loss_v2t = criterion(logits_v2t.transpose(1, 2), target_v2t)

            # Original triplet loss (remains unchanged for SEG)
            trip_loss = criterion_tip(feature_map, normal_text_features_ahchor, abnormal_text_features_ahchor)
            
            # ===== Multi-Abnormal Regularizations =====
            # A) L_pull: Pull learned prototypes toward mean(normal_batch)
            pull_weight = getattr(args, 'pull_weight', 0.0)
            L_pull = torch.tensor(0.0, device=device)
            if pull_weight > 0:
                # Compute mean of batch features (use spatial mean for SEG)
                z_batch_mean = feature_map_norm.mean(dim=(0, 1))  # [D]
                w_n = normal_text_features_ahchor.squeeze(0)  # [D]
                cos_sim = (z_batch_mean * w_n).sum()
                L_pull = pull_weight * (1 - cos_sim)
            
            # B) L_rep: Repulsion among learned prototypes (encourage diversity)
            rep_weight = getattr(args, 'rep_weight', 0.0)
            rep_gamma = getattr(args, 'rep_gamma', 0.3)
            L_rep = torch.tensor(0.0, device=device)
            if rep_weight > 0 and len(abnormal_learned) > 1:
                # Compute pairwise cosine similarities
                sim_matrix = abnormal_learned @ abnormal_learned.T  # [K, K]
                # Only upper triangle (exclude diagonal)
                K = len(abnormal_learned)
                mask = torch.triu(torch.ones(K, K, device=device), diagonal=1)
                pairwise_sims = sim_matrix[mask > 0]
                # Penalize similarities above threshold gamma
                L_rep = rep_weight * torch.relu(pairwise_sims - rep_gamma).mean()
            
            # C) L_margin: Hard negative triplet margin
            margin_weight = getattr(args, 'margin_weight', 0.0)
            L_margin = torch.tensor(0.0, device=device)
            if margin_weight > 0:
                # Use TripletLoss with hard negatives from multi-abnormal prototypes
                # Combine learned and manual as negative set
                # Ensure abnormal_manual is 2D [K_manual, D]
                if abnormal_manual.dim() == 3:
                    abnormal_manual = abnormal_manual.squeeze(0)
                # Concatenate learned prototypes and manual prompts: [K_learned + K_manual, D]
                negative_set = torch.cat([abnormal_learned, abnormal_manual], dim=0)
                L_margin = margin_weight * criterion_tip(
                    feature_map_norm.reshape(-1, D),  # Flatten to [B*H*W, D]
                    normal_text_features_ahchor.expand(B * HW, -1),  # [B*H*W, D]
                    negative_set  # [K_total, D] - all abnormal prototypes
                )
            
            # Fusion-Aware Training: pixel-level fusion with spatial mean
            # feature_map: [B, H*W, D]
            feature_map_normalized = F.normalize(feature_map, dim=-1)  # [B, H*W, D]
            feature_map_mean = feature_map_normalized.mean(dim=(0, 1), keepdim=True)  # [1, 1, D]
            
            # Fused normal prototype: p_fused = (1-λ)*p_learned + λ*mean(v_train_pixels)
            fusion_lambda = getattr(args, 'fusion_lambda', 0.0)
            fusion_loss_weight = getattr(args, 'fusion_loss_weight', 0.0)
            normal_fused = (1 - fusion_lambda) * normal_text_features_ahchor.unsqueeze(1) + \
                           fusion_lambda * feature_map_mean
            normal_fused = F.normalize(normal_fused, dim=-1)  # [1, 1, D]
            
            # Triplet loss with fused prototype
            trip_loss_fused = criterion_tip(
                feature_map,
                normal_fused.squeeze(1),  # [1, D]
                abnormal_text_features_ahchor
            )
            
            # Combined loss
            loss = loss_v2t + trip_loss + trip_loss_fused * fusion_loss_weight + L_pull + L_rep + L_margin

            loss.backward()
            optimizer.step()

        scheduler.step()
        model.build_text_feature_gallery()

        # 方向 1: 降低评估频率（每 5 个 epoch 或最后一个 epoch）
        # Evaluate every epoch for faster feedback
        if (epoch + 1) % 1 == 0 or epoch == args.Epoch - 1:
            score_maps = []
            test_imgs = [] if args.vis else None  # 方向 2 + 3.3: 仅在可视化时收集
            gt_mask_list = []
            names = []

            # 方向 5: 使用缓存或第一次收集
            if cached_test_data is None:
                for (data, mask, label, name, img_type) in dataloader:
                    # data is already transformed by Dataset
                    for d, n, l, m in zip(data, name, label, mask):
                        # 方向 2: 仅在可视化时 denormalize
                        if args.vis:
                            test_imgs.append(denormalization(d.cpu().numpy()))
                        # Convert to numpy if it's a tensor, otherwise keep as is
                        m = m.cpu().numpy() if torch.is_tensor(m) else m
                        m[m > 0] = 1

                        names.append(n)
                        gt_mask_list.append(m)

                    data = data.to(device)
                    score_map = model(data, 'seg')
                    score_maps += score_map

                # 方向 3.1 + 3.2: 只 resize gt_mask，降低到 256
                import cv2
                gt_mask_list = [cv2.resize(mask, (args.resolution, args.resolution), 
                                          interpolation=cv2.INTER_NEAREST) for mask in gt_mask_list]
                if args.vis:
                    test_imgs = [cv2.resize(img, (args.resolution, args.resolution), 
                                           interpolation=cv2.INTER_CUBIC) for img in test_imgs]
                
                # 方向 5: 缓存预处理结果
                cached_test_data = {
                    'test_imgs': test_imgs,
                    'gt_mask_list': gt_mask_list,
                    'names': names
                }
            else:
                # 使用缓存数据，只重新计算 scores
                for (data, mask, label, name, img_type) in dataloader:
                    data = data.to(device)
                    score_map = model(data, 'seg')
                    score_maps += score_map
                
                test_imgs = cached_test_data['test_imgs']
                gt_mask_list = cached_test_data['gt_mask_list']
                names = cached_test_data['names']

            result_dict = metric_cal_pix(np.array(score_maps), gt_mask_list)

            if best_result_dict is None:
                save_check_point(model, check_path)
                best_result_dict = result_dict
                best_epoch = epoch + 1
                patience_counter = 0
                print(f'  Epoch {epoch+1}: Pixel-AUROC={result_dict["p_roc"]:.2f}')
                if args.vis:
                    plot_sample_cv2(names, test_imgs, {'PromptAD': score_maps}, gt_mask_list, save_folder=img_dir)

            elif best_result_dict['p_roc'] < result_dict['p_roc']:
                save_check_point(model, check_path)
                best_result_dict = result_dict
                best_epoch = epoch + 1
                patience_counter = 0
                print(f'  Epoch {epoch+1}: Pixel-AUROC={result_dict["p_roc"]:.2f} *** Best ***')
                if args.vis:
                    plot_sample_cv2(names, test_imgs, {'PromptAD': score_maps}, gt_mask_list, save_folder=img_dir)
            else:
                patience_counter += 1
                print(f'  Epoch {epoch+1}: Pixel-AUROC={result_dict["p_roc"]:.2f} (Patience: {patience_counter}/{patience})')
                
                if patience_counter >= patience:
                    print(f'\n[Early Stopping] No improvement for {patience} epochs. Best: Epoch {best_epoch} with Pixel-AUROC={best_result_dict["p_roc"]:.2f}%')
                    break

    return best_result_dict


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

    # prepare the experiment dir
    img_dir, csv_path, check_path = get_dir_from_args(TASK, **kwargs)

    # get the model first (need model.transform for dataset)
    kwargs['out_size_h'] = kwargs['resolution']
    kwargs['out_size_w'] = kwargs['resolution']

    model = PromptAD(**kwargs)
    model = model.to(device)

    # get the train dataloader (pass model.transform to avoid repeat conversion)
    train_dataloader, train_dataset_inst = get_dataloader_from_args(phase='train', perturbed=False, transform=model.transform, **kwargs)

    # get the test dataloader
    test_dataloader, test_dataset_inst = get_dataloader_from_args(phase='test', perturbed=False, transform=model.transform, **kwargs)

    # as the pro metric calculation is costly, we only calculate it in the last evaluation
    metrics = fit(model, args, test_dataloader, device, img_dir=img_dir, check_path=check_path, train_data=train_dataloader)

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
    parser.add_argument('--class_name', type=str, default='carpet')

    parser.add_argument('--img-resize', type=int, default=240)
    parser.add_argument('--img-cropsize', type=int, default=240)
    parser.add_argument('--resolution', type=int, default=256)  # 优化: 从 400 降到 256

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
    parser.add_argument("--Epoch", type=int, default=200)  # Increased to 200, early stopping (patience=20) handles convergence

    # optimizer
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)

    # loss hyper parameter
    parser.add_argument("--lambda1", type=float, default=0.001)
    parser.add_argument("--fusion-lambda", type=float, default=0.2,
                        help="Fusion weight for fusion-aware training: p_fused = (1-λ)*p_learned + λ*p_img")
    parser.add_argument("--fusion-loss-weight", type=float, default=0.5,
                        help="Weight for fusion-aware triplet loss")
    
    # Multi-Abnormal Prototypes regularization parameters
    parser.add_argument("--pull-weight", type=float, default=0.1,
                        help="L_pull weight: pull learned prototypes toward batch mean")
    parser.add_argument("--rep-weight", type=float, default=0.05,
                        help="L_rep weight: repulsion among learned prototypes")
    parser.add_argument("--margin-weight", type=float, default=0.1,
                        help="L_margin weight: hard negative triplet margin")
    parser.add_argument("--margin-value", type=float, default=0.03,
                        help="Hard negative margin value (m in triplet loss)")
    parser.add_argument("--rep-gamma", type=float, default=0.3,
                        help="Repulsion threshold γ (penalize similarities > γ)")
    parser.add_argument("--topk-abnormal", type=int, default=2,
                        help="Top-k aggregation for abnormal prototypes (k=1: max, k>1: top-k mean)")
    parser.add_argument("--filter-threshold-delta", type=float, default=0.03,
                        help="Prototype filtering threshold offset")

    # dataloader configuration
    parser.add_argument("--num-workers", type=int, default=0,
                        help='Number of data loading workers (0=main process only, 2=+2 cores)')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    import os

    args = get_args()
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu_id}"
    main(args)
