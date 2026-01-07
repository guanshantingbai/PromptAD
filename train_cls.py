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
import random
from tqdm import tqdm

TASK = 'CLS'


def save_check_point(model, path):
    selected_keys = [
        'feature_gallery1',
        'feature_gallery2',
        'text_features',
        'normal_text_features_all',  # 新增：保存所有normal向量
        'abnormal_text_features_all',  # 新增：保存所有abnormal向量
        'training_cls_tokens',  # 🆕 保存训练图像的 CLS tokens
    ]
    state_dict = model.state_dict()
    selected_state_dict = {k: v for k, v in state_dict.items() if k in selected_keys}

    torch.save(selected_state_dict, path)

def fit(model,
        args,
        dataloader: DataLoader,
        device: str,
        check_path: str,
        train_data: DataLoader,
        ):

    # change the model into eval mode
    model.eval_mode()

    # 🆕 Visual Prototypes Mode: 使用训练图像的 CLS tokens 作为 Normal Prototypes
    if hasattr(model, 'use_visual_prototypes') and model.use_visual_prototypes:
        print("\n" + "="*70)
        print("[Visual Prototypes Mode] Setting normal prototypes from training images...")
        print("="*70)
        
        # 收集所有训练图像
        train_images_list = []
        for (data, mask, label, name, img_type) in train_data:
            # data 已经经过 transform
            train_images_list.append(data)
        train_images_tensor = torch.cat(train_images_list, dim=0).to(device)  # [k_shot, 3, H, W]
        
        # 设置视觉原型
        model.set_visual_prototypes(train_images_tensor)
        print(f"\u2705 Visual prototypes set from {train_images_tensor.shape[0]} training images")
        print("="*70 + "\n")

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
    criterion_tip = TripletLoss(margin=0.0)

    best_result_dict = None
    # 方向 5: 缓存测试集预处理结果
    cached_test_data = None
    
    for epoch in range(args.Epoch):
        for (data, mask, label, name, img_type) in train_data:
            # data is already transformed by Dataset
            data = data.to(device)

            normal_text_prompt, abnormal_text_prompt_handle, abnormal_text_prompt_learned = model.prompt_learner()

            optimizer.zero_grad()

            # 🆕 Visual Prototypes Mode: 跳过 normal text features 计算
            if hasattr(model, 'use_visual_prototypes') and model.use_visual_prototypes:
                # 仅计算 abnormal text features
                abnormal_text_features_handle = model.encode_text_embedding(abnormal_text_prompt_handle, model.tokenized_abnormal_prompts_handle)
                abnormal_text_features_learned = model.encode_text_embedding(abnormal_text_prompt_learned, model.tokenized_abnormal_prompts_learned)
                abnormal_text_features = torch.cat([abnormal_text_features_handle, abnormal_text_features_learned], dim=0)
                
                # normal 特征已经在 set_visual_prototypes 中设置，直接使用
                normal_text_features = model.text_features[0:1]  # [1, dim] - 已经是视觉特征
            else:
                # 📖 原始逻辑：计算 normal 和 abnormal text features
                normal_text_features = model.encode_text_embedding(normal_text_prompt, model.tokenized_normal_prompts)

                abnormal_text_features_handle = model.encode_text_embedding(abnormal_text_prompt_handle, model.tokenized_abnormal_prompts_handle)
                abnormal_text_features_learned = model.encode_text_embedding(abnormal_text_prompt_learned, model.tokenized_abnormal_prompts_learned)
                abnormal_text_features = torch.cat([abnormal_text_features_handle, abnormal_text_features_learned], dim=0)

            # compute mean
            mean_ad_handle = torch.mean(F.normalize(abnormal_text_features_handle, dim=-1), dim=0)
            mean_ad_learned = torch.mean(F.normalize(abnormal_text_features_learned, dim=-1), dim=0)

            loss_match_abnormal = (mean_ad_handle - mean_ad_learned).norm(dim=0) ** 2.0

            cls_feature, _, _, _ = model.encode_image(data)

            # compute v2t loss and triplet loss
            normal_text_features_ahchor = normal_text_features.mean(dim=0).unsqueeze(0)
            normal_text_features_ahchor = normal_text_features_ahchor / normal_text_features_ahchor.norm(dim=-1, keepdim=True)

            abnormal_text_features_ahchor = abnormal_text_features.mean(dim=0).unsqueeze(0)
            abnormal_text_features_ahchor = abnormal_text_features_ahchor / abnormal_text_features_ahchor.norm(dim=-1, keepdim=True)
            abnormal_text_features = abnormal_text_features / abnormal_text_features.norm(dim=-1, keepdim=True)

            l_pos = torch.einsum('nc,cm->nm', cls_feature, normal_text_features_ahchor.transpose(0, 1))
            l_neg_v2t = torch.einsum('nc,cm->nm', cls_feature, abnormal_text_features.transpose(0, 1))

            if model.precision == 'fp16':
                logit_scale = model.model.logit_scale.half()
            else:
                logit_scale = model.model.logit_scalef

            logits_v2t = torch.cat([l_pos, l_neg_v2t], dim=-1) * logit_scale

            target_v2t = torch.zeros([logits_v2t.shape[0]], dtype=torch.long).to(device)

            loss_v2t = criterion(logits_v2t, target_v2t)

            trip_loss = criterion_tip(cls_feature, normal_text_features_ahchor, abnormal_text_features_ahchor)
            loss = loss_v2t + trip_loss + loss_match_abnormal * args.lambda1

            loss.backward()
            optimizer.step()
        scheduler.step()
        model.build_text_feature_gallery()

        # 方向 1: 降低评估频率（每 3 个 epoch 或最后一个 epoch）
        if (epoch + 1) % 3 == 0 or epoch == args.Epoch - 1:
            scores_semantic = []
            scores_memory = []
            scores_fusion = []
            score_maps = []
            test_imgs = [] if args.vis else None  # 方向 2 + 3.3: 仅在可视化时收集
            gt_list = []
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
                        l = l.cpu().numpy() if torch.is_tensor(l) else l
                        m = m.cpu().numpy() if torch.is_tensor(m) else m
                        m[m > 0] = 1

                        names.append(n)
                        gt_list.append(l)
                        gt_mask_list.append(m)

                    data = data.to(device)
                    semantic_scores, memory_scores, score_map = model(data, 'cls')
                    score_maps += score_map
                    scores_semantic += semantic_scores
                    scores_memory += memory_scores

                # 方向 3.1 + 3.2: 只 resize gt_mask（score_maps 已是正确尺寸），降低到 256
                import cv2
                gt_mask_list = [cv2.resize(mask, (args.resolution, args.resolution), 
                                          interpolation=cv2.INTER_NEAREST) for mask in gt_mask_list]
                if args.vis:
                    test_imgs = [cv2.resize(img, (args.resolution, args.resolution), 
                                           interpolation=cv2.INTER_CUBIC) for img in test_imgs]
                
                # 方向 5: 缓存预处理结果
                cached_test_data = {
                    'test_imgs': test_imgs,
                    'gt_list': gt_list,
                    'gt_mask_list': gt_mask_list
                }
            else:
                # 使用缓存数据，只重新计算 scores
                for (data, mask, label, name, img_type) in dataloader:
                    data = data.to(device)
                    semantic_scores, memory_scores, score_map = model(data, 'cls')
                    score_maps += score_map
                    scores_semantic += semantic_scores
                    scores_memory += memory_scores
                
                test_imgs = cached_test_data['test_imgs']
                gt_list = cached_test_data['gt_list']
                gt_mask_list = cached_test_data['gt_mask_list']

            # Perform harmonic mean fusion (following original PromptAD paper)
            semantic_img_scores = np.array(scores_semantic)
            memory_img_scores = np.array(scores_memory)
            fusion_img_scores = 1.0 / (1.0 / semantic_img_scores + 1.0 / memory_img_scores)
            
            # Calculate metrics for each branch
            from utils.metrics import metric_cal_img_only
            result_semantic = metric_cal_img_only(semantic_img_scores, gt_list)
            result_memory = metric_cal_img_only(memory_img_scores, gt_list)
            result_fusion = metric_cal_img_only(fusion_img_scores, gt_list)
            
            # Classification task: only image-level metrics (no pixel-level p_roc)
            result_dict = {
                'i_roc': result_fusion['i_roc'],  # Main metric: fusion AUROC
                'semantic_i_roc': result_semantic['i_roc'],
                'memory_i_roc': result_memory['i_roc']
            }

            if best_result_dict is None:
                save_check_point(model, check_path)
                best_result_dict = result_dict
                print(f'  Epoch {epoch+1}: Semantic={result_semantic["i_roc"]:.2f}, Memory={result_memory["i_roc"]:.2f}, Fusion={result_fusion["i_roc"]:.2f}')

            elif best_result_dict['i_roc'] < result_dict['i_roc']:
                save_check_point(model, check_path)
                best_result_dict = result_dict
                print(f'  Epoch {epoch+1}: Semantic={result_semantic["i_roc"]:.2f}, Memory={result_memory["i_roc"]:.2f}, Fusion={result_fusion["i_roc"]:.2f} *** Best ***')
            else:
                print(f'  Epoch {epoch+1}: Semantic={result_semantic["i_roc"]:.2f}, Memory={result_memory["i_roc"]:.2f}, Fusion={result_fusion["i_roc"]:.2f}')

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
    _, csv_path, check_path = get_dir_from_args(TASK, **kwargs)

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
    metrics = fit(model, args, test_dataloader, device, check_path=check_path, train_data=train_dataloader)

    fusion_roc = round(metrics['i_roc'], 2)
    semantic_roc = round(metrics['semantic_i_roc'], 2)
    memory_roc = round(metrics['memory_i_roc'], 2)
    object = kwargs['class_name']
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
    parser.add_argument('--resolution', type=int, default=256)  # 优化: 从 400 降到 256

    parser.add_argument('--batch-size', type=int, default=400)
    parser.add_argument('--vis', type=str2bool, choices=[True, False], default=False)
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

    parser.add_argument("--use-cpu", type=int, default=0)

    # prompt tuning hyper-parameter
    parser.add_argument("--n_ctx", type=int, default=4)
    parser.add_argument("--n_ctx_ab", type=int, default=1)
    parser.add_argument("--n_pro", type=int, default=3)
    parser.add_argument("--n_pro_ab", type=int, default=4)
    parser.add_argument("--Epoch", type=int, default=100)
    
    # MAP/LAP control
    parser.add_argument("--use-lap", type=str2bool, default=True,
                        help="Use LAP (Least Anomalous Patches). Set False for MAP-only mode.")
    
    # Visual Prototypes mode
    parser.add_argument("--use-visual-prototypes", type=str2bool, default=False,
                        help="Use training image CLS tokens as Normal Prototypes (skip prompt learning for normal)")

    # optimizer
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)

    # loss hyper parameter
    parser.add_argument("--lambda1", type=float, default=0.001)

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
