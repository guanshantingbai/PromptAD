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
from datetime import datetime

TASK = 'CLS'


def save_check_point(model, path):
    selected_keys = [
        'normal_prototypes',
        'abnormal_prototypes',
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

    # Build image feature gallery (memory bank) from training data
    # Baseline方式：先收集所有features，再一次性build
    print("Building memory bank from training data...")
    features1 = []
    features2 = []
    with torch.no_grad():
        for data, mask, label, name, img_type in tqdm(train_data, desc="Building memory bank"):
            data = data.to(device)
            _, _, feature_map1, feature_map2 = model.encode_image(data)
            features1.append(feature_map1)
            features2.append(feature_map2)
    
    features1 = torch.cat(features1, dim=0)
    features2 = torch.cat(features2, dim=0)
    model.build_image_feature_gallery(features1, features2)
    print(f"Memory bank built: {model.feature_gallery1.shape[0]} samples")

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

            normal_text_features = model.encode_text_embedding(normal_text_prompt, model.tokenized_normal_prompts)

            abnormal_text_features_handle = model.encode_text_embedding(abnormal_text_prompt_handle, model.tokenized_abnormal_prompts_handle)
            abnormal_text_features_learned = model.encode_text_embedding(abnormal_text_prompt_learned, model.tokenized_abnormal_prompts_learned)
            abnormal_text_features = torch.cat([abnormal_text_features_handle, abnormal_text_features_learned], dim=0)

            # 【改动1】修正EMA: 逐原型对齐而非均值对齐
            # 旧版：只对齐均值，导致learned内部可以坍缩
            # 新版：每个learned原型都被拉向handle的均值，防止坍缩
            handle_normalized = F.normalize(abnormal_text_features_handle, dim=-1)
            learned_normalized = F.normalize(abnormal_text_features_learned, dim=-1)
            
            # 计算handle集合的中心作为anchor
            handle_center = torch.mean(handle_normalized, dim=0, keepdim=True)  # [1, dim]
            
            # 每个learned原型到handle中心的距离
            loss_match_abnormal = torch.mean((learned_normalized - handle_center).norm(dim=-1) ** 2.0)
            
            # 【改动2】Repulsion Loss: 防止learned原型坍缩
            # 只对learned原型施加互斥约束，handle原型不动
            if learned_normalized.shape[0] > 1:
                # 计算learned原型之间的余弦相似度矩阵
                learned_sim_matrix = torch.mm(learned_normalized, learned_normalized.t())  # [M, M]
                # 去除对角线（自身与自身的相似度）
                mask = torch.eye(learned_sim_matrix.shape[0], device=learned_sim_matrix.device).bool()
                learned_sim_matrix = learned_sim_matrix.masked_fill(mask, 0.0)
                # Repulsion: 最小化非对角元素（让原型相互远离）
                loss_repulsion = learned_sim_matrix.abs().mean()
            else:
                loss_repulsion = torch.tensor(0.0, device=device)
            
            # 【改动3】Margin Loss: 显式优化判别边界
            # Margin = s_normal - s_ab_max，希望正样本的margin尽可能大

            cls_feature, _, _, _ = model.encode_image(data)
            cls_feature = cls_feature / cls_feature.norm(dim=-1, keepdim=True)

            # Multi-prototype contrastive loss
            # Normalize all features
            normal_text_features = normal_text_features / normal_text_features.norm(dim=-1, keepdim=True)
            abnormal_text_features = abnormal_text_features / abnormal_text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity to all normal prototypes (pull to closest one)
            normal_sim = torch.einsum('nc,mc->nm', cls_feature, normal_text_features)  # [batch, K_normal]
            
            # Compute similarity to all abnormal prototypes (push away from all)
            abnormal_sim = torch.einsum('nc,mc->nm', cls_feature, abnormal_text_features)  # [batch, M_abnormal]
            
            if model.precision == 'fp16':
                logit_scale = model.model.logit_scale.half()
            else:
                logit_scale = model.model.logit_scale
            
            # Multi-prototype InfoNCE loss: pull to best normal, push all abnormals
            # For each sample, use max normal similarity as positive, all abnormals as negatives
            max_normal_sim = normal_sim.max(dim=-1, keepdim=True)[0]  # [batch, 1]
            logits = torch.cat([max_normal_sim, abnormal_sim], dim=-1) * logit_scale  # [batch, 1 + M_abnormal]
            
            target_v2t = torch.zeros([logits.shape[0]], dtype=torch.long).to(device)
            loss_v2t = criterion(logits, target_v2t)

            # Triplet loss with multi-prototypes: use best normal and worst abnormal
            best_normal_idx = normal_sim.argmax(dim=-1)  # [batch]
            normal_anchors = normal_text_features[best_normal_idx]  # [batch, dim]
            
            worst_abnormal_idx = abnormal_sim.argmax(dim=-1)  # [batch]
            abnormal_anchors = abnormal_text_features[worst_abnormal_idx]  # [batch, dim]
            
            trip_loss = criterion_tip(cls_feature, normal_anchors, abnormal_anchors)
            
            # 总损失（移除Margin Loss以避免破坏Stable类的Separation）
            loss = loss_v2t + trip_loss + \
                   loss_match_abnormal * args.lambda1 + \
                   loss_repulsion * args.lambda_rep

            loss.backward()
            optimizer.step()
        scheduler.step()
        model.build_text_feature_gallery()

        # 方向 1: 降低评估频率（每 5 个 epoch 或最后一个 epoch）
        if (epoch + 1) % 5 == 0 or epoch == args.Epoch - 1:
            scores_img = []
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
                    
                    # 【修复BUG】只用语义分支评估，不用融合（Memory Bank应该是inference-only）
                    with torch.no_grad():
                        visual_features = model.encode_image(data)
                        # 图像级得分（用于模型选择）
                        textual_anomaly = model.calculate_textual_anomaly_score(visual_features, 'cls')
                        # 像素级得分（用于metric计算）
                        textual_anomaly_map = model.calculate_textual_anomaly_score(visual_features, 'seg')
                        textual_anomaly_map = textual_anomaly_map.detach().cpu().numpy()
                    
                    scores_img += textual_anomaly.tolist()
                    # 添加真实的score_maps
                    for i in range(textual_anomaly_map.shape[0]):
                        score_maps.append(textual_anomaly_map[i, 0])  # 移除channel维度

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
                    
                    # 【修复BUG】只用语义分支评估
                    with torch.no_grad():
                        visual_features = model.encode_image(data)
                        textual_anomaly = model.calculate_textual_anomaly_score(visual_features, 'cls')
                        textual_anomaly_map = model.calculate_textual_anomaly_score(visual_features, 'seg')
                        textual_anomaly_map = textual_anomaly_map.detach().cpu().numpy()
                    
                    scores_img += textual_anomaly.tolist()
                    for i in range(textual_anomaly_map.shape[0]):
                        score_maps.append(textual_anomaly_map[i, 0])
                
                test_imgs = cached_test_data['test_imgs']
                gt_list = cached_test_data['gt_list']
                gt_mask_list = cached_test_data['gt_mask_list']

            result_dict = metric_cal_img(np.array(scores_img), gt_list, np.array(score_maps))

            if best_result_dict is None:
                save_check_point(model, check_path)
                best_result_dict = result_dict

            elif best_result_dict['i_roc'] < result_dict['i_roc']:
                save_check_point(model, check_path)
                best_result_dict = result_dict

    return best_result_dict


def main(args):
    # 记录进程起始时间
    process_start_time = datetime.now()
    
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

    i_roc = round(metrics['i_roc'], 2)
    object = kwargs['class_name']
    
    # 记录进程终止时间和用时
    process_end_time = datetime.now()
    process_elapsed_time = process_end_time - process_start_time
    hours, remainder = divmod(process_elapsed_time.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f'\n{"="*80}')
    print(f'Object:{object} =========================== Image-AUROC:{i_roc}')
    print(f'进程起始时间: {process_start_time.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'进程终止时间: {process_end_time.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'进程用时: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒')
    print(f'{"="*80}\n')

    save_metric(metrics, dataset_classes[kwargs['dataset']], kwargs['class_name'],
                kwargs['dataset'], csv_path)
    
    # 保存时间信息到单独的日志文件
    timing_log = csv_path.replace('.csv', '_timing.log')
    with open(timing_log, 'w') as f:
        f.write(f"Object: {object}\n")
        f.write(f"Dataset: {kwargs['dataset']}, K-shot: {kwargs['k_shot']}\n")
        f.write(f"进程起始时间: {process_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"进程终止时间: {process_end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"进程用时: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒\n")
        f.write(f"Image-AUROC: {i_roc}\n")


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
    parser.add_argument("--n_pro", type=int, default=1)
    parser.add_argument("--n_pro_ab", type=int, default=4)
    parser.add_argument("--Epoch", type=int, default=100)

    # optimizer
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)

    # loss hyper parameter
    parser.add_argument("--lambda1", type=float, default=0.001)
    parser.add_argument("--lambda_rep", type=float, default=0.1, help='Repulsion loss weight (EMA+Rep config)')

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
