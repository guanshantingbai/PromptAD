"""最小化聚合对比脚本 - 直接使用get_dataloader_from_args"""
import os, sys, torch
import numpy as np
import torch.nn.functional as F
from PIL import Image

# 设置参数
class Args:
    dataset = 'mvtec'
    data_path = './data/mvtec'
    k_shot = 2
    seed = 111
    batch_size = 1

TEST_CLASSES = ['grid', 'bottle', 'cable', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']

all_results = {}

for cls in TEST_CLASSES:
    print(f"\n{'='*60}\nClass: {cls}\n{'='*60}")
    
    args = Args()
    args.class_name = cls
    
    # 使用项目原有的数据加载函数
    from datasets import get_dataloader_from_args
    from PromptAD.model import PromptAD
    
    # 加载模型
    kwargs = {
        'dataset': 'mvtec', 'class_name': cls, 'k_shot': 2, 'seed': 111,
        'backbone': 'ViT-B-16-plus-240', 'pretrained_dataset': 'laion400m_e32',
        'n_ctx': 12, 'n_pro': 1, 'n_ctx_ab': 12, 'n_pro_ab': 4,
        'device': 'cuda:0', 'resolution': 256, 'img_resize': 256,
        'img_cropsize': 256, 'out_size_h': 256, 'out_size_w': 256,
        'batch_size': 1, 'data_path': './data/mvtec',
    }
    model = PromptAD(**kwargs)
    model.to('cuda:0')
    model.eval()
    
    # 加载checkpoint
    ckpt = torch.load(f'./result/fix_validation/mvtec/k_2/checkpoint/CLS-Seed_111-{cls}-check_point.pt')
    model.text_features = ckpt['text_features'].cuda()
    model.abnormal_text_features_all = ckpt['abnormal_text_features_all'].cuda()
    
    # 加载数据
    test_loader, _ = get_dataloader_from_args(
        phase='test', perturbed=False, transform=model.transform, **kwargs
    )
    
    print(f"Testing with {len(test_loader)} samples...")
    
    # 收集分数
    results_cls = {}
    for agg_name, agg_k in [('mean', None), ('top2', 2), ('max', 1)]:
        norm_scores, abnorm_scores = [], []
        
        with torch.no_grad():
            for data, _, label, _, img_type in test_loader:
                data = data.cuda()
                
                # 提取特征
                feat_list = model.encode_image(data)
                cls_feat = F.normalize(feat_list[0], dim=-1)
                
                # 计算分数
                normal_sim = (cls_feat @ model.text_features[0:1].T).squeeze()
                ab_sims = cls_feat @ model.abnormal_text_features_all.T
                
                if agg_k is None:
                    abnormal_sim = ab_sims.mean()
                elif agg_k == 1:
                    abnormal_sim = ab_sims.max()
                else:
                    abnormal_sim = ab_sims.topk(agg_k)[0].mean()
                
                score = (1 - normal_sim + abnormal_sim).cpu().item()
                
                if img_type[0] == 'good':
                    norm_scores.append(score)
                else:
                    abnorm_scores.append(score)
        
        norm_scores = np.array(norm_scores)
        abnorm_scores = np.array(abnorm_scores)
        
        results_cls[agg_name] = {
            'normal_p95': float(np.percentile(norm_scores, 95)),
            'normal_p99': float(np.percentile(norm_scores, 99)),
            'abnormal_median': float(np.median(abnorm_scores)),
            'abnormal_p95': float(np.percentile(abnorm_scores, 95)),
        }
        
        print(f"\n[{agg_name}]")
        print(f"  Normal  - P95: {results_cls[agg_name]['normal_p95']:.4f}, P99: {results_cls[agg_name]['normal_p99']:.4f}")
        print(f"  Abnormal - Med: {results_cls[agg_name]['abnormal_median']:.4f}, P95: {results_cls[agg_name]['abnormal_p95']:.4f}")
    
    all_results[cls] = results_cls
    
    del model
    torch.cuda.empty_cache()

# 保存结果
import json
with open('aggregation_comparison_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print("\n✅ Done! Results saved to aggregation_comparison_results.json")
