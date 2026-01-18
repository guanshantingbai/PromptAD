"""
同权重多聚合方式对比实验
固定checkpoint，只改变abnormal prototype的聚合方式
"""
import os
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from datasets.dataset import CLIPDataset
from datasets import load_mvtec
from PromptAD.model import PromptAD

# 9个测试类别
TEST_CLASSES = ['grid', 'bottle', 'cable', 'metal_nut', 'pill', 
                'screw', 'toothbrush', 'transistor', 'zipper']

def load_model(class_name):
    """加载fix_validation的checkpoint"""
    kwargs = {
        'dataset': 'mvtec',
        'class_name': class_name,
        'k_shot': 2,
        'seed': 111,
        'backbone': 'ViT-B-16-plus-240',
        'pretrained_dataset': 'laion400m_e32',
        'n_ctx': 12,
        'n_pro': 1,
        'n_ctx_ab': 12,
        'n_pro_ab': 4,
        'device': 'cuda:0',
        'resolution': 256,
        'img_resize': 256,
        'img_cropsize': 256,
        'out_size_h': 256,
        'out_size_w': 256,
    }
    
    model = PromptAD(**kwargs)
    device = torch.device('cuda:0')
    model.to(device)
    model.eval()
    
    # 加载checkpoint - 只加载text features（忽略尺寸不匹配）
    ckpt_path = f'./result/fix_validation/mvtec/k_2/checkpoint/CLS-Seed_111-{class_name}-check_point.pt'
    state_dict = torch.load(ckpt_path, map_location=device)
    
    # 直接使用checkpoint中的text features
    model.text_features = state_dict['text_features'].to(device)
    model.normal_text_features_all = state_dict['normal_text_features_all'].to(device)
    model.abnormal_text_features_all = state_dict['abnormal_text_features_all'].to(device)
    
    return model, device, kwargs

def load_test_data(class_name, kwargs):
    """加载测试数据"""
    data_tuple = load_mvtec(category=class_name, k_shot=kwargs['k_shot'])
    # load_mvtec返回2个元素：(train_data, test_data)
    # 每个data是(img_paths, gt_paths, labels, types)的tuple
    train_data, test_data = data_tuple
    test_paths, test_gts, test_labels, test_types = test_data
    
    dataset = CLIPDataset(
        test_paths,
        test_gts,
        test_labels,
        test_types,
        resize=kwargs['resolution']
    )
    return DataLoader(dataset, batch_size=1, shuffle=False)

def compute_scores_with_aggregation(model, dataloader, device, agg_mode):
    """
    计算异常分数
    agg_mode: 'mean_all', 'top2', 'max', 'filtered_top2'
    """
    normal_scores = []
    abnormal_scores = []
    
    with torch.no_grad():
        for data, _, label, _, img_type in dataloader:
            data = [model.transform(Image.fromarray(f.numpy())) for f in data]
            data = torch.stack(data, dim=0).to(device)
            
            # Encode image - returns list
            features_list = model.encode_image(data)
            cls_feature = features_list[0]  # 取第一个（cls token）
            cls_feature = F.normalize(cls_feature, dim=-1)
            
            # Normal similarity
            normal_sim = (cls_feature @ model.text_features[0:1].T).squeeze()
            
            # Abnormal similarity with different aggregations
            ab_sims = cls_feature @ model.abnormal_text_features_all.T  # [1, K]
            
            if agg_mode == 'mean_all':
                abnormal_sim = ab_sims.mean()
            elif agg_mode == 'top2':
                abnormal_sim = ab_sims.topk(2)[0].mean()
            elif agg_mode == 'max':
                abnormal_sim = ab_sims.max()
            elif agg_mode == 'filtered_top2':
                # 简化：只用前4个learned prototypes
                ab_sims_learned = ab_sims[:, :4]
                abnormal_sim = ab_sims_learned.topk(min(2, ab_sims_learned.shape[1]))[0].mean()
            
            score = (1 - normal_sim + abnormal_sim).cpu().item()
            
            if img_type[0] == 'good':
                normal_scores.append(score)
            else:
                abnormal_scores.append(score)
    
    return np.array(normal_scores), np.array(abnormal_scores)

def analyze_one_class(class_name):
    """分析单个类别"""
    print(f"\n{'='*60}")
    print(f"Class: {class_name}")
    print(f"{'='*60}")
    
    model, device, kwargs = load_model(class_name)
    dataloader = load_test_data(class_name, kwargs)
    
    results = {}
    for agg_mode in ['mean_all', 'top2', 'max']:
        print(f"  Testing {agg_mode}...", end='', flush=True)
        norm_scores, abnorm_scores = compute_scores_with_aggregation(
            model, dataloader, device, agg_mode
        )
        
        results[agg_mode] = {
            'normal_p95': np.percentile(norm_scores, 95),
            'normal_p99': np.percentile(norm_scores, 99),
            'abnormal_median': np.median(abnorm_scores),
            'abnormal_p95': np.percentile(abnorm_scores, 95),
        }
        print(" Done")
    
    del model
    torch.cuda.empty_cache()
    
    return results

if __name__ == '__main__':
    all_results = {}
    for cls in TEST_CLASSES:
        try:
            all_results[cls] = analyze_one_class(cls)
        except Exception as e:
            print(f"  Error: {e}")
    
    # 保存结果
    import json
    with open('aggregation_comparison_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Results saved to aggregation_comparison_results.json")
    print(f"{'='*60}")
