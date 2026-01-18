"""
Multi-Abnormal Prototype 假阳性假设验证
=========================================

假设：Multi-Abnormal实现增强了异常侧表达能力，但显著抬高了正常样本上的
      异常响应（假阳性率），这是高基线类别退化的主要原因。

验证方法：
  (A) 正常样本侧：P95/P99异常分数（判断假阳性风险）
  (B) 异常样本侧：median/P95异常分数（判断表达能力）
  (C) 统一阈值下FPR/TPR（因果证据）

对比对象：
  - Baseline: result/fusion_normal (mean abnormal anchor)
  - Multi-Abnormal: result/fix_validation (多异常原型+修复)
"""

import os
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm

from datasets.dataset import CLIPDataset
from datasets import load_mvtec
from PromptAD.model import PromptAD

def setup_kwargs_for_inference(class_name, root_dir, is_baseline=False):
    """构造推理所需的kwargs字典"""
    kwargs = {
        'dataset': 'mvtec',
        'class_name': class_name,
        'data_path': './data/mvtec',
        'k_shot': 2,
        'seed': 111,
        'backbone': 'ViT-B-16-plus-240',
        'pretrained_dataset': 'laion400m_e32',
        'n_ctx': 12,
        'n_pro': 1,
        'n_ctx_ab': 12,
        'n_pro_ab': 1 if is_baseline else 4,
        'root_dir': root_dir,
        'topk_abnormal': None,
        'filter_threshold': 0.03,
        'device': 'cuda:0',
        'resolution': 256,
        'img_resize': 256,
        'img_cropsize': 256,
        'out_size_h': 256,
        'out_size_w': 256,
        'aggregation': 'average',
        'lse_tau': 1.0,
    }
    return kwargs

def load_model_and_data(class_name, root_dir, is_baseline=False):
    """加载模型和数据"""
    kwargs = setup_kwargs_for_inference(class_name, root_dir, is_baseline)
    device = torch.device(kwargs['device'])
    
    # 加载模型
    model = PromptAD(**kwargs)
    model.to(device)
    model.eval()
    
    # 加载checkpoint
    check_path = os.path.join(
        root_dir, 
        kwargs['dataset'], 
        f'k_{kwargs["k_shot"]}',
        'checkpoint',
        f'CLS-Seed_{kwargs["seed"]}-{kwargs["class_name"]}-check_point.pt'
    )
    
    if not os.path.exists(check_path):
        return None, None, None, None
    
    state_dict = torch.load(check_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    
    # 构建text gallery
    model.build_text_feature_gallery()
    
    # 加载训练数据（support set）
    train_img_paths, _, train_labels, train_types = load_mvtec(
        category=kwargs['class_name'],
        k_shot=kwargs['k_shot']
    )
    train_dataset = CLIPDataset(
        train_img_paths[:kwargs['k_shot']],  # 只取k-shot
        [0] * kwargs['k_shot'],
        [0] * kwargs['k_shot'],
        ['good'] * kwargs['k_shot'],
        resize=kwargs['resolution']
    )
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    
    # 加载测试数据
    test_img_paths, test_gt_paths, test_labels, test_types = load_mvtec(
        category=kwargs['class_name'],
        k_shot=kwargs['k_shot']
    )
    # 测试集跳过前k_shot个样本
    test_dataset = CLIPDataset(
        test_img_paths[kwargs['k_shot']:],
        test_gt_paths[kwargs['k_shot']:],
        test_labels[kwargs['k_shot']:],
        test_types[kwargs['k_shot']:],
        resize=kwargs['resolution']
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    return model, train_loader, test_loader, device

def compute_semantic_scores(model, dataloader, device, is_baseline=False):
    """
    计算语义分支的异常分数
    
    返回：
    - normal_scores: 正常样本的分数
    - abnormal_scores: 异常样本的分数
    """
    normal_scores = []
    abnormal_scores = []
    
    with torch.no_grad():
        for data, mask, label, name, img_type in tqdm(dataloader, desc="Computing scores", leave=False):
            data = [model.transform(Image.fromarray(f.numpy())) for f in data]
            data = torch.stack(data, dim=0).to(device)
            
            # Encode image
            cls_feature, patch_tokens, _, _ = model.encode_image(data)  # [1, D]
            cls_feature = F.normalize(cls_feature, dim=-1)
            
            # Semantic branch similarity
            normal_sim = cls_feature @ model.text_features[0:1].T  # [1, 1]
            
            if is_baseline:
                # Baseline: mean abnormal anchor
                abnormal_sim = cls_feature @ model.text_features[1:2].T  # [1, 1]
                semantic_score = (1 - normal_sim + abnormal_sim).squeeze().cpu().item()
            else:
                # Multi-Abnormal: top-k aggregation (default top-2)
                if hasattr(model, 'abnormal_text_features_all'):
                    abnormal_sims = cls_feature @ model.abnormal_text_features_all.T  # [1, K+M]
                    # Top-2 aggregation
                    topk = min(2, abnormal_sims.shape[-1])
                    topk_sims, _ = torch.topk(abnormal_sims, k=topk, dim=-1)
                    abnormal_sim = topk_sims.mean(dim=-1, keepdim=True)
                else:
                    abnormal_sim = cls_feature @ model.text_features[1:2].T
                
                semantic_score = (1 - normal_sim + abnormal_sim).squeeze().cpu().item()
            
            # Collect scores
            is_normal = (img_type[0] == 'good')
            if is_normal:
                normal_scores.append(semantic_score)
            else:
                abnormal_scores.append(semantic_score)
    
    return np.array(normal_scores), np.array(abnormal_scores)

def analyze_single_class(class_name):
    """分析单个类别"""
    print(f"\n{'='*80}")
    print(f"Analyzing: {class_name}")
    print(f"{'='*80}")
    
    results = {
        'class': class_name,
        'baseline': {},
        'multi_abnormal': {}
    }
    
    # === Baseline ===
    print(f"\n[1/2] Loading Baseline model...")
    baseline_model, train_loader_base, test_loader_base, device_base = load_model_and_data(
        class_name, './result/fusion_normal', is_baseline=True
    )
    
    if baseline_model is None:
        print(f"❌ Baseline checkpoint not found for {class_name}")
        return None
    
    # Compute support set P99 (for threshold)
    print(f"[Baseline] Computing support set scores...")
    support_normal_base, _ = compute_semantic_scores(baseline_model, train_loader_base, device_base, is_baseline=True)
    thr_base = np.percentile(support_normal_base, 99)
    
    # Compute test set scores
    print(f"[Baseline] Computing test set scores...")
    normal_scores_base, abnormal_scores_base = compute_semantic_scores(baseline_model, test_loader_base, device_base, is_baseline=True)
    
    # (A) Normal side metrics
    results['baseline']['normal_median'] = np.median(normal_scores_base)
    results['baseline']['normal_p95'] = np.percentile(normal_scores_base, 95)
    results['baseline']['normal_p99'] = np.percentile(normal_scores_base, 99)
    
    # (B) Abnormal side metrics
    results['baseline']['abnormal_median'] = np.median(abnormal_scores_base)
    results['baseline']['abnormal_p95'] = np.percentile(abnormal_scores_base, 95)
    
    # (C) FPR/TPR at P99 threshold
    results['baseline']['threshold'] = thr_base
    results['baseline']['fpr'] = np.mean(normal_scores_base > thr_base)
    results['baseline']['tpr'] = np.mean(abnormal_scores_base > thr_base)
    
    del baseline_model
    torch.cuda.empty_cache()
    
    # === Multi-Abnormal ===
    print(f"\n[2/2] Loading Multi-Abnormal (Fixed) model...")
    ma_model, train_loader_ma, test_loader_ma, device_ma = load_model_and_data(
        class_name, './result/fix_validation', is_baseline=False
    )
    
    if ma_model is None:
        print(f"❌ Multi-Abnormal checkpoint not found for {class_name}")
        return None
    
    # Compute support set P99
    print(f"[Multi-Abnormal] Computing support set scores...")
    support_normal_ma, _ = compute_semantic_scores(ma_model, train_loader_ma, device_ma, is_baseline=False)
    thr_ma = np.percentile(support_normal_ma, 99)
    
    # Compute test set scores
    print(f"[Multi-Abnormal] Computing test set scores...")
    normal_scores_ma, abnormal_scores_ma = compute_semantic_scores(ma_model, test_loader_ma, device_ma, is_baseline=False)
    
    # (A) Normal side metrics
    results['multi_abnormal']['normal_median'] = np.median(normal_scores_ma)
    results['multi_abnormal']['normal_p95'] = np.percentile(normal_scores_ma, 95)
    results['multi_abnormal']['normal_p99'] = np.percentile(normal_scores_ma, 99)
    
    # (B) Abnormal side metrics
    results['multi_abnormal']['abnormal_median'] = np.median(abnormal_scores_ma)
    results['multi_abnormal']['abnormal_p95'] = np.percentile(abnormal_scores_ma, 95)
    
    # (C) FPR/TPR at P99 threshold
    results['multi_abnormal']['threshold'] = thr_ma
    results['multi_abnormal']['fpr'] = np.mean(normal_scores_ma > thr_ma)
    results['multi_abnormal']['tpr'] = np.mean(abnormal_scores_ma > thr_ma)
    
    # Sample sizes
    results['n_normal'] = len(normal_scores_base)
    results['n_abnormal'] = len(abnormal_scores_base)
    
    del ma_model
    torch.cuda.empty_cache()
    
    return results

def print_class_results(results):
    """打印单个类别的对比表"""
    if results is None:
        return
    
    cls = results['class']
    base = results['baseline']
    ma = results['multi_abnormal']
    
    print(f"\n{'='*80}")
    print(f"Class: {cls} (N_normal={results['n_normal']}, N_abnormal={results['n_abnormal']})")
    print(f"{'='*80}")
    
    print(f"\n{'Metric':<30} {'Baseline':>15} {'Multi-Abnormal':>15} {'Delta':>15}")
    print(f"{'-'*80}")
    
    # (A) Normal side
    print(f"{'--- (A) Normal Side ---':<30}")
    print(f"{'  Median(score|normal)':<30} {base['normal_median']:>15.4f} {ma['normal_median']:>15.4f} {ma['normal_median']-base['normal_median']:>15.4f}")
    print(f"{'  P95(score|normal)':<30} {base['normal_p95']:>15.4f} {ma['normal_p95']:>15.4f} {ma['normal_p95']-base['normal_p95']:>15.4f}")
    print(f"{'  P99(score|normal)':<30} {base['normal_p99']:>15.4f} {ma['normal_p99']:>15.4f} {ma['normal_p99']-base['normal_p99']:>15.4f}")
    
    # (B) Abnormal side
    print(f"\n{'--- (B) Abnormal Side ---':<30}")
    print(f"{'  Median(score|abnormal)':<30} {base['abnormal_median']:>15.4f} {ma['abnormal_median']:>15.4f} {ma['abnormal_median']-base['abnormal_median']:>15.4f}")
    print(f"{'  P95(score|abnormal)':<30} {base['abnormal_p95']:>15.4f} {ma['abnormal_p95']:>15.4f} {ma['abnormal_p95']-base['abnormal_p95']:>15.4f}")
    
    # (C) Threshold-aligned FPR/TPR
    print(f"\n{'--- (C) Threshold-Aligned ---':<30}")
    print(f"{'  Threshold (support P99)':<30} {base['threshold']:>15.4f} {ma['threshold']:>15.4f} {ma['threshold']-base['threshold']:>15.4f}")
    print(f"{'  FPR (test normal)':<30} {base['fpr']:>15.4f} {ma['fpr']:>15.4f} {ma['fpr']-base['fpr']:>15.4f}")
    print(f"{'  TPR (test abnormal)':<30} {base['tpr']:>15.4f} {ma['tpr']:>15.4f} {ma['tpr']-base['tpr']:>15.4f}")
    
    # Diagnosis
    print(f"\n{'--- Diagnosis ---':<30}")
    p99_up = (ma['normal_p99'] - base['normal_p99']) > 0.05
    fpr_up = (ma['fpr'] - base['fpr']) > 0.05
    abnormal_p95_up = (ma['abnormal_p95'] - base['abnormal_p95']) > 0
    
    if p99_up and fpr_up:
        print(f"  ✅ 假阳性假设成立：P99(normal)↑ 且 FPR↑")
    elif fpr_up:
        print(f"  ⚠️  FPR上升但P99变化不显著")
    else:
        print(f"  ❌ 假阳性假设不成立：FPR未显著上升")
    
    if abnormal_p95_up:
        print(f"  ✅ 异常侧表达能力增强：P95(abnormal)↑")
    else:
        print(f"  ❌ 异常侧表达能力未增强")

def main():
    # 从fix_validation结果中提取有数据的类别
    results_csv = './result/fix_validation/mvtec/k_2/csv/Seed_111-results.csv'
    
    test_classes = []
    with open(results_csv, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            parts = line.strip().split(',')
            cls_name = parts[0].replace('mvtec-', '')
            fusion_auroc = float(parts[1])
            if fusion_auroc > 0:  # 有数据
                test_classes.append(cls_name)
    
    print(f"\n{'='*80}")
    print(f"Multi-Abnormal Prototype 假阳性假设验证")
    print(f"{'='*80}")
    print(f"测试类别: {test_classes}")
    print(f"总计: {len(test_classes)} 个类别")
    print(f"{'='*80}\n")
    
    all_results = []
    
    for cls in test_classes:
        result = analyze_single_class(cls)
        if result is not None:
            all_results.append(result)
            print_class_results(result)
    
    # 汇总分析
    print(f"\n\n{'='*80}")
    print(f"汇总分析")
    print(f"{'='*80}")
    
    fp_driven = []
    enhanced_but_fp = []
    not_fp_driven = []
    
    for res in all_results:
        cls = res['class']
        base = res['baseline']
        ma = res['multi_abnormal']
        
        delta_p99 = ma['normal_p99'] - base['normal_p99']
        delta_fpr = ma['fpr'] - base['fpr']
        delta_abnormal_p95 = ma['abnormal_p95'] - base['abnormal_p95']
        
        # 判断类型
        if delta_p99 > 0.05 and delta_fpr > 0.05:
            fp_driven.append(cls)
        elif delta_abnormal_p95 > 0 and delta_fpr > 0.05:
            enhanced_but_fp.append(cls)
        else:
            not_fp_driven.append(cls)
    
    print(f"\n(1) 退化主要由假阳性驱动 (P99↑ & FPR↑):")
    print(f"    {fp_driven if fp_driven else '无'}")
    
    print(f"\n(2) 异常侧增强但伴随假阳性 (P95_abnormal↑ & FPR↑):")
    print(f"    {enhanced_but_fp if enhanced_but_fp else '无'}")
    
    print(f"\n(3) 假阳性假设不成立:")
    print(f"    {not_fp_driven if not_fp_driven else '无'}")
    
    # 最终结论
    print(f"\n{'='*80}")
    print(f"最终结论")
    print(f"{'='*80}")
    
    if len(fp_driven) >= len(all_results) * 0.5:
        print(f"✅ 假设成立：Multi-Abnormal Prototype 在多数类别({len(fp_driven)}/{len(all_results)})")
        print(f"   通过抬高正常样本异常响应（假阳性）导致退化。")
    elif len(enhanced_but_fp) >= len(all_results) * 0.5:
        print(f"⚠️  部分成立：Multi-Abnormal Prototype 增强了异常侧表达，")
        print(f"   但同时显著提高了假阳性率({len(enhanced_but_fp)}/{len(all_results)})。")
    else:
        print(f"❌ 假设不成立：退化不主要由假阳性驱动。")
        print(f"   需要检查其他原因（如异常侧表达能力未增强、特征质量下降等）。")
    
    print(f"\n{'='*80}\n")

if __name__ == '__main__':
    main()
