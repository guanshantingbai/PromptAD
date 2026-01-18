"""验证alpha=0时退化为旧结果 - 使用与mini_agg_test.py相同的测试方式"""
import torch
import numpy as np
import torch.nn.functional as F
from datasets import get_dataloader_from_args
from PromptAD.model import PromptAD
from sklearn.metrics import roc_auc_score


def test_single_class(class_name, expected_auroc, alpha_value=0.0):
    """测试单个类别"""
    print(f"\n{'='*60}")
    print(f"Class: {class_name}")
    print(f"Alpha: {alpha_value}")
    print(f"Expected AUROC: {expected_auroc:.4f}")
    print(f"{'='*60}")
    
    # 模型参数（与mini_agg_test.py一致）
    kwargs = {
        'dataset': 'mvtec', 'class_name': class_name, 'k_shot': 2, 'seed': 111,
        'backbone': 'ViT-B-16-plus-240', 'pretrained_dataset': 'laion400m_e32',
        'n_ctx': 12, 'n_pro': 1, 'n_ctx_ab': 12, 'n_pro_ab': 4,
        'device': 'cuda:0', 'resolution': 256, 'img_resize': 256,
        'img_cropsize': 256, 'out_size_h': 256, 'out_size_w': 256,
        'batch_size': 1, 'data_path': './data/mvtec',
        'alpha_normal_aware': alpha_value,  # 🔑 alpha参数
    }
    
    model = PromptAD(**kwargs)
    model.to('cuda:0')
    model.eval()
    
    # 加载checkpoint
    ckpt_path = f'./result/fix_validation/mvtec/k_2/checkpoint/CLS-Seed_111-{class_name}-check_point.pt'
    ckpt = torch.load(ckpt_path)
    model.text_features = ckpt['text_features'].cuda()
    model.abnormal_text_features_all = ckpt['abnormal_text_features_all'].cuda()
    print(f"✓ Loaded checkpoint")
    print(f"  text_features: {model.text_features.shape}")
    print(f"  abnormal_text_features_all: {model.abnormal_text_features_all.shape}")
    
    # 加载测试数据
    test_loader, _ = get_dataloader_from_args(
        phase='test', perturbed=False, transform=model.transform, **kwargs
    )
    print(f"✓ Loaded {len(test_loader.dataset)} test samples")
    
    # 推理
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for data, _, label, _, img_type in test_loader:
            data = data.cuda()
            
            # 提取特征
            feat_list = model.encode_image(data)
            cls_feat = F.normalize(feat_list[0], dim=-1)  # [1, 640]
            
            # 计算相似度
            normal_sim = (cls_feat @ model.text_features[0:1].T).squeeze()  # 标量
            ab_sims = cls_feat @ model.abnormal_text_features_all.T  # [1, K]
            
            # 🔑 Normal-aware correction (alpha=0时无修正)
            ab_sims_corrected = ab_sims - alpha_value * normal_sim.unsqueeze(0)
            
            # Top-2 aggregation
            abnormal_sim = ab_sims_corrected.topk(2)[0].mean()
            
            # 计算异常分数（与mini_agg_test.py一致）
            score = (1 - normal_sim + abnormal_sim).cpu().item()
            
            all_scores.append(score)
            all_labels.append(1 if img_type[0] != 'good' else 0)
    
    # 计算AUROC
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    auroc = roc_auc_score(all_labels, all_scores) * 100
    
    # 报告结果
    print(f"\n{'='*60}")
    print(f"RESULT:")
    print(f"  Expected: {expected_auroc:.4f}")
    print(f"  Got:      {auroc:.4f}")
    print(f"  Diff:     {abs(auroc - expected_auroc):.4f}")
    
    is_match = abs(auroc - expected_auroc) < 0.01
    print(f"  Status:   {'✅ MATCH' if is_match else '❌ MISMATCH'}")
    print(f"{'='*60}\n")
    
    return {
        'class': class_name,
        'expected': expected_auroc,
        'got': auroc,
        'diff': abs(auroc - expected_auroc),
        'match': is_match
    }


if __name__ == '__main__':
    # 从aggregation_comparison_results.json中选择2个类别
    test_cases = [
        {'class': 'grid', 'expected': 82.0384},
        {'class': 'bottle', 'expected': 85.2778},
    ]
    
    results = []
    for case in test_cases:
        result = test_single_class(case['class'], case['expected'], alpha_value=0.0)
        results.append(result)
    
    # 总结
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    all_match = True
    for r in results:
        status = "✅" if r['match'] else "❌"
        print(f"{status} {r['class']:12s} | Exp: {r['expected']:7.4f} | Got: {r['got']:7.4f} | Diff: {r['diff']:.4f}")
        all_match = all_match and r['match']
    
    print("="*60)
    if all_match:
        print("🎉 All tests passed! Alpha=0 reproduces fix_validation results.")
    else:
        print("⚠️  Some tests failed. Check implementation.")
    print("="*60)
