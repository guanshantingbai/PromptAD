"""验证alpha=0时复现训练时的mean aggregation结果"""
import torch
import numpy as np
import torch.nn.functional as F
from datasets import get_dataloader_from_args
from PromptAD.model import PromptAD
from sklearn.metrics import roc_auc_score


def test_with_train_mode(class_name, expected_auroc):
    """使用训练时的完全相同逻辑测试"""
    print(f"\n{'='*60}")
    print(f"Class: {class_name}")
    print(f"Expected AUROC: {expected_auroc:.4f}")
    print(f"Mode: Training evaluation (mean aggregation, no topk)")
    print(f"{'='*60}")
    
    kwargs = {
        'dataset': 'mvtec', 'class_name': class_name, 'k_shot': 2, 'seed': 111,
        'backbone': 'ViT-B-16-plus-240', 'pretrained_dataset': 'laion400m_e32',
        'n_ctx': 12, 'n_pro': 1, 'n_ctx_ab': 12, 'n_pro_ab': 4,
        'device': 'cuda:0', 'resolution': 256, 'img_resize': 256,
        'img_cropsize': 256, 'out_size_h': 256, 'out_size_w': 256,
        'batch_size': 1, 'data_path': './data/mvtec',
        'alpha_normal_aware': 0.0,  # 🔑 alpha=0
    }
    
    model = PromptAD(**kwargs)
    model.to('cuda:0')
    model.eval()
    
    # 加载checkpoint（只包含5个keys）
    ckpt = torch.load(f'./result/fix_validation/mvtec/k_2/checkpoint/CLS-Seed_111-{class_name}-check_point.pt')
    
    # 🔑 关键：直接加载完整state_dict（不是只加载text_features）
    # 这会覆盖feature_gallery和text_features
    checkpoint_filtered = {k: v for k, v in ckpt.items() if not k.endswith('_all')}
    model.load_state_dict(checkpoint_filtered, strict=False)
    model.normal_text_features_all = ckpt['normal_text_features_all'].cuda()
    model.abnormal_text_features_all = ckpt['abnormal_text_features_all'].cuda()
    
    print(f"✓ Loaded checkpoint")
    print(f"  text_features: {model.text_features.shape}")
    print(f"  normal_text_features_all: {model.normal_text_features_all.shape}")
    print(f"  abnormal_text_features_all: {model.abnormal_text_features_all.shape}")
    
    # ⚠️ 关键：不设置topk_abnormal，让它保持None（使用mean aggregation）
    # model.topk_abnormal = None  # 默认就是None
    
    test_loader, _ = get_dataloader_from_args(
        phase='test', perturbed=False, transform=model.transform, **kwargs
    )
    print(f"✓ Loaded {len(test_loader.dataset)} test samples")
    
    # 使用训练时的完全相同逻辑
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for data, _, label, _, img_type in test_loader:
            data = data.cuda()
            
            # 🔑 使用model(data, 'cls')，完全复现训练时的evaluate逻辑
            semantic_scores, memory_scores, _ = model(data, 'cls')
            
            # semantic_scores已经是list，转numpy
            all_scores.extend(semantic_scores)
            all_labels.append(1 if img_type[0] != 'good' else 0)
    
    auroc = roc_auc_score(all_labels, all_scores) * 100
    
    print(f"\n{'='*60}")
    print(f"RESULT:")
    print(f"  Expected: {expected_auroc:.4f}")
    print(f"  Got:      {auroc:.4f}")
    print(f"  Diff:     {abs(auroc - expected_auroc):.4f}")
    is_match = abs(auroc - expected_auroc) < 0.1
    print(f"  Status:   {'✅ MATCH' if is_match else '❌ MISMATCH'}")
    print(f"{'='*60}\n")
    
    return {'class': class_name, 'expected': expected_auroc, 'got': auroc, 'match': is_match}


if __name__ == '__main__':
    # 测试2个类别
    results = []
    for cls, exp in [('grid', 82.0384), ('bottle', 85.2778)]:
        result = test_with_train_mode(cls, exp)
        results.append(result)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in results:
        status = "✅" if r['match'] else "❌"
        print(f"{status} {r['class']:12s} | Exp: {r['expected']:7.4f} | Got: {r['got']:7.4f}")
    print("="*60)
    
    if all(r['match'] for r in results):
        print("🎉 Alpha=0 successfully reproduces training CSV results!")
    else:
        print("⚠️  Some tests failed")
