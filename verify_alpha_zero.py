"""
验证 alpha=0 时是否严格复现 fix_validation 的结果
选择 2 个类别：grid 和 bottle
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_mvtec
from PromptAD.model import PromptAD
from sklearn.metrics import roc_auc_score
import json


def test_class(class_name, checkpoint_path, alpha_value, expected_semantic_auroc):
    """测试单个类别"""
    print(f"\n{'='*60}")
    print(f"Testing {class_name} with alpha={alpha_value}")
    print(f"Expected semantic AUROC: {expected_semantic_auroc:.4f}")
    print(f"{'='*60}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    print(f"✓ Loaded checkpoint from {checkpoint_path}")
    
    # Initialize model
    model_kwargs = {
        'alpha_normal_aware': alpha_value,  # 🔑 设置 alpha
        'K_ab': 2,  # fix_validation 使用 K=2
        'semantic_topk': 2,  # fix_validation 使用 top-2
        't_train': 10.0,
        'label_smoothing': 0.1
    }
    model = PromptAD(obj=class_name, **model_kwargs).cuda()
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    print(f"✓ Model initialized with alpha={alpha_value}")
    
    # Prepare test dataset
    _, test_dataset = load_mvtec(
        dataset_path='/home/zju/mywork/PromptAD/mvtec',
        class_name=class_name,
        img_size=518,
        k_shot=2,
        seed=111
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    print(f"✓ Test dataset loaded: {len(test_dataset)} samples")
    
    # Inference
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].cuda()
            labels = batch['label'].cpu().numpy()
            
            # Get semantic anomaly score
            results = model(images, topk_abnormal=2)  # 使用 top-2 聚合
            semantic_scores = results['semantic_anomaly_scores'].cpu().numpy()
            
            all_scores.extend(semantic_scores.tolist())
            all_labels.extend(labels.tolist())
    
    # Compute AUROC
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    semantic_auroc = roc_auc_score(all_labels, all_scores) * 100
    
    print(f"\n{'='*60}")
    print(f"Results for {class_name}:")
    print(f"  Expected: {expected_semantic_auroc:.4f}")
    print(f"  Got:      {semantic_auroc:.4f}")
    print(f"  Diff:     {abs(semantic_auroc - expected_semantic_auroc):.4f}")
    
    # 判断是否复现
    is_match = abs(semantic_auroc - expected_semantic_auroc) < 0.01  # 允许 0.01 误差
    if is_match:
        print(f"  Status:   ✅ MATCH (diff < 0.01)")
    else:
        print(f"  Status:   ❌ MISMATCH (diff >= 0.01)")
    print(f"{'='*60}\n")
    
    return {
        'class_name': class_name,
        'expected': expected_semantic_auroc,
        'got': semantic_auroc,
        'diff': abs(semantic_auroc - expected_semantic_auroc),
        'match': is_match
    }


def main():
    # 从 Seed_111-results.csv 中选择的 2 个类别
    test_cases = [
        {
            'class_name': 'grid',
            'checkpoint': 'result/fix_validation/mvtec/k_2/checkpoint/CLS-Seed_111-grid-check_point.pt',
            'expected_semantic_auroc': 82.0384
        },
        {
            'class_name': 'bottle',
            'checkpoint': 'result/fix_validation/mvtec/k_2/checkpoint/CLS-Seed_111-bottle-check_point.pt',
            'expected_semantic_auroc': 85.2778
        }
    ]
    
    alpha = 0.0  # 🔑 验证 alpha=0 时的退化行为
    
    results = []
    for case in test_cases:
        result = test_class(
            class_name=case['class_name'],
            checkpoint_path=case['checkpoint'],
            alpha_value=alpha,
            expected_semantic_auroc=case['expected_semantic_auroc']
        )
        results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    for res in results:
        status = "✅ PASS" if res['match'] else "❌ FAIL"
        print(f"{res['class_name']:15s} | Exp: {res['expected']:.4f} | Got: {res['got']:.4f} | Diff: {res['diff']:.4f} | {status}")
    
    all_match = all(r['match'] for r in results)
    print("="*60)
    if all_match:
        print("🎉 ALL TESTS PASSED - alpha=0 successfully reproduces fix_validation results!")
    else:
        print("⚠️  SOME TESTS FAILED - check implementation")
    print("="*60 + "\n")
    
    # Save results
    with open('alpha_zero_verification.json', 'w') as f:
        json.dump({
            'alpha': alpha,
            'test_cases': results,
            'all_match': all_match
        }, f, indent=2)
    print("Results saved to alpha_zero_verification.json")


if __name__ == '__main__':
    main()
