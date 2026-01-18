"""
最小化测试：只测试 grid 一个类别，alpha=0
目标：严格复现 82.0384 的 semantic AUROC
"""
import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_mvtec
from PromptAD.model import PromptAD
from sklearn.metrics import roc_auc_score


def main():
    class_name = 'grid'
    checkpoint_path = 'result/fix_validation/mvtec/k_2/checkpoint/CLS-Seed_111-grid-check_point.pt'
    expected_auroc = 82.0384
    alpha = 0.0
    
    print(f"Testing {class_name} with alpha={alpha}")
    print(f"Expected: {expected_auroc:.4f}")
    print("-" * 60)
    
    # 1. Load checkpoint
    print("Step 1: Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cuda')
    print(f"  ✓ Checkpoint keys: {list(checkpoint.keys())[:5]}...")
    
    # 2. Initialize model
    print("Step 2: Initializing model...")
    model_kwargs = {
        'out_size_h': 400,
        'out_size_w': 400,
        'device': 'cuda',
        'backbone': 'ViT-B-16-plus-240',
        'pretrained_dataset': 'laion400m_e32',
        'n_ctx': 4,
        'n_pro': 1,
        'n_ctx_ab': 1,
        'n_pro_ab': 4,
        'class_name': class_name,
        'alpha_normal_aware': alpha,
        'K_ab': 18,  # 修正：fix_validation使用18个abnormal prompts
        'semantic_topk': 2,
        't_train': 10.0,
        'label_smoothing': 0.1,
        'k_shot': 2,
        'img_resize': 240,
        'img_cropsize': 240,
    }
    model = PromptAD(**model_kwargs).cuda()
    print(f"  ✓ Model created")
    
    # 3. Load state dict
    print("Step 3: Loading state dict...")
    # Checkpoint is direct state_dict, not wrapped in 'model_state_dict'
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    print(f"  ✓ State dict loaded")
    
    # 4. Prepare dataset
    print("Step 4: Preparing dataset...")
    _, test_dataset = load_mvtec(
        dataset_path='/home/zju/mywork/PromptAD/mvtec',
        class_name=class_name,
        img_size=518,
        k_shot=2,
        seed=111
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    print(f"  ✓ Test dataset: {len(test_dataset)} samples")
    
    # 5. Run inference
    print("Step 5: Running inference...")
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch['image'].cuda()
            labels = batch['label'].cpu().numpy()
            
            # Get semantic score
            results = model(images, topk_abnormal=2)
            semantic_scores = results['semantic_anomaly_scores'].cpu().numpy()
            
            all_scores.extend(semantic_scores.tolist())
            all_labels.extend(labels.tolist())
            
            if i == 0:
                print(f"  First batch: {len(images)} images")
                print(f"  Score range: [{semantic_scores.min():.4f}, {semantic_scores.max():.4f}]")
    
    print(f"  ✓ Inference done: {len(all_scores)} predictions")
    
    # 6. Compute AUROC
    print("Step 6: Computing AUROC...")
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    n_normal = (all_labels == 0).sum()
    n_abnormal = (all_labels == 1).sum()
    print(f"  Normal samples: {n_normal}")
    print(f"  Abnormal samples: {n_abnormal}")
    
    semantic_auroc = roc_auc_score(all_labels, all_scores) * 100
    
    # 7. Report
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Class:    {class_name}")
    print(f"Alpha:    {alpha}")
    print(f"Expected: {expected_auroc:.4f}")
    print(f"Got:      {semantic_auroc:.4f}")
    print(f"Diff:     {abs(semantic_auroc - expected_auroc):.4f}")
    
    if abs(semantic_auroc - expected_auroc) < 0.01:
        print("Status:   ✅ MATCH")
    else:
        print("Status:   ❌ MISMATCH")
    print("=" * 60)


if __name__ == '__main__':
    main()
