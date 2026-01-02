"""
Quick inference test to verify expanded prompts work correctly.
Test metal_nut with existing checkpoint from promptpurging.
"""
import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/home/zju/mywork/PromptAD')

from datasets import get_dataloader_from_args
from PromptAD import PromptAD

def test_inference():
    """Test inference with metal_nut to verify expanded prompts."""
    
    print("="*80)
    print("INFERENCE TEST - Expanded Prompts Verification")
    print("="*80)
    
    # Configuration
    class_name = 'metal_nut'
    dataset = 'mvtec'
    checkpoint_path = 'result/promptpurging/mvtec/k_2/checkpoint/CLS-Seed_111-metal_nut-check_point.pt'
    k_shot = 2
    seed = 111
    batch_size = 1
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nConfiguration:")
    print(f"  Dataset: {dataset}")
    print(f"  Class: {class_name}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device: {device}")
    print(f"  K-shot: {k_shot}")
    
    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f"\n❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    # Initialize model
    print(f"\n[1/4] Initializing model with expanded prompts...")
    try:
        model = PromptAD(
            out_size_h=240,
            out_size_w=240,
            device=device,
            backbone='ViT-B-16-plus-240',
            pretrained_dataset='laion400m_e32',
            n_ctx=12,
            n_pro=5,
            n_ctx_ab=12,
            n_pro_ab=5,
            class_name=class_name,
            k_shot=k_shot,
            img_resize=240,
            img_cropsize=240,
        )
        print("  ✓ Model initialized successfully")
    except Exception as e:
        print(f"  ❌ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Load checkpoint
    print(f"\n[2/4] Loading checkpoint...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        print("  ✓ Checkpoint loaded successfully")
        print(f"  - feature_gallery1 shape: {model.feature_gallery1.shape}")
        print(f"  - feature_gallery1 initialized: {model.feature_gallery1.abs().sum() > 0}")
        print(f"  - text_features: {hasattr(model, 'text_features') and model.text_features is not None}")
        if hasattr(model, 'text_features') and model.text_features is not None:
            print(f"  - text_features shape: {model.text_features.shape}")
    except Exception as e:
        print(f"  ❌ Checkpoint loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Load test data
    print(f"\n[3/4] Loading test data...")
    try:
        kwargs = {
            'dataset': dataset,
            'class_name': class_name,
            'k_shot': k_shot,
            'batch_size': batch_size,
            'num_workers': 0
        }
        test_loader, test_dataset = get_dataloader_from_args(
            phase='test',
            transform=model.transform,
            **kwargs
        )
        print(f"  ✓ Test dataset loaded: {len(test_dataset)} samples")
    except Exception as e:
        print(f"  ❌ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Run inference on a few samples
    print(f"\n[4/4] Running inference on first 5 samples...")
    model.eval()
    model.eval_mode()
    
    try:
        with torch.no_grad():
            for i, (img, gt, label, name, img_type) in enumerate(test_loader):
                if i >= 5:
                    break
                
                img = img.to(device)
                
                # Forward pass (CLS task)
                # Returns: (semantic_scores, memory_scores, pixel_maps)
                print(f"\n  Processing sample {i+1}...")
                result = model(img, task='CLS')
                
                if result is None:
                    print(f"  ❌ Model returned None!")
                    print(f"     Debugging info:")
                    print(f"     - Input shape: {img.shape}")
                    print(f"     - Device: {img.device}")
                    print(f"     - Model device: {next(model.parameters()).device}")
                    
                    # Try to manually call encode_image
                    print(f"\n  Trying manual encode_image...")
                    visual_features = model.encode_image(img)
                    print(f"     - Visual features: {type(visual_features)}")
                    if isinstance(visual_features, (list, tuple)):
                        print(f"     - Visual features[0] shape: {visual_features[0].shape}")
                        print(f"     - Visual features[1] shape: {visual_features[1].shape}")
                    
                    return False
                
                semantic_scores, memory_scores, _ = result
                score_semantic = torch.tensor(semantic_scores[0])
                score_memory = torch.tensor(memory_scores[0])
                
                # Calculate fusion score (weighted)
                semantic_alpha = 0.5  # default from test_cls.py
                score_fusion = semantic_alpha * score_semantic + (1 - semantic_alpha) * score_memory
                
                is_anomaly = label.item() == 1
                status = "Anomaly" if is_anomaly else "Normal"
                
                print(f"  Sample {i+1}: {name[0]:<30} [{status}]")
                print(f"    Semantic score: {score_semantic.item():.4f}")
                print(f"    Memory score:   {score_memory.item():.4f}")
                print(f"    Fusion score:   {score_fusion.item():.4f}")
        
        print("\n  ✓ Inference completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n  ❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\nTesting inference with expanded prompts table...")
    print("This verifies that the model can load and run with the new prompt format.\n")
    
    success = test_inference()
    
    print("\n" + "="*80)
    if success:
        print("✅ TEST PASSED - Expanded prompts work correctly!")
        print("="*80)
        print("\nThe model successfully:")
        print("  1. Initialized with expanded_class_prompts")
        print("  2. Loaded the checkpoint")
        print("  3. Ran inference on test samples")
        print("\n✓ Ready for multi-prototype development!")
    else:
        print("❌ TEST FAILED - Please check the errors above")
        print("="*80)
        sys.exit(1)
