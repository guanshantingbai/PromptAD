"""
Test script to verify the new experimental architectures work correctly.
"""
import torch
import sys
sys.path.append('.')

from PromptAD.CLIPAD import model as clip_model
from experimental_configs import get_config, list_configs


def test_architecture(config_name, batch_size=2, image_size=224):
    """
    Test a specific architecture configuration.
    
    Args:
        config_name: Configuration name
        batch_size: Batch size for test input
        image_size: Image size
    """
    print(f"\n{'='*60}")
    print(f"Testing configuration: {config_name}")
    print(f"{'='*60}")
    
    # Get configuration
    config = get_config(config_name)
    print(f"Config: {config}")
    
    # Extract description and remove it from config before passing to CLIPVisionCfg
    description = config.pop('description', '')
    
    # Create vision config
    vision_cfg = clip_model.CLIPVisionCfg(
        layers=12,
        width=768,
        head_width=64,
        patch_size=16,
        image_size=image_size,
        **config
    )
    
    # Build vision tower
    embed_dim = 512
    try:
        visual = clip_model._build_vision_tower(
            embed_dim=embed_dim,
            vision_cfg=vision_cfg,
            quick_gelu=False,
            cast_dtype=None,
        )
        print(f"✓ Model built successfully")
        print(f"  Model type: {type(visual).__name__}")
        
        # Count parameters
        total_params = sum(p.numel() for p in visual.parameters())
        trainable_params = sum(p.numel() for p in visual.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        visual.eval()
        with torch.no_grad():
            x = torch.randn(batch_size, 3, image_size, image_size)
            output = visual(x)
            
            if isinstance(output, tuple):
                print(f"✓ Forward pass successful")
                print(f"  Number of outputs: {len(output)}")
                for i, out in enumerate(output):
                    if out is not None:
                        print(f"  Output[{i}] shape: {out.shape}")
                    else:
                        print(f"  Output[{i}]: None")
            else:
                print(f"✓ Forward pass successful")
                print(f"  Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all_configs():
    """Test all experimental configurations."""
    print("\n" + "="*60)
    print("Testing All Experimental Configurations")
    print("="*60)
    
    list_configs()
    
    configs_to_test = [
        'original',
        'qq_residual',
        'kk_residual',
        'vv_residual',
        'qq_no_residual',
        'kk_no_residual',
        'vv_no_residual',
    ]
    
    results = {}
    for config_name in configs_to_test:
        success = test_architecture(config_name, batch_size=2, image_size=224)
        results[config_name] = success
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for config_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {config_name}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed")
    
    return all_passed


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test experimental architectures')
    parser.add_argument('--config', type=str, default='all',
                       help='Configuration to test (default: all)')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Batch size for testing (default: 2)')
    parser.add_argument('--image-size', type=int, default=224,
                       help='Image size (default: 224)')
    
    args = parser.parse_args()
    
    if args.config == 'all':
        test_all_configs()
    else:
        test_architecture(args.config, args.batch_size, args.image_size)
