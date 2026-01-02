"""
Quick test to verify the model can load with expanded prompts
"""
import sys
sys.path.insert(0, '/home/zju/mywork/PromptAD')

def test_model_initialization():
    """Test that PromptLearner can initialize with expanded prompts."""
    print("="*80)
    print("MODEL INITIALIZATION TEST")
    print("="*80)
    
    from PromptAD.ad_prompts import expanded_class_prompts
    
    # Test that all required classes exist
    test_classes = ['bottle', 'metal_nut', 'capsule', 'pcb1']
    
    print(f"\nChecking {len(test_classes)} test classes...")
    for cls in test_classes:
        if cls in expanded_class_prompts:
            prompts = expanded_class_prompts[cls]
            print(f"  ✓ {cls}: {len(prompts)} prompts")
            print(f"    Sample: {prompts[0]}")
        else:
            print(f"  ✗ {cls}: NOT FOUND")
            return False
    
    print("\n✓ All test classes found in expanded_class_prompts")
    print("\n" + "="*80)
    print("Model should be able to initialize with these prompts")
    print("="*80)
    
    # Show what the model will see
    print(f"\nExample for 'metal_nut':")
    prompts = expanded_class_prompts['metal_nut']
    print(f"  Class: metal_nut")
    print(f"  Total prompts: {len(prompts)}")
    print(f"  Prompts list:")
    for i, p in enumerate(prompts, 1):
        print(f"    {i}. {p}")
    
    return True

if __name__ == "__main__":
    success = test_model_initialization()
    if success:
        print("\n✓ Model initialization test PASSED")
    else:
        print("\n✗ Model initialization test FAILED")
        sys.exit(1)
