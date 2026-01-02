"""
Test the expanded prompts table in ad_prompts.py
Verify that all classes have complete prompt lists
"""
import sys
sys.path.insert(0, '/home/zju/mywork/PromptAD')

from PromptAD.ad_prompts import expanded_class_prompts

def test_expanded_prompts():
    """Test expanded prompts structure."""
    print("="*80)
    print("EXPANDED PROMPTS TABLE TEST")
    print("="*80)
    
    print(f"\nTotal classes: {len(expanded_class_prompts)}")
    
    # Statistics
    total_prompts = 0
    min_prompts = float('inf')
    max_prompts = 0
    
    print("\n{:<20} {:>10} {:>15}".format("Class", "Prompts", "Comment"))
    print("-" * 50)
    
    for cls in sorted(expanded_class_prompts.keys()):
        prompts = expanded_class_prompts[cls]
        count = len(prompts)
        
        total_prompts += count
        min_prompts = min(min_prompts, count)
        max_prompts = max(max_prompts, count)
        
        comment = ""
        if count < 5:
            comment = "⚠ Few prompts"
        elif count > 10:
            comment = "✓ Rich"
        
        print("{:<20} {:>10} {:>15}".format(cls, count, comment))
    
    print("-" * 50)
    print(f"Average: {total_prompts / len(expanded_class_prompts):.2f}")
    print(f"Min: {min_prompts}, Max: {max_prompts}")

def test_specific_classes():
    """Test specific classes in detail."""
    print("\n\n" + "="*80)
    print("DETAILED CLASS INSPECTION")
    print("="*80)
    
    test_classes = ['metal_nut', 'capsule', 'pill', 'cable', 'transistor']
    
    for cls in test_classes:
        if cls not in expanded_class_prompts:
            print(f"\n❌ Class '{cls}' NOT FOUND!")
            continue
        
        prompts = expanded_class_prompts[cls]
        print(f"\n[{cls.upper()}] - {len(prompts)} prompts")
        print("-" * 60)
        
        lap_count = 0
        map_count = 0
        commented_count = 0
        
        for i, prompt in enumerate(prompts, 1):
            # Count LAP vs MAP (approximate based on common patterns)
            if i <= 2:  # First 2 are typically LAP
                lap_count += 1
                print(f"  LAP {lap_count}: {prompt}")
            else:
                map_count += 1
                print(f"  MAP {map_count}: {prompt}")
        
        print(f"  Summary: {lap_count} LAP + {map_count} MAP = {len(prompts)} total")

def test_purge_annotations():
    """Check if Purge1/2/3 annotations are preserved in comments."""
    print("\n\n" + "="*80)
    print("PURGE ANNOTATION CHECK")
    print("="*80)
    
    import inspect
    import PromptAD.ad_prompts as module
    
    source = inspect.getsource(module)
    
    # Count comment markers
    purge1_count = source.count('❌ Purge1 removed')
    purge2_count = source.count('❌ Purge2 removed')
    purge3_count = source.count('✅ Purge3 restored')
    
    print(f"\n✓ Purge1 annotations: {purge1_count}")
    print(f"✓ Purge2 annotations: {purge2_count}")
    print(f"✓ Purge3 annotations: {purge3_count}")
    
    if purge1_count > 0 and purge2_count > 0:
        print("\n✓ All purge history preserved in comments!")
    else:
        print("\n⚠ Warning: Some purge annotations may be missing")

def test_class_mapping():
    """Test that class_mapping is properly applied."""
    print("\n\n" + "="*80)
    print("CLASS MAPPING TEST")
    print("="*80)
    
    # Check classes that should have mapping
    mappings = {
        'metal_nut': 'metal nut',
        'pcb1': 'printed circuit board',
        'macaroni1': 'macaroni',
        'chewinggum': 'chewing gum',
        'pipe_fryum': 'pipe fryum'
    }
    
    print("\n{:<20} {:<30} {:>10}".format("Class Key", "Expected Display Name", "Status"))
    print("-" * 65)
    
    for cls_key, expected_name in mappings.items():
        if cls_key in expanded_class_prompts:
            prompts = expanded_class_prompts[cls_key]
            # Check if first LAP prompt contains the expected name
            first_prompt = prompts[0]
            
            if expected_name in first_prompt:
                status = "✓"
            else:
                status = f"✗ ({first_prompt})"
            
            print("{:<20} {:<30} {:>10}".format(cls_key, expected_name, status))
        else:
            print("{:<20} {:<30} {:>10}".format(cls_key, expected_name, "✗ Missing"))

def compare_purge_config():
    """Compare with Purge3 expected configuration."""
    print("\n\n" + "="*80)
    print("PURGE3 CONFIGURATION VERIFICATION")
    print("="*80)
    
    # Expected counts after Purge3
    expected = {
        'metal_nut': 4,  # 2 LAP + 2 MAP (Purge2)
        'pill': 7,       # 2 LAP + 5 MAP (Purge2)
        'cable': 5,      # 2 LAP + 3 MAP (Purge2)
        'capsule': 7,    # 2 LAP + 5 MAP (Purge3 restored)
        'transistor': 4, # 2 LAP + 2 MAP (Purge2)
        'screw': 5       # 2 LAP + 3 MAP (baseline)
    }
    
    print("\n{:<20} {:>10} {:>10} {:>10}".format("Class", "Expected", "Actual", "Status"))
    print("-" * 55)
    
    all_match = True
    for cls, exp_count in expected.items():
        actual_count = len(expanded_class_prompts.get(cls, []))
        status = "✓" if actual_count == exp_count else "✗"
        
        if actual_count != exp_count:
            all_match = False
        
        print("{:<20} {:>10} {:>10} {:>10}".format(cls, exp_count, actual_count, status))
    
    if all_match:
        print("\n✓ All Purge3 configurations match!")
    else:
        print("\n⚠ Some configurations don't match expected Purge3 state")

if __name__ == "__main__":
    test_expanded_prompts()
    test_specific_classes()
    test_purge_annotations()
    test_class_mapping()
    compare_purge_config()
    
    print("\n\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80)
