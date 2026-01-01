"""
Test script to visualize expanded prompts and verify the new structure.
"""
import sys
sys.path.insert(0, '/home/zju/mywork/PromptAD')

from PromptAD.ad_prompts_expanded import (
    get_all_prompts_for_class, 
    print_prompt_table,
    generic_lap_prompts,
    class_specific_map_prompts
)

def test_basic_functionality():
    """Test basic prompt retrieval."""
    print("="*80)
    print("TEST 1: Basic Functionality")
    print("="*80)
    
    # Test a few classes
    test_classes = ['metal_nut', 'capsule', 'bottle', 'pcb1']
    
    for cls in test_classes:
        prompts_with_lap = get_all_prompts_for_class(cls, use_lap=True)
        prompts_no_lap = get_all_prompts_for_class(cls, use_lap=False)
        
        print(f"\n[{cls}]")
        print(f"  With LAP: {len(prompts_with_lap)} prompts")
        print(f"  Without LAP: {len(prompts_no_lap)} prompts")
        print(f"  Samples:")
        for i, p in enumerate(prompts_with_lap[:3], 1):
            print(f"    {i}. {p}")

def test_prompt_table():
    """Test prompt table printing."""
    print("\n\n")
    print("="*80)
    print("TEST 2: Prompt Table Visualization")
    print("="*80)
    
    # Print table for a subset of classes
    test_classes = ['metal_nut', 'pill', 'cable', 'capsule', 'transistor']
    for cls in test_classes:
        print_prompt_table(cls)

def test_purge3_configuration():
    """Verify Purge3 configuration matches expected state."""
    print("\n\n")
    print("="*80)
    print("TEST 3: Purge3 Configuration Verification")
    print("="*80)
    
    # Verify LAP count
    print(f"\nGeneric LAP prompts: {len(generic_lap_prompts)}")
    assert len(generic_lap_prompts) == 2, "Should have 2 LAP prompts after Purge1"
    print("✓ LAP count correct")
    
    # Verify specific classes after Purge2/3
    purge_results = {
        'metal_nut': 2,  # Removed 2 prompts
        'pill': 5,       # Removed 1 prompt
        'cable': 3,      # Removed 2 prompts
        'capsule': 5,    # Restored to Purge1
        'transistor': 2, # Removed 2 prompts
        'screw': 3       # No changes
    }
    
    print(f"\nClass-specific MAP counts:")
    for cls, expected_count in purge_results.items():
        actual_count = len(class_specific_map_prompts[cls])
        status = "✓" if actual_count == expected_count else "✗"
        print(f"  {status} {cls}: {actual_count} (expected {expected_count})")
        
        if actual_count != expected_count:
            print(f"    Prompts: {class_specific_map_prompts[cls]}")

def compare_with_original():
    """Compare with original ad_prompts.py structure."""
    print("\n\n")
    print("="*80)
    print("TEST 4: Comparison with Original ad_prompts.py")
    print("="*80)
    
    from PromptAD.ad_prompts import state_anomaly, class_state_abnormal, class_mapping as orig_mapping
    
    print(f"\nOriginal state_anomaly (LAP): {len(state_anomaly)} prompts")
    print(f"Expanded generic_lap_prompts: {len(generic_lap_prompts)} prompts")
    print(f"Match: {len(state_anomaly) == len(generic_lap_prompts)}")
    
    print(f"\nOriginal class_state_abnormal: {len(class_state_abnormal)} classes")
    print(f"Expanded class_specific_map_prompts: {len(class_specific_map_prompts)} classes")
    print(f"Match: {len(class_state_abnormal) == len(class_specific_map_prompts)}")
    
    # Check specific class (with mapping consideration)
    test_cls = 'metal_nut'
    display_name = orig_mapping.get(test_cls, test_cls)
    original_prompts = [p.format(display_name) for p in state_anomaly + class_state_abnormal[test_cls]]
    expanded_prompts = get_all_prompts_for_class(test_cls, use_lap=True)
    
    print(f"\n[{test_cls}] Comparison (display name: '{display_name}'):")
    print(f"  Original (formatted): {len(original_prompts)} prompts")
    print(f"  Expanded (static): {len(expanded_prompts)} prompts")
    print(f"  Match: {len(original_prompts) == len(expanded_prompts)}")
    
    if len(original_prompts) == len(expanded_prompts):
        print("\n  Content comparison:")
        all_match = True
        for i, (orig, exp) in enumerate(zip(original_prompts, expanded_prompts), 1):
            if orig != exp:
                print(f"    ✗ Prompt {i} mismatch:")
                print(f"      Original: {orig}")
                print(f"      Expanded: {exp}")
                all_match = False
        if all_match:
            print("    ✓ All prompts match!")
    
    # Also test a class without mapping
    test_cls2 = 'bottle'
    original_prompts2 = [p.format(test_cls2) for p in state_anomaly + class_state_abnormal[test_cls2]]
    expanded_prompts2 = get_all_prompts_for_class(test_cls2, use_lap=True)
    
    print(f"\n[{test_cls2}] Comparison:")
    print(f"  Original (formatted): {len(original_prompts2)} prompts")
    print(f"  Expanded (static): {len(expanded_prompts2)} prompts")
    
    all_match2 = True
    for orig, exp in zip(original_prompts2, expanded_prompts2):
        if orig != exp:
            all_match2 = False
            break
    
    print(f"  Content match: {all_match2}")

def generate_summary_table():
    """Generate summary statistics for all classes."""
    print("\n\n")
    print("="*80)
    print("TEST 5: Summary Statistics")
    print("="*80)
    
    print("\n{:<20} {:>8} {:>8} {:>8}".format("Class", "LAP", "MAP", "Total"))
    print("-" * 48)
    
    total_lap = 0
    total_map = 0
    
    for cls in sorted(class_specific_map_prompts.keys()):
        lap_count = len(generic_lap_prompts)
        map_count = len(class_specific_map_prompts[cls])
        total_count = lap_count + map_count
        
        print("{:<20} {:>8} {:>8} {:>8}".format(cls, lap_count, map_count, total_count))
        
        total_lap += lap_count
        total_map += map_count
    
    print("-" * 48)
    print("{:<20} {:>8} {:>8} {:>8}".format(
        f"Total ({len(class_specific_map_prompts)} classes)",
        total_lap,
        total_map,
        total_lap + total_map
    ))
    
    avg_map = total_map / len(class_specific_map_prompts)
    print(f"\nAverage MAP prompts per class: {avg_map:.2f}")

if __name__ == "__main__":
    test_basic_functionality()
    test_prompt_table()
    test_purge3_configuration()
    compare_with_original()
    generate_summary_table()
    
    print("\n\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80)
