#!/usr/bin/env python3
"""
验证 Phase 1 清洗效果
测试一个类别，验证禁用的 prompts 是否真的不再加载
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.abspath('.'))

from PromptAD.ad_prompts import load_prompts_from_table

# 6个被禁用的 prompt templates
DISABLED_PROMPTS = [
    "imperfect {}",
    "flawed {}",
    "{} with defect",
    "blemished {}",
    "abnormal {}",
    "{} with flaw",
]


def test_prompt_loading(classname='bottle'):
    """测试 prompt 加载"""
    
    print("="*70)
    print(f"Testing Prompt Loading for '{classname}'")
    print("="*70)
    
    # 加载 prompts
    full_texts, prompt_info = load_prompts_from_table(classname)
    
    print(f"\nLoaded {len(full_texts)} prompts")
    print(f"\nPrompt details:")
    print("-"*70)
    
    for i, (text, info) in enumerate(zip(full_texts, prompt_info)):
        print(f"{i+1:2d}. [{info['type']:8s}] {info['template']:20s} -> {text}")
    
    # 检查是否有被禁用的 prompts
    print(f"\n{'='*70}")
    print("Validation:")
    print("="*70)
    
    templates_loaded = [info['template'] for info in prompt_info]
    
    # 检查每个被禁用的 template
    found_disabled = []
    for template in DISABLED_PROMPTS:
        if template in templates_loaded:
            found_disabled.append(template)
    
    if len(found_disabled) > 0:
        print(f"❌ ERROR: Found {len(found_disabled)} disabled prompts still loaded:")
        for template in found_disabled:
            print(f"   - '{template}'")
    else:
        print(f"✅ SUCCESS: All {len(DISABLED_PROMPTS)} disabled prompts are excluded")
        print(f"   Remaining prompts: {len(full_texts)}")
    
    # 显示被禁用的 prompts（期望不在列表中）
    print(f"\n{'='*70}")
    print("Expected to be ABSENT (disabled prompts):")
    print("="*70)
    for template in DISABLED_PROMPTS:
        status = "❌ PRESENT" if template in templates_loaded else "✓ Absent"
        print(f"  {status}: '{template}'")
    
    return len(found_disabled) == 0


def test_multiple_classes():
    """测试多个类别"""
    
    test_classes = ['bottle', 'cable', 'grid', 'toothbrush']
    
    print("\n" + "="*70)
    print("Testing Multiple Classes")
    print("="*70)
    
    results = {}
    
    for classname in test_classes:
        print(f"\n--- {classname} ---")
        full_texts, _ = load_prompts_from_table(classname)
        results[classname] = len(full_texts)
        print(f"  Loaded {len(full_texts)} prompts")
    
    print(f"\n{'='*70}")
    print("Summary:")
    print("="*70)
    for classname, count in results.items():
        print(f"  {classname:15s}: {count} prompts")
    
    print(f"\nExpected: 5-9 prompts per class (after removing 6 generic prompts)")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify Phase 1 prompt cleaning')
    parser.add_argument('--class', dest='classname', type=str, default='bottle',
                        help='Class name to test')
    parser.add_argument('--multi', action='store_true',
                        help='Test multiple classes')
    
    args = parser.parse_args()
    
    if args.multi:
        test_multiple_classes()
    else:
        success = test_prompt_loading(args.classname)
        
        if success:
            print(f"\n{'='*70}")
            print("✅ Phase 1 Verification PASSED")
            print("="*70)
            print("  Disabled prompts are correctly excluded")
            print("  Ready to run re-inference with cleaned prompts")
            print("\nNext step:")
            print("  bash bash/phase1_reinference.sh mvtec 2")
            print("="*70)
        else:
            print(f"\n{'='*70}")
            print("❌ Phase 1 Verification FAILED")
            print("="*70)
            print("  Some disabled prompts are still being loaded")
            print("  Check prompts/manual_prompts_master_table.csv")
            print("="*70)
            sys.exit(1)
