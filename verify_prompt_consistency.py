"""
验证新旧方式生成的prompts是否完全一致
"""

from PromptAD.ad_prompts import state_anomaly, class_state_abnormal, load_prompts_from_table

def verify_prompt_consistency():
    """验证新旧方式的一致性"""
    
    classname = 'bottle'
    
    print("="*80)
    print("验证Prompt生成的一致性")
    print("="*80)
    
    # 旧方式（代码拼接）
    print("\n【旧方式】代码拼接:")
    state_anomaly1_old = state_anomaly + class_state_abnormal[classname]
    display_name = classname  # bottle没有映射
    
    old_prompts = []
    for state in state_anomaly1_old:
        prompt = state.format(display_name)
        old_prompts.append(prompt)
        print(f"  {prompt}")
    
    # 新方式（表格读取）
    print("\n【新方式】表格读取:")
    full_texts, prompt_info = load_prompts_from_table(classname)
    
    for text in full_texts:
        print(f"  {text}")
    
    # 比较
    print("\n" + "="*80)
    print("对比结果:")
    print("="*80)
    
    if old_prompts == full_texts:
        print("✅ 完全一致！新旧方式生成的prompts完全相同")
        print(f"   共 {len(old_prompts)} 个prompts")
    else:
        print("❌ 存在差异！")
        print(f"   旧方式: {len(old_prompts)} 个")
        print(f"   新方式: {len(full_texts)} 个")
        
        # 找出差异
        print("\n差异详情:")
        for i, (old, new) in enumerate(zip(old_prompts, full_texts)):
            if old != new:
                print(f"  [{i}] 旧: {old}")
                print(f"  [{i}] 新: {new}")
    
    # 测试完整的prompt构造（包含prefix）
    print("\n" + "="*80)
    print("测试完整Prompt构造（带prefix）:")
    print("="*80)
    
    n_ctx = 12
    n_pro = 4
    normal_prompt_prefix = " ".join(["N"] * n_ctx)
    
    # 旧方式
    old_full_prompts = [
        normal_prompt_prefix + " " + state.format(display_name) + "." 
        for state in state_anomaly1_old 
        for _ in range(n_pro)
    ]
    
    # 新方式
    new_full_prompts = [
        normal_prompt_prefix + " " + text + "."
        for text in full_texts
        for _ in range(n_pro)
    ]
    
    print(f"旧方式生成: {len(old_full_prompts)} 个完整prompts")
    print(f"新方式生成: {len(new_full_prompts)} 个完整prompts")
    
    # 显示前3个对比
    print("\n前3个完整prompts对比:")
    for i in range(min(3, len(old_full_prompts))):
        print(f"\n[{i}] 旧: {old_full_prompts[i]}")
        print(f"[{i}] 新: {new_full_prompts[i]}")
        if old_full_prompts[i] == new_full_prompts[i]:
            print("     ✓ 一致")
        else:
            print("     ✗ 不一致！")
    
    if old_full_prompts == new_full_prompts:
        print("\n✅ 完整prompts完全一致！")
        print("   新旧方式在模型中的行为完全相同")
    else:
        print("\n❌ 存在差异！")


if __name__ == '__main__':
    verify_prompt_consistency()
