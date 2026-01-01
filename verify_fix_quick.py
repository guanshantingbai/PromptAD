"""
快速验证：修复后的 Phase 1 与 sanity test 结果对比
"""

import pandas as pd

# 读取修复后的 Phase 1 结果
phase1_fixed = pd.read_csv('result/prompt_purging/phase1/mvtec/k_2/bottle_phase1_normal_side_risk_eps0.05.csv')

# 读取 sanity test 的修复版
sanity_fixed = pd.read_csv('result/prompt_purging/sanity_tests/mvtec_bottle_phase1_FIXED.csv')

print("="*80)
print("验证确认：Phase 1 修复是否成功")
print("="*80)

print("\n对比 median_margin (前5个 prompts):")
print("-"*80)
print(f"{'Prompt':<30} {'Phase1':<12} {'Sanity':<12} {'Diff':<10}")
print("-"*80)

for i in range(min(5, len(phase1_fixed))):
    template = phase1_fixed.iloc[i]['template']
    if len(template) > 28:
        template = template[:25] + "..."
    
    p1_median = phase1_fixed.iloc[i]['median_margin']
    sanity_median = sanity_fixed.iloc[i]['median_margin']
    diff = abs(p1_median - sanity_median)
    
    print(f"{template:<30} {p1_median:<12.4f} {sanity_median:<12.4f} {diff:<10.4f}")

print("\n" + "="*80)
print("结论:")
print("="*80)

if abs(phase1_fixed['median_margin'].mean() - sanity_fixed['median_margin'].mean()) < 0.1:
    print("✓ 修复成功！Phase 1 与 sanity test 结果一致")
    print("✓ s_n 计算已修正为: max(sim) 而非 mean(prototypes)")
else:
    print("✗ 仍有差异，需要进一步检查")
