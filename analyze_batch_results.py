"""
分析批量测试结果，对比baseline和prompt1_memory性能
"""
import pandas as pd
import numpy as np

# 读取prompt1_memory批量测试结果
prompt1_results = pd.read_csv('test_logs/batch_testing/final_results.csv')

# 手动添加之前测试的bottle和screw结果（从对话历史中获取）
bottle_screw = pd.DataFrame({
    'class': ['bottle', 'screw'],
    'auroc': [99.52, 68.96]  # 之前测试的结果
})

# 合并所有prompt1结果
all_prompt1 = pd.concat([prompt1_results, bottle_screw], ignore_index=True).sort_values('class')

# 读取baseline结果（k=2列）
baseline_data = """carpet,100.0
grid,99.08
leather,100.0
tile,100.0
wood,99.82
bottle,100.0
cable,96.38
capsule,79.94
hazelnut,99.93
metal_nut,100.0
pill,95.61
screw,58.66
toothbrush,98.89
transistor,89.79
zipper,96.4"""

baseline_lines = baseline_data.strip().split('\n')
baseline_dict = {}
for line in baseline_lines:
    parts = line.split(',')
    baseline_dict[parts[0]] = float(parts[1])

# 创建完整对比表
comparison = []
for _, row in all_prompt1.iterrows():
    cls_name = row['class']
    prompt1_score = row['auroc']
    baseline_score = baseline_dict.get(cls_name, 0)
    diff = prompt1_score - baseline_score
    comparison.append({
        'Class': cls_name,
        'Baseline (k=2)': baseline_score,
        'Prompt1_Memory': prompt1_score,
        'Difference': diff,
        'Diff%': f"{diff:+.2f}"
    })

df_comparison = pd.DataFrame(comparison)

print("=" * 80)
print("MVTec CLS (k=2) 完整对比分析")
print("=" * 80)
print()
print(df_comparison.to_string(index=False))
print()
print("-" * 80)
print(f"Baseline 平均:      {df_comparison['Baseline (k=2)'].mean():.2f}%")
print(f"Prompt1_Memory 平均: {df_comparison['Prompt1_Memory'].mean():.2f}%")
print(f"整体差异:            {df_comparison['Difference'].mean():+.2f}%")
print("-" * 80)
print()

# 分析改进和退化情况
improved = df_comparison[df_comparison['Difference'] > 0]
degraded = df_comparison[df_comparison['Difference'] < 0]
unchanged = df_comparison[df_comparison['Difference'] == 0]

print("性能变化统计:")
print(f"  改进类别: {len(improved)}/15 ({len(improved)/15*100:.1f}%)")
print(f"  退化类别: {len(degraded)}/15 ({len(degraded)/15*100:.1f}%)")
print(f"  持平类别: {len(unchanged)}/15 ({len(unchanged)/15*100:.1f}%)")
print()

if len(improved) > 0:
    print(f"改进类别平均提升: +{improved['Difference'].mean():.2f}%")
    print(f"最大改进: {improved.loc[improved['Difference'].idxmax(), 'Class']} (+{improved['Difference'].max():.2f}%)")
print()

if len(degraded) > 0:
    print(f"退化类别平均下降: {degraded['Difference'].mean():.2f}%")
    print(f"最大退化: {degraded.loc[degraded['Difference'].idxmin(), 'Class']} ({degraded['Difference'].min():.2f}%)")
print()

print("=" * 80)
print("关键发现:")
print("=" * 80)
print(f"1. Baseline 声称值: 95.7 ± 1.5% (范围: 94.2% - 97.2%)")
print(f"2. Baseline 复现值: {df_comparison['Baseline (k=2)'].mean():.2f}%")
print(f"3. Prompt1_Memory:  {df_comparison['Prompt1_Memory'].mean():.2f}%")
print()

baseline_avg = df_comparison['Baseline (k=2)'].mean()
prompt1_avg = df_comparison['Prompt1_Memory'].mean()
baseline_claimed = 95.7

print("位置关系分析:")
if prompt1_avg > baseline_claimed + 1.5:
    print(f"  ✅ Prompt1_Memory ({prompt1_avg:.2f}%) > Baseline声称上限 ({baseline_claimed + 1.5}%)")
    print("     结论: 方法显著优于baseline")
elif prompt1_avg > baseline_claimed:
    print(f"  ⚠️  Prompt1_Memory ({prompt1_avg:.2f}%) 在 Baseline声称值 ({baseline_claimed}%) 和上限 ({baseline_claimed + 1.5}%) 之间")
    print("     结论: 方法略优于baseline声称值")
elif prompt1_avg > baseline_avg:
    print(f"  ⚠️  Baseline复现 ({baseline_avg:.2f}%) < Prompt1_Memory ({prompt1_avg:.2f}%) < Baseline声称 ({baseline_claimed}%)")
    print("     结论: 方法优于复现baseline，但未达到论文声称值")
elif prompt1_avg > baseline_claimed - 1.5:
    print(f"  ⚠️  Prompt1_Memory ({prompt1_avg:.2f}%) 在 Baseline声称下限 ({baseline_claimed - 1.5}%) 和声称值 ({baseline_claimed}%) 之间")
    print("     结论: 方法处于baseline声称范围内")
else:
    print(f"  ❌ Prompt1_Memory ({prompt1_avg:.2f}%) < Baseline声称下限 ({baseline_claimed - 1.5}%)")
    print("     结论: 方法低于baseline")
print()

# 验证假设保留率
print("=" * 80)
print("假设验证: 多原型语义改进在融合中的保留率")
print("=" * 80)
print()
print("注: 需要多原型语义单独结果来计算保留率")
print("从之前的screw测试得到的保留率: 78.3% (10.30% / 13.15%)")
print()
print("根据整体结果推测:")
print(f"  如果语义改进约为 {(prompt1_avg - baseline_avg) / 0.75:.2f}%")
print(f"  则融合保留率约为 75% (实际融合改进 {prompt1_avg - baseline_avg:.2f}%)")
print("=" * 80)

