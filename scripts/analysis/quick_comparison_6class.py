#!/usr/bin/env python3
"""
快速对比6类代表性类别的三版本性能
Baseline vs Prompt2 vs Ours
"""

import pandas as pd

# Ours训练结果（刚完成）
ours_results = {
    'toothbrush': 88.61,
    'capsule': 67.41,
    'carpet': 100.0,
    'leather': 99.93,
    'screw': 75.49,
    'pcb2': 66.61,
}

# 从full_metrics_k2.csv提取Baseline和Prompt2
df_full = pd.read_csv('analysis/full_metrics_k2.csv')

print("="*100)
print("6类代表性类别 - 三版本快速对比 (Image-AUROC %)")
print("="*100)
print()
print(f"{'类别':<20} {'性能组':<15} {'Baseline':<10} {'Prompt2':<10} {'Ours':<10} {'Δ(P2-B)':<12} {'Δ(Ours-P2)':<15} {'状态':<15}")
print("-"*100)

total_delta_prompt2 = 0
total_delta_ours = 0
improve_count = 0

for cls_name, ours_auroc in ours_results.items():
    # 从full_metrics找到对应行
    if cls_name == 'pcb2':
        full_cls_name = 'visa-pcb2'
    else:
        full_cls_name = f'mvtec-{cls_name}'
    
    row = df_full[df_full['class'] == full_cls_name]
    
    if len(row) == 0:
        print(f"⚠️  未找到 {full_cls_name} 的诊断数据")
        continue
    
    baseline = row['baseline_acc'].values[0]
    prompt2 = row['prompt2_acc'].values[0]
    group = row['performance_group'].values[0]
    
    delta_prompt2 = prompt2 - baseline
    delta_ours = ours_auroc - prompt2
    
    total_delta_prompt2 += delta_prompt2
    total_delta_ours += delta_ours
    
    if delta_ours > 0:
        improve_count += 1
        status = "✅ 改善"
    elif delta_ours > -1:
        status = "⚖️  微降"
    else:
        status = "❌ 退化"
    
    print(f"{full_cls_name:<20} {group:<15} {baseline:<10.2f} {prompt2:<10.2f} {ours_auroc:<10.2f} {delta_prompt2:<+12.2f} {delta_ours:<+15.2f} {status:<15}")

print("-"*100)
print(f"{'平均':<20} {'':<15} {'':<10} {'':<10} {'':<10} {total_delta_prompt2/6:<+12.2f} {total_delta_ours/6:<+15.2f} {f'{improve_count}/6改善':<15}")
print("="*100)
print()

# 分组统计
print("="*100)
print("按性能组统计")
print("="*100)
print()

groups = {
    'Severe Degrade': ['toothbrush', 'capsule', 'pcb2'],
    'Stable': ['carpet', 'leather'],
    'Improved': ['screw'],
}

for group_name, classes in groups.items():
    print(f"【{group_name}】(n={len(classes)})")
    
    group_delta_ours = []
    for cls in classes:
        if cls in ours_results:
            if cls == 'pcb2':
                full_cls_name = 'visa-pcb2'
            else:
                full_cls_name = f'mvtec-{cls}'
            
            row = df_full[df_full['class'] == full_cls_name]
            if len(row) > 0:
                prompt2 = row['prompt2_acc'].values[0]
                delta = ours_results[cls] - prompt2
                group_delta_ours.append(delta)
                print(f"  {cls:<15} Δ(Ours-Prompt2): {delta:+.2f}")
    
    if group_delta_ours:
        avg = sum(group_delta_ours) / len(group_delta_ours)
        print(f"  {'平均':<15} {avg:+.2f}")
    print()

print("="*100)
print("💡 初步结论")
print("="*100)
print()

# 计算各组平均
severe_deltas = []
for cls in groups['Severe Degrade']:
    if cls in ours_results:
        if cls == 'pcb2':
            full_cls_name = 'visa-pcb2'
        else:
            full_cls_name = f'mvtec-{cls}'
        row = df_full[df_full['class'] == full_cls_name]
        if len(row) > 0:
            severe_deltas.append(ours_results[cls] - row['prompt2_acc'].values[0])

if severe_deltas:
    avg_severe = sum(severe_deltas) / len(severe_deltas)
    if avg_severe > 2:
        print(f"✅ Severe组显著改善: 平均提升 {avg_severe:.2f}%")
        print(f"   → 三项改动（EMA修正+Repulsion+Margin）对严重退化类别有效")
    elif avg_severe > 0:
        print(f"⚖️  Severe组略有改善: 平均提升 {avg_severe:.2f}%")
        print(f"   → 改动方向正确但效果有限，可能需要调整超参数")
    else:
        print(f"❌ Severe组未改善: 平均变化 {avg_severe:.2f}%")
        print(f"   → 当前策略对严重退化类别效果不佳")

# Screw检查
if 'screw' in ours_results:
    row = df_full[df_full['class'] == 'mvtec-screw']
    if len(row) > 0:
        screw_delta = ours_results['screw'] - row['prompt2_acc'].values[0]
        print()
        if screw_delta >= -2:
            print(f"✅ Screw保持改进: 相对Prompt2变化 {screw_delta:+.2f}%")
            print(f"   → 改动未破坏困难类的Prompt2提升效果")
        else:
            print(f"⚠️  Screw显著回退: 相对Prompt2变化 {screw_delta:+.2f}%")
            print(f"   → 需要分析为何改动对Improved类别不利")

print()
print("="*100)
print("📋 下一步行动")
print("="*100)
print()

if avg_severe > 2 and improve_count >= 4:
    print("✅ 结果令人鼓舞！建议：")
    print("   1. 运行完整评估：bash evaluate_6class_comparison.sh")
    print("   2. 分析extended metrics（margin/separation/collapse）")
    print("   3. 如果diagnostic metrics也改善，扩展到27类全量验证")
elif avg_severe > 0:
    print("⚖️  结果部分积极，建议：")
    print("   1. 先运行extended evaluation确认margin是否改善")
    print("   2. 如果margin改善但AUROC提升不明显，可能是超参数问题")
    print("   3. 考虑调整 lambda_rep/lambda_margin 后重训")
else:
    print("❌ 结果不理想，建议：")
    print("   1. 检查训练日志中的loss曲线（各项loss是否收敛）")
    print("   2. 诊断是哪项改动引入了负面影响")
    print("   3. 考虑单项改动的小规模测试（先只改EMA，或只加Margin）")

print()
