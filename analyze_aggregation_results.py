"""分析聚合方式对比结果"""
import json
import pandas as pd

# 读取结果
with open('aggregation_comparison_results.json', 'r') as f:
    results = json.load(f)

print(f"\n{'='*100}")
print(f"同权重多聚合方式对比分析报告")
print(f"{'='*100}\n")

# 创建对比表
rows = []
for cls, agg_results in results.items():
    for agg_name in ['mean', 'top2', 'max']:
        r = agg_results[agg_name]
        rows.append({
            'Class': cls,
            'Aggregation': agg_name,
            'Normal_P95': r['normal_p95'],
            'Normal_P99': r['normal_p99'],
            'Abnormal_Median': r['abnormal_median'],
            'Abnormal_P95': r['abnormal_p95'],
        })

df = pd.DataFrame(rows)

# 分类别打印
for cls in results.keys():
    print(f"\n{'='*100}")
    print(f"Class: {cls}")
    print(f"{'='*100}")
    
    cls_df = df[df['Class'] == cls]
    
    # 提取数据
    mean_data = cls_df[cls_df['Aggregation'] == 'mean'].iloc[0]
    top2_data = cls_df[cls_df['Aggregation'] == 'top2'].iloc[0]
    max_data = cls_df[cls_df['Aggregation'] == 'max'].iloc[0]
    
    # 对比表
    print(f"\n{'Metric':<25} {'Mean':>12} {'Top-2':>12} {'Max':>12} {'Δ(Top2-Mean)':>15} {'Δ(Max-Mean)':>15}")
    print(f"{'-'*100}")
    
    # Normal P95
    print(f"{'Normal P95':<25} {mean_data['Normal_P95']:>12.4f} {top2_data['Normal_P95']:>12.4f} {max_data['Normal_P95']:>12.4f} "
          f"{top2_data['Normal_P95']-mean_data['Normal_P95']:>15.4f} {max_data['Normal_P95']-mean_data['Normal_P95']:>15.4f}")
    
    # Normal P99
    print(f"{'Normal P99':<25} {mean_data['Normal_P99']:>12.4f} {top2_data['Normal_P99']:>12.4f} {max_data['Normal_P99']:>12.4f} "
          f"{top2_data['Normal_P99']-mean_data['Normal_P99']:>15.4f} {max_data['Normal_P99']-mean_data['Normal_P99']:>15.4f}")
    
    # Abnormal Median
    print(f"{'Abnormal Median':<25} {mean_data['Abnormal_Median']:>12.4f} {top2_data['Abnormal_Median']:>12.4f} {max_data['Abnormal_Median']:>12.4f} "
          f"{top2_data['Abnormal_Median']-mean_data['Abnormal_Median']:>15.4f} {max_data['Abnormal_Median']-mean_data['Abnormal_Median']:>15.4f}")
    
    # Abnormal P95
    print(f"{'Abnormal P95':<25} {mean_data['Abnormal_P95']:>12.4f} {top2_data['Abnormal_P95']:>12.4f} {max_data['Abnormal_P95']:>12.4f} "
          f"{top2_data['Abnormal_P95']-mean_data['Abnormal_P95']:>15.4f} {max_data['Abnormal_P95']-mean_data['Abnormal_P95']:>15.4f}")
    
    # 判断
    print(f"\n{'Diagnosis':<25}")
    normal_p99_increase_top2 = top2_data['Normal_P99'] - mean_data['Normal_P99']
    normal_p99_increase_max = max_data['Normal_P99'] - mean_data['Normal_P99']
    
    if normal_p99_increase_top2 > 0.05:
        print(f"  ✅ Top-2聚合显著抬高正常样本右尾 (+{normal_p99_increase_top2:.4f})")
    if normal_p99_increase_max > 0.05:
        print(f"  ✅ Max聚合显著抬高正常样本右尾 (+{normal_p99_increase_max:.4f})")
    
    if normal_p99_increase_top2 < 0.05 and normal_p99_increase_max < 0.05:
        print(f"  ❌ 聚合方式对正常样本影响较小")

# 汇总统计
print(f"\n\n{'='*100}")
print(f"汇总统计")
print(f"{'='*100}\n")

summary_rows = []
for cls in results.keys():
    mean_data = df[(df['Class'] == cls) & (df['Aggregation'] == 'mean')].iloc[0]
    top2_data = df[(df['Class'] == cls) & (df['Aggregation'] == 'top2')].iloc[0]
    max_data = df[(df['Class'] == cls) & (df['Aggregation'] == 'max')].iloc[0]
    
    summary_rows.append({
        'Class': cls,
        'ΔP99(Top2-Mean)': top2_data['Normal_P99'] - mean_data['Normal_P99'],
        'ΔP99(Max-Mean)': max_data['Normal_P99'] - mean_data['Normal_P99'],
        'ΔAbnormal_P95(Top2-Mean)': top2_data['Abnormal_P95'] - mean_data['Abnormal_P95'],
    })

summary_df = pd.DataFrame(summary_rows)

print(f"{'Class':<15} {'ΔP99(Top2-Mean)':>18} {'ΔP99(Max-Mean)':>18} {'ΔAbnormal_P95':>18} {'FP Risk':>12}")
print(f"{'-'*100}")

for _, row in summary_df.iterrows():
    fp_risk = "⚠️ 高" if row['ΔP99(Top2-Mean)'] > 0.05 else "✅ 低"
    print(f"{row['Class']:<15} {row['ΔP99(Top2-Mean)']:>18.4f} {row['ΔP99(Max-Mean)']:>18.4f} "
          f"{row['ΔAbnormal_P95(Top2-Mean)']:>18.4f} {fp_risk:>12}")

# 最终结论
print(f"\n{'='*100}")
print(f"结论")
print(f"{'='*100}\n")

high_fp_classes = summary_df[summary_df['ΔP99(Top2-Mean)'] > 0.05]['Class'].tolist()
print(f"1. 假阳性风险类别（{len(high_fp_classes)}/{len(summary_df)}）:")
print(f"   {high_fp_classes}")

avg_increase = summary_df['ΔP99(Top2-Mean)'].mean()
print(f"\n2. Top-2聚合平均抬高正常样本P99: {avg_increase:.4f}")

if len(high_fp_classes) >= len(summary_df) * 0.7:
    print(f"\n3. ✅ 假设成立：Top-k/Max聚合在多数类别上显著抬高正常样本异常响应")
    print(f"   → 这解释了Multi-Abnormal性能退化的主要原因")
else:
    print(f"\n3. ⚠️ 假设部分成立：只有{len(high_fp_classes)}个类别出现显著假阳性")
    print(f"   → 需要进一步分析其他退化原因")

print(f"\n{'='*100}\n")
