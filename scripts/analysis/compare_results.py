"""
对比 Prompt1_Memory (多原型+记忆库) vs Baseline (单原型+记忆库)
"""
import pandas as pd
import numpy as np

# Baseline数据
baseline_data = {
    'MVTec_CLS': {
        'k1': [100.0,99.0,100.0,99.93,100.0,100.0,96.63,68.33,97.5,99.85,95.2,60.48,95.56,90.33,96.69],
        'k2': [100.0,99.08,100.0,100.0,99.82,100.0,96.38,79.94,99.93,100.0,95.61,58.66,98.89,89.79,96.4],
        'k4': [100.0,99.16,100.0,100.0,99.74,100.0,97.23,83.81,100.0,99.66,95.06,70.01,98.06,94.42,97.22]
    },
    'MVTec_SEG': {
        'k1': [99.51,98.15,99.47,96.77,95.13,98.53,97.05,93.22,98.76,90.98,94.42,93.71,98.48,90.09,95.73],
        'k2': [99.48,97.81,99.42,96.53,95.65,98.73,96.77,96.13,98.86,93.76,95.45,93.12,98.95,88.66,94.46],
        'k4': [99.49,98.14,99.42,96.65,97.41,98.78,97.19,96.28,99.06,93.01,95.18,94.67,99.15,93.07,89.91]
    },
    'VisA_CLS': {
        'k1': [93.89,74.76,87.81,96.11,88.99,85.65,73.19,94.48,77.09,76.14,87.85,97.99],
        'k2': [94.86,72.8,90.4,97.19,89.96,85.75,74.74,94.15,77.09,79.78,83.45,98.8],
        'k4': [95.65,74.25,88.34,96.48,92.31,88.1,75.99,93.11,82.83,80.59,92.66,98.88]
    },
    'VisA_SEG': {
        'k1': [94.55,93.41,99.48,99.62,97.23,97.55,96.58,98.5,95.2,94.95,95.24,99.48],
        'k2': [96.16,94.39,99.43,99.61,96.91,97.61,94.61,98.45,0,0,0,99.61],  # 有缺失值
        'k4': [95.78,94.96,99.34,99.63,97.72,98.16,0,0,0,95.67,97.41,99.58]  # 有缺失值
    }
}

# Prompt1数据
prompt1_data = {
    'MVTec_CLS': {
        'k1': [100.0,99.16,100.0,99.82,99.47,100.0,91.15,65.1,99.82,99.22,94.35,56.89,94.17,84.06,95.19],
        'k2': [100.0,98.33,100.0,99.93,99.74,99.84,94.38,86.1,99.96,99.66,94.37,68.75,95.56,88.88,92.32],
        'k4': [100.0,98.58,100.0,100.0,99.21,99.84,98.59,91.4,100.0,100.0,95.53,68.06,97.5,92.19,96.6]
    },
    'MVTec_SEG': {
        'k1': [99.45,97.57,99.47,96.32,95.9,98.7,96.75,95.52,98.84,91.96,93.81,92.59,98.51,90.03,93.88],
        'k2': [99.49,98.0,99.41,96.54,97.15,98.75,96.15,96.31,98.93,93.98,95.1,93.23,98.73,87.9,92.71],
        'k4': [99.45,98.09,99.46,96.63,96.73,98.86,96.95,96.86,98.78,95.19,94.88,94.33,99.13,92.76,96.12]
    },
    'VisA_CLS': {
        'k1': [93.23,75.29,89.61,92.81,92.09,84.33,70.12,94.04,82.07,73.15,93.49,99.28],
        'k2': [93.86,76.54,89.44,96.1,87.5,86.63,77.47,93.33,77.65,74.99,86.12,98.74],
        'k4': [89.88,74.55,85.38,94.97,90.91,88.9,74.24,94.84,77.71,80.54,90.27,98.67]
    },
    'VisA_SEG': {
        'k1': [94.76,93.78,99.22,99.6,96.56,96.86,94.64,98.87,95.95,95.74,97.04,99.55],
        'k2': [95.21,94.17,99.1,99.59,96.79,97.76,95.57,98.25,94.89,94.78,96.79,99.75],
        'k4': [95.31,95.16,99.38,99.65,97.54,97.48,96.47,98.36,96.0,96.6,98.07,99.67]
    }
}

classes_mvtec = ['carpet','grid','leather','tile','wood','bottle','cable','capsule','hazelnut','metal_nut','pill','screw','toothbrush','transistor','zipper']
classes_visa = ['candle','capsules','cashew','chewinggum','fryum','macaroni1','macaroni2','pcb1','pcb2','pcb3','pcb4','pipe_fryum']

print("="*100)
print("Prompt1_Memory (多原型+记忆库) vs Baseline (单原型+记忆库) 完整对比")
print("="*100)
print()

# MVTec CLS 对比
print("="*100)
print("1. MVTec Classification (Image AUROC)")
print("="*100)
for k in ['k1', 'k2', 'k4']:
    baseline = np.array(baseline_data['MVTec_CLS'][k])
    prompt1 = np.array(prompt1_data['MVTec_CLS'][k])
    diff = prompt1 - baseline
    
    print(f"\n--- k={k[1]} Shot ---")
    print(f"{'Class':<15} {'Baseline':<10} {'Prompt1':<10} {'Diff':<10} {'Change'}")
    print("-"*60)
    
    for i, cls in enumerate(classes_mvtec):
        symbol = "✅" if diff[i] > 0.5 else "⚠️" if diff[i] < -0.5 else "➖"
        print(f"{cls:<15} {baseline[i]:>8.2f}% {prompt1[i]:>8.2f}% {diff[i]:>+8.2f}% {symbol}")
    
    print("-"*60)
    print(f"{'Average':<15} {baseline.mean():>8.2f}% {prompt1.mean():>8.2f}% {diff.mean():>+8.2f}%")
    
    improved = (diff > 0.5).sum()
    degraded = (diff < -0.5).sum()
    unchanged = len(diff) - improved - degraded
    print(f"改进: {improved}/15, 退化: {degraded}/15, 持平: {unchanged}/15")

# MVTec SEG 对比
print("\n" + "="*100)
print("2. MVTec Segmentation (Pixel AUROC)")
print("="*100)
for k in ['k1', 'k2', 'k4']:
    baseline = np.array(baseline_data['MVTec_SEG'][k])
    prompt1 = np.array(prompt1_data['MVTec_SEG'][k])
    diff = prompt1 - baseline
    
    print(f"\n--- k={k[1]} Shot ---")
    print(f"{'Class':<15} {'Baseline':<10} {'Prompt1':<10} {'Diff':<10} {'Change'}")
    print("-"*60)
    
    for i, cls in enumerate(classes_mvtec):
        symbol = "✅" if diff[i] > 0.5 else "⚠️" if diff[i] < -0.5 else "➖"
        print(f"{cls:<15} {baseline[i]:>8.2f}% {prompt1[i]:>8.2f}% {diff[i]:>+8.2f}% {symbol}")
    
    print("-"*60)
    print(f"{'Average':<15} {baseline.mean():>8.2f}% {prompt1.mean():>8.2f}% {diff.mean():>+8.2f}%")
    
    improved = (diff > 0.5).sum()
    degraded = (diff < -0.5).sum()
    unchanged = len(diff) - improved - degraded
    print(f"改进: {improved}/15, 退化: {degraded}/15, 持平: {unchanged}/15")

# VisA CLS 对比
print("\n" + "="*100)
print("3. VisA Classification (Image AUROC)")
print("="*100)
for k in ['k1', 'k2', 'k4']:
    baseline = np.array(baseline_data['VisA_CLS'][k])
    prompt1 = np.array(prompt1_data['VisA_CLS'][k])
    diff = prompt1 - baseline
    
    print(f"\n--- k={k[1]} Shot ---")
    print(f"{'Class':<15} {'Baseline':<10} {'Prompt1':<10} {'Diff':<10} {'Change'}")
    print("-"*60)
    
    for i, cls in enumerate(classes_visa):
        symbol = "✅" if diff[i] > 0.5 else "⚠️" if diff[i] < -0.5 else "➖"
        print(f"{cls:<15} {baseline[i]:>8.2f}% {prompt1[i]:>8.2f}% {diff[i]:>+8.2f}% {symbol}")
    
    print("-"*60)
    print(f"{'Average':<15} {baseline.mean():>8.2f}% {prompt1.mean():>8.2f}% {diff.mean():>+8.2f}%")
    
    improved = (diff > 0.5).sum()
    degraded = (diff < -0.5).sum()
    unchanged = len(diff) - improved - degraded
    print(f"改进: {improved}/12, 退化: {degraded}/12, 持平: {unchanged}/12")

# VisA SEG 对比 (注意baseline有缺失值)
print("\n" + "="*100)
print("4. VisA Segmentation (Pixel AUROC)")
print("="*100)
print("注意: Baseline k=2和k=4部分数据缺失(显示为0)，仅对比有效数据")
print()

for k in ['k1', 'k2', 'k4']:
    baseline = np.array(baseline_data['VisA_SEG'][k])
    prompt1 = np.array(prompt1_data['VisA_SEG'][k])
    
    # 过滤掉baseline为0的无效数据
    valid_mask = baseline > 10  # 假设有效值都>10
    baseline_valid = baseline[valid_mask]
    prompt1_valid = prompt1[valid_mask]
    diff_valid = prompt1_valid - baseline_valid
    
    print(f"\n--- k={k[1]} Shot ---")
    print(f"{'Class':<15} {'Baseline':<10} {'Prompt1':<10} {'Diff':<10} {'Change'}")
    print("-"*60)
    
    valid_classes = [cls for i, cls in enumerate(classes_visa) if valid_mask[i]]
    for i, cls in enumerate(valid_classes):
        symbol = "✅" if diff_valid[i] > 0.5 else "⚠️" if diff_valid[i] < -0.5 else "➖"
        print(f"{cls:<15} {baseline_valid[i]:>8.2f}% {prompt1_valid[i]:>8.2f}% {diff_valid[i]:>+8.2f}% {symbol}")
    
    print("-"*60)
    print(f"{'Average':<15} {baseline_valid.mean():>8.2f}% {prompt1_valid.mean():>8.2f}% {diff_valid.mean():>+8.2f}%")
    print(f"有效数据: {len(baseline_valid)}/12")
    
    improved = (diff_valid > 0.5).sum()
    degraded = (diff_valid < -0.5).sum()
    unchanged = len(diff_valid) - improved - degraded
    print(f"改进: {improved}/{len(valid_classes)}, 退化: {degraded}/{len(valid_classes)}, 持平: {unchanged}/{len(valid_classes)}")

# 总结
print("\n" + "="*100)
print("总结")
print("="*100)

all_results = []
for dataset, task in [('MVTec', 'CLS'), ('MVTec', 'SEG'), ('VisA', 'CLS')]:
    key = f"{dataset}_{task}"
    for k in ['k1', 'k2', 'k4']:
        baseline = np.array(baseline_data[key][k])
        prompt1 = np.array(prompt1_data[key][k])
        diff = (prompt1 - baseline).mean()
        all_results.append({
            'Dataset': dataset,
            'Task': task,
            'k': k[1],
            'Baseline': baseline.mean(),
            'Prompt1': prompt1.mean(),
            'Diff': diff
        })

# VisA SEG特殊处理
for k in ['k1', 'k2', 'k4']:
    baseline = np.array(baseline_data['VisA_SEG'][k])
    prompt1 = np.array(prompt1_data['VisA_SEG'][k])
    valid_mask = baseline > 10
    baseline_valid = baseline[valid_mask]
    prompt1_valid = prompt1[valid_mask]
    diff = (prompt1_valid - baseline_valid).mean()
    all_results.append({
        'Dataset': 'VisA',
        'Task': 'SEG',
        'k': k[1],
        'Baseline': baseline_valid.mean(),
        'Prompt1': prompt1_valid.mean(),
        'Diff': diff
    })

df_summary = pd.DataFrame(all_results)
print("\n各配置平均性能对比:")
print(df_summary.to_string(index=False))

print("\n" + "="*100)
print("关键发现:")
print("="*100)
print(f"1. 总体平均差异: {df_summary['Diff'].mean():+.2f}%")
print(f"2. 改进的配置数: {(df_summary['Diff'] > 0).sum()}/{len(df_summary)}")
print(f"3. 最大改进: {df_summary['Diff'].max():+.2f}% ({df_summary.loc[df_summary['Diff'].idxmax(), 'Dataset']} {df_summary.loc[df_summary['Diff'].idxmax(), 'Task']} k={df_summary.loc[df_summary['Diff'].idxmax(), 'k']})")
print(f"4. 最大退化: {df_summary['Diff'].min():+.2f}% ({df_summary.loc[df_summary['Diff'].idxmin(), 'Dataset']} {df_summary.loc[df_summary['Diff'].idxmin(), 'Task']} k={df_summary.loc[df_summary['Diff'].idxmin(), 'k']})")
print()
print("结论:")
if df_summary['Diff'].mean() > 0.5:
    print("✅ Prompt1_Memory (多原型+记忆库) 整体优于 Baseline (单原型+记忆库)")
elif df_summary['Diff'].mean() < -0.5:
    print("⚠️ Prompt1_Memory (多原型+记忆库) 整体弱于 Baseline (单原型+记忆库)")
else:
    print("➖ Prompt1_Memory (多原型+记忆库) 与 Baseline (单原型+记忆库) 性能相当")
print("="*100)
