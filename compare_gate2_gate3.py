#!/usr/bin/env python3
"""
对比 Gate2 vs Gate3 训练结果

Gate2: Memory分支参与训练（原始实现）
Gate3: Memory分支不参与训练（修复后）

对比内容：
1. Image-level AUROC
2. Pixel-level AUROC (如果有)
3. 训练稳定性
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path


def find_result_files(root_dir, pattern="*AUROC.csv"):
    """查找所有结果文件"""
    results = []
    for path in Path(root_dir).rglob(pattern):
        results.append(str(path))
    return sorted(results)


def parse_result_file(file_path):
    """解析结果文件"""
    try:
        df = pd.read_csv(file_path)
        # 通常格式: class_name, image_auroc, pixel_auroc, ...
        return df
    except Exception as e:
        print(f"⚠ 无法解析 {file_path}: {e}")
        return None


def extract_class_info(file_path):
    """从文件路径提取类别信息"""
    parts = Path(file_path).parts
    
    # 查找数据集和类别
    dataset = None
    for i, part in enumerate(parts):
        if part in ['mvtec', 'visa']:
            dataset = part
            # 类别通常在文件名中
            class_name = Path(file_path).stem.split('_')[0]
            break
    
    return dataset, class_name


def compare_results(gate2_dir, gate3_dir):
    """对比两个版本的结果"""
    
    print("=" * 80)
    print("Gate2 vs Gate3 训练结果对比")
    print("=" * 80)
    print()
    
    # 查找结果文件
    gate2_files = find_result_files(gate2_dir)
    gate3_files = find_result_files(gate3_dir)
    
    print(f"Gate2 结果文件: {len(gate2_files)} 个")
    print(f"Gate3 结果文件: {len(gate3_files)} 个")
    print()
    
    if len(gate2_files) == 0:
        print("❌ 未找到Gate2结果文件")
        return
    
    if len(gate3_files) == 0:
        print("❌ 未找到Gate3结果文件")
        print("   请先运行: ./train_gate3_full.sh")
        return
    
    # 构建类别到文件的映射
    gate2_map = {}
    for f in gate2_files:
        dataset, cls = extract_class_info(f)
        if dataset and cls:
            key = f"{dataset}/{cls}"
            gate2_map[key] = f
    
    gate3_map = {}
    for f in gate3_files:
        dataset, cls = extract_class_info(f)
        if dataset and cls:
            key = f"{dataset}/{cls}"
            gate3_map[key] = f
    
    # 找到共同的类别
    common_classes = sorted(set(gate2_map.keys()) & set(gate3_map.keys()))
    
    print(f"共同类别数: {len(common_classes)}")
    print()
    
    if len(common_classes) == 0:
        print("⚠ 没有共同的类别可以对比")
        print()
        print("Gate2 类别:", sorted(gate2_map.keys())[:5], "...")
        print("Gate3 类别:", sorted(gate3_map.keys())[:5], "...")
        return
    
    # 对比结果
    results = []
    
    for cls_key in common_classes:
        gate2_file = gate2_map[cls_key]
        gate3_file = gate3_map[cls_key]
        
        gate2_df = parse_result_file(gate2_file)
        gate3_df = parse_result_file(gate3_file)
        
        if gate2_df is None or gate3_df is None:
            continue
        
        # 提取AUROC值（假设列名为 'image_auroc' 或类似）
        gate2_auroc = None
        gate3_auroc = None
        
        # 尝试不同的列名
        for col in ['image_auroc', 'Image-AUROC', 'img_auroc', 'auroc']:
            if col in gate2_df.columns:
                gate2_auroc = gate2_df[col].values[0]
                break
        
        for col in ['image_auroc', 'Image-AUROC', 'img_auroc', 'auroc']:
            if col in gate3_df.columns:
                gate3_auroc = gate3_df[col].values[0]
                break
        
        if gate2_auroc is not None and gate3_auroc is not None:
            diff = gate3_auroc - gate2_auroc
            results.append({
                'class': cls_key,
                'gate2_auroc': gate2_auroc,
                'gate3_auroc': gate3_auroc,
                'diff': diff,
                'diff_pct': diff / gate2_auroc * 100 if gate2_auroc > 0 else 0
            })
    
    if len(results) == 0:
        print("⚠ 无法提取对比数据")
        return
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 统计
    print("=" * 80)
    print("整体统计")
    print("=" * 80)
    print(f"对比类别数: {len(df)}")
    print(f"Gate2 平均AUROC: {df['gate2_auroc'].mean():.2f}%")
    print(f"Gate3 平均AUROC: {df['gate3_auroc'].mean():.2f}%")
    print(f"平均差异: {df['diff'].mean():.2f}% ({df['diff_pct'].mean():.2f}%)")
    print()
    
    # 改进/退化统计
    improved = (df['diff'] > 0).sum()
    degraded = (df['diff'] < 0).sum()
    unchanged = (df['diff'] == 0).sum()
    
    print(f"改进: {improved}/{len(df)} ({improved/len(df)*100:.1f}%)")
    print(f"退化: {degraded}/{len(df)} ({degraded/len(df)*100:.1f}%)")
    print(f"不变: {unchanged}/{len(df)} ({unchanged/len(df)*100:.1f}%)")
    print()
    
    # Top 5 改进
    print("=" * 80)
    print("Top 5 改进最大")
    print("=" * 80)
    top_improved = df.nlargest(5, 'diff')
    for _, row in top_improved.iterrows():
        print(f"{row['class']:20s}  Gate2: {row['gate2_auroc']:5.2f}%  "
              f"Gate3: {row['gate3_auroc']:5.2f}%  Δ: +{row['diff']:.2f}%")
    print()
    
    # Top 5 退化
    if degraded > 0:
        print("=" * 80)
        print("Top 5 退化最大")
        print("=" * 80)
        top_degraded = df.nsmallest(5, 'diff')
        for _, row in top_degraded.iterrows():
            print(f"{row['class']:20s}  Gate2: {row['gate2_auroc']:5.2f}%  "
                  f"Gate3: {row['gate3_auroc']:5.2f}%  Δ: {row['diff']:.2f}%")
        print()
    
    # 详细结果
    print("=" * 80)
    print("详细结果")
    print("=" * 80)
    print(f"{'Class':<20s} {'Gate2':>8s} {'Gate3':>8s} {'Diff':>8s} {'Diff%':>8s}")
    print("-" * 80)
    for _, row in df.sort_values('diff', ascending=False).iterrows():
        status = "↑" if row['diff'] > 0 else "↓" if row['diff'] < 0 else "="
        print(f"{row['class']:<20s} {row['gate2_auroc']:7.2f}% {row['gate3_auroc']:7.2f}% "
              f"{status}{abs(row['diff']):6.2f}% {row['diff_pct']:7.2f}%")
    print()
    
    # 保存结果
    output_file = "result/gate2_vs_gate3_comparison.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"✓ 对比结果已保存: {output_file}")
    print()
    
    # 结论
    print("=" * 80)
    print("结论")
    print("=" * 80)
    
    avg_diff = df['diff'].mean()
    if abs(avg_diff) < 0.5:
        print("📊 Gate2 和 Gate3 性能基本一致（差异 < 0.5%）")
        print("   修复没有显著影响性能，符合预期")
    elif avg_diff > 0.5:
        print("📈 Gate3 性能优于 Gate2（平均提升 {:.2f}%）".format(avg_diff))
        print("   修复后semantic branch优化更纯粹，带来性能提升")
    else:
        print("📉 Gate3 性能略低于 Gate2（平均下降 {:.2f}%）".format(abs(avg_diff)))
        print("   但这是符合设计的，因为memory branch原本不应参与训练")
    
    print()
    print("技术解释：")
    print("  - Gate2: Memory branch参与训练（bug，可能引入额外信号）")
    print("  - Gate3: Memory branch不参与训练（正确实现）")
    print("  - 性能差异反映了memory branch在训练中的影响")
    print()
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='对比Gate2和Gate3训练结果')
    parser.add_argument('--gate2-dir', default='result_gate',
                        help='Gate2结果目录（默认: result_gate）')
    parser.add_argument('--gate3-dir', default='result_gate3',
                        help='Gate3结果目录（默认: result_gate3）')
    
    args = parser.parse_args()
    
    compare_results(args.gate2_dir, args.gate3_dir)
