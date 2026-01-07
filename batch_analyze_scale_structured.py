#!/usr/bin/env python3
"""
Batch Scale Analysis with Structured Output

批量分析所有类别的 E_geom 和 E_sem 统计信息，并保存为CSV
"""

import argparse
import subprocess
import os
import re
import pandas as pd
from datetime import datetime

def parse_scale_stats(output):
    """
    从输出中解析scale统计信息
    
    Returns:
        dict with keys:
            - E_geom_normal_mean, E_geom_normal_std, E_geom_normal_min, E_geom_normal_max
            - E_geom_abnormal_mean, E_geom_abnormal_std, E_geom_abnormal_min, E_geom_abnormal_max
            - E_sem_normal_mean, E_sem_normal_std, E_sem_normal_min, E_sem_normal_max
            - E_sem_abnormal_mean, E_sem_abnormal_std, E_sem_abnormal_min, E_sem_abnormal_max
            - scale_ratio, warning
    """
    stats = {}
    
    # Extract E_geom Normal stats
    match = re.search(r'E_geom.*?Normal.*?Mean:\s+([\d.]+).*?Std:\s+([\d.]+).*?Range:\s+\[([\d.]+),\s+([\d.]+)\]', 
                     output, re.DOTALL)
    if match:
        stats['E_geom_normal_mean'] = float(match.group(1))
        stats['E_geom_normal_std'] = float(match.group(2))
        stats['E_geom_normal_min'] = float(match.group(3))
        stats['E_geom_normal_max'] = float(match.group(4))
    
    # Extract E_geom Abnormal stats
    match = re.search(r'E_geom.*?Abnormal.*?Mean:\s+([\d.]+).*?Std:\s+([\d.]+).*?Range:\s+\[([\d.]+),\s+([\d.]+)\]', 
                     output, re.DOTALL)
    if match:
        stats['E_geom_abnormal_mean'] = float(match.group(1))
        stats['E_geom_abnormal_std'] = float(match.group(2))
        stats['E_geom_abnormal_min'] = float(match.group(3))
        stats['E_geom_abnormal_max'] = float(match.group(4))
    
    # Extract E_sem Normal stats
    match = re.search(r'E_sem.*?Normal.*?Mean:\s+([\d.]+).*?Std:\s+([\d.]+).*?Range:\s+\[([\d.]+),\s+([\d.]+)\]', 
                     output, re.DOTALL)
    if match:
        stats['E_sem_normal_mean'] = float(match.group(1))
        stats['E_sem_normal_std'] = float(match.group(2))
        stats['E_sem_normal_min'] = float(match.group(3))
        stats['E_sem_normal_max'] = float(match.group(4))
    
    # Extract E_sem Abnormal stats
    match = re.search(r'E_sem.*?Abnormal.*?Mean:\s+([\d.]+).*?Std:\s+([\d.]+).*?Range:\s+\[([\d.]+),\s+([\d.]+)\]', 
                     output, re.DOTALL)
    if match:
        stats['E_sem_abnormal_mean'] = float(match.group(1))
        stats['E_sem_abnormal_std'] = float(match.group(2))
        stats['E_sem_abnormal_min'] = float(match.group(3))
        stats['E_sem_abnormal_max'] = float(match.group(4))
    
    # Extract scale ratio
    match = re.search(r'Ratio \(sem/geom\):\s+([\d.]+)', output)
    if match:
        stats['scale_ratio'] = float(match.group(1))
    
    # Extract warning
    if 'WARNING' in output:
        if 'E_sem scale is' in output and 'larger' in output:
            match = re.search(r'E_sem scale is ([\d.]+)x larger', output)
            if match:
                stats['warning'] = f'E_sem {match.group(1)}x larger'
        elif 'E_sem scale is' in output and 'smaller' in output:
            match = re.search(r'E_sem scale is ([\d.]+)x smaller', output)
            if match:
                stats['warning'] = f'E_sem {match.group(1)}x smaller'
    else:
        stats['warning'] = 'Balanced'
    
    # Extract sample counts
    match = re.search(r'Normal\s+\(n=(\d+)\)', output)
    if match:
        stats['n_normal'] = int(match.group(1))
    
    match = re.search(r'Abnormal\s+\(n=(\d+)\)', output)
    if match:
        stats['n_abnormal'] = int(match.group(1))
    
    # Extract AUROC
    match = re.search(r'Fusion-AUROC:([\d.]+),\s+Semantic:([\d.]+),\s+Memory:([\d.]+)', output)
    if match:
        stats['fusion_auroc'] = float(match.group(1))
        stats['semantic_auroc'] = float(match.group(2))
        stats['memory_auroc'] = float(match.group(3))
    
    return stats


def analyze_class(dataset, class_name, k_shot, seed, alpha, checkpoint_dir):
    """
    分析单个类别并返回统计信息
    """
    output_dir = "./result/scale_analysis"
    
    cmd = [
        "python", "analyze_scale.py",
        "--dataset", dataset,
        "--class-name", class_name,
        "--k-shot", str(k_shot),
        "--seed", str(seed),
        "--checkpoint-dir", checkpoint_dir,
        "--alpha", str(alpha)
    ]
    
    print(f"  Analyzing {class_name}...", end='', flush=True)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        stats = parse_scale_stats(result.stdout)
        if stats:
            print(f" ✅ (AUROC: {stats.get('semantic_auroc', 0):.2f})")
            return stats
        else:
            print(" ⚠️  Failed to parse")
            return None
    else:
        print(" ❌ Error")
        return None


def main():
    parser = argparse.ArgumentParser(description='Batch Scale Analysis with CSV Output')
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa'])
    parser.add_argument('--k-shot', type=int, default=2)
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='./result/scale_analysis/scale_stats.csv',
                       help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Define class lists
    mvtec_classes = [
        'carpet', 'grid', 'leather', 'tile', 'wood',
        'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
        'pill', 'screw', 'toothbrush', 'transistor', 'zipper'
    ]
    
    visa_classes = [
        'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
        'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'
    ]
    
    classes = mvtec_classes if args.dataset == 'mvtec' else visa_classes
    checkpoint_dir = f"./result/nq_mvp/{args.dataset}/k_{args.k_shot}"
    
    print(f"\n{'='*70}")
    print(f"Batch Scale Analysis")
    print(f"{'='*70}")
    print(f"Dataset: {args.dataset}")
    print(f"K-shot: {args.k_shot}")
    print(f"Seed: {args.seed}")
    print(f"Alpha: {args.alpha}")
    print(f"Classes: {len(classes)}")
    print(f"Output: {args.output}")
    print(f"{'='*70}\n")
    
    # Analyze all classes
    results = []
    for class_name in classes:
        stats = analyze_class(
            dataset=args.dataset,
            class_name=class_name,
            k_shot=args.k_shot,
            seed=args.seed,
            alpha=args.alpha,
            checkpoint_dir=checkpoint_dir
        )
        
        if stats:
            stats['class'] = class_name
            stats['dataset'] = args.dataset
            stats['k_shot'] = args.k_shot
            stats['alpha'] = args.alpha
            results.append(stats)
    
    # Create DataFrame
    if results:
        df = pd.DataFrame(results)
        
        # Reorder columns for better readability
        col_order = [
            'dataset', 'class', 'k_shot', 'alpha',
            'n_normal', 'n_abnormal',
            'fusion_auroc', 'semantic_auroc', 'memory_auroc',
            'E_geom_normal_mean', 'E_geom_normal_std', 'E_geom_normal_min', 'E_geom_normal_max',
            'E_geom_abnormal_mean', 'E_geom_abnormal_std', 'E_geom_abnormal_min', 'E_geom_abnormal_max',
            'E_sem_normal_mean', 'E_sem_normal_std', 'E_sem_normal_min', 'E_sem_normal_max',
            'E_sem_abnormal_mean', 'E_sem_abnormal_std', 'E_sem_abnormal_min', 'E_sem_abnormal_max',
            'scale_ratio', 'warning'
        ]
        
        # Only keep columns that exist
        col_order = [c for c in col_order if c in df.columns]
        df = df[col_order]
        
        # Create output directory
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        # Save to CSV
        df.to_csv(args.output, index=False, float_format='%.4f')
        
        print(f"\n{'='*70}")
        print(f"✅ Analysis Complete!")
        print(f"{'='*70}")
        print(f"Total classes: {len(results)}/{len(classes)}")
        print(f"Output saved to: {args.output}")
        print(f"\n📊 Summary Statistics:")
        print(f"  Average semantic AUROC: {df['semantic_auroc'].mean():.2f}")
        print(f"  Average scale ratio: {df['scale_ratio'].mean():.2f}")
        print(f"  Classes with warnings: {(df['warning'] != 'Balanced').sum()}")
        print(f"{'='*70}\n")
        
        # Display first few rows
        print("Preview (first 5 classes):")
        print(df[['class', 'semantic_auroc', 'E_geom_normal_mean', 'E_geom_abnormal_mean', 
                  'E_sem_normal_mean', 'E_sem_abnormal_mean', 'scale_ratio']].head())
        
    else:
        print("\n❌ No results collected. Check for errors above.")


if __name__ == '__main__':
    main()
