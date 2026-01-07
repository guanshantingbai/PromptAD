#!/usr/bin/env python3
"""
Scale Analysis Tool for n(q) MVP

分析 E_geom 和 E_sem 在正常/异常样本上的统计分布
用于诊断尺度不匹配问题
"""

import argparse
import subprocess
import os

def analyze_class(dataset, class_name, k_shot, seed, alpha, checkpoint_dir):
    """
    分析单个类别的scale统计
    """
    print(f"\n{'='*70}")
    print(f"Analyzing: {dataset}/{class_name} (k={k_shot}, alpha={alpha})")
    print(f"{'='*70}\n")
    
    # Use dedicated debug output directory
    output_dir = "./result/scale_analysis"
    
    cmd = [
        "python", "test_cls.py",
        "--dataset", dataset,
        "--class_name", class_name,
        "--k-shot", str(k_shot),
        "--seed", str(seed),
        "--checkpoint-dir", checkpoint_dir,
        "--output-dir", output_dir,
        "--semantic-weight", str(alpha),
        "--use-visual-prototypes", "True",
        "--return-scale-stats", "True"
    ]
    
    # Run and capture output
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print only the scale statistics part
    output = result.stdout
    if "[Scale Statistics]" in output:
        stats_start = output.find("[Scale Statistics]")
        stats_section = output[stats_start:]
        print(stats_section)
    else:
        print("⚠️  No scale statistics found. Check if semantic fusion is enabled.")
        print("\nFull output:")
        print(output)
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Scale Analysis for n(q) MVP')
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa'])
    parser.add_argument('--class-name', type=str, default=None,
                       help='Class name to analyze. If None, analyze all classes')
    parser.add_argument('--k-shot', type=int, default=2)
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='Semantic weight for analysis')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                       help='Checkpoint directory. Default: ./result/nq_mvp/{dataset}/k_{k_shot}')
    
    args = parser.parse_args()
    
    # Determine checkpoint directory
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"./result/nq_mvp/{args.dataset}/k_{args.k_shot}"
    
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
    
    # Determine classes to analyze
    if args.class_name is not None:
        classes = [args.class_name]
    else:
        classes = mvtec_classes if args.dataset == 'mvtec' else visa_classes
    
    print(f"\n{'='*70}")
    print(f"Scale Analysis Tool")
    print(f"{'='*70}")
    print(f"Dataset: {args.dataset}")
    print(f"K-shot: {args.k_shot}")
    print(f"Seed: {args.seed}")
    print(f"Alpha: {args.alpha}")
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Classes: {len(classes)}")
    print(f"Output: ./result/scale_analysis (won't pollute nq_mvp results)")
    print(f"{'='*70}\n")
    
    # Analyze each class
    success_count = 0
    for class_name in classes:
        success = analyze_class(
            dataset=args.dataset,
            class_name=class_name,
            k_shot=args.k_shot,
            seed=args.seed,
            alpha=args.alpha,
            checkpoint_dir=args.checkpoint_dir
        )
        if success:
            success_count += 1
    
    print(f"\n{'='*70}")
    print(f"Analysis Complete: {success_count}/{len(classes)} classes succeeded")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
