#!/usr/bin/env python3
"""
测试真正的 LSE 聚合实现
验证不同 τ 值对 MVTec 和 VISA 数据集的影响
"""

import os
import subprocess
import time

# 测试配置
DATASETS = {
    'mvtec': [
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
    ],
    'visa': [
        'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
        'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'
    ]
}

# 测试不同的 τ 值
TAU_VALUES = [5, 10, 20]

def run_test(dataset_name, class_name, tau, k=2):
    """运行单个测试"""
    # 使用fusion_normal的baseline checkpoint
    checkpoint_path = f'./result/fusion_normal/{dataset_name}/k_{k}/checkpoint/CLS-Seed_111-{class_name}-check_point.pt'
    
    cmd = [
        'python', 'test_cls.py',
        '--dataset', dataset_name,
        '--root-dir', f'./data/{dataset_name}',
        '--output-dir', f'./result/test_true_lse_tau{tau}',
        '--checkpoint', checkpoint_path,
        '--class_name', class_name,
        '--k-shot', str(k),
        '--aggregation', 'lse',
        '--lse-tau', str(tau)
    ]
    
    print(f"  Running: {dataset_name}/{class_name} with τ={tau}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  ❌ Failed: {result.stderr[:200]}")
        return False
    
    return True

def main():
    print("=" * 60)
    print("Testing True LSE Aggregation")
    print("=" * 60)
    print()
    
    total_tests = sum(len(classes) for classes in DATASETS.values()) * len(TAU_VALUES)
    current_test = 0
    
    for tau in TAU_VALUES:
        print(f"\n{'='*60}")
        print(f"Testing τ = {tau}")
        print(f"{'='*60}")
        
        for dataset_name, classes in DATASETS.items():
            print(f"\n[{dataset_name.upper()}]")
            
            for class_name in classes:
                current_test += 1
                print(f"[{current_test}/{total_tests}] {dataset_name}/{class_name} (τ={tau})")
                
                success = run_test(dataset_name, class_name, tau)
                
                if not success:
                    print(f"  Skipping due to error...")
                
                # 短暂延迟避免GPU冲突
                time.sleep(0.5)
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
    print("\nResults saved in:")
    for tau in TAU_VALUES:
        print(f"  - result/test_true_lse_tau{tau}/")
    print("\nTo compare results, run:")
    print("  python compare_lse_results.py")

if __name__ == '__main__':
    main()
