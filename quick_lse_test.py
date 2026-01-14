#!/usr/bin/env python3
"""
快速测试真正的LSE实现
仅测试少数类别以快速验证效果
"""

import os
import subprocess
import time

# 快速测试：每个数据集选2个类别
QUICK_TEST = {
    'mvtec': ['bottle', 'cable'],
    'visa': ['candle', 'capsules']
}

TAU_VALUES = [5, 10, 20]
K = 2

def run_test(dataset_name, class_name, tau):
    """运行单个测试"""
    cmd = [
        'python', 'test_cls.py',
        '--dataset', dataset_name,
        '--data-path', f'./data/{dataset_name}',
        '--save-path', f'./result/test_true_lse_tau{tau}',
        '--checkpoint', f'./result/{dataset_name}/k_{K}/checkpoints/{class_name}/epoch_best.pth',
        '--cls-name', class_name,
        '-k', str(K),
        '--aggregation', 'lse',
        '--lse-tau', str(tau)
    ]
    
    print(f"  Testing {dataset_name}/{class_name} with τ={tau}...", end=' ')
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Failed")
        return False
    
    print(f"✓")
    return True

def main():
    print("="*60)
    print("Quick LSE Test (True Mathematical LSE)")
    print("="*60)
    
    for tau in TAU_VALUES:
        print(f"\n[τ = {tau}]")
        for dataset_name, classes in QUICK_TEST.items():
            for class_name in classes:
                run_test(dataset_name, class_name, tau)
                time.sleep(0.3)
    
    print("\n" + "="*60)
    print("Quick test completed!")
    print("="*60)

if __name__ == '__main__':
    main()
