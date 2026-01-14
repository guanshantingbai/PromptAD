#!/usr/bin/env python3
"""
检查LSE测试进度
"""

import os
import glob

TAU_VALUES = [5, 10, 20]

DATASETS = {
    'mvtec': ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 
              'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 
              'transistor', 'wood', 'zipper'],
    'visa': ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
             'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
}

def check_progress():
    total = sum(len(classes) for classes in DATASETS.values()) * len(TAU_VALUES)
    
    print("="*60)
    print("LSE测试进度检查")
    print("="*60)
    
    for tau in TAU_VALUES:
        print(f"\n[τ = {tau}]")
        completed = 0
        
        for dataset_name, classes in DATASETS.items():
            dataset_completed = 0
            for class_name in classes:
                csv_path = f"result/test_true_lse_tau{tau}/{dataset_name}/k_2/csv/{class_name}.csv"
                if os.path.exists(csv_path):
                    dataset_completed += 1
                    completed += 1
            
            print(f"  {dataset_name}: {dataset_completed}/{len(classes)} 完成")
        
        print(f"  小计: {completed}/{len(DATASETS['mvtec']) + len(DATASETS['visa'])} 完成")
    
    # 总进度
    total_completed = 0
    for tau in TAU_VALUES:
        for dataset_name, classes in DATASETS.items():
            for class_name in classes:
                csv_path = f"result/test_true_lse_tau{tau}/{dataset_name}/k_2/csv/{class_name}.csv"
                if os.path.exists(csv_path):
                    total_completed += 1
    
    print(f"\n{'='*60}")
    print(f"总进度: {total_completed}/{total} ({100*total_completed/total:.1f}%)")
    print("="*60)
    
    if total_completed == 0:
        print("\n⏳ 测试刚启动，请稍候...")
    elif total_completed < total:
        print(f"\n⏳ 测试进行中... 剩余 {total - total_completed} 个")
    else:
        print("\n✅ 所有测试已完成！")
        print("\n运行以下命令查看结果分析:")
        print("  python compare_true_lse_results.py")

if __name__ == '__main__':
    check_progress()
