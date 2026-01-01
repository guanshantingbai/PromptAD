"""
Phase 2: 类别级 Prompt 清洗脚本
禁用特定类别下的 Dangerous-and-Useless prompts
"""

import pandas as pd
import argparse
from datetime import datetime
import shutil
import os

# Phase 2 清洗列表：24 个 Useless Prompts
PHASE2_CLEANING_LIST = {
    'metal_nut': [
        'metal nut with defect',
        'metal nut with a bent shape ',  # 注意：末尾有空格
        'metal nut with a flipped orientation'
    ],
    'pill': [
        'abnormal pill',
        'imperfect pill',
        'blemished pill',
        'pill with scratch'
    ],
    'cable': [
        'cable with flaw',
        'cable with defect',
        'cable with missing part',
        'cable with missing wire'
    ],
    'screw': [
        'imperfect screw',
        'blemished screw'
    ],
    'capsule': [
        'flawed capsule',
        'abnormal capsule',
        'capsule with flaw',
        'capsule with poke',
        'capsule squeezed with compression'
    ],
    'transistor': [
        'flawed transistor',
        'imperfect transistor',
        'blemished transistor',
        'transistor with damage',
        'transistor with misplaced transistor'
    ]
}

def disable_class_prompts(csv_path='prompts/manual_prompts_master_table.csv',
                          target_classes=None,
                          backup=True,
                          dry_run=False):
    """
    禁用指定类别的 Useless prompts
    
    Args:
        csv_path: prompt master table 路径
        target_classes: 目标类别列表，None 表示所有类别
        backup: 是否创建备份
        dry_run: 是否只模拟运行
    """
    
    # 读取 CSV
    df = pd.read_csv(csv_path)
    
    # 创建备份
    if backup and not dry_run:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = csv_path.replace('.csv', f'_backup_phase2_{timestamp}.csv')
        shutil.copy2(csv_path, backup_path)
        print(f"✓ 备份已创建: {backup_path}")
    
    # 如果没有指定类别，清洗所有类别
    if target_classes is None:
        target_classes = list(PHASE2_CLEANING_LIST.keys())
    
    # 统计信息
    total_disabled = 0
    class_stats = {}
    
    print("\n" + "="*80)
    print("Phase 2: 类别级 Prompt 清洗")
    print("="*80)
    
    if dry_run:
        print("\n⚠️  DRY RUN MODE - 不会实际修改文件\n")
    
    # 逐类别处理
    for class_name in target_classes:
        if class_name not in PHASE2_CLEANING_LIST:
            print(f"\n⚠️  警告: 类别 '{class_name}' 不在清洗列表中")
            continue
        
        prompts_to_disable = PHASE2_CLEANING_LIST[class_name]
        disabled_count = 0
        
        print(f"\n{'='*80}")
        print(f"类别: {class_name.upper()}")
        print(f"{'='*80}")
        print(f"计划清洗 {len(prompts_to_disable)} 个 Useless Prompts:")
        
        for prompt_text in prompts_to_disable:
            # 查找匹配的 prompt
            # 需要同时匹配 class 和 full_text
            mask = (df['class'] == class_name) & (df['full_text'] == prompt_text)
            matched_rows = df[mask]
            
            if len(matched_rows) == 0:
                print(f"  ❌ 未找到: {prompt_text}")
                continue
            
            # 检查当前状态
            current_enabled = matched_rows['enabled'].iloc[0]
            
            if not dry_run:
                # 禁用该 prompt
                df.loc[mask, 'enabled'] = False
            
            if current_enabled:
                print(f"  ✓ 已禁用: {prompt_text}")
                disabled_count += 1
            else:
                print(f"  ⊗ 已是禁用状态: {prompt_text}")
        
        class_stats[class_name] = {
            'total': len(prompts_to_disable),
            'disabled': disabled_count
        }
        total_disabled += disabled_count
    
    # 打印汇总
    print("\n" + "="*80)
    print("清洗汇总")
    print("="*80)
    
    for class_name, stats in class_stats.items():
        print(f"  {class_name:<15} : {stats['disabled']}/{stats['total']} prompts 已禁用")
    
    print(f"\n  总计: {total_disabled} 个 prompts 已禁用")
    
    # 保存修改
    if not dry_run:
        df.to_csv(csv_path, index=False)
        print(f"\n✓ 修改已保存到: {csv_path}")
    else:
        print(f"\n⚠️  DRY RUN - 未保存修改")
    
    # 统计当前启用的 prompts 数量
    total_prompts = len(df)
    enabled_prompts = df['enabled'].sum()
    disabled_prompts = total_prompts - enabled_prompts
    
    print(f"\n当前 Prompt 状态:")
    print(f"  总数: {total_prompts}")
    print(f"  启用: {enabled_prompts} ({enabled_prompts/total_prompts*100:.1f}%)")
    print(f"  禁用: {disabled_prompts} ({disabled_prompts/total_prompts*100:.1f}%)")
    
    return class_stats

def verify_phase2_cleaning(csv_path='prompts/manual_prompts_master_table.csv',
                           target_classes=None):
    """验证 Phase 2 清洗效果"""
    
    df = pd.read_csv(csv_path)
    
    if target_classes is None:
        target_classes = list(PHASE2_CLEANING_LIST.keys())
    
    print("\n" + "="*80)
    print("Phase 2 清洗验证")
    print("="*80)
    
    all_correct = True
    
    for class_name in target_classes:
        if class_name not in PHASE2_CLEANING_LIST:
            continue
        
        prompts_to_check = PHASE2_CLEANING_LIST[class_name]
        
        print(f"\n类别: {class_name}")
        print(f"  检查 {len(prompts_to_check)} 个应该被禁用的 prompts:")
        
        for prompt_text in prompts_to_check:
            mask = (df['class'] == class_name) & (df['full_text'] == prompt_text)
            matched_rows = df[mask]
            
            if len(matched_rows) == 0:
                print(f"    ❌ 未找到: {prompt_text}")
                all_correct = False
                continue
            
            is_enabled = matched_rows['enabled'].iloc[0]
            
            if is_enabled:
                print(f"    ❌ 仍然启用: {prompt_text}")
                all_correct = False
            else:
                print(f"    ✓ 已禁用: {prompt_text}")
    
    print("\n" + "="*80)
    if all_correct:
        print("✓ 验证通过：所有目标 prompts 已正确禁用")
    else:
        print("❌ 验证失败：部分 prompts 状态不正确")
    print("="*80)
    
    return all_correct

def main():
    parser = argparse.ArgumentParser(description='Phase 2: 类别级 Prompt 清洗')
    parser.add_argument('--csv-path', type=str, 
                       default='prompts/manual_prompts_master_table.csv',
                       help='Prompt master table CSV 路径')
    parser.add_argument('--classes', type=str, nargs='+',
                       help='目标类别列表（不指定则清洗所有 Phase 2 目标类别）')
    parser.add_argument('--tier', type=str, choices=['1', '2', 'all'],
                       default='all',
                       help='清洗哪一层：1 (Tier 1), 2 (Tier 2), all (全部)')
    parser.add_argument('--dry-run', action='store_true',
                       help='模拟运行，不实际修改文件')
    parser.add_argument('--no-backup', action='store_true',
                       help='不创建备份')
    parser.add_argument('--verify', action='store_true',
                       help='验证清洗效果')
    
    args = parser.parse_args()
    
    # 确定目标类别
    if args.classes:
        target_classes = args.classes
    elif args.tier == '1':
        target_classes = ['metal_nut', 'pill', 'cable']
    elif args.tier == '2':
        target_classes = ['screw', 'capsule', 'transistor']
    else:  # all
        target_classes = None  # 清洗所有
    
    if args.verify:
        verify_phase2_cleaning(args.csv_path, target_classes)
    else:
        disable_class_prompts(
            csv_path=args.csv_path,
            target_classes=target_classes,
            backup=not args.no_backup,
            dry_run=args.dry_run
        )
        
        # 自动验证
        if not args.dry_run:
            print("\n" + "="*80)
            print("自动验证清洗效果...")
            print("="*80)
            verify_phase2_cleaning(args.csv_path, target_classes)

if __name__ == "__main__":
    main()
