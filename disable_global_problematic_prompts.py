#!/usr/bin/env python3
"""
禁用 Phase 1.5 分析中发现的全局问题 prompts
基于 MVTEC_PHASE1_5_SUMMARY.md 中的分析结果
"""

import pandas as pd
import os
from datetime import datetime

# 6个全局问题 prompt templates
PROBLEMATIC_PROMPTS = [
    "imperfect {}",      # 7类失效, gap=-0.37, R_eps=1.000
    "flawed {}",         # 6类失效, gap=-0.40, R_eps=0.980
    "{} with defect",    # 6类失效, gap=-0.36, R_eps=1.000
    "blemished {}",      # 5类失效, gap=-0.34, R_eps=0.945
    "abnormal {}",       # 4类失效, gap=-0.44, R_eps=0.813
    "{} with flaw",      # 4类失效, gap=-0.20, R_eps=0.812
]


def disable_prompts(
    table_path='prompts/manual_prompts_master_table.csv',
    backup=True,
    dry_run=False
):
    """
    禁用指定的 prompts
    
    Args:
        table_path: 主表路径
        backup: 是否备份原表
        dry_run: 是否只显示将要修改的内容而不实际修改
    """
    
    # 读取表格
    if not os.path.exists(table_path):
        print(f"❌ Error: Table not found at {table_path}")
        return None
    
    df = pd.read_csv(table_path)
    print(f"✓ Loaded table: {table_path}")
    print(f"  Total prompts: {len(df)}")
    print(f"  Currently enabled: {df['enabled'].sum()}")
    
    # 备份原表
    if backup and not dry_run:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = table_path.replace('.csv', f'_backup_{timestamp}.csv')
        df.to_csv(backup_path, index=False)
        print(f"✓ Backup created: {backup_path}")
    
    # 找到要禁用的prompts
    mask = df['template'].isin(PROBLEMATIC_PROMPTS) & (df['type'] == 'generic')
    affected = df[mask].copy()
    
    print(f"\n{'='*70}")
    print(f"Found {len(affected)} prompts to disable:")
    print(f"{'='*70}")
    
    # 按 template 分组统计
    for template in PROBLEMATIC_PROMPTS:
        count = len(affected[affected['template'] == template])
        if count > 0:
            classes = affected[affected['template'] == template]['class'].tolist()
            print(f"\n📌 '{template}'")
            print(f"   Affects {count} classes: {', '.join(classes[:5])}", end='')
            if len(classes) > 5:
                print(f" + {len(classes)-5} more", end='')
            print()
    
    # 显示详细信息
    if not affected.empty:
        print(f"\n{'='*70}")
        print("Detailed list of prompts to be disabled:")
        print(f"{'='*70}\n")
        
        display_cols = ['prompt_id', 'class', 'template', 'full_text', 'enabled']
        print(affected[display_cols].to_string(index=False))
    
    # 执行修改
    if dry_run:
        print(f"\n{'='*70}")
        print("🔍 DRY RUN - No changes made")
        print(f"{'='*70}")
        print(f"\nTo apply changes, run with dry_run=False")
        return df
    
    # 实际禁用
    df.loc[mask, 'enabled'] = False
    
    # 保存
    df.to_csv(table_path, index=False)
    
    print(f"\n{'='*70}")
    print("✅ SUCCESS - Prompts disabled")
    print(f"{'='*70}")
    print(f"  Disabled: {len(affected)} prompts")
    print(f"  Remaining enabled: {df['enabled'].sum()}")
    print(f"  Saved to: {table_path}")
    
    return df


def verify_changes(table_path='prompts/manual_prompts_master_table.csv'):
    """验证修改结果"""
    
    df = pd.read_csv(table_path)
    
    # 检查问题prompts是否已禁用
    mask = df['template'].isin(PROBLEMATIC_PROMPTS) & (df['type'] == 'generic')
    problematic = df[mask]
    
    still_enabled = problematic[problematic['enabled'] == True]
    
    print(f"\n{'='*70}")
    print("Verification Report")
    print(f"{'='*70}")
    
    if len(still_enabled) > 0:
        print(f"⚠️  Warning: {len(still_enabled)} problematic prompts still enabled:")
        print(still_enabled[['prompt_id', 'class', 'template', 'enabled']])
    else:
        print(f"✅ All {len(problematic)} problematic prompts are disabled")
    
    # 统计每个类别剩余的prompts数量
    print(f"\n{'='*70}")
    print("Enabled prompts per class:")
    print(f"{'='*70}\n")
    
    enabled_by_class = df[df['enabled'] == True].groupby('class').size()
    print(enabled_by_class.to_string())
    
    print(f"\n{'='*70}")
    print(f"Total enabled prompts: {df['enabled'].sum()}/{len(df)}")
    print(f"{'='*70}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Disable global problematic prompts identified in Phase 1.5 analysis'
    )
    parser.add_argument(
        '--table_path',
        type=str,
        default='prompts/manual_prompts_master_table.csv',
        help='Path to prompt master table'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip backup creation'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without actually modifying the table'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify changes after disabling prompts'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Phase 1: Disable Global Problematic Prompts")
    print("=" * 70)
    print(f"\nBased on Phase 1.5 analysis (MVTEC_PHASE1_5_SUMMARY.md)")
    print(f"Disabling {len(PROBLEMATIC_PROMPTS)} generic prompt templates:\n")
    for i, template in enumerate(PROBLEMATIC_PROMPTS, 1):
        print(f"  {i}. '{template}'")
    print()
    
    # 执行禁用
    df = disable_prompts(
        table_path=args.table_path,
        backup=not args.no_backup,
        dry_run=args.dry_run
    )
    
    # 验证
    if not args.dry_run and args.verify:
        verify_changes(args.table_path)


if __name__ == '__main__':
    main()
