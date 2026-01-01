"""
演示如何通过修改表格来控制prompts
"""

import pandas as pd
import shutil

def demo_prompt_control():
    """演示prompt控制功能"""
    
    table_path = 'prompts/manual_prompts_master_table.csv'
    demo_path = 'prompts/manual_prompts_master_table_demo.csv'
    
    # 复制原表格作为demo
    shutil.copy(table_path, demo_path)
    
    # 读取表格
    df = pd.read_csv(demo_path)
    
    print("="*80)
    print("Demo: Controlling Prompts via Table")
    print("="*80)
    
    # 示例1: 查看bottle的所有prompts
    print("\n1. Original prompts for 'bottle':")
    print("-"*80)
    bottle_prompts = df[df['class'] == 'bottle']
    print(bottle_prompts[['index_in_class', 'type', 'full_text', 'enabled']].to_string(index=False))
    
    # 示例2: 禁用某些低质量的generic prompts
    print("\n\n2. Disabling some generic prompts (e.g., 'blemished', 'imperfect'):")
    print("-"*80)
    
    # 为bottle类别禁用"blemished"和"imperfect"
    df.loc[(df['class'] == 'bottle') & (df['template'] == 'blemished {}'), 'enabled'] = False
    df.loc[(df['class'] == 'bottle') & (df['template'] == 'imperfect {}'), 'enabled'] = False
    
    print("✓ Disabled prompts: 'blemished bottle', 'imperfect bottle'")
    
    # 示例3: 添加人工评分
    print("\n\n3. Adding manual scores and notes:")
    print("-"*80)
    
    # 为bottle的specific prompts打分
    df.loc[(df['class'] == 'bottle') & (df['template'] == '{} with large breakage'), 'manual_score'] = 9
    df.loc[(df['class'] == 'bottle') & (df['template'] == '{} with large breakage'), 'relevance'] = 'high'
    df.loc[(df['class'] == 'bottle') & (df['template'] == '{} with large breakage'), 'notes'] = 'Very relevant for bottle defects'
    
    df.loc[(df['class'] == 'bottle') & (df['template'] == '{} with small breakage'), 'manual_score'] = 9
    df.loc[(df['class'] == 'bottle') & (df['template'] == '{} with small breakage'), 'relevance'] = 'high'
    
    df.loc[(df['class'] == 'bottle') & (df['template'] == '{} with contamination'), 'manual_score'] = 8
    df.loc[(df['class'] == 'bottle') & (df['template'] == '{} with contamination'), 'relevance'] = 'high'
    
    print("✓ Added scores and notes for specific prompts")
    
    # 示例4: 标记清洗决策
    print("\n\n4. Marking cleanup actions:")
    print("-"*80)
    
    # 标记保留和删除
    df.loc[(df['class'] == 'bottle') & (df['type'] == 'specific'), 'action'] = 'keep'
    df.loc[(df['class'] == 'bottle') & (df['template'] == 'blemished {}'), 'action'] = 'remove'
    df.loc[(df['class'] == 'bottle') & (df['template'] == 'imperfect {}'), 'action'] = 'remove'
    
    print("✓ Marked actions: 'keep' for specific prompts, 'remove' for low-quality ones")
    
    # 保存修改后的表格
    df.to_csv(demo_path, index=False)
    
    # 显示最终结果
    print("\n\n5. Final state of bottle prompts:")
    print("-"*80)
    bottle_final = df[df['class'] == 'bottle']
    display_cols = ['index_in_class', 'type', 'full_text', 'enabled', 'manual_score', 'relevance', 'action']
    print(bottle_final[display_cols].to_string(index=False))
    
    print("\n" + "="*80)
    print("Demo Summary:")
    print("="*80)
    print(f"✓ Created demo table: {demo_path}")
    print(f"✓ Original: 11 prompts for bottle (all enabled)")
    print(f"✓ Modified: 9 prompts enabled, 2 disabled")
    print(f"✓ Added manual scores for specific prompts")
    print(f"✓ Marked cleanup actions")
    print("\nTo use the modified table:")
    print(f"  1. Review the changes in: {demo_path}")
    print(f"  2. If satisfied, rename it to: {table_path}")
    print(f"  3. Run training with the updated prompts")
    print(f"  4. The model will automatically use only enabled=True prompts\n")


if __name__ == '__main__':
    demo_prompt_control()
