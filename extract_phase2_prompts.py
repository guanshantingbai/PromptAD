"""
提取 Phase 2 目标类别需要清洗的 prompts
"""

import pandas as pd
import os

# 目标类别列表（按优先级排序）
TARGET_CLASSES = {
    'Tier 1 (必做)': ['metal_nut', 'pill', 'cable'],
    'Tier 2 (推荐)': ['screw', 'capsule', 'transistor']
}

def extract_useless_prompts(dataset='mvtec', k_shot=2):
    """提取每个目标类别的 Useless prompts"""
    
    results = {}
    
    for tier, classes in TARGET_CLASSES.items():
        results[tier] = {}
        
        for cls in classes:
            csv_path = f"result/prompt_purging/phase1_5/{dataset}/k_{k_shot}/{cls}_phase1_5_classification.csv"
            
            if not os.path.exists(csv_path):
                print(f"⚠️  文件不存在: {csv_path}")
                continue
            
            df = pd.read_csv(csv_path)
            
            # 筛选 Useless prompts (separation_gap < 0)
            useless = df[df['prompt_classification'] == 'dangerous_useless'].copy()
            
            # 统计信息
            total = len(df)
            safe = len(df[df['prompt_classification'] == 'safe'])
            useful = len(df[df['prompt_classification'] == 'dangerous_useful'])
            useless_count = len(useless)
            
            results[tier][cls] = {
                'total': total,
                'safe': safe,
                'useful': useful,
                'useless': useless_count,
                'prompts': useless[['full_text', 'type', 'separation_gap', 
                                   'mean_margin_normal', 'mean_margin_abnormal']].to_dict('records')
            }
    
    return results

def print_report(results):
    """打印清洗计划报告"""
    
    print("="*100)
    print("Phase 2 类别级 Prompt 清洗计划")
    print("="*100)
    print()
    
    total_to_clean = 0
    
    for tier, classes_data in results.items():
        print(f"\n{'='*100}")
        print(f"{tier}")
        print(f"{'='*100}\n")
        
        for cls, data in classes_data.items():
            print(f"📦 类别: {cls.upper()}")
            print(f"   总 Prompts: {data['total']} | Safe: {data['safe']} | Useful: {data['useful']} | Useless: {data['useless']}")
            print(f"   Useful 占比: {data['useful']/data['total']*100:.1f}%")
            print()
            
            if data['useless'] > 0:
                print(f"   🗑️  计划清洗 {data['useless']} 个 Useless Prompts:")
                print(f"   {'-'*96}")
                print(f"   {'#':<4} {'Prompt':<50} {'Type':<10} {'Gap':<8} {'Normal':<10} {'Abnormal':<10}")
                print(f"   {'-'*96}")
                
                for i, prompt in enumerate(data['prompts'], 1):
                    prompt_text = prompt['full_text']
                    if len(prompt_text) > 48:
                        prompt_text = prompt_text[:45] + '...'
                    
                    ptype = prompt['type']
                    gap = prompt['separation_gap']
                    norm = prompt['mean_margin_normal']
                    abnorm = prompt['mean_margin_abnormal']
                    
                    print(f"   {i:<4} {prompt_text:<50} {ptype:<10} {gap:<8.3f} {norm:<10.3f} {abnorm:<10.3f}")
                
                total_to_clean += data['useless']
            else:
                print(f"   ✅ 没有需要清洗的 Useless prompts")
            
            print()
    
    print("="*100)
    print(f"总计: 计划清洗 {total_to_clean} 个 Useless Prompts")
    print("="*100)

def export_to_markdown(results, output_file='PHASE2_CLEANING_LIST.md'):
    """导出为 Markdown 表格"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Phase 2 类别级 Prompt 清洗详细列表\n\n")
        f.write("**日期**: 2025-12-31  \n")
        f.write("**数据集**: MVTec-AD  \n")
        f.write("**策略**: 删除 Dangerous-and-Useless prompts (separation_gap < 0)  \n\n")
        f.write("---\n\n")
        
        total_to_clean = 0
        
        for tier, classes_data in results.items():
            f.write(f"\n## {tier}\n\n")
            
            for cls, data in classes_data.items():
                f.write(f"### 📦 {cls.upper()}\n\n")
                f.write(f"**统计信息**:\n")
                f.write(f"- 总 Prompts: {data['total']}\n")
                f.write(f"- Safe: {data['safe']} ({data['safe']/data['total']*100:.1f}%)\n")
                f.write(f"- Dangerous-but-Useful: {data['useful']} ({data['useful']/data['total']*100:.1f}%)\n")
                f.write(f"- Dangerous-and-Useless: {data['useless']} ({data['useless']/data['total']*100:.1f}%)\n\n")
                
                if data['useless'] > 0:
                    f.write(f"**🗑️ 计划清洗 {data['useless']} 个 Useless Prompts**:\n\n")
                    f.write("| # | Prompt | Type | Gap | Normal Margin | Abnormal Margin |\n")
                    f.write("|---|--------|------|-----|---------------|------------------|\n")
                    
                    for i, prompt in enumerate(data['prompts'], 1):
                        prompt_text = prompt['full_text']
                        ptype = prompt['type']
                        gap = prompt['separation_gap']
                        norm = prompt['mean_margin_normal']
                        abnorm = prompt['mean_margin_abnormal']
                        
                        f.write(f"| {i} | `{prompt_text}` | {ptype} | {gap:.3f} | {norm:.3f} | {abnorm:.3f} |\n")
                    
                    total_to_clean += data['useless']
                    f.write("\n")
                else:
                    f.write("✅ **没有需要清洗的 Useless prompts**\n\n")
                
                f.write("---\n\n")
        
        f.write(f"\n## 📊 清洗总结\n\n")
        f.write(f"- **总计划清洗**: {total_to_clean} 个 Useless Prompts\n")
        f.write(f"- **覆盖类别**: {sum(len(classes_data) for classes_data in results.values())} 个\n")
        f.write(f"- **预期效果**: Semantic AUROC 平均提升 +3~5 points\n\n")
        
        f.write("---\n\n")
        f.write("## 🔍 Gap 含义解释\n\n")
        f.write("```\n")
        f.write("separation_gap = mean_margin_abnormal - mean_margin_normal\n")
        f.write("\n")
        f.write("Gap < 0: Dangerous-and-Useless\n")
        f.write("  → 异常样本的 margin 反而比正常样本高\n")
        f.write("  → Prompt 无法区分正常/异常，是噪声\n")
        f.write("  → 需要清洗\n")
        f.write("\n")
        f.write("Gap > 0: Dangerous-but-Useful 或 Safe\n")
        f.write("  → 异常样本的 margin 比正常样本低\n")
        f.write("  → Prompt 方向正确，保留\n")
        f.write("```\n")
    
    print(f"\n✓ Markdown 报告已保存到: {output_file}")

if __name__ == "__main__":
    results = extract_useless_prompts()
    print_report(results)
    export_to_markdown(results)
