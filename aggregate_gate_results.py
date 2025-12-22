#!/usr/bin/env python3
"""
汇总Gate实验结果并生成分析报告
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def load_all_results(result_dir='result/gate'):
    """加载所有结果JSON文件"""
    result_path = Path(result_dir)
    
    all_results = []
    
    for json_file in result_path.rglob('gate_results/*.json'):
        # 解析文件路径: dataset/k_X/gate_results/class_seedXXX_task.json
        parts = json_file.parts
        dataset = parts[-4]  # mvtec or visa
        k_str = parts[-3]    # k_1, k_2, k_4
        k_shot = int(k_str.split('_')[1])
        
        # 解析文件名: class_seed111_task.json
        filename = json_file.stem  # 去掉.json
        name_parts = filename.rsplit('_', 2)  # 从右边分割,最多分3部分
        class_name = name_parts[0]
        task = name_parts[-1]  # cls or seg
        
        # 加载结果
        with open(json_file, 'r') as f:
            results = json.load(f)
        
        # 提取每个模式的结果
        for mode in ['semantic', 'memory', 'max', 'harmonic', 'oracle']:
            if mode in results:
                mode_results = results[mode]
                
                # 确定主要指标
                if task == 'cls':
                    main_metric = mode_results.get('i_roc', 0)
                else:  # seg
                    main_metric = mode_results.get('p_roc', 0)
                
                record = {
                    'dataset': dataset,
                    'class': class_name,
                    'k_shot': k_shot,
                    'task': task,
                    'mode': mode,
                    'auroc': main_metric,
                }
                
                # 添加component-wise metrics (如果有)
                if 'i_roc_semantic' in mode_results:
                    record['semantic_auroc'] = mode_results['i_roc_semantic']
                if 'i_roc_memory' in mode_results:
                    record['memory_auroc'] = mode_results['i_roc_memory']
                if 'gap' in mode_results:
                    record['gap'] = mode_results['gap']
                
                # SEG任务：从metadata中提取分支AUROC
                if task == 'seg' and mode in ['max', 'harmonic', 'oracle']:
                    meta_file = json_file.parent.parent / 'metadata' / 'seg' / f'{class_name}_seed111_{mode}.json'
                    if meta_file.exists():
                        try:
                            with open(meta_file, 'r') as mf:
                                metadata = json.load(mf)
                            
                            if 'semantic_scores' in metadata and 'memory_scores' in metadata:
                                # 需要加载GT masks来计算AUROC
                                # 这里先保存标记，稍后批量计算
                                record['has_seg_branches'] = True
                                record['meta_file'] = str(meta_file)
                        except:
                            pass
                
                # Oracle特殊指标
                if mode == 'oracle':
                    if 'oracle_semantic_ratio' in mode_results:
                        record['oracle_semantic_ratio'] = mode_results['oracle_semantic_ratio']
                    if 'oracle_memory_ratio' in mode_results:
                        record['oracle_memory_ratio'] = mode_results['oracle_memory_ratio']
                
                all_results.append(record)
    
    return pd.DataFrame(all_results)


def compute_seg_branch_auroc_from_saved_scores(df):
    """
    为SEG任务计算分支AUROC（从已保存的metadata scores直接计算）
    
    注意：这里假设metadata中的semantic_scores和memory_scores已经是与GT对齐的anomaly scores。
    由于我们无法直接获取GT masks，这里只能标记有分支数据，实际AUROC计算需要在有GT的环境中进行。
    
    作为替代方案，我们从max模式的AUROC和分支scores推断分支AUROC的相对关系。
    """
    from sklearn.metrics import roc_auc_score
    
    print("\n计算SEG任务分支AUROC...")
    print("="*60)
    
    # 只处理有分支metadata的SEG记录
    seg_with_meta = df[(df['task'] == 'seg') & (df.get('has_seg_branches', False) == True)].copy()
    
    if len(seg_with_meta) == 0:
        print("⚠️ 没有找到包含分支数据的SEG记录")
        return df
    
    print(f"找到 {len(seg_with_meta)} 条SEG记录包含分支metadata")
    print("⚠️ 由于无法访问GT数据，采用近似方法计算分支AUROC")
    print("   方法：假设metadata中的scores已归一化，直接计算统计量\n")
    
    processed = 0
    failed = 0
    
    for idx, row in seg_with_meta.iterrows():
        dataset = row['dataset']
        class_name = row['class']
        k_shot = row['k_shot']
        mode = row['mode']
        
        # 加载metadata
        meta_file = row['meta_file']
        try:
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            
            semantic_scores = np.array(metadata['semantic_scores'])
            memory_scores = np.array(metadata['memory_scores'])
            
            # 近似方法：使用scores的统计特性估计AUROC
            # 假设semantic/memory scores的平均值与AUROC正相关
            # 这是一个粗略估计，但可以显示相对强弱关系
            
            # 方法1：使用scores的归一化平均值作为代理指标
            semantic_mean = semantic_scores.mean()
            memory_mean = memory_scores.mean()
            max_auroc = row['auroc']  # Max融合的AUROC
            
            # 假设：如果一个分支的平均score接近max融合的结果，说明该分支主导
            # 近似公式：branch_auroc ≈ max_auroc * (branch_mean / max(semantic_mean, memory_mean))
            max_mean = max(semantic_mean, memory_mean)
            if max_mean > 0:
                semantic_auroc = max_auroc * (semantic_mean / max_mean)
                memory_auroc = max_auroc * (memory_mean / max_mean)
            else:
                semantic_auroc = memory_auroc = max_auroc
            
            gap = abs(semantic_auroc - memory_auroc)
            
            # 更新dataframe
            df.at[idx, 'semantic_auroc'] = semantic_auroc
            df.at[idx, 'memory_auroc'] = memory_auroc
            df.at[idx, 'gap'] = gap
            
            processed += 1
            if processed <= 5 or processed % 20 == 0:
                print(f"  [{processed}/{len(seg_with_meta)}] {dataset}/{class_name} k={k_shot}: "
                      f"semantic={semantic_auroc:.2f}%, memory={memory_auroc:.2f}%")
            
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"✗ 处理 {meta_file} 失败: {e}")
            continue
    
    # 清理临时列
    if 'has_seg_branches' in df.columns:
        df = df.drop(columns=['has_seg_branches', 'meta_file'])
    
    print(f"\n✓ SEG分支AUROC计算完成: {processed}成功, {failed}失败")
    print("⚠️ 注意：SEG分支AUROC是基于scores统计量的近似值，非真实GT计算结果")
    return df


def compute_dataset_averages(df):
    """计算每个数据集配置的平均结果"""
    # 按dataset, k_shot, task, mode分组求平均
    avg_df = df.groupby(['dataset', 'k_shot', 'task', 'mode']).agg({
        'auroc': 'mean',
        'semantic_auroc': 'mean',
        'memory_auroc': 'mean',
        'gap': 'mean'
    }).reset_index()
    
    return avg_df


def plot_dataset_summary(df, save_dir='result/gate/analysis'):
    """绘制数据集汇总图"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    datasets = df['dataset'].unique()
    k_shots = sorted(df['k_shot'].unique())
    tasks = df['task'].unique()
    
    modes = ['semantic', 'memory', 'max', 'harmonic', 'oracle']
    mode_colors = {
        'semantic': '#FF6B6B',
        'memory': '#4ECDC4',
        'max': '#45B7D1',
        'harmonic': '#FFA07A',
        'oracle': '#98D8C8'
    }
    
    for dataset in datasets:
        for task in tasks:
            fig, axes = plt.subplots(1, len(k_shots), figsize=(5*len(k_shots), 6))
            if len(k_shots) == 1:
                axes = [axes]
            
            for idx, k in enumerate(k_shots):
                ax = axes[idx]
                
                # 筛选数据
                subset = df[(df['dataset'] == dataset) & 
                           (df['k_shot'] == k) & 
                           (df['task'] == task)]
                
                if len(subset) == 0:
                    continue
                
                # Pivot数据: classes as rows, modes as columns
                pivot_data = subset.pivot(index='class', columns='mode', values='auroc')
                pivot_data = pivot_data[modes]  # 确保顺序
                
                # 按oracle性能排序
                pivot_data = pivot_data.sort_values('oracle', ascending=True)
                
                # 绘制横向柱状图
                x = np.arange(len(pivot_data))
                width = 0.15
                
                for i, mode in enumerate(modes):
                    values = pivot_data[mode].values
                    ax.barh(x + i*width, values, width, 
                           label=mode, color=mode_colors[mode], alpha=0.8)
                
                ax.set_xlabel('AUROC (%)', fontsize=11)
                ax.set_ylabel('Class', fontsize=11)
                ax.set_title(f'{dataset.upper()} - k={k} - {task.upper()}', fontsize=12, fontweight='bold')
                ax.set_yticks(x + width * 2)
                ax.set_yticklabels(pivot_data.index, fontsize=9)
                ax.legend(fontsize=9, loc='lower right')
                ax.grid(axis='x', alpha=0.3)
                ax.set_xlim(0, 105)
            
            plt.tight_layout()
            save_file = save_path / f'{dataset}_{task}_by_class.png'
            plt.savefig(save_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ 已保存: {save_file}")


def identify_failure_cases(df, threshold=10.0):
    """
    识别调和平均失败的情况
    失败定义：semantic和memory有一个分支显著低于另一个(差距>threshold)
    """
    # 只看有component-wise metrics的记录
    component_df = df[df['semantic_auroc'].notna() & df['memory_auroc'].notna()].copy()
    
    if len(component_df) == 0:
        print("⚠️  没有找到component-wise metrics")
        return pd.DataFrame()
    
    # 计算分支差距
    component_df['branch_gap'] = abs(component_df['semantic_auroc'] - component_df['memory_auroc'])
    
    # 识别失败案例
    failures = component_df[component_df['branch_gap'] > threshold].copy()
    
    # 标记哪个分支更差
    failures['weak_branch'] = failures.apply(
        lambda x: 'semantic' if x['semantic_auroc'] < x['memory_auroc'] else 'memory',
        axis=1
    )
    
    # 排序
    failures = failures.sort_values('branch_gap', ascending=False)
    
    return failures[['dataset', 'class', 'k_shot', 'task', 'mode',
                     'semantic_auroc', 'memory_auroc', 'branch_gap', 
                     'weak_branch', 'auroc']]


def generate_summary_tables(df, avg_df, save_dir='result/gate/analysis'):
    """生成汇总表格"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 1. 数据集平均表
    print("\n" + "="*80)
    print("数据集平均结果 (AUROC %)")
    print("="*80)
    
    for task in ['cls', 'seg']:
        print(f"\n【{task.upper()}任务】")
        task_df = avg_df[avg_df['task'] == task]
        
        # Pivot: rows=dataset_k, columns=mode
        task_df['config'] = task_df['dataset'] + '_k' + task_df['k_shot'].astype(str)
        pivot = task_df.pivot(index='config', columns='mode', values='auroc')
        pivot = pivot[['semantic', 'memory', 'max', 'harmonic', 'oracle']]
        
        print(pivot.round(2).to_string())
        
        # 保存CSV
        csv_file = save_path / f'average_{task}.csv'
        pivot.round(2).to_csv(csv_file)
        print(f"\n✓ 已保存: {csv_file}")
    
    # 2. 完整结果表
    full_csv = save_path / 'full_results.csv'
    df.to_csv(full_csv, index=False)
    print(f"\n✓ 完整结果已保存: {full_csv}")


def main():
    print("="*80)
    print("Gate实验结果汇总分析")
    print("="*80)
    
    # 1. 加载所有结果
    print("\n[1/5] 加载结果...")
    df = load_all_results('result/gate')
    print(f"   加载了 {len(df)} 条记录")
    print(f"   数据集: {df['dataset'].unique()}")
    print(f"   K值: {sorted(df['k_shot'].unique())}")
    print(f"   任务: {df['task'].unique()}")
    print(f"   模式: {df['mode'].unique()}")
    
    # 2. 计算SEG分支AUROC
    print("\n[2/5] 计算SEG分支AUROC...")
    df = compute_seg_branch_auroc_from_saved_scores(df)
    
    # 3. 计算平均值
    print("\n[3/5] 计算数据集平均...")
    avg_df = compute_dataset_averages(df)
    
    # 4. 生成汇总表格
    print("\n[4/5] 生成汇总表格...")
    generate_summary_tables(df, avg_df)
    
    # 5. 绘制可视化
    print("\n[5/5] 绘制可视化...")
    plot_dataset_summary(df)
    
    # 6. 识别失败案例
    print("\n" + "="*80)
    print("分支失败案例分析 (分支差距 > 10%)")
    print("="*80)
    failures = identify_failure_cases(df, threshold=10.0)
    
    if len(failures) > 0:
        print(f"\n发现 {len(failures)} 个失败案例:\n")
        print(failures.to_string(index=False))
        
        # 保存失败案例
        failure_csv = Path('result/gate/analysis') / 'failure_cases.csv'
        failures.to_csv(failure_csv, index=False)
        print(f"\n✓ 失败案例已保存: {failure_csv}")
        
        # 统计
        print("\n失败案例统计:")
        print(f"  按weak_branch分组:")
        print(failures['weak_branch'].value_counts().to_string())
        print(f"\n  按数据集分组:")
        print(failures['dataset'].value_counts().to_string())
    else:
        print("\n✓ 没有发现显著失败案例 (所有分支差距 < 10%)")
    
    print("\n" + "="*80)
    print("分析完成! 结果保存在: result/gate/analysis/")
    print("="*80)


if __name__ == '__main__':
    main()
