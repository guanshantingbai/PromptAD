"""
聚合实验结果：从指定目录读取所有CSV，生成4张表格的汇总报告
- 2个任务（分类i_roc，分割p_roc）
- 2个数据集（mvtec，visa）
- 3个k值（1, 2, 4）作为列
- 各类别作为行，底部添加平均值
"""

import os
import pandas as pd
import argparse
from datasets import dataset_classes


def load_csv_data(base_dir, dataset, k_shot, seed=111):
    """加载指定数据集和k值的CSV文件"""
    csv_path = os.path.join(base_dir, dataset, f'k_{k_shot}', 'csv', f'Seed_{seed}-results.csv')
    
    if not os.path.exists(csv_path):
        print(f"警告: 文件不存在 {csv_path}")
        return None
    
    df = pd.read_csv(csv_path, index_col=0)
    return df


def create_aggregated_table(base_dir, dataset, metric, k_values=[1, 2, 4], seed=111):
    """
    创建聚合表格
    
    Args:
        base_dir: 基础目录（如 output_max_fusion）
        dataset: 数据集名称（mvtec 或 visa）
        metric: 指标名称（i_roc 或 p_roc）
        k_values: k-shot值列表
        seed: 种子值
    
    Returns:
        DataFrame: 聚合后的表格
    """
    classes = dataset_classes[dataset]
    
    # 初始化结果字典
    results = {f'k={k}': [] for k in k_values}
    
    # 收集每个类别在不同k值下的结果
    for cls in classes:
        for k in k_values:
            df = load_csv_data(base_dir, dataset, k, seed)
            
            if df is None:
                results[f'k={k}'].append(None)
                continue
            
            # 查找当前类别的行
            row_key = f'{dataset}-{cls}'
            if row_key in df.index and metric in df.columns:
                value = df.loc[row_key, metric]
                results[f'k={k}'].append(value)
            else:
                results[f'k={k}'].append(None)
    
    # 创建DataFrame
    result_df = pd.DataFrame(results, index=classes)
    
    # 计算平均值（忽略None/NaN）
    mean_row = result_df.mean(axis=0, skipna=True)
    result_df.loc['Average'] = mean_row
    
    return result_df


def generate_report(base_dir, output_path=None, seed=111):
    """
    生成完整的汇总报告
    
    Args:
        base_dir: 基础目录
        output_path: 输出CSV路径（默认为base_dir/aggregated_results.csv）
        seed: 种子值
    """
    if output_path is None:
        output_path = os.path.join(base_dir, 'aggregated_results.csv')
    
    k_values = [1, 2, 4]
    
    # 创建Excel writer以支持多sheet（如果需要）或者拼接成一个大CSV
    results_text = []
    
    # 1. MVTec 分类 (i_roc)
    results_text.append("="*80)
    results_text.append("MVTec Dataset - Classification (i_roc)")
    results_text.append("="*80)
    mvtec_cls = create_aggregated_table(base_dir, 'mvtec', 'i_roc', k_values, seed)
    results_text.append(mvtec_cls.to_string())
    results_text.append("")
    
    # 2. MVTec 分割 (p_roc)
    results_text.append("="*80)
    results_text.append("MVTec Dataset - Segmentation (p_roc)")
    results_text.append("="*80)
    mvtec_seg = create_aggregated_table(base_dir, 'mvtec', 'p_roc', k_values, seed)
    results_text.append(mvtec_seg.to_string())
    results_text.append("")
    
    # 3. VisA 分类 (i_roc)
    results_text.append("="*80)
    results_text.append("VisA Dataset - Classification (i_roc)")
    results_text.append("="*80)
    visa_cls = create_aggregated_table(base_dir, 'visa', 'i_roc', k_values, seed)
    results_text.append(visa_cls.to_string())
    results_text.append("")
    
    # 4. VisA 分割 (p_roc)
    results_text.append("="*80)
    results_text.append("VisA Dataset - Segmentation (p_roc)")
    results_text.append("="*80)
    visa_seg = create_aggregated_table(base_dir, 'visa', 'p_roc', k_values, seed)
    results_text.append(visa_seg.to_string())
    results_text.append("")
    
    # 保存为文本格式（方便查看）
    txt_path = output_path.replace('.csv', '.txt')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(results_text))
    
    print(f"✅ 文本报告已保存到: {txt_path}")
    
    # 保存为CSV格式（使用分隔符分开各表）
    with open(output_path, 'w') as f:
        f.write("MVTec Dataset - Classification (i_roc)\n")
        mvtec_cls.to_csv(f)
        f.write("\n")
        
        f.write("MVTec Dataset - Segmentation (p_roc)\n")
        mvtec_seg.to_csv(f)
        f.write("\n")
        
        f.write("VisA Dataset - Classification (i_roc)\n")
        visa_cls.to_csv(f)
        f.write("\n")
        
        f.write("VisA Dataset - Segmentation (p_roc)\n")
        visa_seg.to_csv(f)
    
    print(f"✅ CSV报告已保存到: {output_path}")
    
    # 打印到控制台
    print("\n" + "\n".join(results_text))
    
    return {
        'mvtec_cls': mvtec_cls,
        'mvtec_seg': mvtec_seg,
        'visa_cls': visa_cls,
        'visa_seg': visa_seg
    }


def main():
    parser = argparse.ArgumentParser(description='聚合实验结果')
    parser.add_argument('--dir', type=str, default='./output_max_fusion',
                        help='实验结果目录（包含mvtec和visa子目录）')
    parser.add_argument('--output', type=str, default=None,
                        help='输出CSV路径（默认为dir/aggregated_results.csv）')
    parser.add_argument('--seed', type=int, default=111,
                        help='种子值')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dir):
        print(f"错误: 目录不存在 {args.dir}")
        return
    
    generate_report(args.dir, args.output, args.seed)


if __name__ == '__main__':
    main()
