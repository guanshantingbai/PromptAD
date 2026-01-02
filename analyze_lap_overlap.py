import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_checkpoint(ckpt_path):
    """加载checkpoint并提取LAP向量"""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    # 提取abnormal context vectors (LAP)
    if 'prompt_learner.abnormal_ctx' in ckpt:
        lap_vectors = ckpt['prompt_learner.abnormal_ctx']
    else:
        print(f"Keys in checkpoint: {list(ckpt.keys())}")
        return None
    
    return lap_vectors

def analyze_lap_diversity(lap_vectors, name):
    """分析LAP向量的多样性"""
    print(f"\n{'='*80}")
    print(f"分析: {name}")
    print(f"{'='*80}")
    
    n_pro_ab, n_ctx_ab, dim = lap_vectors.shape
    print(f"LAP向量形状: {lap_vectors.shape}")
    print(f"  - n_pro_ab (LAP原型数量): {n_pro_ab}")
    print(f"  - n_ctx_ab (每个LAP的token数): {n_ctx_ab}")
    print(f"  - dim (embedding维度): {dim}")
    
    # 计算每个LAP的平均embedding (跨token维度)
    lap_avg = lap_vectors.mean(dim=1)  # [n_pro_ab, dim]
    lap_avg_norm = lap_avg / lap_avg.norm(dim=1, keepdim=True)
    
    # 计算LAP之间的余弦相似度矩阵
    similarity_matrix = lap_avg_norm @ lap_avg_norm.T
    similarity_matrix = similarity_matrix.numpy()
    
    # 统计分析
    # 提取上三角（不包括对角线）
    triu_indices = np.triu_indices(n_pro_ab, k=1)
    pairwise_similarities = similarity_matrix[triu_indices]
    
    print(f"\nLAP原型间余弦相似度统计:")
    print(f"  - 平均值: {pairwise_similarities.mean():.4f}")
    print(f"  - 标准差: {pairwise_similarities.std():.4f}")
    print(f"  - 最小值: {pairwise_similarities.min():.4f}")
    print(f"  - 最大值: {pairwise_similarities.max():.4f}")
    print(f"  - 中位数: {np.median(pairwise_similarities):.4f}")
    
    # 统计高相似度对
    high_sim_threshold = 0.9
    high_sim_pairs = np.sum(pairwise_similarities > high_sim_threshold)
    print(f"\n高相似度对 (> {high_sim_threshold}): {high_sim_pairs} / {len(pairwise_similarities)} ({high_sim_pairs/len(pairwise_similarities)*100:.1f}%)")
    
    very_high_sim_threshold = 0.95
    very_high_sim_pairs = np.sum(pairwise_similarities > very_high_sim_threshold)
    print(f"极高相似度对 (> {very_high_sim_threshold}): {very_high_sim_pairs} / {len(pairwise_similarities)} ({very_high_sim_pairs/len(pairwise_similarities)*100:.1f}%)")
    
    # 找出最相似的几对
    top_k = min(5, len(pairwise_similarities))
    top_indices = np.argsort(pairwise_similarities)[-top_k:][::-1]
    print(f"\n最相似的{top_k}对LAP:")
    for idx in top_indices:
        i, j = triu_indices[0][idx], triu_indices[1][idx]
        sim = pairwise_similarities[idx]
        print(f"  LAP-{i} <-> LAP-{j}: {sim:.4f}")
    
    return similarity_matrix, pairwise_similarities

def visualize_similarity(similarity_matrix, name, save_path):
    """可视化相似度矩阵"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, 
                annot=True, 
                fmt='.2f', 
                cmap='RdYlGn_r',
                vmin=0.0, 
                vmax=1.0,
                square=True,
                cbar_kws={'label': 'Cosine Similarity'})
    plt.title(f'LAP Prototype Similarity Matrix\n{name}')
    plt.xlabel('LAP Index')
    plt.ylabel('LAP Index')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n相似度矩阵已保存到: {save_path}")
    plt.close()

def main():
    # 分析baseline和promptpurging的checkpoint
    datasets = [
        ('baseline', 'mvtec', 'metal_nut', 2),
        ('promptpurging', 'mvtec', 'metal_nut', 2),
        ('baseline', 'visa', 'pcb2', 2),
        ('promptpurging', 'visa', 'pcb2', 2),
    ]
    
    results = []
    
    for result_dir, dataset, classname, k_shot in datasets:
        ckpt_path = f"result/{result_dir}/{dataset}/k_{k_shot}/checkpoint/CLS-Seed_111-{classname}-check_point.pt"
        
        if not Path(ckpt_path).exists():
            print(f"\n⚠️  Checkpoint不存在: {ckpt_path}")
            continue
        
        name = f"{result_dir}/{dataset}/{classname}/k_{k_shot}"
        lap_vectors = load_checkpoint(ckpt_path)
        
        if lap_vectors is None:
            continue
        
        sim_matrix, pairwise_sims = analyze_lap_diversity(lap_vectors, name)
        
        # 保存可视化
        save_dir = Path("analysis/lap_diversity")
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{result_dir}_{dataset}_{classname}_k{k_shot}.png"
        visualize_similarity(sim_matrix, name, save_path)
        
        # 收集结果
        results.append({
            'result_dir': result_dir,
            'dataset': dataset,
            'class': classname,
            'k_shot': k_shot,
            'n_lap': lap_vectors.shape[0],
            'mean_sim': pairwise_sims.mean(),
            'std_sim': pairwise_sims.std(),
            'max_sim': pairwise_sims.max(),
            'high_sim_ratio': np.sum(pairwise_sims > 0.9) / len(pairwise_sims),
        })
    
    # 汇总结果
    if results:
        print(f"\n{'='*80}")
        print("汇总结果")
        print(f"{'='*80}")
        df = pd.DataFrame(results)
        print(df.to_string(index=False))
        
        # 保存CSV
        csv_path = "analysis/lap_diversity/summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n结果已保存到: {csv_path}")

if __name__ == "__main__":
    main()
