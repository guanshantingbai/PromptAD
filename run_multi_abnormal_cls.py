"""
Multi-Abnormal Prototypes Training Script for CLS Task
训练MVTec + ViSA两个数据集，k-shot=2，使用Multi-Abnormal Prototypes方法
"""
import os
from datasets import dataset_classes
from multiprocessing import Pool

# 限制 CPU 线程数，避免多进程并行时 CPU 超载
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

if __name__ == '__main__':

    # ========== Configuration ==========
    pool = Pool(processes=2)  # 并行数=2（根据计算资源设定）
    
    datasets = ['mvtec', 'visa']
    shots = [2]  # 只训练k=2
    output_dir = './result/new_paradigm_1'  # Multi-Abnormal Prototypes输出目录
    gpu_id = 0
    
    # Multi-Abnormal Prototypes超参数
    pull_weight = 0.1      # L_pull: 稳定正常原型
    rep_weight = 0.05      # L_rep: 异常原型排斥
    margin_weight = 1.0    # L_margin: 硬负样本margin
    rep_gamma = 0.3        # 排斥阈值
    topk_abnormal = 2      # 推理时top-k聚合
    
    # Fusion-aware参数
    fusion_lambda = 0.1    # 与记忆分支融合权重
    fusion_loss_weight = 0.5  # Fusion-aware triplet loss权重

    # 创建日志目录
    log_dir = os.path.join(output_dir, 'cls_logs')
    os.makedirs(log_dir, exist_ok=True)
    print(f"\n{'='*80}")
    print(f"Multi-Abnormal Prototypes CLS Training")
    print(f"{'='*80}")
    print(f"Output Directory: {output_dir}")
    print(f"Datasets: {datasets}")
    print(f"K-shot: {shots}")
    print(f"Parallel processes: 2")
    print(f"Hyperparameters:")
    print(f"  - pull_weight: {pull_weight}")
    print(f"  - rep_weight: {rep_weight}")
    print(f"  - margin_weight: {margin_weight}")
    print(f"  - rep_gamma: {rep_gamma}")
    print(f"  - topk_abnormal: {topk_abnormal}")
    print(f"  - fusion_lambda: {fusion_lambda}")
    print(f"  - fusion_loss_weight: {fusion_loss_weight}")
    print(f"{'='*80}\n")

    # ========== Training Loop ==========
    total_tasks = 0
    for shot in shots:
        for dataset in datasets:
            classes = dataset_classes[dataset]
            for cls in classes:
                total_tasks += 1
                
                log_file = os.path.join(log_dir, f'k{shot}_{dataset}_{cls}.log')
                
                sh_method = f'python train_cls.py ' \
                            f'--dataset {dataset} ' \
                            f'--gpu-id {gpu_id} ' \
                            f'--k-shot {shot} ' \
                            f'--class_name {cls} ' \
                            f'--root-dir {output_dir} ' \
                            f'--pull-weight {pull_weight} ' \
                            f'--rep-weight {rep_weight} ' \
                            f'--margin-weight {margin_weight} ' \
                            f'--rep-gamma {rep_gamma} ' \
                            f'--topk-abnormal {topk_abnormal} ' \
                            f'--fusion-lambda {fusion_lambda} ' \
                            f'--fusion-loss-weight {fusion_loss_weight} ' \
                            f'--Epoch 200 ' \
                            f'--resolution 256 ' \
                            f'--vis False ' \
                            f'> {log_file} 2>&1'

                print(f"[{total_tasks:02d}] {dataset:8s} | {cls:15s} | k={shot} | Log: {log_file}")
                pool.apply_async(os.system, (sh_method,))

    print(f"\n{'='*80}")
    print(f"Total tasks submitted: {total_tasks}")
    print(f"Training started with {pool._processes} parallel processes...")
    print(f"{'='*80}\n")
    
    pool.close()
    pool.join()
    
    print(f"\n{'='*80}")
    print(f"All training tasks completed!")
    print(f"Results saved in: {output_dir}")
    print(f"Logs saved in: {log_dir}")
    print(f"{'='*80}\n")
