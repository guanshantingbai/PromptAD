import os
from datasets import dataset_classes
from multiprocessing import Pool

# 限制 CPU 线程数，避免多进程并行时 CPU 超载
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

if __name__ == '__main__':

    pool = Pool(processes=2)

    datasets = ['visa']  # 只训练visa
    shots = [2]  # 只训练k=2
    output_dir = './result/verify_expanded_retrain'  # 重新训练的输出目录
    gpu_id = 0

    # 创建日志目录
    log_dir = os.path.join(output_dir, 'cls_logs')
    os.makedirs(log_dir, exist_ok=True)

    print("="*80)
    print("VisA K=2 重新训练（验证展开的prompts）")
    print(f"输出目录: {output_dir}")
    print("="*80)

    for shot in shots:
        for dataset in datasets:
            classes = dataset_classes[dataset]
            for cls in classes[:]:
                log_file = os.path.join(log_dir, f'k{shot}_{dataset}_{cls}.log')
                sh_method = f'python train_cls.py ' \
                            f'--dataset {dataset} ' \
                            f'--gpu-id {gpu_id} ' \
                            f'--k-shot {shot} ' \
                            f'--class_name {cls} ' \
                            f'--root-dir {output_dir} ' \
                            f'> {log_file} 2>&1'

                print(f"Training: {dataset}/{cls}")
                pool.apply_async(os.system, (sh_method,))

    pool.close()
    pool.join()
    
    print("="*80)
    print("分类训练完成！")
    print(f"结果保存在: {output_dir}/visa/k_2/csv/Seed_111-results.csv")
    print("="*80)
