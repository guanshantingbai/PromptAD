import os
from datasets import dataset_classes
from multiprocessing import Pool

# 限制 CPU 线程数
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

if __name__ == '__main__':

    pool = Pool(processes=2)

    datasets = ['mvtec', 'visa']
    shot = 2  # k=2
    gpu_id = 0
    
    # 输出目录
    output_dir = './result/test_lse_tau2_k2'
    checkpoint_dir = './result/fusion_normal'  # baseline checkpoints
    
    # LSE参数
    aggregation = 'lse'
    lse_tau = 2.0
    
    # 创建日志目录
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    print('='*80)
    print(f'批量测试LSE聚合 (k={shot}, τ={lse_tau})')
    print('='*80)
    print(f'Checkpoint目录: {checkpoint_dir}')
    print(f'输出目录: {output_dir}')
    print(f'并行进程: 2')
    print('='*80)
    print()
    
    for dataset in datasets:
        classes = dataset_classes[dataset]
        for cls in classes:
            checkpoint_path = f'{checkpoint_dir}/{dataset}/k_{shot}/checkpoint/CLS-Seed_111-{cls}-check_point.pt'
            
            # 检查checkpoint是否存在
            if not os.path.exists(checkpoint_path):
                print(f'⚠️  跳过 {dataset}/{cls}: checkpoint不存在')
                continue
            
            log_file = os.path.join(log_dir, f'{dataset}_{cls}_k{shot}_lse_tau2.log')
            sh_method = f'python test_cls.py ' \
                f'--dataset {dataset} ' \
                f'--checkpoint {checkpoint_path} ' \
                f'--gpu-id {gpu_id} ' \
                f'--k-shot {shot} ' \
                f'--class_name {cls} ' \
                f'--root-dir {output_dir} ' \
                f'--seed 111 ' \
                f'--aggregation {aggregation} ' \
                f'--lse-tau {lse_tau} ' \
                f'> {log_file} 2>&1'
            
            print(f'提交测试: {dataset}/{cls} (τ={lse_tau})')
            pool.apply_async(os.system, (sh_method,))

    pool.close()
    pool.join()
    
    print()
    print('='*80)
    print(f'所有测试完成！结果保存在: {output_dir}')
    print('='*80)
    print()
    print('查看结果:')
    print(f'  MVTec: {output_dir}/mvtec/k_{shot}/csv/Seed_111-results.csv')
    print(f'  VISA:  {output_dir}/visa/k_{shot}/csv/Seed_111-results.csv')
