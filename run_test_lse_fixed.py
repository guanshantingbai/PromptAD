import os
from datasets import dataset_classes

# 限制 CPU 线程数
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

if __name__ == '__main__':

    datasets = ['mvtec', 'visa']
    shot = 2  # k=2
    gpu_id = 0
    
    checkpoint_dir = './result/fusion_normal'
    
    # 修复后的LSE测试 - 3个有意义的τ值
    test_configs = [
        {'tau': 0.1, 'output_dir': './result/test_lse_tau0.1_fixed'},    # 接近max
        {'tau': 1.0, 'output_dir': './result/test_lse_tau1_fixed'},      # 标准LSE
        {'tau': 10.0, 'output_dir': './result/test_lse_tau10_fixed'},    # 接近mean
    ]
    
    print('='*80)
    print('LSE修复后测试 - 串行执行')
    print('='*80)
    print('τ值范围: 0.1 (接近max), 1.0 (标准), 10.0 (接近mean)')
    print(f'Checkpoint: {checkpoint_dir}')
    print('='*80)
    print()
    
    for config in test_configs:
        tau = config['tau']
        output_dir = config['output_dir']
        
        print(f'\n▶ 开始测试 τ={tau}')
        print(f'  输出: {output_dir}\n')
        
        # 创建日志目录
        log_dir = os.path.join(output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        for dataset in datasets:
            classes = dataset_classes[dataset]
            for cls in classes:
                checkpoint_path = f'{checkpoint_dir}/{dataset}/k_{shot}/checkpoint/CLS-Seed_111-{cls}-check_point.pt'
                
                if not os.path.exists(checkpoint_path):
                    print(f'  ⚠️  跳过 {dataset}/{cls}: checkpoint不存在')
                    continue
                
                log_file = os.path.join(log_dir, f'{dataset}_{cls}_k{shot}_lse_tau{tau}.log')
                cmd = f'python test_cls.py ' \
                      f'--dataset {dataset} ' \
                      f'--checkpoint {checkpoint_path} ' \
                      f'--gpu-id {gpu_id} ' \
                      f'--k-shot {shot} ' \
                      f'--class_name {cls} ' \
                      f'--root-dir {output_dir} ' \
                      f'--seed 111 ' \
                      f'--aggregation lse ' \
                      f'--lse-tau {tau} ' \
                      f'> {log_file} 2>&1'
                
                print(f'  测试: {dataset}/{cls}')
                os.system(cmd)
        
        print(f'\n✓ τ={tau} 测试完成')
    
    print()
    print('='*80)
    print('所有测试完成！')
    print('='*80)
    print()
    print('查看结果:')
    for config in test_configs:
        tau = config['tau']
        output_dir = config['output_dir']
        print(f'\nτ={tau}:')
        print(f'  MVTec: {output_dir}/mvtec/k_2/csv/Seed_111-results.csv')
        print(f'  VISA:  {output_dir}/visa/k_2/csv/Seed_111-results.csv')
