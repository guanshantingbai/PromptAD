import os
from datasets import dataset_classes
from multiprocessing import Pool

# 限制 CPU 线程数，避免多进程并行时 CPU 超载
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

if __name__ == '__main__':

    pool = Pool(processes=2)

    datasets = ['mvtec', 'visa']
    shots = [1, 2, 4]
    output_dir = './output_max_fusion'  # 指定输出目录
    gpu_id = 0
    
    # 创建日志目录
    log_dir = os.path.join(output_dir, 'seg_logs')
    os.makedirs(log_dir, exist_ok=True)
    
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

                print(sh_method)
                pool.apply_async(os.system, (sh_method,))

    pool.close()
    pool.join()




