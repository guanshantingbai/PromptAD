import os
from datasets import dataset_classes
from multiprocessing import Pool
from datetime import datetime

# 限制 CPU 线程数，避免多进程并行时 CPU 超载
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

if __name__ == '__main__':

    # 记录主程序起始时间
    main_start_time = datetime.now()
    print(f"\n{'='*80}")
    print(f"主程序起始时间: {main_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    pool = Pool(processes=3)

    datasets = ['mvtec', 'visa']
#    datasets = ['mvtec']
    shots = [2, 4]
#    shots = [1]
    output_dir = './result/baseline'  # 指定输出目录
    gpu_id = 1

    # 创建日志目录
    log_dir = os.path.join(output_dir, 'cls_logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建主日志文件记录时间信息
    main_log_file = os.path.join(log_dir, 'main_timing.log')
    with open(main_log_file, 'w') as f:
        f.write(f"主程序起始时间: {main_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n")

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
                            f'--root-dir {output_dir}' \

                print(sh_method)
                pool.apply_async(os.system, (sh_method,))

    pool.close()
    pool.join()
    
    # 记录主程序终止时间和总用时
    main_end_time = datetime.now()
    main_elapsed_time = main_end_time - main_start_time
    hours, remainder = divmod(main_elapsed_time.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n{'='*80}")
    print(f"主程序终止时间: {main_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"主程序总用时: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
    print(f"{'='*80}\n")
    
    # 记录到日志文件
    with open(main_log_file, 'a') as f:
        f.write(f"\n主程序终止时间: {main_end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"主程序总用时: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒\n")
        f.write(f"{'='*80}\n")

