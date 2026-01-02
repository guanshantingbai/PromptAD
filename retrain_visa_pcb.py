import os
from multiprocessing import Pool

# 限制 CPU 线程数
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

if __name__ == '__main__':

    pool = Pool(processes=2)

    dataset = 'visa'
    pcb_classes = ['pcb1', 'pcb2', 'pcb3', 'pcb4']  # 只训练PCB类别
    shots = [1, 2, 4]  # 全K-shot
    output_dir = './result/promptpurging2'  # 覆盖原目录
    gpu_id = 0

    # 创建日志目录
    log_dir = os.path.join(output_dir, 'pcb_retrain_logs')
    os.makedirs(log_dir, exist_ok=True)

    print("="*80)
    print("VisA PCB类别重新训练 (修复class_mapping BUG)")
    print(f"类别: {pcb_classes}")
    print(f"K-shot: {shots}")
    print(f"输出目录: {output_dir}")
    print("="*80)

    # 先训练分类任务
    print("\n[阶段1] 分类任务训练...")
    for shot in shots:
        for cls in pcb_classes:
            log_file = os.path.join(log_dir, f'cls_k{shot}_{cls}.log')
            sh_method = f'python train_cls.py ' \
                        f'--dataset {dataset} ' \
                        f'--gpu-id {gpu_id} ' \
                        f'--k-shot {shot} ' \
                        f'--class_name {cls} ' \
                        f'--root-dir {output_dir} ' \
                        f'> {log_file} 2>&1'

            print(f"  Training: {dataset}/{cls} K={shot}")
            pool.apply_async(os.system, (sh_method,))

    pool.close()
    pool.join()
    
    print("\n[阶段1完成] 分类训练完成！")
    
    # 分割任务
    pool = Pool(processes=2)
    print("\n[阶段2] 分割任务训练...")
    for shot in shots:
        for cls in pcb_classes:
            log_file = os.path.join(log_dir, f'seg_k{shot}_{cls}.log')
            sh_method = f'python train_seg.py ' \
                        f'--dataset {dataset} ' \
                        f'--gpu-id {gpu_id} ' \
                        f'--k-shot {shot} ' \
                        f'--class_name {cls} ' \
                        f'--root-dir {output_dir} ' \
                        f'> {log_file} 2>&1'

            print(f"  Training: {dataset}/{cls} K={shot}")
            pool.apply_async(os.system, (sh_method,))

    pool.close()
    pool.join()

    print("\n" + "="*80)
    print("PCB类别重新训练完成！")
    print(f"结果已保存到: {output_dir}/visa/")
    print("="*80)
