"""
统一的批量训练脚本 - 覆盖所有剩余实验
包括:
1. MVTec CLS k=1,2,4 (已完成k=2)
2. MVTec SEG k=1,2,4
3. VisA CLS k=1,2,4
4. VisA SEG k=1,2,4 (如果需要)

输出目录: result/prompt1/
"""
import os
from datasets import dataset_classes
from multiprocessing import Pool
from datetime import datetime
import argparse

# 限制 CPU 线程数，避免多进程并行时 CPU 超载
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'


def run_experiments(args):
    """执行所有实验配置"""
    
    # 记录主程序起始时间
    main_start_time = datetime.now()
    print(f"\n{'='*80}")
    print(f"批量训练起始时间: {main_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    pool = Pool(processes=args.num_processes)

    output_dir = './result/prompt1'  # 统一输出目录
    gpu_id = args.gpu_id
    
    # 创建主日志目录
    os.makedirs(output_dir, exist_ok=True)
    main_log_file = os.path.join(output_dir, 'all_experiments_timing.log')
    
    with open(main_log_file, 'w') as f:
        f.write(f"批量训练起始时间: {main_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"配置: GPU={gpu_id}, 并行进程数={args.num_processes}\n")
        f.write(f"{'='*80}\n\n")
    
    total_jobs = 0
    
    # ========== MVTec 分类任务 ==========
    if args.run_mvtec_cls:
        print(f"\n{'='*80}")
        print("配置 MVTec 分类任务 (CLS)")
        print(f"{'='*80}")
        
        cls_log_dir = os.path.join(output_dir, 'cls_logs')
        os.makedirs(cls_log_dir, exist_ok=True)
        
        for shot in args.shots:
            classes = dataset_classes['mvtec']
            for cls in classes:
                log_file = os.path.join(cls_log_dir, f'k{shot}_mvtec_{cls}.log')
                sh_method = f'python train_cls.py ' \
                            f'--dataset mvtec ' \
                            f'--gpu-id {gpu_id} ' \
                            f'--k-shot {shot} ' \
                            f'--class_name {cls} ' \
                            f'--root-dir {output_dir} ' \
                            f'> {log_file} 2>&1'
                
                print(f"  [CLS-MVTec] k={shot}, class={cls}")
                pool.apply_async(os.system, (sh_method,))
                total_jobs += 1
    
    # ========== MVTec 分割任务 ==========
    if args.run_mvtec_seg:
        print(f"\n{'='*80}")
        print("配置 MVTec 分割任务 (SEG)")
        print(f"{'='*80}")
        
        seg_log_dir = os.path.join(output_dir, 'seg_logs')
        os.makedirs(seg_log_dir, exist_ok=True)
        
        for shot in args.shots:
            classes = dataset_classes['mvtec']
            for cls in classes:
                log_file = os.path.join(seg_log_dir, f'k{shot}_mvtec_{cls}.log')
                sh_method = f'python train_seg.py ' \
                            f'--dataset mvtec ' \
                            f'--gpu-id {gpu_id} ' \
                            f'--k-shot {shot} ' \
                            f'--class_name {cls} ' \
                            f'--root-dir {output_dir} ' \
                            f'> {log_file} 2>&1'
                
                print(f"  [SEG-MVTec] k={shot}, class={cls}")
                pool.apply_async(os.system, (sh_method,))
                total_jobs += 1
    
    # ========== VisA 分类任务 ==========
    if args.run_visa_cls:
        print(f"\n{'='*80}")
        print("配置 VisA 分类任务 (CLS)")
        print(f"{'='*80}")
        
        cls_log_dir = os.path.join(output_dir, 'cls_logs')
        os.makedirs(cls_log_dir, exist_ok=True)
        
        for shot in args.shots:
            classes = dataset_classes['visa']
            for cls in classes:
                log_file = os.path.join(cls_log_dir, f'k{shot}_visa_{cls}.log')
                sh_method = f'python train_cls.py ' \
                            f'--dataset visa ' \
                            f'--gpu-id {gpu_id} ' \
                            f'--k-shot {shot} ' \
                            f'--class_name {cls} ' \
                            f'--root-dir {output_dir} ' \
                            f'> {log_file} 2>&1'
                
                print(f"  [CLS-VisA] k={shot}, class={cls}")
                pool.apply_async(os.system, (sh_method,))
                total_jobs += 1
    
    # ========== VisA 分割任务 ==========
    if args.run_visa_seg:
        print(f"\n{'='*80}")
        print("配置 VisA 分割任务 (SEG)")
        print(f"{'='*80}")
        
        seg_log_dir = os.path.join(output_dir, 'seg_logs')
        os.makedirs(seg_log_dir, exist_ok=True)
        
        for shot in args.shots:
            classes = dataset_classes['visa']
            for cls in classes:
                log_file = os.path.join(seg_log_dir, f'k{shot}_visa_{cls}.log')
                sh_method = f'python train_seg.py ' \
                            f'--dataset visa ' \
                            f'--gpu-id {gpu_id} ' \
                            f'--k-shot {shot} ' \
                            f'--class_name {cls} ' \
                            f'--root-dir {output_dir} ' \
                            f'> {log_file} 2>&1'
                
                print(f"  [SEG-VisA] k={shot}, class={cls}")
                pool.apply_async(os.system, (sh_method,))
                total_jobs += 1
    
    print(f"\n{'='*80}")
    print(f"总任务数: {total_jobs}")
    print(f"{'='*80}\n")
    
    with open(main_log_file, 'a') as f:
        f.write(f"总任务数: {total_jobs}\n")
        f.write(f"{'='*80}\n\n")
    
    # 等待所有任务完成
    pool.close()
    pool.join()
    
    # 记录主程序终止时间和总用时
    main_end_time = datetime.now()
    main_elapsed_time = main_end_time - main_start_time
    hours, remainder = divmod(main_elapsed_time.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n{'='*80}")
    print(f"批量训练终止时间: {main_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总用时: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
    print(f"{'='*80}\n")
    
    # 记录到日志文件
    with open(main_log_file, 'a') as f:
        f.write(f"\n批量训练终止时间: {main_end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总用时: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒\n")
        f.write(f"{'='*80}\n")


def get_args():
    parser = argparse.ArgumentParser(description='批量训练所有实验配置')
    
    # 基础配置
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID')
    parser.add_argument('--num-processes', type=int, default=2, 
                        help='并行进程数 (默认2)')
    parser.add_argument('--shots', type=int, nargs='+', default=[1, 2, 4],
                        help='k-shot配置 (默认: 1 2 4)')
    
    # 任务选择
    parser.add_argument('--run-mvtec-cls', action='store_true', default=False,
                        help='运行MVTec分类任务')
    parser.add_argument('--run-mvtec-seg', action='store_true', default=False,
                        help='运行MVTec分割任务')
    parser.add_argument('--run-visa-cls', action='store_true', default=False,
                        help='运行VisA分类任务')
    parser.add_argument('--run-visa-seg', action='store_true', default=False,
                        help='运行VisA分割任务')
    parser.add_argument('--run-all', action='store_true', default=False,
                        help='运行所有任务')
    
    args = parser.parse_args()
    
    # 如果指定了run-all，启用所有任务
    if args.run_all:
        args.run_mvtec_cls = True
        args.run_mvtec_seg = True
        args.run_visa_cls = True
        args.run_visa_seg = True
    
    # 如果没有指定任何任务，提示用户
    if not any([args.run_mvtec_cls, args.run_mvtec_seg, 
                args.run_visa_cls, args.run_visa_seg]):
        print("\n警告: 未指定任何任务！请使用以下选项之一:")
        print("  --run-mvtec-cls    MVTec分类")
        print("  --run-mvtec-seg    MVTec分割")
        print("  --run-visa-cls     VisA分类")
        print("  --run-visa-seg     VisA分割")
        print("  --run-all          所有任务")
        print("\n示例: python run_all_experiments.py --run-mvtec-seg --shots 1 2")
        exit(1)
    
    return args


if __name__ == '__main__':
    args = get_args()
    
    print("\n" + "="*80)
    print("批量训练配置汇总")
    print("="*80)
    print(f"GPU ID: {args.gpu_id}")
    print(f"并行进程数: {args.num_processes}")
    print(f"k-shot配置: {args.shots}")
    print(f"输出目录: ./result/prompt1/")
    print("-"*80)
    print("任务清单:")
    if args.run_mvtec_cls:
        print("  ✓ MVTec 分类 (CLS) - 15类")
    if args.run_mvtec_seg:
        print("  ✓ MVTec 分割 (SEG) - 15类")
    if args.run_visa_cls:
        print("  ✓ VisA 分类 (CLS) - 12类")
    if args.run_visa_seg:
        print("  ✓ VisA 分割 (SEG) - 12类")
    
    total_classes = 0
    if args.run_mvtec_cls: total_classes += 15 * len(args.shots)
    if args.run_mvtec_seg: total_classes += 15 * len(args.shots)
    if args.run_visa_cls: total_classes += 12 * len(args.shots)
    if args.run_visa_seg: total_classes += 12 * len(args.shots)
    
    print(f"\n预计总任务数: {total_classes}")
    print("="*80 + "\n")
    
    run_experiments(args)
