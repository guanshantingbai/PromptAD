"""
结构性修复验证实验
只跑代表性类别快速验证修复效果
"""
import os
from multiprocessing import Pool

# 限制 CPU 线程数
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

if __name__ == '__main__':

    # ========== Configuration ==========
    pool = Pool(processes=2)
    
    # 代表性类别选择
    test_classes = {
        'mvtec': [
            # 严重退化类
            'zipper',      # -30.92% semantic
            'pill',        # -25.80%
            'cable',       # -21.18%
            'transistor',  # -12.31%
            'metal_nut',   # -10.02%
            'grid',        # -10.57%
            # 提升/稳定类
            'toothbrush',  # +19.58%
            'bottle',      # +0.36%
            'screw',       # -11.93% (低基线)
        ],
        # ViSA严重退化的两个类（可选，如果想快速看全局）
        # 'visa': ['macaroni1', 'pcb1']
    }
    
    output_dir = './result/fix_validation'
    gpu_id = 0
    shot = 2
    
    # 修复后的超参数（保持不变，验证结构性修复效果）
    pull_weight = 0.1
    rep_weight = 0.05
    margin_weight = 1.0
    rep_gamma = 0.3
    topk_abnormal = 2
    fusion_lambda = 0.1
    fusion_loss_weight = 0.5

    # 创建日志目录
    log_dir = os.path.join(output_dir, 'cls_logs')
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"🔧 Multi-Abnormal Prototypes: 结构性修复验证实验")
    print(f"{'='*80}")
    print(f"修复内容:")
    print(f"  1. 训练阶段 t_train = 10.0 (降低10倍，防止CE饱和)")
    print(f"  2. Margin loss 改为 top-2 hardest (更有效抑制假阳性)")
    print(f"  3. Label smoothing = 0.1 (辅助防止饱和)")
    print(f"\n测试类别: {sum(len(v) for v in test_classes.values())}个")
    for dataset, classes in test_classes.items():
        print(f"  {dataset}: {classes}")
    print(f"Output Directory: {output_dir}")
    print(f"{'='*80}\n")

    # ========== Training Loop ==========
    total_tasks = 0
    for dataset, classes in test_classes.items():
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

            print(f"[{total_tasks:02d}] {dataset:8s} | {cls:15s} | Log: {log_file}")
            pool.apply_async(os.system, (sh_method,))

    print(f"\n{'='*80}")
    print(f"Total tasks: {total_tasks}")
    print(f"Training started with 2 parallel processes...")
    print(f"{'='*80}\n")
    
    pool.close()
    pool.join()
    
    print(f"\n{'='*80}")
    print(f"✅ Validation training completed!")
    print(f"Results saved in: {output_dir}")
    print(f"{'='*80}\n")
