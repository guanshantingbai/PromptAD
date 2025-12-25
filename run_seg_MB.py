"""
run_seg_MB.py - Parallel execution script for Memory Bank only segmentation
This script runs test_seg_MB.py in parallel for multiple datasets and classes.
"""
import os
from datasets import dataset_classes
from multiprocessing import Pool

# Limit CPU threads to avoid CPU overload when running in parallel
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

if __name__ == '__main__':

    # Configuration
    num_parallel_processes = 2  # Number of parallel GPU processes
    datasets = ['mvtec', 'visa']
    shots = [1]  # Memory bank doesn't depend on k-shot, but keep for consistency
    output_dir = './result_MB'  # Output directory for MB results
    gpu_ids = [0, 1]  # Available GPUs
    
    # Create log directory
    log_dir = os.path.join(output_dir, 'seg_MB_logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Collect all tasks
    tasks = []
    for shot in shots:
        for dataset in datasets:
            classes = dataset_classes[dataset]
            for cls in classes:
                tasks.append((shot, dataset, cls))
    
    print(f"Total tasks: {len(tasks)}")
    print(f"Parallel processes: {num_parallel_processes}")
    print(f"Output directory: {output_dir}")
    print(f"GPU IDs: {gpu_ids}")
    print("-" * 80)
    
    # Create process pool
    pool = Pool(processes=num_parallel_processes)
    
    # Submit tasks
    for idx, (shot, dataset, cls) in enumerate(tasks):
        gpu_id = gpu_ids[idx % len(gpu_ids)]  # Round-robin GPU assignment
        
        log_file = os.path.join(log_dir, f'k{shot}_{dataset}_{cls}.log')
        
        sh_method = f'python test_seg_MB.py ' \
                    f'--dataset {dataset} ' \
                    f'--gpu-id {gpu_id} ' \
                    f'--k-shot {shot} ' \
                    f'--class_name {cls} ' \
                    f'--root-dir {output_dir} ' \
                    f'--batch-size 32 ' \
                    f'--resolution 240 ' \
                    f'--img-resize 240 ' \
                    f'--img-cropsize 240 ' \
                    f'--vis False ' \
                    f'--num-workers 4 ' \
                    f'> {log_file} 2>&1'

        print(f"[{idx+1}/{len(tasks)}] GPU{gpu_id} - {dataset}/{cls}")
        pool.apply_async(os.system, (sh_method,))

    pool.close()
    pool.join()
    
    print("\n" + "=" * 80)
    print("All tasks completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Logs saved to: {log_dir}")
