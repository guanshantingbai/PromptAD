"""
run_seg_MB_single_dataset.py - Run Memory Bank test on a single dataset
Useful for testing or running on just MVTec or VISA
"""
import os
import sys
from datasets import dataset_classes
from multiprocessing import Pool

# Limit CPU threads to avoid CPU overload when running in parallel
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

if __name__ == '__main__':
    
    # Configuration - can be modified via command line
    if len(sys.argv) > 1:
        target_dataset = sys.argv[1]  # e.g., 'mvtec' or 'visa'
    else:
        target_dataset = 'mvtec'  # Default
    
    if len(sys.argv) > 2:
        num_parallel_processes = int(sys.argv[2])
    else:
        num_parallel_processes = 2
    
    if len(sys.argv) > 3:
        gpu_ids = [int(x) for x in sys.argv[3].split(',')]
    else:
        gpu_ids = [0, 1]

    output_dir = './result_MB'
    
    # Create log directory
    log_dir = os.path.join(output_dir, 'seg_MB_logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Get classes for the target dataset
    if target_dataset not in dataset_classes:
        print(f"Error: Unknown dataset '{target_dataset}'")
        print(f"Available datasets: {list(dataset_classes.keys())}")
        sys.exit(1)
    
    classes = dataset_classes[target_dataset]
    
    print("=" * 80)
    print(f"Memory Bank Segmentation Test - {target_dataset.upper()}")
    print("=" * 80)
    print(f"Dataset: {target_dataset}")
    print(f"Classes: {len(classes)}")
    print(f"Parallel processes: {num_parallel_processes}")
    print(f"GPU IDs: {gpu_ids}")
    print(f"Output directory: {output_dir}")
    print("-" * 80)
    
    # Create process pool
    pool = Pool(processes=num_parallel_processes)
    
    # Submit tasks
    for idx, cls in enumerate(classes):
        gpu_id = gpu_ids[idx % len(gpu_ids)]  # Round-robin GPU assignment
        
        log_file = os.path.join(log_dir, f'{target_dataset}_{cls}.log')
        
        sh_method = f'python test_seg_MB.py ' \
                    f'--dataset {target_dataset} ' \
                    f'--gpu-id {gpu_id} ' \
                    f'--class_name {cls} ' \
                    f'--root-dir {output_dir} ' \
                    f'--batch-size 32 ' \
                    f'--resolution 240 ' \
                    f'--img-resize 240 ' \
                    f'--img-cropsize 240 ' \
                    f'--vis False ' \
                    f'--num-workers 4 ' \
                    f'> {log_file} 2>&1'

        print(f"[{idx+1}/{len(classes)}] GPU{gpu_id} - {target_dataset}/{cls}")
        pool.apply_async(os.system, (sh_method,))

    pool.close()
    pool.join()
    
    print("\n" + "=" * 80)
    print("All tasks completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Logs saved to: {log_dir}")
    print("\nUsage: python run_seg_MB_single_dataset.py [dataset] [num_processes] [gpu_ids]")
    print("Example: python run_seg_MB_single_dataset.py mvtec 4 0,1,2,3")
