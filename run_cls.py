import os
from datasets import dataset_classes
from multiprocessing import Pool

if __name__ == '__main__':

    pool = Pool(processes=2)

#    datasets = ['mvtec', 'visa']
    datasets = ['mvtec']
    shots = [1, 2, 4]
#    shots = [1]
    gpu_id = 1
    for shot in shots:
        for dataset in datasets:
            classes = dataset_classes[dataset]
            for cls in classes[:]:
                sh_method = f'python train_cls.py ' \
                            f'--dataset {dataset} ' \
                            f'--gpu-id {gpu_id} ' \
                            f'--k-shot {shot} ' \
                            f'--class_name {cls} ' \

                print(sh_method)
                pool.apply_async(os.system, (sh_method,))

    pool.close()
    pool.join()

