"""
提取baseline模型的semantic分支AUROC
从baseline checkpoint加载权重，只使用semantic分支进行评估
"""
import argparse
import torch
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from datasets import get_dataloader_from_args, dataset_classes
from utils.training_utils import setup_seed
from utils.metrics import metric_cal_img, metric_cal_pix
from PromptAD import PromptAD

DATASETS = {
    'mvtec': ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 
              'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 
              'transistor', 'wood', 'zipper'],
    'visa': ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 
             'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
}

def evaluate_semantic_branch(model, dataloader, task, device):
    """只使用semantic分支进行评估"""
    model.eval()
    
    if task == 'cls':
        scores = []
        gt_labels = []
        
        with torch.no_grad():
            for (data, mask, label, name, img_type) in dataloader:
                data = data.to(device)
                
                # 只获取semantic分支的分数
                visual_features = model.encode_image(data)
                if isinstance(visual_features, tuple):
                    visual_features = visual_features[0]
                
                textual_score = model.calculate_textual_anomaly_score(visual_features, 'cls')
                
                scores.extend(textual_score.cpu().numpy())
                gt_labels.extend(label.numpy())
        
        # 计算AUROC
        result_dict = metric_cal_img(np.array(scores), gt_labels, None)
        return result_dict['i_roc']
    
    else:  # seg
        score_maps = []
        gt_masks = []
        
        with torch.no_grad():
            for (data, mask, label, name, img_type) in dataloader:
                data = data.to(device)
                
                # 只获取semantic分支的异常图
                visual_features = model.encode_image(data)
                textual_anomaly_map = model.calculate_textual_anomaly_score(visual_features, 'seg')
                
                score_maps.extend(textual_anomaly_map.cpu().numpy())
                
                for m in mask:
                    m = m.numpy()
                    m[m > 0] = 1
                    gt_masks.append(m)
        
        # 计算pixel-level AUROC
        result_dict = metric_cal_pix(np.array(score_maps), gt_masks)
        return result_dict['p_roc']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--backbone', type=str, default='ViT-B-16-plus-240')
    parser.add_argument('--pretrained_dataset', type=str, default='laion400m_e32')
    parser.add_argument('--img-resize', type=int, default=240)
    parser.add_argument('--img-cropsize', type=int, default=240)
    parser.add_argument('--resolution', type=int, default=400)
    parser.add_argument('--n_ctx', type=int, default=4)
    parser.add_argument('--n_ctx_ab', type=int, default=1)
    parser.add_argument('--n_pro', type=int, default=1)
    parser.add_argument('--n_pro_ab', type=int, default=4)
    
    args = parser.parse_args()
    
    setup_seed(args.seed)
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    
    baseline_dir = Path('result/baseline')
    results = []
    
    total_tasks = len(DATASETS['mvtec']) * 3 * 2 + len(DATASETS['visa']) * 3 * 2
    pbar = tqdm(total=total_tasks, desc="提取semantic分支AUROC")
    
    for dataset_name, classes in DATASETS.items():
        for k_shot in [1, 2, 4]:
            checkpoint_dir = baseline_dir / dataset_name / f'k_{k_shot}' / 'checkpoint'
            
            for class_name in classes:
                for task in ['cls', 'seg']:
                    # 构建checkpoint路径
                    task_upper = task.upper()
                    ckpt_file = checkpoint_dir / f'{task_upper}-Seed_111-{class_name}-check_point.pt'
                    
                    if not ckpt_file.exists():
                        print(f"⚠️ 缺失: {ckpt_file}")
                        pbar.update(1)
                        continue
                    
                    try:
                        # 创建模型
                        model = PromptAD(
                            device=device,
                            dataset=dataset_name,
                            class_name=class_name,
                            k_shot=k_shot,
                            backbone=args.backbone,
                            pretrained_dataset=args.pretrained_dataset,
                            out_size_h=args.resolution,
                            out_size_w=args.resolution,
                            n_ctx=args.n_ctx,
                            n_ctx_ab=args.n_ctx_ab,
                            n_pro=args.n_pro,
                            n_pro_ab=args.n_pro_ab,
                        )
                        model = model.to(device)
                        
                        # 加载权重
                        state_dict = torch.load(ckpt_file, map_location=device)
                        model.load_state_dict(state_dict, strict=False)
                        model.eval()
                        
                        # 获取数据加载器
                        dataloader, _ = get_dataloader_from_args(
                            phase='test',
                            dataset=dataset_name,
                            class_name=class_name,
                            k_shot=k_shot,
                            batch_size=args.batch_size,
                            img_resize=args.img_resize,
                            img_cropsize=args.img_cropsize,
                            perturbed=False,
                            transform=model.transform
                        )
                        
                        # 评估semantic分支
                        semantic_auroc = evaluate_semantic_branch(model, dataloader, task, device)
                        
                        results.append({
                            'dataset': dataset_name,
                            'class': class_name,
                            'k_shot': k_shot,
                            'task': task,
                            'semantic_auroc': round(semantic_auroc, 2)
                        })
                        
                        pbar.set_postfix({
                            'current': f"{dataset_name}/{class_name}/k{k_shot}/{task}",
                            'auroc': f"{semantic_auroc:.2f}"
                        })
                        
                        # 释放显存
                        del model
                        torch.cuda.empty_cache()
                        
                    except Exception as e:
                        print(f"\n❌ 错误: {dataset_name}/{class_name}/k{k_shot}/{task}: {e}")
                    
                    pbar.update(1)
    
    pbar.close()
    
    # 保存结果
    df = pd.DataFrame(results)
    output_file = 'result/baseline_semantic_branch_auroc.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ 结果已保存到: {output_file}")
    print(f"总计: {len(results)}/{total_tasks} 任务完成")
    
    # 打印统计信息
    print("\n" + "=" * 80)
    print("📊 Baseline Semantic分支统计")
    print("=" * 80)
    
    for dataset_name in ['mvtec', 'visa']:
        for task in ['cls', 'seg']:
            subset = df[(df['dataset'] == dataset_name) & (df['task'] == task)]
            if len(subset) > 0:
                avg_auroc = subset['semantic_auroc'].mean()
                print(f"{dataset_name.upper()} {task.upper()}: 平均 {avg_auroc:.2f}% ({len(subset)}个类别)")


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
