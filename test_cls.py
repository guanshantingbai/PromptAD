import argparse

import torch.optim.lr_scheduler

from datasets import *
from datasets import dataset_classes
from utils.csv_utils import *
from utils.metrics import *
from utils.training_utils import *
from PromptAD import *
from utils.eval_utils import *
from torchvision import transforms
from tqdm import tqdm

TASK = 'CLS'

def test(model,
        args,
        dataloader: DataLoader,
        train_dataloader: DataLoader,  # Add train dataloader to build memory bank
        device: str,
        img_dir: str,
        check_path: str,
        ):

    # change the model into eval mode
    model.eval_mode()

    # Load checkpoint
    checkpoint = torch.load(check_path)
    
    # Handle prototype size mismatch - resize buffers to match checkpoint
    if 'normal_prototypes' in checkpoint:
        model.normal_prototypes = torch.zeros_like(checkpoint['normal_prototypes'])
        if model.precision == 'fp16':
            model.normal_prototypes = model.normal_prototypes.half()
    
    if 'abnormal_prototypes' in checkpoint:
        model.abnormal_prototypes = torch.zeros_like(checkpoint['abnormal_prototypes'])
        if model.precision == 'fp16':
            model.abnormal_prototypes = model.abnormal_prototypes.half()
    
    # Handle memory bank size mismatch - will be rebuilt from training data
    if 'feature_gallery1' in checkpoint:
        model.feature_gallery1 = torch.zeros_like(checkpoint['feature_gallery1'])
        if model.precision == 'fp16':
            model.feature_gallery1 = model.feature_gallery1.half()
    
    if 'feature_gallery2' in checkpoint:
        model.feature_gallery2 = torch.zeros_like(checkpoint['feature_gallery2'])
        if model.precision == 'fp16':
            model.feature_gallery2 = model.feature_gallery2.half()
    
    model.load_state_dict(checkpoint, strict=False)
    
    # Build memory bank from training data if not in checkpoint
    if 'feature_gallery1' not in checkpoint or checkpoint['feature_gallery1'].sum() == 0:
        print("Building memory bank from training data...")
        with torch.no_grad():
            for (data, _, _, _, _) in tqdm(train_dataloader, desc="Building memory bank"):
                data = [model.transform(Image.fromarray(f.numpy())) for f in data]
                data = torch.stack(data, dim=0).to(device)
                model.build_image_feature_gallery(data)
        print(f"Memory bank built: {model.feature_gallery1.shape[0]} samples")

    scores_img = []
    score_maps = []
    test_imgs = []
    gt_list = []
    gt_mask_list = []
    names = []

    for (data, mask, label, name, img_type) in dataloader:

        data = [model.transform(Image.fromarray(f.numpy())) for f in data]
        data = torch.stack(data, dim=0)

        for d, n, l, m in zip(data, name, label, mask):
            test_imgs += [denormalization(d.cpu().numpy())]
            l = l.numpy()
            m = m.numpy()
            m[m > 0] = 1

            names += [n]
            gt_list += [l]
            gt_mask_list += [m]

        data = data.to(device)
        score_img, score_map = model(data, 'cls')
        score_maps += score_map
        scores_img += score_img

    test_imgs, score_maps, gt_mask_list = specify_resolution(test_imgs, score_maps, gt_mask_list,
                                                             resolution=(args.resolution, args.resolution))
    result_dict = metric_cal_img(np.array(scores_img), gt_list, np.array(score_maps))

    return result_dict


def main(args):
    kwargs = vars(args)

    if kwargs['seed'] is None:
        kwargs['seed'] = 222

    setup_seed(kwargs['seed'])

    if kwargs['use_cpu'] == 0:
        device = f"cuda:0"
    else:
        device = f"cpu"
    kwargs['device'] = device

    # prepare the experiment dir
    img_dir, csv_path, check_path = get_dir_from_args(TASK, **kwargs)

    # get the test dataloader
    test_dataloader, test_dataset_inst = get_dataloader_from_args(phase='test', perturbed=False, **kwargs)
    
    # get the train dataloader for building memory bank
    train_dataloader, train_dataset_inst = get_dataloader_from_args(phase='train', perturbed=False, **kwargs)

    kwargs['out_size_h'] = kwargs['resolution']
    kwargs['out_size_w'] = kwargs['resolution']

    # get the model
    model = PromptAD(**kwargs)
    model = model.to(device)

    # as the pro metric calculation is costly, we only calculate it in the last evaluation
    metrics = test(model, args, test_dataloader, train_dataloader, device, img_dir=img_dir, check_path=check_path)

    p_roc = round(metrics['i_roc'], 2)
    object = kwargs['class_name']
    print(f'Object:{object} =========================== Pixel-AUROC:{p_roc}\n')

    save_metric(metrics, dataset_classes[kwargs['dataset']], kwargs['class_name'],
                kwargs['dataset'], csv_path)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_args():
    parser = argparse.ArgumentParser(description='Anomaly detection')
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa'])
    parser.add_argument('--class_name', type=str, default='carpet')

    parser.add_argument('--img-resize', type=int, default=240)
    parser.add_argument('--img-cropsize', type=int, default=240)
    parser.add_argument('--resolution', type=int, default=400)

    parser.add_argument('--batch-size', type=int, default=400)
    parser.add_argument('--vis', type=str2bool, choices=[True, False], default=True)
    parser.add_argument("--root-dir", type=str, default="./result")
    parser.add_argument("--load-memory", type=str2bool, default=True)
    parser.add_argument("--cal-pro", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--gpu-id", type=int, default=0)

    # pure test
    parser.add_argument("--pure-test", type=str2bool, default=False)

    # method related parameters
    parser.add_argument('--k-shot', type=int, default=1)
    parser.add_argument("--backbone", type=str, default="ViT-B-16-plus-240",
                        choices=['ViT-B-16-plus-240', 'ViT-B-16'])
    parser.add_argument("--pretrained_dataset", type=str, default="laion400m_e32")
    parser.add_argument("--version", type=str, default='')

    parser.add_argument("--use-cpu", type=int, default=0)

    # prompt tuning hyper-parameter
    parser.add_argument("--n_ctx", type=int, default=4)
    parser.add_argument("--n_ctx_ab", type=int, default=1)
    parser.add_argument("--n_pro", type=int, default=1)
    parser.add_argument("--n_pro_ab", type=int, default=4)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    import os

    args = get_args()
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu_id}"
    main(args)
