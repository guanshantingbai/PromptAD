"""调试训练过程，看看为什么结果是50%"""
import sys
import torch
import argparse
from train_cls import main

# 修改train_cls以添加更多输出
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Debug training')
    parser.add_argument('--dataset', type=str, default='mvtec')
    parser.add_argument('--class_name', type=str, default='screw')
    parser.add_argument('--k-shot', type=int, default=2)
    parser.add_argument('--n_pro', type=int, default=3)
    parser.add_argument('--n_pro_ab', type=int, default=4)
    parser.add_argument('--Epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--root-dir', type=str, default='result/debug_train')
    parser.add_argument('--vis', type=str, default='False')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--resolution', type=int, default=240)
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--use_cpu', type=int, default=0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--lambda1', type=float, default=0.1)
    parser.add_argument('--image_size', type=int, nargs='+', default=[240, 240])
    
    args = parser.parse_args()
    args.vis = args.vis == 'True'
    
    print(f"\n=== Training {args.class_name} with k={args.k_shot} ===\n")
    main(args)
