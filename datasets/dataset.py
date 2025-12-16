import os

import cv2
import numpy as np
from torch.utils.data import Dataset


class CLIPDataset(Dataset):
    def __init__(self, load_function, category, phase, k_shot, transform=None):

        self.load_function = load_function
        self.phase = phase
        self.transform = transform

        self.category = category

        # load datasets
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset(k_shot)  # self.labels => good : 0, anomaly : 1

    def load_dataset(self, k_shot):

        (train_img_tot_paths, train_gt_tot_paths, train_tot_labels, train_tot_types), \
        (test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types) = self.load_function(self.category,
                                                                                                      k_shot)
        if self.phase == 'train':

            return train_img_tot_paths, \
                   train_gt_tot_paths, \
                   train_tot_labels, \
                   train_tot_types
        else:
            return test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if gt == 0:
            gt = np.zeros([img.shape[0], img.shape[0]])
        else:
            gt = cv2.imread(gt, cv2.IMREAD_GRAYSCALE)
            gt[gt > 0] = 255

        img = cv2.resize(img, (256, 256))
        gt = cv2.resize(gt, (256, 256), interpolation=cv2.INTER_NEAREST)

        # 如果提供了 transform，在这里应用（避免训练循环中重复转换）
        if self.transform is not None:
            import torch
            from PIL import Image
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = self.transform(img)
            # 将 gt 也转换为 Tensor，保持类型一致
            gt = torch.from_numpy(gt).long()
            label = torch.tensor(label, dtype=torch.long)

        img_name = f'{self.category}-{img_type}-{os.path.basename(img_path[:-4])}'

        return img, gt, label, img_name, img_type
