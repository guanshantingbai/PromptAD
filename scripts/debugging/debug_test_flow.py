"""详细调试测试流程"""
import torch
from datasets.dataset import get_dataloader
from PromptAD.model import PromptAD
from utils.metrics import metric_cal_img
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

# 设置
class_name = "screw"
k_shot = 2
device = 'cuda:0'

print("=" * 80)
print(f"详细调试测试流程: {class_name} k={k_shot}")
print("=" * 80)

# 1. 创建模型
kwargs = {
    'dataset': 'mvtec',
    'class_name': class_name,
    'k_shot': k_shot,
    'n_pro': 3,
    'n_pro_ab': 4,
    'resolution': 240,
    'out_size_h': 240,
    'out_size_w': 240,
    'seed': 111,
    'backbone': 'ViT-B-16-plus-240',
    'pretrained_dataset': 'laion400m_e32',
    'n_ctx': 4,
    'n_ctx_ab': 4,
    'device': device,
    'root': './datasets',
    'image_size': (240, 240),
    'batch_size': 1,
}

print("\n1. 初始化模型...")
model = PromptAD(**kwargs)
model = model.to(device)
model.eval_mode()

# 2. 加载checkpoint
ckpt_path = f'result/prompt1_fixed/mvtec/k_{k_shot}/checkpoint/CLS-Seed_111-{class_name}-check_point.pt'
print(f"\n2. 加载checkpoint: {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location='cpu')

print(f"   Normal prototypes shape: {checkpoint['normal_prototypes'].shape}")
print(f"   Abnormal prototypes shape: {checkpoint['abnormal_prototypes'].shape}")

# 直接加载到模型
model.normal_prototypes = checkpoint['normal_prototypes'].clone().half().to(device)
model.abnormal_prototypes = checkpoint['abnormal_prototypes'].clone().half().to(device)

print(f"   已加载到模型: normal={model.normal_prototypes.shape}, abnormal={model.abnormal_prototypes.shape}")

# 3. 构建memory bank
print(f"\n3. 构建memory bank...")
train_loader = get_dataloader('mvtec', class_name, k_shot, (240, 240), 'train', False, transform=model.transform)
with torch.no_grad():
    for (data, _, _, _, _) in tqdm(train_loader, desc="Building memory bank"):
        data = data.to(device)
        model.build_image_feature_gallery(data)
print(f"   Memory bank大小: {model.feature_gallery1.shape[0] if hasattr(model, 'feature_gallery1') and model.feature_gallery1 is not None else 0}")

# 4. 测试
print(f"\n4. 运行测试...")
test_loader = get_dataloader('mvtec', class_name, k_shot, (240, 240), 'test', False, transform=model.transform)

scores_img = []
score_maps = []
gt_list = []
gt_mask_list = []

with torch.no_grad():
    for (data, mask, label, name, img_type) in tqdm(test_loader, desc="Testing"):
        for l, m in zip(label, mask):
            gt_list.append(l.item() if torch.is_tensor(l) else l)
            m_np = m.numpy() if torch.is_tensor(m) else m
            m_np[m_np > 0] = 1
            gt_mask_list.append(m_np)
        
        data = data.to(device)
        
        # 纯语义评估
        visual_features = model.encode_image(data)
        textual_anomaly = model.calculate_textual_anomaly_score(visual_features, 'cls')
        textual_anomaly_map = model.calculate_textual_anomaly_score(visual_features, 'seg')
        textual_anomaly_map = textual_anomaly_map.detach().cpu().numpy()
        
        scores_img += textual_anomaly.tolist()
        for i in range(textual_anomaly_map.shape[0]):
            score_maps.append(textual_anomaly_map[i, 0])

# 5. Resize
print(f"\n5. Resize score maps...")
gt_mask_list = [cv2.resize(mask, (240, 240), interpolation=cv2.INTER_NEAREST) for mask in gt_mask_list]
score_maps = [cv2.resize(s, (240, 240), interpolation=cv2.INTER_CUBIC) if s.shape != (240, 240) else s for s in score_maps]

# 6. 计算指标
print(f"\n6. 计算指标...")
print(f"   scores_img数量: {len(scores_img)}, 样本: {scores_img[:5]}")
print(f"   gt_list数量: {len(gt_list)}, 样本: {gt_list[:5]}")
print(f"   score_maps数量: {len(score_maps)}, shape: {score_maps[0].shape if len(score_maps) > 0 else 'N/A'}")

result_dict = metric_cal_img(np.array(scores_img), gt_list, np.array(score_maps))

print(f"\n" + "=" * 80)
print(f"结果:")
print(f"=" * 80)
print(f"Image-AUROC: {result_dict['i_roc']:.2f}%")
print(f"Pixel-AUROC: {result_dict['p_roc']:.2f}%")
print(f"PRO: {result_dict['p_pro']:.2f}%")
