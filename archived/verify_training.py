"""验证训练是否真正学到了东西"""
import torch
import sys
import os

# 检查训练前后的prototypes变化
class_name = "screw"
k_shot = 2

# 1. 初始化一个模型看prototypes的初始值
print("=" * 80)
print("验证训练是否改变了prototypes")
print("=" * 80)

# 模拟初始化
from PromptAD.model import PromptAD
import argparse

kwargs = {
    'dataset': 'mvtec',
    'class_name': class_name,
    'k_shot': k_shot,
    'n_pro': 3,
    'n_pro_ab': 4,
    'resolution': 240,
    'out_size_h': 240,
    'out_size_w': 240,
    'device': 'cuda:0',
    'seed': 111,
    'backbone': 'ViT-B-16-plus-240',
    'pretrained_dataset': 'laion400m_e32',
    'n_ctx': 4,
    'n_ctx_ab': 4,
}

print(f"\n初始化模型 {class_name} (k={k_shot})...")
model_init = PromptAD(**kwargs)
model_init = model_init.to('cuda:0')
model_init.eval_mode()

# 获取初始prototypes
with torch.no_grad():
    normal_text_prompt, abnormal_text_prompt_handle, abnormal_text_prompt_learned = model_init.prompt_learner()
    initial_normal = model_init.encode_text_embedding(normal_text_prompt, model_init.tokenized_normal_prompts)
    abnormal_handle = model_init.encode_text_embedding(abnormal_text_prompt_handle, model_init.tokenized_abnormal_prompts_handle)
    abnormal_learned = model_init.encode_text_embedding(abnormal_text_prompt_learned, model_init.tokenized_abnormal_prompts_learned)
    initial_abnormal = torch.cat([abnormal_handle, abnormal_learned], dim=0)
    
    # Normalize
    initial_normal = initial_normal / initial_normal.norm(dim=-1, keepdim=True)
    initial_abnormal = initial_abnormal / initial_abnormal.norm(dim=-1, keepdim=True)

print(f"初始 normal prototypes shape: {initial_normal.shape}")
print(f"初始 normal prototypes[0, :5]: {initial_normal[0, :5]}")
print(f"初始 abnormal prototypes shape: {initial_abnormal.shape}")
print(f"初始 abnormal prototypes[0, :5]: {initial_abnormal[0, :5]}")

# 2. 加载训练后的checkpoint
ckpt_path = f'result/prompt1_fixed/mvtec/k_{k_shot}/checkpoint/CLS-Seed_111-{class_name}-check_point.pt'
if not os.path.exists(ckpt_path):
    print(f"\n❌ Checkpoint不存在: {ckpt_path}")
    sys.exit(1)

checkpoint = torch.load(ckpt_path, map_location='cpu')
trained_normal = checkpoint['normal_prototypes']
trained_abnormal = checkpoint['abnormal_prototypes']

print(f"\n训练后 normal prototypes shape: {trained_normal.shape}")
print(f"训练后 normal prototypes[0, :5]: {trained_normal[0, :5]}")
print(f"训练后 abnormal prototypes shape: {trained_abnormal.shape}")
print(f"训练后 abnormal prototypes[0, :5]: {trained_abnormal[0, :5]}")

# 3. 比较差异
initial_normal_cpu = initial_normal.cpu().float()
initial_abnormal_cpu = initial_abnormal.cpu().float()
trained_normal_float = trained_normal.float()
trained_abnormal_float = trained_abnormal.float()

normal_diff = torch.norm(initial_normal_cpu - trained_normal_float, dim=-1).mean()
abnormal_diff = torch.norm(initial_abnormal_cpu - trained_abnormal_float, dim=-1).mean()

print(f"\n📊 变化分析:")
print(f"  Normal prototypes平均L2距离: {normal_diff:.6f}")
print(f"  Abnormal prototypes平均L2距离: {abnormal_diff:.6f}")

if normal_diff < 0.001 and abnormal_diff < 0.001:
    print(f"\n  ❌ 严重问题：prototypes几乎没有变化！训练可能失败了")
elif normal_diff < 0.01 and abnormal_diff < 0.01:
    print(f"\n  ⚠️  警告：prototypes变化很小，训练可能不充分")
else:
    print(f"\n  ✅ prototypes发生了明显变化，训练正常")

# 4. 检查是否所有prototypes都相同（未学习的标志）
normal_std = trained_normal_float.std(dim=0).mean()
abnormal_std = trained_abnormal_float.std(dim=0).mean()

print(f"\n  Normal prototypes标准差: {normal_std:.6f}")
print(f"  Abnormal prototypes标准差: {abnormal_std:.6f}")

if normal_std < 0.001:
    print(f"  ❌ Normal prototypes几乎相同！可能初始化有问题")
if abnormal_std < 0.001:
    print(f"  ❌ Abnormal prototypes几乎相同！可能初始化有问题")
