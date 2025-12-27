"""简单对比：检查两个checkpoint是否相同（说明没训练）"""
import torch
import numpy as np

class_name = "screw"
k_shot = 2

# 加载两个不同k值的checkpoint，看它们是否相似（不该相似）
ckpt1 = torch.load(f'result/prompt1_fixed/mvtec/k_1/checkpoint/CLS-Seed_111-{class_name}-check_point.pt', map_location='cpu')
ckpt2 = torch.load(f'result/prompt1_fixed/mvtec/k_2/checkpoint/CLS-Seed_111-{class_name}-check_point.pt', map_location='cpu')

normal1 = ckpt1['normal_prototypes'].float()
normal2 = ckpt2['normal_prototypes'].float()

abnormal1 = ckpt1['abnormal_prototypes'].float()
abnormal2 = ckpt2['abnormal_prototypes'].float()

# 计算差异
normal_diff = torch.norm(normal1 - normal2, dim=-1).mean()
abnormal_diff = torch.norm(abnormal1 - abnormal2, dim=-1).mean()

print("=" * 80)
print(f"对比 {class_name} k=1 vs k=2 的checkpoints")
print("=" * 80)
print(f"\nNormal prototypes L2距离: {normal_diff:.6f}")
print(f"Abnormal prototypes L2距离: {abnormal_diff:.6f}")

# 检查是否完全相同
if torch.allclose(normal1, normal2, atol=1e-6):
    print(f"\n❌ 严重问题：k=1和k=2的normal prototypes完全相同！")
    print(f"   这说明训练根本没有学习，或者都使用了相同的初始化")
else:
    print(f"\n✅ k=1和k=2的normal prototypes不同，说明训练有效果")

if torch.allclose(abnormal1, abnormal2, atol=1e-6):
    print(f"❌ 严重问题：k=1和k=2的abnormal prototypes完全相同！")
else:
    print(f"✅ k=1和k=2的abnormal prototypes不同")

# 检查prototypes的多样性
print(f"\nk=1 Normal prototypes标准差: {normal1.std(dim=0).mean():.6f}")
print(f"k=2 Normal prototypes标准差: {normal2.std(dim=0).mean():.6f}")
print(f"k=1 Abnormal prototypes标准差: {abnormal1.std(dim=0).mean():.6f}")
print(f"k=2 Abnormal prototypes标准差: {abnormal2.std(dim=0).mean():.6f}")

# 打印具体数值
print(f"\nk=1 Normal prototypes[0, :10]:")
print(normal1[0, :10])
print(f"\nk=2 Normal prototypes[0, :10]:")
print(normal2[0, :10])
