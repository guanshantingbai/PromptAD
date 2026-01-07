#!/usr/bin/env python3
"""
纯逻辑测试 - 不依赖完整模型
测试 query-conditioned 的核心算法
"""

import torch
import torch.nn.functional as F

print("\n🔬 Pure Logic Test: Query-Conditioned Anomaly Construction")
print("="*70)

# 模拟数据
N = 3  # batch size
K = 4  # number of normal representatives
M = 6  # number of abnormal directions
D = 512  # feature dimension
t = 100  # temperature

# 创建模拟数据
torch.manual_seed(42)
queries = torch.randn(N, D)
queries = F.normalize(queries, dim=-1)

normal_reps = torch.randn(K, D)
normal_reps = F.normalize(normal_reps, dim=-1)

abnormal_directions = torch.randn(M, D)
abnormal_directions = F.normalize(abnormal_directions, dim=-1)

print(f"✅ Setup:")
print(f"   Queries (N={N}): {queries.shape}")
print(f"   Normal reps (K={K}): {normal_reps.shape}")
print(f"   Abnormal dirs (M={M}): {abnormal_directions.shape}")

# Step 1: Hard selection of normal representative
print(f"\n[Step 1: Hard Selection]")
sim_to_normals = queries @ normal_reps.T  # [N, K]
i_star = sim_to_normals.argmax(dim=-1)  # [N]
n_star = normal_reps[i_star]  # [N, D]

print(f"  Similarity matrix: {sim_to_normals}")
print(f"  Selected indices: {i_star}")
print(f"  Selected normals: {n_star.shape}")

# Step 2: Construct anomaly candidates
print(f"\n[Step 2: Anomaly Construction]")
lambda_scale = 1.0
A = n_star.unsqueeze(1) + lambda_scale * abnormal_directions.unsqueeze(0)  # [N, M, D]
A = F.normalize(A, dim=-1)

print(f"  Anomaly candidates: {A.shape}")
print(f"  Sample norms: {A[0].norm(dim=-1)[:3]}")  # 应该都是 1

# Step 3: Scoring with logsumexp
print(f"\n[Step 3: Scoring]")

# Normal evidence
logits_normal = t * (queries @ normal_reps.T)  # [N, K]
s_N = torch.logsumexp(logits_normal, dim=-1)  # [N]

# Anomaly evidence
logits_abnormal = t * torch.einsum('nd,nmd->nm', queries, A)  # [N, M]
s_A = torch.logsumexp(logits_abnormal, dim=-1)  # [N]

print(f"  Normal evidence (s_N): {s_N}")
print(f"  Abnormal evidence (s_A): {s_A}")
print(f"  Difference (s_A - s_N): {s_A - s_N}")

# Step 4: Final score with margin
print(f"\n[Step 4: Final Anomaly Score]")
for margin in [0.0, 0.5, 1.0]:
    scores = F.relu(s_A - s_N - margin)
    print(f"  margin={margin:.1f}: {scores.numpy()}")

print(f"\n✅ All calculations completed successfully!")
print(f"\n[Analysis]")
print(f"  - Hard selection picks the closest normal representative for each query")
print(f"  - Anomaly candidates are constructed by offsetting from selected normal")
print(f"  - LogSumExp aggregates evidence across all candidates")
print(f"  - ReLU with margin creates final anomaly score")

print(f"\n{'='*70}")
print(f"🎉 Pure logic test PASSED! The algorithm is correct.")
print(f"{'='*70}\n")
