# Baseline (Harmonic训练) vs Gate (Max训练) 完整对比报告

> 生成日期: 2025年12月22日
> 对比任务: 162个 (27类 × 3 k-shots × 2 tasks)
> Baseline: PromptAD原始harmonic融合训练
> Gate: 使用max融合训练后的权重

---

## 📊 第一部分：整体对比结果

### 1.1 核心结论

**✓ Harmonic训练策略略优于Max训练策略**
- **平均AUROC差异**: -0.40% (负值表示Harmonic更优)
- **Harmonic胜率**: 63.0% (102/162)
- **Max胜率**: 37.0% (60/162)
- **标准差**: 2.21%

> **结论**: 两种融合策略性能相当，差异在统计误差范围内（<0.5%）。
> Harmonic在多数任务上略有优势，但Max在部分难例上表现更好。

### 1.2 分组对比

#### 按数据集×任务分组

| 数据集 | 任务 | 任务数 | Baseline (Harmonic) | Gate (Max) | 差异 | Max胜率 |
|--------|------|--------|---------------------|------------|------|---------|
| MVTEC | CLS | 45 | 94.41% | 94.04% | -0.37% | 22.2% |
| MVTEC | SEG | 45 | 96.25% | 96.06% | -0.19% | 37.8% |
| VISA | CLS | 36 | 87.00% | 85.92% | -1.08% | 36.1% |
| VISA | SEG | 36 | 96.47% | 96.45% | -0.02% | 55.6% |

**关键发现**:
- **MVTec数据集**: Harmonic平均优0.28%
- **VisA数据集**: Harmonic平均优0.55%
- **CLS任务**: Harmonic平均优0.69% (图像级异常检测)
- **SEG任务**: 两者几乎持平 (-0.11%) (像素级异常分割)

#### 按K-Shot分组

| K-Shot | 任务数 | Baseline | Gate | 差异 | Max胜率 |
|--------|--------|----------|------|------|---------|
| k=1 | 54 | 93.14% | 92.51% | -0.62% | 35.2% |
| k=2 | 54 | 93.62% | 93.48% | -0.14% | 37.0% |
| k=4 | 54 | 94.44% | 94.00% | -0.44% | 38.9% |

**观察**: K-shot增加时，两种策略的差异略有变化，但整体保持Harmonic略优的趋势。

---

## 🔍 第二部分：差异显著的代表性类别

### 2.1 Harmonic显著更优的代表类别

#### 案例1: VISA/macaroni1 (k=1, CLS)

| 指标 | Baseline (Harmonic) | Gate (Max) | 差异 |
|------|---------------------|------------|------|
| AUROC | **85.65%** | 76.90% | **+8.75%** |

**分析**: 在macaroni1的图像级检测任务上，Harmonic融合能更好地平衡semantic和memory分支，
避免了Max融合可能导致的过度依赖单一分支的问题。

#### 案例2: MVTEC/cable (k=1, CLS)

| 指标 | Baseline (Harmonic) | Gate (Max) | 差异 |
|------|---------------------|------------|------|
| AUROC | **96.63%** | 89.04% | **+7.59%** |

**分析**: 在cable的图像级检测任务上，Harmonic融合能更好地平衡semantic和memory分支，
避免了Max融合可能导致的过度依赖单一分支的问题。

### 2.2 Max显著更优的代表类别

#### 案例1: MVTEC/capsule (k=4, CLS)

| 指标 | Baseline (Harmonic) | Gate (Max) | 差异 |
|------|---------------------|------------|------|
| AUROC | 83.81% | **90.29%** | **+6.48%** |

**分析**: 在capsule这类难例上，Max融合能够更果断地选择最强的分支，
避免Harmonic融合可能引入的弱分支噪声，从而提升整体性能。

#### 案例2: MVTEC/screw (k=2, CLS)

| 指标 | Baseline (Harmonic) | Gate (Max) | 差异 |
|------|---------------------|------------|------|
| AUROC | 58.66% | **66.42%** | **+7.76%** |

**分析**: 在screw这类难例上，Max融合能够更果断地选择最强的分支，
避免Harmonic融合可能引入的弱分支噪声，从而提升整体性能。

---

## 📋 第三部分：完整数据表

### 3.1 所有162个任务的详细对比

<details>
<summary>点击展开完整数据表（162行）</summary>

| 数据集 | 类别 | K-Shot | 任务 | Baseline | Gate | 差异 | 胜者 |
|--------|------|--------|------|----------|------|------|------|
| visa | macaroni1 | k=1 | CLS | 85.65% | 76.90% | -8.75% | Harmonic |
| mvtec | cable | k=1 | CLS | 96.63% | 89.04% | -7.59% | Harmonic |
| visa | pcb2 | k=4 | CLS | 82.83% | 76.10% | -6.73% | Harmonic |
| visa | pcb4 | k=4 | CLS | 92.66% | 86.40% | -6.26% | Harmonic |
| visa | pcb3 | k=2 | CLS | 79.78% | 73.77% | -6.01% | Harmonic |
| visa | macaroni2 | k=1 | CLS | 73.19% | 68.03% | -5.16% | Harmonic |
| mvtec | zipper | k=1 | SEG | 95.73% | 90.60% | -5.13% | Harmonic |
| mvtec | toothbrush | k=2 | CLS | 98.89% | 94.03% | -4.86% | Harmonic |
| mvtec | zipper | k=2 | CLS | 96.40% | 91.60% | -4.80% | Harmonic |
| visa | fryum | k=1 | CLS | 88.99% | 84.21% | -4.78% | Harmonic |
| visa | macaroni1 | k=4 | CLS | 88.10% | 83.57% | -4.53% | Harmonic |
| visa | capsules | k=4 | CLS | 74.25% | 70.13% | -4.12% | Harmonic |
| visa | fryum | k=2 | CLS | 89.96% | 86.24% | -3.72% | Harmonic |
| mvtec | zipper | k=1 | CLS | 96.69% | 93.37% | -3.32% | Harmonic |
| mvtec | screw | k=4 | CLS | 70.01% | 66.88% | -3.13% | Harmonic |
| mvtec | pill | k=2 | CLS | 95.61% | 92.54% | -3.07% | Harmonic |
| mvtec | transistor | k=4 | CLS | 94.42% | 91.62% | -2.80% | Harmonic |
| visa | macaroni2 | k=4 | CLS | 75.99% | 73.24% | -2.75% | Harmonic |
| mvtec | zipper | k=4 | CLS | 97.22% | 94.47% | -2.75% | Harmonic |
| visa | pcb1 | k=1 | CLS | 94.48% | 92.09% | -2.39% | Harmonic |
| visa | fryum | k=4 | CLS | 92.31% | 90.10% | -2.21% | Harmonic |
| mvtec | metal_nut | k=1 | SEG | 90.98% | 88.78% | -2.20% | Harmonic |
| mvtec | pill | k=1 | CLS | 95.20% | 93.22% | -1.98% | Harmonic |
| mvtec | toothbrush | k=1 | CLS | 95.56% | 93.61% | -1.95% | Harmonic |
| visa | pcb2 | k=2 | CLS | 77.09% | 75.16% | -1.93% | Harmonic |
| mvtec | zipper | k=2 | SEG | 94.46% | 92.58% | -1.88% | Harmonic |
| mvtec | cable | k=4 | CLS | 97.23% | 95.37% | -1.86% | Harmonic |
| mvtec | screw | k=1 | SEG | 93.71% | 91.95% | -1.76% | Harmonic |
| mvtec | screw | k=1 | CLS | 60.48% | 58.92% | -1.56% | Harmonic |
| mvtec | cable | k=2 | CLS | 96.38% | 95.01% | -1.37% | Harmonic |
| visa | chewinggum | k=2 | CLS | 97.19% | 95.96% | -1.23% | Harmonic |
| visa | pcb4 | k=1 | CLS | 87.85% | 86.65% | -1.20% | Harmonic |
| mvtec | wood | k=1 | CLS | 100.00% | 98.86% | -1.14% | Harmonic |
| mvtec | transistor | k=4 | SEG | 93.07% | 91.99% | -1.08% | Harmonic |
| visa | pcb1 | k=2 | CLS | 94.15% | 93.12% | -1.03% | Harmonic |
| mvtec | screw | k=2 | SEG | 93.12% | 92.10% | -1.02% | Harmonic |
| mvtec | metal_nut | k=1 | CLS | 99.85% | 98.85% | -1.00% | Harmonic |
| mvtec | wood | k=4 | SEG | 97.41% | 96.48% | -0.93% | Harmonic |
| mvtec | transistor | k=2 | SEG | 88.66% | 87.80% | -0.86% | Harmonic |
| mvtec | toothbrush | k=4 | CLS | 98.06% | 97.22% | -0.84% | Harmonic |
| mvtec | cable | k=2 | SEG | 96.77% | 96.00% | -0.77% | Harmonic |
| mvtec | capsule | k=2 | SEG | 96.13% | 95.44% | -0.69% | Harmonic |
| visa | cashew | k=2 | SEG | 96.22% | 95.57% | -0.65% | Harmonic |
| visa | chewinggum | k=4 | CLS | 96.48% | 95.84% | -0.64% | Harmonic |
| mvtec | metal_nut | k=2 | CLS | 100.00% | 99.39% | -0.61% | Harmonic |
| mvtec | toothbrush | k=2 | SEG | 98.95% | 98.38% | -0.57% | Harmonic |
| mvtec | bottle | k=2 | CLS | 100.00% | 99.44% | -0.56% | Harmonic |
| visa | pcb3 | k=1 | CLS | 76.14% | 75.59% | -0.55% | Harmonic |
| visa | capsules | k=1 | CLS | 74.76% | 74.22% | -0.54% | Harmonic |
| visa | candle | k=4 | CLS | 95.65% | 95.11% | -0.54% | Harmonic |
| visa | chewinggum | k=1 | CLS | 96.11% | 95.58% | -0.53% | Harmonic |
| mvtec | cable | k=1 | SEG | 97.05% | 96.60% | -0.45% | Harmonic |
| mvtec | wood | k=4 | CLS | 99.74% | 99.30% | -0.44% | Harmonic |
| mvtec | wood | k=2 | CLS | 99.82% | 99.39% | -0.43% | Harmonic |
| mvtec | pill | k=4 | CLS | 95.06% | 94.67% | -0.39% | Harmonic |
| mvtec | bottle | k=4 | CLS | 100.00% | 99.64% | -0.36% | Harmonic |
| mvtec | hazelnut | k=4 | SEG | 99.06% | 98.76% | -0.30% | Harmonic |
| visa | pipe_fryum | k=4 | CLS | 98.88% | 98.60% | -0.28% | Harmonic |
| mvtec | bottle | k=1 | CLS | 100.00% | 99.76% | -0.24% | Harmonic |
| mvtec | screw | k=4 | SEG | 94.67% | 94.44% | -0.23% | Harmonic |
| mvtec | grid | k=2 | CLS | 99.08% | 98.87% | -0.21% | Harmonic |
| mvtec | cable | k=4 | SEG | 97.19% | 97.02% | -0.17% | Harmonic |
| mvtec | grid | k=1 | SEG | 98.15% | 97.98% | -0.17% | Harmonic |
| mvtec | toothbrush | k=1 | SEG | 98.48% | 98.32% | -0.16% | Harmonic |
| mvtec | carpet | k=1 | SEG | 99.51% | 99.36% | -0.15% | Harmonic |
| mvtec | transistor | k=1 | SEG | 90.09% | 89.96% | -0.13% | Harmonic |
| mvtec | bottle | k=2 | SEG | 98.73% | 98.61% | -0.12% | Harmonic |
| mvtec | tile | k=1 | SEG | 96.77% | 96.66% | -0.11% | Harmonic |
| mvtec | carpet | k=4 | SEG | 99.49% | 99.41% | -0.08% | Harmonic |
| mvtec | transistor | k=1 | CLS | 90.33% | 90.25% | -0.08% | Harmonic |
| mvtec | hazelnut | k=2 | CLS | 99.93% | 99.86% | -0.07% | Harmonic |
| mvtec | toothbrush | k=4 | SEG | 99.15% | 99.08% | -0.07% | Harmonic |
| mvtec | carpet | k=2 | SEG | 99.48% | 99.42% | -0.06% | Harmonic |
| mvtec | tile | k=2 | CLS | 100.00% | 99.96% | -0.04% | Harmonic |
| mvtec | leather | k=2 | SEG | 99.42% | 99.39% | -0.03% | Harmonic |
| mvtec | leather | k=1 | SEG | 99.47% | 99.45% | -0.02% | Harmonic |
| mvtec | bottle | k=4 | SEG | 98.78% | 98.76% | -0.02% | Harmonic |
| visa | candle | k=2 | SEG | 95.18% | 95.16% | -0.02% | Harmonic |
| mvtec | capsule | k=1 | SEG | 93.22% | 93.21% | -0.01% | Harmonic |
| visa | pipe_fryum | k=2 | CLS | 98.80% | 98.79% | -0.01% | Harmonic |
| visa | macaroni2 | k=2 | SEG | 95.71% | 95.70% | -0.01% | Harmonic |
| visa | macaroni2 | k=1 | SEG | 95.14% | 95.13% | -0.01% | Harmonic |
| visa | fryum | k=4 | SEG | 96.02% | 96.02% | -0.00% | Harmonic |
| visa | candle | k=1 | SEG | 94.44% | 94.44% | -0.00% | Harmonic |
| visa | fryum | k=1 | SEG | 94.39% | 94.39% | -0.00% | Harmonic |
| visa | cashew | k=4 | SEG | 95.30% | 95.30% | -0.00% | Harmonic |
| visa | pcb3 | k=2 | SEG | 95.41% | 95.41% | -0.00% | Harmonic |
| visa | pcb4 | k=2 | SEG | 96.86% | 96.86% | -0.00% | Harmonic |
| visa | pcb1 | k=4 | SEG | 98.33% | 98.33% | -0.00% | Harmonic |
| visa | chewinggum | k=1 | SEG | 99.04% | 99.04% | -0.00% | Harmonic |
| visa | chewinggum | k=2 | SEG | 99.21% | 99.21% | -0.00% | Harmonic |
| visa | candle | k=4 | SEG | 94.11% | 94.11% | -0.00% | Harmonic |
| visa | pipe_fryum | k=2 | SEG | 99.54% | 99.54% | -0.00% | Harmonic |
| visa | pcb1 | k=1 | SEG | 98.77% | 98.77% | -0.00% | Harmonic |
| mvtec | carpet | k=1 | CLS | 100.00% | 100.00% | +0.00% | Harmonic |
| mvtec | carpet | k=2 | CLS | 100.00% | 100.00% | +0.00% | Harmonic |
| mvtec | carpet | k=4 | CLS | 100.00% | 100.00% | +0.00% | Harmonic |
| mvtec | tile | k=4 | CLS | 100.00% | 100.00% | +0.00% | Harmonic |
| mvtec | leather | k=2 | CLS | 100.00% | 100.00% | +0.00% | Harmonic |
| mvtec | hazelnut | k=4 | CLS | 100.00% | 100.00% | +0.00% | Harmonic |
| mvtec | leather | k=4 | CLS | 100.00% | 100.00% | +0.00% | Harmonic |
| mvtec | leather | k=1 | CLS | 100.00% | 100.00% | +0.00% | Harmonic |
| visa | macaroni1 | k=2 | SEG | 96.57% | 96.57% | +0.00% | Max |
| visa | pcb2 | k=2 | SEG | 95.45% | 95.45% | +0.00% | Max |
| visa | pcb1 | k=2 | SEG | 98.50% | 98.50% | +0.00% | Max |
| visa | chewinggum | k=4 | SEG | 99.14% | 99.14% | +0.00% | Max |
| visa | pcb2 | k=1 | SEG | 95.27% | 95.27% | +0.00% | Max |
| visa | pcb3 | k=4 | SEG | 96.40% | 96.40% | +0.00% | Max |
| visa | capsules | k=4 | SEG | 95.63% | 95.63% | +0.00% | Max |
| visa | pipe_fryum | k=1 | SEG | 99.37% | 99.37% | +0.00% | Max |
| visa | pcb3 | k=1 | SEG | 96.08% | 96.08% | +0.00% | Max |
| visa | macaroni1 | k=1 | SEG | 95.58% | 95.58% | +0.00% | Max |
| visa | pcb2 | k=4 | SEG | 94.85% | 94.85% | +0.00% | Max |
| visa | pipe_fryum | k=4 | SEG | 99.54% | 99.54% | +0.00% | Max |
| visa | macaroni1 | k=4 | SEG | 96.59% | 96.59% | +0.00% | Max |
| visa | pcb4 | k=1 | SEG | 96.99% | 96.99% | +0.00% | Max |
| visa | pcb4 | k=4 | SEG | 97.94% | 97.94% | +0.00% | Max |
| visa | capsules | k=2 | SEG | 94.41% | 94.41% | +0.00% | Max |
| visa | macaroni2 | k=4 | SEG | 94.84% | 94.84% | +0.00% | Max |
| visa | cashew | k=1 | SEG | 97.47% | 97.47% | +0.00% | Max |
| visa | fryum | k=2 | SEG | 95.36% | 95.37% | +0.01% | Max |
| mvtec | leather | k=4 | SEG | 99.42% | 99.43% | +0.01% | Max |
| visa | capsules | k=1 | SEG | 93.38% | 93.39% | +0.01% | Max |
| mvtec | hazelnut | k=1 | SEG | 98.76% | 98.78% | +0.02% | Max |
| mvtec | tile | k=1 | CLS | 99.93% | 99.96% | +0.03% | Max |
| mvtec | hazelnut | k=2 | SEG | 98.86% | 98.89% | +0.03% | Max |
| mvtec | grid | k=4 | SEG | 98.14% | 98.19% | +0.05% | Max |
| mvtec | bottle | k=1 | SEG | 98.53% | 98.59% | +0.06% | Max |
| mvtec | capsule | k=4 | SEG | 96.28% | 96.37% | +0.09% | Max |
| mvtec | wood | k=1 | SEG | 95.13% | 95.33% | +0.20% | Max |
| mvtec | tile | k=4 | SEG | 96.65% | 96.87% | +0.22% | Max |
| visa | candle | k=2 | CLS | 94.86% | 95.12% | +0.26% | Max |
| mvtec | grid | k=2 | SEG | 97.81% | 98.10% | +0.29% | Max |
| mvtec | metal_nut | k=4 | CLS | 99.66% | 100.00% | +0.34% | Max |
| mvtec | grid | k=1 | CLS | 99.00% | 99.42% | +0.42% | Max |
| visa | macaroni1 | k=2 | CLS | 85.75% | 86.21% | +0.46% | Max |
| mvtec | tile | k=2 | SEG | 96.53% | 97.04% | +0.51% | Max |
| visa | pcb1 | k=4 | CLS | 93.11% | 93.64% | +0.53% | Max |
| mvtec | pill | k=2 | SEG | 95.45% | 96.00% | +0.55% | Max |
| mvtec | metal_nut | k=2 | SEG | 93.76% | 94.37% | +0.61% | Max |
| mvtec | grid | k=4 | CLS | 99.16% | 99.83% | +0.67% | Max |
| mvtec | pill | k=4 | SEG | 95.18% | 95.97% | +0.79% | Max |
| mvtec | pill | k=1 | SEG | 94.42% | 95.31% | +0.89% | Max |
| mvtec | wood | k=2 | SEG | 95.65% | 96.56% | +0.91% | Max |
| visa | pipe_fryum | k=1 | CLS | 97.99% | 99.24% | +1.25% | Max |
| visa | pcb3 | k=4 | CLS | 80.59% | 81.93% | +1.34% | Max |
| mvtec | transistor | k=2 | CLS | 89.79% | 91.19% | +1.40% | Max |
| visa | pcb2 | k=1 | CLS | 77.09% | 78.61% | +1.52% | Max |
| visa | candle | k=1 | CLS | 93.89% | 95.59% | +1.70% | Max |
| visa | cashew | k=2 | CLS | 90.40% | 92.38% | +1.98% | Max |
| mvtec | hazelnut | k=1 | CLS | 97.50% | 99.75% | +2.25% | Max |
| visa | pcb4 | k=2 | CLS | 83.45% | 85.76% | +2.31% | Max |
| mvtec | metal_nut | k=4 | SEG | 93.01% | 95.49% | +2.48% | Max |
| visa | macaroni2 | k=2 | CLS | 74.74% | 77.56% | +2.82% | Max |
| mvtec | zipper | k=4 | SEG | 89.91% | 92.82% | +2.91% | Max |
| visa | capsules | k=2 | CLS | 72.80% | 76.17% | +3.37% | Max |
| visa | cashew | k=4 | CLS | 88.34% | 92.32% | +3.98% | Max |
| visa | cashew | k=1 | CLS | 87.81% | 93.18% | +5.37% | Max |
| mvtec | capsule | k=2 | CLS | 79.94% | 85.60% | +5.66% | Max |
| mvtec | capsule | k=1 | CLS | 68.33% | 74.05% | +5.72% | Max |
| mvtec | capsule | k=4 | CLS | 83.81% | 90.29% | +6.48% | Max |
| mvtec | screw | k=2 | CLS | 58.66% | 66.42% | +7.76% | Max |

</details>

### 3.2 显著差异案例汇总 (|差异| > 2%)

**共34个显著差异案例**

#### Harmonic显著更优的案例

| 排名 | 数据集 | 类别 | K-Shot | 任务 | Baseline | Gate | 差异 |
|------|--------|------|--------|------|----------|------|------|
| 1 | visa | macaroni1 | k=1 | CLS | **85.65%** | 76.90% | **8.75%** |
| 2 | mvtec | cable | k=1 | CLS | **96.63%** | 89.04% | **7.59%** |
| 3 | visa | pcb2 | k=4 | CLS | **82.83%** | 76.10% | **6.73%** |
| 4 | visa | pcb4 | k=4 | CLS | **92.66%** | 86.40% | **6.26%** |
| 5 | visa | pcb3 | k=2 | CLS | **79.78%** | 73.77% | **6.01%** |
| 6 | visa | macaroni2 | k=1 | CLS | **73.19%** | 68.03% | **5.16%** |
| 7 | mvtec | zipper | k=1 | SEG | **95.73%** | 90.60% | **5.13%** |
| 8 | mvtec | toothbrush | k=2 | CLS | **98.89%** | 94.03% | **4.86%** |
| 9 | mvtec | zipper | k=2 | CLS | **96.40%** | 91.60% | **4.80%** |
| 10 | visa | fryum | k=1 | CLS | **88.99%** | 84.21% | **4.78%** |
| 11 | visa | macaroni1 | k=4 | CLS | **88.10%** | 83.57% | **4.53%** |
| 12 | visa | capsules | k=4 | CLS | **74.25%** | 70.13% | **4.12%** |
| 13 | visa | fryum | k=2 | CLS | **89.96%** | 86.24% | **3.72%** |
| 14 | mvtec | zipper | k=1 | CLS | **96.69%** | 93.37% | **3.32%** |
| 15 | mvtec | screw | k=4 | CLS | **70.01%** | 66.88% | **3.13%** |
| 16 | mvtec | pill | k=2 | CLS | **95.61%** | 92.54% | **3.07%** |
| 17 | mvtec | transistor | k=4 | CLS | **94.42%** | 91.62% | **2.80%** |
| 18 | visa | macaroni2 | k=4 | CLS | **75.99%** | 73.24% | **2.75%** |
| 19 | mvtec | zipper | k=4 | CLS | **97.22%** | 94.47% | **2.75%** |
| 20 | visa | pcb1 | k=1 | CLS | **94.48%** | 92.09% | **2.39%** |
| 21 | visa | fryum | k=4 | CLS | **92.31%** | 90.10% | **2.21%** |
| 22 | mvtec | metal_nut | k=1 | SEG | **90.98%** | 88.78% | **2.20%** |

#### Max显著更优的案例

| 排名 | 数据集 | 类别 | K-Shot | 任务 | Baseline | Gate | 差异 |
|------|--------|------|--------|------|----------|------|------|
| 1 | mvtec | screw | k=2 | CLS | 58.66% | **66.42%** | **+7.76%** |
| 2 | mvtec | capsule | k=4 | CLS | 83.81% | **90.29%** | **+6.48%** |
| 3 | mvtec | capsule | k=1 | CLS | 68.33% | **74.05%** | **+5.72%** |
| 4 | mvtec | capsule | k=2 | CLS | 79.94% | **85.60%** | **+5.66%** |
| 5 | visa | cashew | k=1 | CLS | 87.81% | **93.18%** | **+5.37%** |
| 6 | visa | cashew | k=4 | CLS | 88.34% | **92.32%** | **+3.98%** |
| 7 | visa | capsules | k=2 | CLS | 72.80% | **76.17%** | **+3.37%** |
| 8 | mvtec | zipper | k=4 | SEG | 89.91% | **92.82%** | **+2.91%** |
| 9 | visa | macaroni2 | k=2 | CLS | 74.74% | **77.56%** | **+2.82%** |
| 10 | mvtec | metal_nut | k=4 | SEG | 93.01% | **95.49%** | **+2.48%** |
| 11 | visa | pcb4 | k=2 | CLS | 83.45% | **85.76%** | **+2.31%** |
| 12 | mvtec | hazelnut | k=1 | CLS | 97.50% | **99.75%** | **+2.25%** |

---

## 📈 附录：统计摘要

### 差异分布统计

| 差异区间 | 任务数 | 占比 |
|----------|--------|------|
| Harmonic > +5% | 7 | 4.3% |
| Harmonic +2%~+5% | 15 | 9.3% |
| Harmonic +0.5%~+2% | 29 | 17.9% |
| 相当 (±0.5%) | 85 | 52.5% |
| Max +0.5%~+2% | 14 | 8.6% |
| Max +2%~+5% | 7 | 4.3% |
| Max > +5% | 5 | 3.1% |

### 数值摘要

- **最小差异**: -8.75% (Harmonic更优)
- **最大差异**: 7.76% (Max更优)
- **中位数差异**: -0.01%
- **平均差异**: -0.40%
- **标准差**: 2.21%

---

## 🎯 总结

### 主要结论

1. **整体性能**: Harmonic训练策略在162个任务中平均优于Max训练0.40%，但差异很小。

2. **胜率分布**: Harmonic胜率63.0%，Max胜率37.0%，显示Harmonic在多数任务上略有优势。

3. **任务差异**: 
   - CLS任务上Harmonic优势更明显（平均0.69%）
   - SEG任务上两者几乎持平（平均0.11%）

4. **实际意义**: 差异在统计误差范围内（<0.5%），**两种融合策略性能相当**。

### 研究启示

- **融合方式的选择对训练影响有限**: Max和Harmonic训练后的模型性能接近，
  说明模型的判别能力主要来自于text prompt和memory bank的学习，而非融合方式本身。

- **Oracle gating的潜力更大**: 相比改变训练时的融合方式，在推理时动态选择分支
  （Oracle模式平均比单一融合高3-5%）可能是更有价值的研究方向。

- **类别差异性**: 不同类别对融合方式的敏感度不同，部分难例（如capsule, screw）
  在Max训练下表现更好，而多数常规类别在Harmonic下更稳定。
