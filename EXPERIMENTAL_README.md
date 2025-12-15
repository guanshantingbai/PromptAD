# 实验性架构变体 - 快速参考

## 🎯 实现目标

在 PromptAD 的基础上实现 6 个实验方案：

1. ✅ 取消 dual-path 架构
2. ✅ 将 attention 中的 QK 替换为 QQ/KK/VV（3 个独立方案）
3. ✅ 取消 FFN 层
4. ✅ 可选：舍弃残差连接（每个方案 × 2）

**结果：6 个实验配置 + 1 个原始配置（对照组）**

## ✅ 测试状态 

```bash
$ python test_experimental.py --config all
✓ All tests passed! (7/7 configurations)
```

## 📊 配置速览

| 配置名 | Attention | 残差 | 参数量 | 说明 |
|--------|-----------|------|--------|------|
| `original` | V2V dual | ✓ | 86.2M | 原始架构（对照组） |
| `qq_residual` | QQ | ✓ | 29.5M | Q 替换 K |
| `kk_residual` | KK | ✓ | 29.5M | K 替换 Q |
| `vv_residual` | VV | ✓ | 29.5M | V 替换 Q+K |
| `qq_no_residual` | QQ | ✗ | 29.5M | QQ + 无残差 |
| `kk_no_residual` | KK | ✗ | 29.5M | KK + 无残差 |
| `vv_no_residual` | VV | ✗ | 29.5M | VV + 无残差 |

**共同特点**（实验配置）：
- ✗ 无 dual-path
- ✗ 无 FFN 层
- ✓ 66% 参数减少

## 🚀 快速开始

### 1. 测试配置
```bash
# 激活环境
conda activate prompt_ad

# 测试单个配置
python test_experimental.py --config qq_residual

# 测试全部
python test_experimental.py --config all
```

### 2. 代码使用
```python
from PromptAD.CLIPAD import model as clip_model
from experimental_configs import get_config

# 获取配置
config = get_config('vv_residual')

# 创建模型
vision_cfg = clip_model.CLIPVisionCfg(
    layers=12, width=768, patch_size=16, 
    image_size=224, **config
)

visual = clip_model._build_vision_tower(
    embed_dim=512, vision_cfg=vision_cfg
)
```

### 3. 查看示例
```bash
python example_usage.py
```

## 📁 新增文件

```
experimental_configs.py      # 7 个配置定义
test_experimental.py        # 自动测试脚本
example_usage.py           # 使用示例
USAGE_GUIDE.py            # 详细文档
IMPLEMENTATION_SUMMARY.md  # 实现总结
EXPERIMENTAL_README.md     # 本文件
```

## 🔧 核心修改

### `PromptAD/CLIPAD/transformer.py`
```python
+ ModifiedAttention          # QQ/KK/VV attention
+ ModifiedResidualAttentionBlock  # 可配置 FFN/残差
+ ModifiedTransformer        # 单路径 transformer
+ SinglePathTransformer      # 完整实现
```

### `PromptAD/CLIPAD/model.py`
```python
+ CLIPVisionCfg: use_single_path, attn_type, use_ffn, use_residual
+ _build_vision_tower: 根据配置选择架构
```

## 💡 推荐实验顺序

1. **qq_residual** ← 从这里开始（最保守）
2. **vv_residual** ← V2V 的简化版本
3. **kk_residual** ← 探索反向注意力
4. **qq_no_residual** ← 如果有残差版本好，尝试无残差
5. **vv_no_residual** ← 最激进配置
6. **kk_no_residual** ← 完整性测试

## 📖 详细文档

- **快速参考**: 本文件（`EXPERIMENTAL_README.md`）
- **使用指南**: `USAGE_GUIDE.py`（详细 API 和集成说明）
- **实现总结**: `IMPLEMENTATION_SUMMARY.md`（技术细节）
- **代码示例**: `example_usage.py`（可执行示例）

## 🎓 理论背景

### Attention 变体
- **QQ**: Q @ Q^T @ V - 自查询相似性
- **KK**: K @ K^T @ V - 自键相似性  
- **VV**: V @ V^T @ V - 自值相似性（类似 V2V）

### 设计选择
- **无 FFN**: 纯注意力机制，减少参数
- **无残差**: 更激进的特征转换，可能不稳定
- **单路径**: 简化架构，减少计算

## ⚠️ 注意事项

1. **预训练**: 实验配置需要从头训练
2. **稳定性**: 无残差版本可能需要调整超参数
3. **特征**: 单路径不提供中间层特征（mid_features）
4. **内存**: 实验配置显著减少 GPU 内存使用

## 🔗 相关资源

- 原始论文: [PromptAD (CVPR 2024)](http://arxiv.org/abs/2404.05231)
- GitHub: [FuNz-0/PromptAD](https://github.com/FuNz-0/PromptAD)

---

**完成日期**: 2025-12-15  
**测试状态**: ✅ 全部通过  
**准备就绪**: ✅ 可以开始实验
