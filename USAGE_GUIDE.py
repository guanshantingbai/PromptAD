"""
使用指南：实验性架构配置
===============================================

本指南说明如何在 PromptAD 中使用新的实验性架构变体。

## 概述

我们实现了 6 个实验性架构变体，所有变体都：
- ✓ 取消了 dual-path 架构（使用单路径）
- ✓ 移除了 FFN 层
- ✓ 使用修改后的 attention 机制（QQ/KK/VV）

**6 个实验方案：**
1. QQ + Residual：Q 替换 K，保留残差连接
2. KK + Residual：K 替换 Q，保留残差连接
3. VV + Residual：V 同时替换 Q 和 K，保留残差连接
4. QQ + No Residual：Q 替换 K，移除残差连接
5. KK + No Residual：K 替换 Q，移除残差连接
6. VV + No Residual：V 同时替换 Q 和 K，移除残差连接

## 快速开始

### 1. 列出所有可用配置

```python
from experimental_configs import list_configs

list_configs()
```

### 2. 在代码中使用实验配置

```python
from PromptAD.CLIPAD import model as clip_model
from experimental_configs import get_config

# 选择一个配置
config = get_config('qq_residual')

# 创建 Vision Config
vision_cfg = clip_model.CLIPVisionCfg(
    layers=12,
    width=768,
    head_width=64,
    patch_size=16,
    image_size=224,
    use_single_path=config['use_single_path'],
    attn_type=config['attn_type'],
    use_ffn=config['use_ffn'],
    use_residual=config['use_residual'],
)

# 构建模型
visual = clip_model._build_vision_tower(
    embed_dim=512,
    vision_cfg=vision_cfg,
    quick_gelu=False,
    cast_dtype=None,
)
```

### 3. 测试特定配置

```bash
# 测试单个配置
python test_experimental.py --config qq_residual

# 测试所有配置
python test_experimental.py --config all
```

## 在训练脚本中使用

### 方法 1: 修改 PromptAD/model.py 中的 get_model 函数

在 `PromptAD/model.py` 的 `get_model` 函数中添加配置参数：

```python
def get_model(self, n_ctx, n_pro, n_ctx_ab, n_pro_ab, class_name, backbone, 
              pretrained_dataset, exp_config='original'):
    
    assert backbone in valid_backbones
    assert pretrained_dataset in valid_pretrained_datasets

    # 导入配置
    from experimental_configs import get_config
    config = get_config(exp_config)
    
    # 创建模型时应用配置
    model, _, _ = CLIPAD.create_model_and_transforms(
        model_name=backbone, 
        pretrained=pretrained_dataset, 
        precision=self.precision,
        vision_cfg_override={
            'use_single_path': config['use_single_path'],
            'attn_type': config['attn_type'],
            'use_ffn': config['use_ffn'],
            'use_residual': config['use_residual'],
        }
    )
    
    # ... 其余代码保持不变
```

### 方法 2: 通过命令行参数

在训练脚本（如 `train_cls.py`）中添加参数：

```python
parser.add_argument('--exp_config', type=str, default='original',
                   choices=['original', 'qq_residual', 'kk_residual', 
                           'vv_residual', 'qq_no_residual', 'kk_no_residual',
                           'vv_no_residual'],
                   help='Experimental configuration to use')

# 然后传递给模型
model = PromptAD(..., exp_config=args.exp_config, ...)
```

## 配置详细说明

### Attention 类型解释

| 配置 | Q | K | 说明 |
|------|---|---|------|
| QQ | Q | Q | Query 替换 Key，自相似性匹配 |
| KK | K | K | Key 替换 Query，反向自相似性 |
| VV | V | V | Value 替换 Q 和 K，内容驱动的注意力 |

### 残差连接的影响

- **有残差连接**: 梯度流动更稳定，训练更容易，但可能限制表达能力
- **无残差连接**: 更激进的特征转换，可能发现新的表示，但训练可能更困难

## 模型参数对比

| 配置 | 参数量 | 相对原始 |
|------|--------|----------|
| Original (V2V) | 86,192,640 | 100% |
| 所有实验配置 | 29,505,024 | 34.2% |

**减少的原因：**
- 移除了 FFN 层（约 2/3 的参数）
- 移除了 dual-path 架构的重复部分

## 性能基准

运行基准测试：

```bash
python test_experimental.py --config all
```

**测试输出示例：**
```
✓ Model built successfully
  Model type: SinglePathTransformer
  Total parameters: 29,505,024
  Trainable parameters: 29,505,024
✓ Forward pass successful
  Number of outputs: 4
  Output[0] shape: torch.Size([2, 512])  # Pooled features
  Output[1] shape: torch.Size([2, 196, 512])  # Patch tokens
  Output[2]: None  # No mid_feature1 in single-path
  Output[3]: None  # No mid_feature2 in single-path
```

## 注意事项

1. **中间特征**: SinglePathTransformer 不提供中间层特征（mid_feature1/2），因为它是单路径架构
2. **内存使用**: 实验配置使用更少的 GPU 内存（约 34% 的原始模型）
3. **训练稳定性**: 无残差连接的配置可能需要更小的学习率
4. **预训练权重**: 这些实验配置需要从头训练，不能直接加载原始的预训练权重

## 推荐实验顺序

1. **QQ + Residual**: 最保守，建议首先尝试
2. **VV + Residual**: 理论上最有趣，V2V 概念的简化版
3. **KK + Residual**: 探索反向注意力机制
4. **QQ + No Residual**: 如果有残差表现好，尝试无残差版本
5. **VV + No Residual**: 最激进的配置
6. **KK + No Residual**: 完整性测试

## 常见问题

**Q: 为什么移除 FFN？**
A: 根据您的要求，探索纯注意力机制的表现。

**Q: 可以只移除 dual-path 但保留 FFN 吗？**
A: 可以！修改配置：`use_single_path=True, use_ffn=True, use_residual=True`

**Q: 如何恢复原始架构？**
A: 使用 `config='original'` 或设置 `use_single_path=False`

## 代码位置

- `PromptAD/CLIPAD/transformer.py`: 核心实现
  - `ModifiedAttention`: QQ/KK/VV attention 实现
  - `ModifiedResidualAttentionBlock`: 支持可配置 FFN/残差的 block
  - `ModifiedTransformer`: 单路径 transformer
  - `SinglePathTransformer`: 完整的视觉 transformer
  
- `PromptAD/CLIPAD/model.py`: 模型构建
  - `CLIPVisionCfg`: 添加了新的配置参数
  - `_build_vision_tower`: 根据配置选择 transformer 类型
  
- `experimental_configs.py`: 配置定义
- `test_experimental.py`: 测试脚本

## 扩展建议

如果您想添加更多变体：

1. 在 `experimental_configs.py` 中添加新配置
2. 在 `ModifiedAttention` 中实现新的 attention 机制
3. 运行 `test_experimental.py` 验证

例如，添加标准的 self-attention：

```python
EXPERIMENTAL_CONFIGS['standard_attention'] = {
    'use_single_path': True,
    'attn_type': 'standard',  # 需要在 ModifiedAttention 中实现
    'use_ffn': True,
    'use_residual': True,
    'description': 'Standard self-attention without modifications',
}
```
"""

if __name__ == '__main__':
    print(__doc__)
