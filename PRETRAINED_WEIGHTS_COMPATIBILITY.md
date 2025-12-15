# 实验配置与预训练权重的兼容性说明

## 问题

实验配置（QQ/KK/VV, 无FFN等）改变了模型架构，导致与原始 CLIP 预训练权重不兼容：

```
Error: Missing/Unexpected keys in state_dict
- Missing: visual.transformer.resblocks.*.attn.qkv.weight (新架构的权重)
- Unexpected: visual.transformer.resblocks.*.attn.in_proj_weight (预训练的权重)
```

## 解决方案

### 方案 1: 从头训练（推荐用于实验）

实验配置需要从头训练，**不加载预训练权重**：

```python
# 修改 PromptAD/model.py 的 get_model 方法
def get_model(self, ...):
    config = get_config(self.exp_config)
    
    # 对于实验配置，不加载预训练权重
    if config['use_single_path']:
        pretrained_to_use = None  # 不加载预训练
    else:
        pretrained_to_use = pretrained_dataset
    
    model, _, _ = CLIPAD.create_model_and_transforms(
        model_name=backbone, 
        pretrained=pretrained_to_use,  # None 或 pretrained
        ...
    )
```

### 方案 2: 部分加载预训练权重（推荐用于快速实验）

只加载可以匹配的部分（如 patch embedding, text encoder等）：

```python
# 修改 factory.py 的 load_checkpoint 函数
def load_checkpoint(model, checkpoint_path, strict=True):
    ...
    # 对于实验配置，使用 strict=False
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    
    if not strict:
        print(f"Loaded checkpoint with {len(incompatible_keys.missing_keys)} missing keys")
        print(f"and {len(incompatible_keys.unexpected_keys)} unexpected keys")
```

### 方案 3: 权重转换（最复杂，但效果可能最好）

手动转换预训练权重以匹配新架构：

```python
def convert_weights_for_modified_attention(state_dict, attn_type='qq'):
    """
    将原始的 in_proj_weight 转换为 qkv.weight
    in_proj_weight shape: [3*dim, dim] (Q, K, V 连接)
    qkv.weight shape: [3*dim, dim]
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        if 'attn.in_proj_weight' in key:
            # 转换为 qkv.weight
            new_key = key.replace('in_proj_weight', 'qkv.weight')
            new_state_dict[new_key] = value
        elif 'attn.in_proj_bias' in key:
            new_key = key.replace('in_proj_bias', 'qkv.bias')
            new_state_dict[new_key] = value
        elif 'attn.out_proj' in key:
            # out_proj -> proj
            new_key = key.replace('out_proj', 'proj')
            new_state_dict[new_key] = value
        elif 'mlp' in key or 'ln_2' in key:
            # 跳过 FFN 相关的权重（如果 use_ffn=False）
            if use_ffn:
                new_state_dict[key] = value
        else:
            new_state_dict[key] = value
    
    return new_state_dict
```

## 实现建议

### 最简单的实现（方案 1 + 2 结合）

修改 `PromptAD/model.py`:

```python
def get_model(self, n_ctx, n_pro, n_ctx_ab, n_pro_ab, class_name, backbone, pretrained_dataset):
    assert backbone in valid_backbones
    assert pretrained_dataset in valid_pretrained_datasets

    from experimental_configs import get_config
    config = get_config(self.exp_config)
    
    # 判断是否使用实验配置
    use_experimental = config['use_single_path']
    
    if use_experimental:
        print(f"⚠️  Using experimental config: {self.exp_config}")
        print("   Training from scratch (no pretrained weights)")
        pretrained_to_use = None
    else:
        pretrained_to_use = pretrained_dataset
    
    model, _, _ = CLIPAD.create_model_and_transforms(
        model_name=backbone, 
        pretrained=pretrained_to_use,
        precision=self.precision,
        force_custom_clip=use_experimental,
        vision_cfg={
            'use_single_path': config['use_single_path'],
            'attn_type': config['attn_type'],
            'use_ffn': config['use_ffn'],
            'use_residual': config['use_residual'],
        } if use_experimental else None
    )
    
    ...
```

### 使用方法

```bash
# 方案 1: 从头训练（推荐）
python train_cls.py --exp_config qq_residual \\
    --class_name bottle \\
    --Epoch 200  # 可能需要更多 epoch

# 使用原始配置（正常加载预训练权重）
python train_cls.py --exp_config original \\
    --class_name bottle
```

## 性能影响

| 方案 | 训练速度 | 收敛性 | 最终性能 |
|------|---------|--------|---------|
| 从头训练 | 慢 | 需要更多 epoch | 取决于数据量 |
| 部分加载 | 中 | 较快 | 可能较好 |
| 权重转换 | 快 | 最快 | 可能最好 |

## 注意事项

1. **Few-Shot 场景**: 从头训练在 few-shot 设置下可能效果不好
2. **数据增强**: 建议增加数据增强来补偿缺少预训练
3. **学习率**: 从头训练时可能需要调整学习率
4. **Warm-up**: 建议使用 learning rate warm-up

## 快速验证

测试模型是否能正常前向传播（不需要预训练权重）：

```bash
python test_experimental.py --config qq_residual
# ✓ 如果通过，说明架构本身没问题
```

## 下一步

1. 实现方案 1（最简单）
2. 运行小规模实验验证
3. 如果效果不好，再考虑方案 2 或 3
