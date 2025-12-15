"""
示例：如何在 PromptAD 中使用实验性配置

这个脚本展示了如何集成新的实验配置到现有的 PromptAD 训练流程中。
"""

import torch
from experimental_configs import get_config, list_configs
from PromptAD.model import PromptAD


def create_experimental_model(exp_config='qq_residual', **kwargs):
    """
    创建带有实验配置的 PromptAD 模型
    
    Args:
        exp_config: 实验配置名称
        **kwargs: 其他 PromptAD 参数
    
    Returns:
        PromptAD 模型实例
    """
    # 获取实验配置
    from experimental_configs import get_config
    config = get_config(exp_config)
    
    print(f"\n使用配置: {exp_config}")
    print(f"配置详情: {config}")
    
    # 创建模型（需要修改 PromptAD 类以接受这些参数）
    # 这里展示概念，实际使用需要修改 PromptAD/model.py
    
    # 默认参数
    default_kwargs = {
        'out_size_h': 224,
        'out_size_w': 224,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'backbone': 'ViT-B-16',
        'pretrained_dataset': 'laion400m_e32',
        'n_ctx': 12,
        'n_pro': 4,
        'n_ctx_ab': 12,
        'n_pro_ab': 4,
        'class_name': 'bottle',
        'k_shot': 4,
        'img_resize': 240,
        'img_cropsize': 224,
    }
    
    # 更新参数
    default_kwargs.update(kwargs)
    
    # 注意：这需要修改 PromptAD 类以支持 exp_config 参数
    # model = PromptAD(**default_kwargs, exp_config=exp_config)
    
    print(f"\n模型配置:")
    print(f"  - Backbone: {default_kwargs['backbone']}")
    print(f"  - Single Path: {config['use_single_path']}")
    print(f"  - Attention Type: {config['attn_type']}")
    print(f"  - Use FFN: {config['use_ffn']}")
    print(f"  - Use Residual: {config['use_residual']}")
    
    return None  # 返回 model


def compare_configurations():
    """比较不同配置的特点"""
    print("\n" + "="*60)
    print("配置对比")
    print("="*60)
    
    configs = [
        'original',
        'qq_residual',
        'kk_residual', 
        'vv_residual',
        'qq_no_residual',
    ]
    
    print(f"\n{'配置':<20} {'路径':<10} {'Attn':<6} {'FFN':<6} {'Residual':<10}")
    print("-" * 60)
    
    for name in configs:
        cfg = get_config(name)
        path = 'Dual' if not cfg['use_single_path'] else 'Single'
        attn = cfg['attn_type'].upper()
        ffn = '✓' if cfg['use_ffn'] else '✗'
        residual = '✓' if cfg['use_residual'] else '✗'
        
        print(f"{name:<20} {path:<10} {attn:<6} {ffn:<6} {residual:<10}")


def integration_guide():
    """集成指南"""
    print("\n" + "="*60)
    print("集成到训练脚本的步骤")
    print("="*60)
    
    print("""
步骤 1: 修改 PromptAD/model.py 的 __init__ 方法
----------------------------------------
在 PromptAD 类的 __init__ 中添加 exp_config 参数：

    def __init__(self, ..., exp_config='original', **kwargs):
        super(PromptAD, self).__init__()
        
        # 存储实验配置
        self.exp_config = exp_config
        
        # ... 其他初始化代码

步骤 2: 修改 get_model 方法
----------------------------------------
在 get_model 方法中应用配置：

    def get_model(self, ..., exp_config=None):
        from experimental_configs import get_config
        
        # 使用传入的配置或实例配置
        config_name = exp_config or self.exp_config
        config = get_config(config_name)
        
        # 应用配置到模型创建
        # 注意：需要修改 CLIPAD.create_model_and_transforms
        # 以支持 vision_cfg 覆盖

步骤 3: 修改训练脚本（train_cls.py 或 train_seg.py）
----------------------------------------
添加命令行参数：

    parser.add_argument('--exp_config', 
                       type=str, 
                       default='original',
                       help='Experimental configuration')
    
    # 创建模型时传入配置
    model = PromptAD(..., exp_config=args.exp_config, ...)

步骤 4: 运行实验
----------------------------------------
    # 原始配置
    python train_cls.py --exp_config original
    
    # QQ attention with residual
    python train_cls.py --exp_config qq_residual
    
    # VV attention without residual
    python train_cls.py --exp_config vv_no_residual
    """)


def example_training_comparison():
    """训练对比示例"""
    print("\n" + "="*60)
    print("实验对比建议")
    print("="*60)
    
    experiments = {
        '基线实验': ['original'],
        '有残差连接的变体': ['qq_residual', 'kk_residual', 'vv_residual'],
        '无残差连接的变体': ['qq_no_residual', 'kk_no_residual', 'vv_no_residual'],
    }
    
    for group, configs in experiments.items():
        print(f"\n{group}:")
        for i, config in enumerate(configs, 1):
            cfg = get_config(config)
            print(f"  {i}. {config}")
            print(f"     描述: {cfg['description']}")


if __name__ == '__main__':
    print("="*60)
    print("PromptAD 实验配置使用示例")
    print("="*60)
    
    # 列出所有配置
    list_configs()
    
    # 配置对比
    compare_configurations()
    
    # 集成指南
    integration_guide()
    
    # 实验建议
    example_training_comparison()
    
    print("\n" + "="*60)
    print("下一步:")
    print("1. 运行 'python test_experimental.py --config all' 测试所有配置")
    print("2. 选择一个配置开始实验")
    print("3. 查看 USAGE_GUIDE.py 了解详细信息")
    print("="*60)
