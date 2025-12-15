"""
Experimental configurations for modified PromptAD architectures.

This file defines 6 experimental variants:
1. QQ + Residual (QQ attention with residual connections)
2. KK + Residual (KK attention with residual connections)
3. VV + Residual (VV attention with residual connections)
4. QQ + No Residual (QQ attention without residual connections)
5. KK + No Residual (KK attention without residual connections)
6. VV + No Residual (VV attention without residual connections)

All variants:
- Remove dual-path architecture (use_single_path=True)
- Remove FFN layers (use_ffn=False)
"""

# Base configuration shared by all experiments
BASE_CONFIG = {
    'use_single_path': True,  # Use SinglePathTransformer instead of V2VTransformer
    'use_ffn': False,          # Remove FFN layers
}

# Define 6 experimental configurations
EXPERIMENTAL_CONFIGS = {
    # With residual connections
    'qq_residual': {
        **BASE_CONFIG,
        'attn_type': 'qq',
        'use_residual': True,
        'description': 'QQ attention (Q replaces K) with residual connections',
    },
    'kk_residual': {
        **BASE_CONFIG,
        'attn_type': 'kk',
        'use_residual': True,
        'description': 'KK attention (K replaces Q) with residual connections',
    },
    'vv_residual': {
        **BASE_CONFIG,
        'attn_type': 'vv',
        'use_residual': True,
        'description': 'VV attention (V replaces both Q and K) with residual connections',
    },
    
    # Without residual connections
    'qq_no_residual': {
        **BASE_CONFIG,
        'attn_type': 'qq',
        'use_residual': False,
        'description': 'QQ attention (Q replaces K) without residual connections',
    },
    'kk_no_residual': {
        **BASE_CONFIG,
        'attn_type': 'kk',
        'use_residual': False,
        'description': 'KK attention (K replaces Q) without residual connections',
    },
    'vv_no_residual': {
        **BASE_CONFIG,
        'attn_type': 'vv',
        'use_residual': False,
        'description': 'VV attention (V replaces both Q and K) without residual connections',
    },
}

# Original configuration (for comparison)
ORIGINAL_CONFIG = {
    'use_single_path': False,
    'attn_type': 'qq',  # Not used in V2VTransformer
    'use_ffn': True,
    'use_residual': True,
    'description': 'Original PromptAD with V2V dual-path architecture',
}


def get_config(config_name):
    """
    Get configuration by name.
    
    Args:
        config_name: One of 'qq_residual', 'kk_residual', 'vv_residual',
                     'qq_no_residual', 'kk_no_residual', 'vv_no_residual',
                     or 'original'
    
    Returns:
        dict: Configuration dictionary
    """
    if config_name == 'original':
        return ORIGINAL_CONFIG.copy()
    elif config_name in EXPERIMENTAL_CONFIGS:
        return EXPERIMENTAL_CONFIGS[config_name].copy()
    else:
        raise ValueError(f"Unknown config: {config_name}. Available configs: "
                        f"{list(EXPERIMENTAL_CONFIGS.keys()) + ['original']}")


def list_configs():
    """List all available configurations."""
    print("Available configurations:")
    print("\n=== Original Configuration ===")
    print(f"original: {ORIGINAL_CONFIG['description']}")
    
    print("\n=== Experimental Configurations (with residual) ===")
    for name in ['qq_residual', 'kk_residual', 'vv_residual']:
        print(f"{name}: {EXPERIMENTAL_CONFIGS[name]['description']}")
    
    print("\n=== Experimental Configurations (without residual) ===")
    for name in ['qq_no_residual', 'kk_no_residual', 'vv_no_residual']:
        print(f"{name}: {EXPERIMENTAL_CONFIGS[name]['description']}")


if __name__ == '__main__':
    list_configs()
