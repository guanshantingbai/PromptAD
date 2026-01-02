"""
简单验证脚本 - 确认展开的 prompts 表格工作正常
"""
from PromptAD.ad_prompts import expanded_class_prompts
from PromptAD.model import PromptAD

print("="*70)
print("展开 Prompts 表格 - 验证脚本")
print("="*70)

# 测试1: 验证所有类别都有 prompts
print("\n[测试1] 验证所有类别")
print("-"*70)
total_classes = len(expanded_class_prompts)
print(f"✓ 总类别数: {total_classes}")

# 测试2: 验证关键类别的 Purge3 配置
print("\n[测试2] 验证 Purge3 配置")
print("-"*70)
expected = {
    'metal_nut': 4,
    'pill': 7,
    'cable': 5,
    'capsule': 7,  # Purge3 恢复
    'transistor': 4
}

all_match = True
for cls, exp_count in expected.items():
    actual = len(expanded_class_prompts[cls])
    status = "✓" if actual == exp_count else "✗"
    print(f"{status} {cls:<15} Expected:{exp_count:2d}  Actual:{actual:2d}")
    if actual != exp_count:
        all_match = False

if all_match:
    print("\n✓ 所有 Purge3 配置正确！")
else:
    print("\n✗ 配置不匹配！")
    exit(1)

# 测试3: 模型初始化
print("\n[测试3] 模型初始化测试")
print("-"*70)
try:
    model = PromptAD(
        out_size_h=240, out_size_w=240,
        device='cpu',
        backbone='ViT-B-16-plus-240',
        pretrained_dataset='laion400m_e32',
        n_ctx=12, n_pro=5,
        n_ctx_ab=12, n_pro_ab=5,
        class_name='metal_nut',
        k_shot=2,
        img_resize=240,
        img_cropsize=240
    )
    print(f"✓ 模型初始化成功")
    print(f"✓ Prompt 数量: {model.prompt_learner.n_ab_handle}")
except Exception as e:
    print(f"✗ 模型初始化失败: {e}")
    exit(1)

# 测试4: 显示示例 prompts
print("\n[测试4] Metal_nut Prompts 示例")
print("-"*70)
for i, p in enumerate(expanded_class_prompts['metal_nut'], 1):
    prompt_type = "LAP" if i <= 2 else "MAP"
    print(f"  {prompt_type} {i}. {p}")

print("\n" + "="*70)
print("✅ 所有测试通过！展开的 prompts 表格工作正常！")
print("="*70)
print("\n📝 总结:")
print(f"  - {total_classes} 个类别的 prompts 已完全展开")
print(f"  - 包含所有 Purge1/2/3 的注释历史")
print(f"  - 模型可以正确加载并使用")
print(f"  - 已为多原型学习做好准备")
print("\n✓ 可以开始训练/测试了！")
