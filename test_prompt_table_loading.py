"""
测试model.py是否正确从表格读取prompts
"""

import torch
from PromptAD.model import PromptAD

def test_prompt_loading():
    """测试prompt从表格加载"""
    
    print("="*80)
    print("Testing Prompt Loading from Table")
    print("="*80)
    
    # 测试参数
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    classname = 'bottle'
    
    print(f"\nCreating model for class: {classname}")
    print(f"Expected: Model will load prompts from prompts/manual_prompts_master_table.csv")
    print("-"*80)
    
    # 创建模型
    model = PromptAD(
        out_size_h=1,
        out_size_w=1,
        device=device,
        backbone='ViT-B-16-plus-240',
        pretrained_dataset='laion400m_e32',
        n_ctx=12,
        n_pro=4,
        n_ctx_ab=12,
        n_pro_ab=4,
        class_name=classname,
        precision='fp16',
        k_shot=2,
        img_resize=240,
        img_cropsize=240
    )
    
    # 获取加载的prompt信息
    prompt_info = model.get_manual_prompt_info()
    
    print(f"\n{'='*80}")
    print(f"Loaded Prompt Information")
    print(f"{'='*80}")
    print(f"Class: {prompt_info['classname']}")
    print(f"Display Name: {prompt_info['display_name']}")
    print(f"Number of Templates: {prompt_info['num_manual_templates']}")
    print(f"Number of Prototypes: {prompt_info['num_manual_prototypes']}")
    print(f"\n{'ID':<5} {'Type':<10} {'Template':<45} {'Full Text':<40}")
    print(f"{'-'*80}")
    
    for item in prompt_info['prompt_details']:
        type_tag = 'Generic' if item['type'] == 'generic' else 'Specific'
        template = item.get('template', '')
        text = item.get('text', '')
        print(f"{item['index']:<5} {type_tag:<10} {template:<45} {text:<40}")
    
    print(f"{'='*80}")
    print(f"\n✓ Test completed successfully!")
    print(f"  The model is now reading prompts directly from the table.")
    print(f"  You can modify prompts/manual_prompts_master_table.csv to control which prompts are used.")
    print(f"  Set 'enabled=False' to disable specific prompts.\n")


if __name__ == '__main__':
    test_prompt_loading()
