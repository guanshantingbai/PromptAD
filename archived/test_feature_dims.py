import torch
from PromptAD import PromptAD

model = PromptAD(
    backbone='ViT-B-16-plus-240',
    pretrained_dataset='laion400m_e32',
    n_pro=3, n_pro_ab=4, n_ctx=4, n_ctx_ab=1,
    Epoch=1, lambda1=0.001, device='cuda:0',
    class_name='screw',
    out_size_h=400, out_size_w=400,
    k_shot=1
).cuda()

# Test dimensions
dummy_img = torch.randn(1, 3, 240, 240).cuda()
features = model.encode_image(dummy_img)

print('Number of feature maps:', len(features))
for i, f in enumerate(features):
    print(f'  features[{i}].shape:', f.shape)
print()
print('model.visual.output_dim:', model.model.visual.output_dim)
if hasattr(model.model.visual, 'embed_dim'):
    print('model.visual.embed_dim:', model.model.visual.embed_dim)
