import torch
import random
import numpy as np
import torch.nn as nn
from . import CLIPAD
from torch.nn import functional as F
from .ad_prompts import state_anomaly, class_state_abnormal, class_mapping
from .ad_prompts_expanded import class_specific_map_prompts, generic_lap_prompts, class_mapping as expanded_class_mapping
from PIL import Image
from scipy.ndimage import gaussian_filter

from .CLIPAD import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()   # local tokenizer, no padding, no sos, no eos

valid_backbones = ['ViT-B-16-plus-240', "ViT-B-16"]
valid_pretrained_datasets = ['laion400m_e32']

from torchvision import transforms


mean_train = [0.48145466, 0.4578275, 0.40821073]
std_train = [0.26862954, 0.26130258, 0.27577711]


def _convert_to_rgb(image):
    return image.convert('RGB')


class PromptLearner(nn.Module):
    """
    Pure Anomaly Direction Generator (Refactored Architecture)
    
    🔥 New Paradigm:
    - Normal: Visual manifold features from training images
    - Abnormal: Text prompts generating anomaly directions Δ
    
    MAP (Manual Abnormal Prompts):
        Template: "a photo of {classname} {anomaly_word}."
        Example: "a photo of carpet with hole."
    
    LAP (Learnable Abnormal Prompts):
        Template: "a photo of {classname} [learnable_ctx]."
        Learnable tokens inserted after classname.
    
    ⚠️ NO normal_ctx in MAP/LAP anymore!
    """
    def __init__(self, n_ctx, n_pro, n_ctx_ab, n_pro_ab, classname, clip_model, pre, use_manifold_normal=False):
        super().__init__()
        
        self.use_manifold_normal = use_manifold_normal
        self.n_ctx = n_ctx
        self.n_pro = n_pro
        self.n_ctx_ab = n_ctx_ab
        self.n_pro_ab = n_pro_ab

        if pre == 'fp16':
            dtype = torch.float16
        else:
            dtype = torch.float32

        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        self.dtype = dtype

        # Apply class mapping for display names
        display_name = expanded_class_mapping.get(classname, classname)
        self.classname = classname
        self.display_name = display_name
        
        # ========================================
        # Normal Prototype (for query-conditioned mode only)
        # ========================================
        if use_manifold_normal:
            # 使用流形特征：不创建可学习参数，注册为 buffer（稍后由外部提供）
            normal_ctx_placeholder = torch.zeros(n_pro, n_ctx, ctx_dim, dtype=dtype)
            self.register_buffer("normal_ctx", normal_ctx_placeholder)
            print(f"\n[Manifold Mode] Normal prototype will be replaced by manifold features")
            print(f"  Expected shape: [{n_pro}, {n_ctx}, {ctx_dim}]")
        else:
            # 原有逻辑：随机初始化可学习参数
            normal_ctx_vectors = torch.empty(n_pro, n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(normal_ctx_vectors, std=0.02)
            self.normal_ctx = nn.Parameter(normal_ctx_vectors)  # to be optimized
            print(f"\n[Learnable Mode] Normal prototype is a learnable parameter")
        
        # Normal prompt placeholder (only used for query-conditioned scoring)
        normal_prompt_prefix = " ".join(["N"] * n_ctx)
        normal_prompts = [normal_prompt_prefix + " " + display_name + "." for _ in range(n_pro)]
        tokenized_normal_prompts = CLIPAD.tokenize(normal_prompts)
        
        with torch.no_grad():
            normal_embedding = clip_model.token_embedding(tokenized_normal_prompts).type(dtype)
        
        self.register_buffer("normal_token_prefix", normal_embedding[:, :1, :])  # SOS
        self.register_buffer("normal_token_suffix", normal_embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        self.tokenized_normal_prompts = tokenized_normal_prompts
        
        # ========================================
        # 🔥 MAP: Manual Abnormal Prompts
        # Template: "a photo of {classname} {anomaly_word}."
        # Structure: Generic LAP + Class-specific MAP
        # ========================================
        
        # 1. Generic abnormal prompts (originally state_anomaly)
        generic_map_prompts = [tmpl.format(display_name) for tmpl in generic_lap_prompts]
        
        # 2. Class-specific MAP from expanded prompts
        if classname in class_specific_map_prompts:
            specific_map_prompts = class_specific_map_prompts[classname]
        else:
            specific_map_prompts = []
        
        # 3. Combine: Generic + Specific (same as original state_anomaly + class_state_abnormal)
        map_templates = generic_map_prompts + specific_map_prompts
        
        self.n_map = len(map_templates)
        self.n_generic_map = len(generic_map_prompts)
        self.n_specific_map = len(specific_map_prompts)
        
        print(f"\n{'='*60}")
        print(f"[MAP Configuration] Class: {display_name}")
        print(f"  - Generic MAP: {self.n_generic_map} prompts")
        print(f"  - Specific MAP: {self.n_specific_map} prompts")
        print(f"  - Total MAP: {self.n_map}")
        print(f"  - Sample generic: {generic_map_prompts}")
        print(f"  - Sample specific: {specific_map_prompts[:2] if specific_map_prompts else 'None'}")
        print(f"{'='*60}\n")
        
        # Build MAP prompts: "a photo of {prompt}"
        map_prompts = [f"a photo of {tmpl}." for tmpl in map_templates]
        tokenized_map = torch.cat([CLIPAD.tokenize(p) for p in map_prompts])
        
        with torch.no_grad():
            map_embedding = clip_model.token_embedding(tokenized_map).type(dtype)
        
        # Store MAP tokens (fixed, no learnable parts)
        self.register_buffer("map_token_prefix", map_embedding[:, :1, :])  # SOS
        self.register_buffer("map_token_suffix", map_embedding[:, 1:, :])  # rest of prompt
        self.tokenized_map = tokenized_map
        
        # ========================================
        # 🔥 LAP: Learnable Abnormal Prompts
        # Template: "a photo of {classname} [learnable_ctx]."
        # ========================================
        
        # Initialize learnable context vectors
        abnormal_ctx_vectors = torch.empty(n_pro_ab, n_ctx_ab, ctx_dim, dtype=dtype)
        nn.init.normal_(abnormal_ctx_vectors, std=0.02)
        self.abnormal_ctx = nn.Parameter(abnormal_ctx_vectors)  # 🔥 Only learnable part
        
        # LAP template structure: "a photo of {classname} A A A ."
        # A A A will be replaced by learnable vectors
        abnormal_prompt_prefix = " ".join(["A"] * n_ctx_ab)
        lap_prompts = [f"a photo of {display_name} {abnormal_prompt_prefix}." for _ in range(n_pro_ab)]
        
        tokenized_lap = torch.cat([CLIPAD.tokenize(p) for p in lap_prompts])
        
        with torch.no_grad():
            lap_embedding = clip_model.token_embedding(tokenized_lap).type(dtype)
        
        # Dissect LAP structure
        # Format: [SOS] "a photo of {cls}" [A A A] "." [EOS]
        # We need to extract positions to insert learnable ctx
        
        # Find where the learnable tokens start (after "a photo of {classname}")
        sample_text = f"a photo of {display_name}"
        sample_tokens = CLIPAD.tokenize([sample_text])[0]
        n_prefix_tokens = (sample_tokens != 0).sum().item() - 1  # exclude SOS
        
        self.register_buffer("lap_token_prefix", lap_embedding[:, :1, :])  # SOS
        self.register_buffer("lap_token_middle", lap_embedding[:, 1:1+n_prefix_tokens, :])  # "a photo of {cls}"
        self.register_buffer("lap_token_suffix", lap_embedding[:, 1+n_prefix_tokens+n_ctx_ab:, :])  # "." + EOS
        self.tokenized_lap = tokenized_lap
        
        print(f"\n[LAP Configuration]")
        print(f"  - Total LAP: {n_pro_ab}")
        print(f"  - Learnable context length: {n_ctx_ab}")
        print(f"  - Template: 'a photo of {display_name} [ctx_ab].'")
        print(f"{'='*60}\n")

    def forward(self):
        """
        Generate anomaly direction embeddings (text-only).
        
        Returns:
            map_embeddings: [n_map, seq_len, dim] - Manual anomaly prompts (fixed)
            lap_embeddings: [n_pro_ab, seq_len, dim] - Learnable anomaly prompts
        
        Note: NO normal_ctx is included. These are pure anomaly directions.
        """
        
        # ========================================
        # MAP: Manual Abnormal Prompts (Fixed)
        # Structure: [SOS] + "a photo of {cls} {anomaly}." + [EOS]
        # ========================================
        
        map_prompts = torch.cat(
            [
                self.map_token_prefix,  # [n_map, 1, dim] - SOS
                self.map_token_suffix,  # [n_map, *, dim] - rest of prompt
            ],
            dim=1,
        )  # [n_map, seq_len, dim]
        
        # ========================================
        # LAP: Learnable Abnormal Prompts
        # Structure: [SOS] + "a photo of {cls}" + [learnable_ctx] + "." + [EOS]
        # ========================================
        
        abnormal_ctx = self.abnormal_ctx  # [n_pro_ab, n_ctx_ab, dim]
        
        lap_prompts = torch.cat(
            [
                self.lap_token_prefix,   # [n_pro_ab, 1, dim] - SOS
                self.lap_token_middle,   # [n_pro_ab, *, dim] - "a photo of {cls}"
                abnormal_ctx,            # [n_pro_ab, n_ctx_ab, dim] - 🔥 Learnable
                self.lap_token_suffix,   # [n_pro_ab, *, dim] - "." + EOS
            ],
            dim=1,
        )  # [n_pro_ab, seq_len, dim]
        
        return map_prompts, lap_prompts

    def build_normal_prompts(self):
        """
        Build normal prompts (for legacy mode compatibility).
        
        Returns:
            normal_prompts: [n_pro, seq_len, dim] - Normal text embeddings
        """
        normal_ctx = self.normal_ctx  # [n_pro, n_ctx, dim]
        
        normal_prompts = torch.cat(
            [
                self.normal_token_prefix,  # [n_pro, 1, dim] - SOS
                normal_ctx,                # [n_pro, n_ctx, dim] - Learnable/Manifold
                self.normal_token_suffix,  # [n_pro, *, dim] - CLS, EOS
            ],
            dim=1,
        )  # [n_pro, seq_len, dim]
        
        return normal_prompts


class PromptAD(torch.nn.Module):
    def __init__(self, out_size_h, out_size_w, device, backbone, pretrained_dataset, n_ctx, n_pro, n_ctx_ab, n_pro_ab, class_name,  precision='fp16', use_manifold_normal=False, use_visual_prototypes=False, **kwargs):
        '''
        :param out_size_h:
        :param out_size_w:
        :param device:
        :param backbone:
        :param pretrained_dataset:
        :param use_manifold_normal: 是否使用流形特征替代 Normal Prototype
        :param use_visual_prototypes: 是否直接使用训练图像的 CLS tokens 作为 Normal Prototypes
        '''
        super(PromptAD, self).__init__()
        
        self.use_manifold_normal = use_manifold_normal
        self.use_visual_prototypes = use_visual_prototypes
        self.shot = kwargs['k_shot']

        self.out_size_h = out_size_h
        self.out_size_w = out_size_w
        self.precision = 'fp16' #precision  -40% GPU memory (2.8G->1.6G) with slight performance drop

        self.device = device
        self.get_model(n_ctx, n_pro, n_ctx_ab, n_pro_ab, class_name, backbone, pretrained_dataset, use_manifold_normal)
        self.phrase_form = '{}'
        self.device = device
        
        print(f"\n[PromptAD] Initializing with expanded table prompts")

        # version v1: no norm for each of linguistic embedding
        # version v1:    norm for each of linguistic embedding
        self.version = 'V1' # V1:
        # visual textual, textual_visual

        self.transform = transforms.Compose([
            transforms.Resize((kwargs['img_resize'], kwargs['img_resize']), Image.BICUBIC),
            transforms.CenterCrop(kwargs['img_cropsize']),
            _convert_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_train, std=std_train)])

        self.gt_transform = transforms.Compose([
            transforms.Resize((kwargs['img_resize'], kwargs['img_resize']), Image.NEAREST),
            transforms.CenterCrop(kwargs['img_cropsize']),
            transforms.ToTensor()])

    def get_model(self, n_ctx, n_pro, n_ctx_ab, n_pro_ab, class_name, backbone, pretrained_dataset, use_manifold_normal=False):

        assert backbone in valid_backbones
        assert pretrained_dataset in valid_pretrained_datasets

        model, _, _ = CLIPAD.create_model_and_transforms(
            model_name=backbone, 
            pretrained=pretrained_dataset,
            precision=self.precision
        )
        tokenizer = CLIPAD.get_tokenizer(backbone)
        model.eval()

        self.prompt_learner = PromptLearner(n_ctx, n_pro, n_ctx_ab, n_pro_ab, class_name, model, self.precision, use_manifold_normal)
        self.model = model.to(self.device)

        self.tokenizer = tokenizer
        self.normal_text_features = None
        self.abnormal_text_features = None
        self.grid_size = model.visual.grid_size
        self.visual_gallery = None

        visual_gallery1 = torch.zeros((self.shot*self.grid_size[0]*self.grid_size[1], self.model.visual.embed_dim))
        self.register_buffer("feature_gallery1", visual_gallery1)

        visual_gallery2 = torch.zeros((self.shot*self.grid_size[0]*self.grid_size[1], self.model.visual.embed_dim))
        self.register_buffer("feature_gallery2", visual_gallery2)

        text_features = torch.zeros((2, self.model.visual.output_dim))
        self.register_buffer("text_features", text_features)
        
        # 🆕 保存所有文本特征向量（用于maxpooling等聚合方式）
        # 初始化为空，将在build_text_feature_gallery中填充
        normal_features_all = torch.zeros((n_pro, self.model.visual.output_dim))
        # MAP + LAP abnormal features
        n_map = len(self.prompt_learner.tokenized_map) if hasattr(self.prompt_learner, 'tokenized_map') else 0
        n_lap = n_pro_ab
        abnormal_features_all = torch.zeros((n_map + n_lap, self.model.visual.output_dim))
        self.register_buffer("normal_text_features_all", normal_features_all)
        self.register_buffer("abnormal_text_features_all", abnormal_features_all)
        
        # 🆕 保存训练图像的 CLS tokens（用于推理阶段的语义融合）
        # 初始化为空，将在 set_visual_prototypes 中填充
        training_cls_tokens = torch.zeros((1, self.model.visual.output_dim))  # placeholder
        self.register_buffer("training_cls_tokens", training_cls_tokens)

        if self.precision == 'fp16':
            self.feature_gallery1  = self.feature_gallery1.half()
            self.feature_gallery2  = self.feature_gallery2.half()
            self.text_features  = text_features.half()
            self.normal_text_features_all = self.normal_text_features_all.half()
            self.abnormal_text_features_all = self.abnormal_text_features_all.half()
            self.training_cls_tokens = self.training_cls_tokens.half()

        # # for testing
        # p1, p2 = self.prompt_learner()
        self.tokenized_normal_prompts = self.prompt_learner.tokenized_normal_prompts
        self.tokenized_map = self.prompt_learner.tokenized_map
        self.tokenized_lap = self.prompt_learner.tokenized_lap
        self.tokenized_abnormal_prompts = torch.cat([self.tokenized_map, self.tokenized_lap], dim=0)

    def set_manifold_normal_features(self, manifold_features):
        """
        设置从训练图像提取的流形特征作为 normal prototype
        
        Args:
            manifold_features: torch.Tensor, shape [n_pro, n_ctx, ctx_dim]
                              表示从正常训练图像中提取的流形特征
        
        Note:
            - 仅在 use_manifold_normal=True 时生效
            - 会自动转换为正确的数据类型（fp16/fp32）
            - 特征会被复制到 prompt_learner.normal_ctx buffer 中
        """
        if not self.use_manifold_normal:
            print("[Warning] set_manifold_normal_features() called but use_manifold_normal=False. Ignoring.")
            return
        
        # 验证形状
        expected_shape = (self.prompt_learner.n_pro, self.prompt_learner.n_ctx, self.prompt_learner.ctx_dim)
        if manifold_features.shape != expected_shape:
            raise ValueError(
                f"Manifold features shape mismatch! "
                f"Expected {expected_shape}, got {manifold_features.shape}"
            )
        
        # 转换数据类型并复制
        manifold_features = manifold_features.to(dtype=self.prompt_learner.dtype, device=self.device)
        self.prompt_learner.normal_ctx.copy_(manifold_features)
        
        print(f"\n[Manifold Features Set]")
        print(f"  Shape: {manifold_features.shape}")
        print(f"  Device: {manifold_features.device}")
        print(f"  Dtype: {manifold_features.dtype}")
        print(f"  Norm (mean): {manifold_features.norm(dim=-1).mean().item():.4f}")

    def set_visual_prototypes(self, train_images):
        """
        直接使用训练图像的 CLS tokens 作为 Normal Prototypes
        
        Args:
            train_images: torch.Tensor or list of PIL.Image
                         形状 [k_shot, 3, H, W] 或 k_shot 个 PIL.Image
        
        Note:
            - 仅在 use_visual_prototypes=True 时生效
            - 会提取 k_shot 张图像的 CLS tokens
            - 平均后作为 Normal Anchor
            - 跳过 Prompt Learning，直接设置 self.text_features[0]
        """
        if not self.use_visual_prototypes:
            print("[Warning] set_visual_prototypes() called but use_visual_prototypes=False. Ignoring.")
            return
        
        # 处理输入格式
        if not isinstance(train_images, torch.Tensor):
            # 如果是 PIL.Image 列表，转换为 tensor
            from PIL import Image
            if isinstance(train_images[0], Image.Image):
                train_images = torch.stack([self.transform(img) for img in train_images])
        
        # 移动到设备
        train_images = train_images.to(self.device)
        if self.precision == 'fp16':
            train_images = train_images.half()
        
        # 提取所有图像的 CLS tokens
        with torch.no_grad():
            visual_features = self.encode_image(train_images)  # returns list
            cls_tokens = visual_features[0]  # [k_shot, dim]
        
        # 归一化
        cls_tokens = cls_tokens / cls_tokens.norm(dim=-1, keepdim=True)
        
        # 计算平均作为 Normal Anchor
        normal_anchor = cls_tokens.mean(dim=0, keepdim=True)  # [1, dim]
        normal_anchor = normal_anchor / normal_anchor.norm(dim=-1, keepdim=True)
        
        # 保存所有 CLS tokens（用于 MaxPooling 等聚合方式）
        if self.normal_text_features_all.shape[0] != cls_tokens.shape[0]:
            self.normal_text_features_all = torch.zeros_like(cls_tokens)
        self.normal_text_features_all.copy_(cls_tokens)
        
        # 🆕 保存训练图像的 CLS tokens 到独立 buffer（用于推理阶段）
        if self.training_cls_tokens.shape[0] != cls_tokens.shape[0]:
            self.training_cls_tokens = torch.zeros_like(cls_tokens)
        self.training_cls_tokens.copy_(cls_tokens)
        
        # 更新 self.text_features[0] (保持 [1] 为 abnormal)
        # 注意：此时 abnormal 仍然是文本特征，但 normal 是视觉特征
        self.text_features[0].copy_(normal_anchor[0])
        
        print(f"\n[Visual Prototypes Set]")
        print(f"  Number of training images: {cls_tokens.shape[0]}")
        print(f"  CLS tokens shape: {cls_tokens.shape}")
        print(f"  Normal anchor (averaged): {normal_anchor.shape}")
        print(f"  Device: {normal_anchor.device}")
        print(f"  Dtype: {normal_anchor.dtype}")
        print(f"  Similarity between prototypes:")
        if cls_tokens.shape[0] > 1:
            sim_matrix = cls_tokens @ cls_tokens.T
            print(f"    Mean: {sim_matrix.mean().item():.4f}")
            print(f"    Min: {sim_matrix.min().item():.4f}")
            print(f"    Max: {sim_matrix.max().item():.4f}")

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor):

        if self.precision == "fp16":
            image = image.half()
        image_features = self.model.encode_image(image)
        return [f / f.norm(dim=-1, keepdim=True) for f in image_features]

    @torch.no_grad()
    def encode_text(self, text: torch.Tensor):
        text_features = self.model.encode_text(text)
        # return [f / f.norm(dim=-1, keepdim=True) for f in text_features]
        return text_features

    def encode_text_embedding(self, text_embedding, original_tokens):
        text_features = self.model.encode_text_embeddings(text_embedding, original_tokens)
        return text_features

    @torch.no_grad()
    def build_text_feature_gallery(self):
        """
        🔥 Pure Anomaly Direction Generator (Refactored)
        
        Architecture:
        - Normal: Visual manifold features (no text prompt learning)
        - Abnormal: MAP + LAP text prompts (pure anomaly directions)
        
        Output:
        - self.abnormal_directions: [n_map + n_lap, D] - L2-normalized anomaly directions
        - self.text_features[1]: Average abnormal direction
        """
        
        # 🆕 如果使用视觉原型，跳过 prompt learning，只更新 abnormal 特征
        if self.use_visual_prototypes:
            print("\n[Visual Prototype Mode] Building pure abnormal directions (MAP + LAP)...")
            
            # Get MAP and LAP embeddings from prompt learner
            map_embeddings, lap_embeddings = self.prompt_learner()
            print(f"  MAP embeddings shape: {map_embeddings.shape}")
            print(f"  LAP embeddings shape: {lap_embeddings.shape}")
            
            # Combine all abnormal embeddings
            abnormal_embeddings = torch.cat([map_embeddings, lap_embeddings], dim=0)
            print(f"  Combined abnormal embeddings shape: {abnormal_embeddings.shape}")
            
            # Encode to text features
            if self.version == "V1":
                print(f"  Encoding with V1...")
                abnormal_text_features = self.encode_text_embedding(abnormal_embeddings, self.tokenized_abnormal_prompts)
                print(f"  Encoded features shape: {abnormal_text_features.shape}")
            elif self.version == "V2":
                abnormal_text_features = []
                for phrase_id in range(abnormal_embeddings.size()[0]):
                    abnormal_text_feature = self.encode_text_embedding(
                        abnormal_embeddings[phrase_id].unsqueeze(0), 
                        self.tokenized_abnormal_prompts[phrase_id].unsqueeze(0)
                    )
                    abnormal_text_feature = abnormal_text_feature / abnormal_text_feature.norm(dim=-1, keepdim=True)
                    abnormal_text_features.append(abnormal_text_feature)
                abnormal_text_features = torch.cat(abnormal_text_features, 0).half()
            else:
                raise NotImplementedError
            
            # L2-normalize all abnormal directions
            abnormal_directions = abnormal_text_features / abnormal_text_features.norm(dim=-1, keepdim=True)
            
            # Compute average abnormal direction as anchor
            avr_abnormal_direction = torch.mean(abnormal_directions, dim=0, keepdim=True)
            avr_abnormal_direction = avr_abnormal_direction / avr_abnormal_direction.norm(dim=-1, keepdim=True)
            
            # Update anchors
            self.text_features[1].copy_(avr_abnormal_direction[0])
            
            # Store all abnormal directions
            if self.abnormal_text_features_all.shape[0] != abnormal_directions.shape[0]:
                self.abnormal_text_features_all = torch.zeros_like(abnormal_directions)
                if self.precision == 'fp16':
                    self.abnormal_text_features_all = self.abnormal_text_features_all.half()
            self.abnormal_text_features_all.copy_(abnormal_directions)
            
            print(f"\n{'='*60}")
            print(f"[Abnormal Directions Built]")
            print(f"  - MAP count: {map_embeddings.shape[0]}")
            print(f"  - LAP count: {lap_embeddings.shape[0]}")
            print(f"  - Total directions: {abnormal_directions.shape[0]}")
            print(f"  - Average abnormal anchor: {avr_abnormal_direction.shape}")
            print(f"  - Normal anchor: [VISUAL MANIFOLD - from training images]")
            print(f"{'='*60}\n")
            return
        
        # 📖 原始逻辑：同时构建 normal 和 abnormal 文本特征 (兼容性保留)
        print("\n[Legacy Mode] Building text features with prompt learning...")
        
        # Get embeddings
        normal_text_embeddings = self.prompt_learner.build_normal_prompts()
        map_embeddings, lap_embeddings = self.prompt_learner()
        abnormal_embeddings = torch.cat([map_embeddings, lap_embeddings], dim=0)

        if self.version == "V1":
            normal_text_features = self.encode_text_embedding(normal_text_embeddings, self.tokenized_normal_prompts)
            abnormal_text_features = self.encode_text_embedding(abnormal_embeddings, self.tokenized_abnormal_prompts)
        elif self.version == "V2":
            normal_text_features = []
            for phrase_id in range(normal_text_embeddings.size()[0]):
                normal_text_feature = self.encode_text_embedding(
                    normal_text_embeddings[phrase_id].unsqueeze(0), 
                    self.tokenized_normal_prompts[phrase_id].unsqueeze(0)
                )
                normal_text_feature = normal_text_feature / normal_text_feature.norm(dim=-1, keepdim=True)
                normal_text_features.append(normal_text_feature)
            normal_text_features = torch.cat(normal_text_features, 0).half()
            
            abnormal_text_features = []
            for phrase_id in range(abnormal_embeddings.size()[0]):
                abnormal_text_feature = self.encode_text_embedding(
                    abnormal_embeddings[phrase_id].unsqueeze(0), 
                    self.tokenized_abnormal_prompts[phrase_id].unsqueeze(0)
                )
                abnormal_text_feature = abnormal_text_feature / abnormal_text_feature.norm(dim=-1, keepdim=True)
                abnormal_text_features.append(abnormal_text_feature)
            abnormal_text_features = torch.cat(abnormal_text_features, 0).half()
        else:
            raise NotImplementedError

        avr_normal_text_features = torch.mean(normal_text_features, dim=0, keepdim=True)
        avr_abnormal_text_features = torch.mean(abnormal_text_features, dim=0, keepdim=True)

        text_features_all = torch.cat([normal_text_features, abnormal_text_features], dim=0)
        text_features_all /= text_features_all.norm(dim=-1, keepdim=True)

        avr_normal_text_features = avr_normal_text_features
        avr_abnormal_text_features = avr_abnormal_text_features
        text_features = torch.cat([avr_normal_text_features, avr_abnormal_text_features], dim=0)
        self.text_features.copy_(text_features / text_features.norm(dim=-1, keepdim=True))
        
        # 🆕 保存所有文本特征向量（归一化后）
        normal_text_features_normed = normal_text_features / normal_text_features.norm(dim=-1, keepdim=True)
        abnormal_text_features_normed = abnormal_text_features / abnormal_text_features.norm(dim=-1, keepdim=True)
        
        # 动态调整buffer大小（如果形状不匹配）
        if self.normal_text_features_all.shape[0] != normal_text_features_normed.shape[0]:
            self.normal_text_features_all = torch.zeros_like(normal_text_features_normed)
            if self.precision == 'fp16':
                self.normal_text_features_all = self.normal_text_features_all.half()
        
        if self.abnormal_text_features_all.shape[0] != abnormal_text_features_normed.shape[0]:
            self.abnormal_text_features_all = torch.zeros_like(abnormal_text_features_normed)
            if self.precision == 'fp16':
                self.abnormal_text_features_all = self.abnormal_text_features_all.half()
        
        self.normal_text_features_all.copy_(normal_text_features_normed)
        self.abnormal_text_features_all.copy_(abnormal_text_features_normed)

    def build_image_feature_gallery(self, features1, features2):
        b1, n1, d1 = features1.shape
        features1_flat = F.normalize(features1.reshape(-1, d1), dim=-1)
        
        # Dynamically resize gallery if needed
        if self.feature_gallery1.shape[0] != features1_flat.shape[0]:
            self.feature_gallery1 = torch.zeros_like(features1_flat)
            if self.precision == 'fp16':
                self.feature_gallery1 = self.feature_gallery1.half()
        
        self.feature_gallery1.copy_(features1_flat)

        b2, n2, d2 = features2.shape
        features2_flat = F.normalize(features2.reshape(-1, d2), dim=-1)
        
        # Dynamically resize gallery if needed
        if self.feature_gallery2.shape[0] != features2_flat.shape[0]:
            self.feature_gallery2 = torch.zeros_like(features2_flat)
            if self.precision == 'fp16':
                self.feature_gallery2 = self.feature_gallery2.half()
        
        self.feature_gallery2.copy_(features2_flat)

    def calculate_textual_anomaly_score(self, visual_features, task, return_logits=False, aggregation='average', lambda_scale=1.0, margin=0.0, semantic_weight=0.0):
        """
        \u8ba1\u7b97\u5f02\u5e38\u5206\u6570
        
        Args:
            visual_features: \u89c6\u89c9\u7279\u5f81
            task: 'cls' \u6216 'seg'
            return_logits: \u662f\u5426\u8fd4\u56delogits
            aggregation: \u805a\u5408\u65b9\u5f0f
                - 'average': \u4f7f\u7528\u5e73\u5747\u9501\u70b9 (self.text_features) [\u9ed8\u8ba4]
                - 'maxpooling': \u4f7f\u7528MaxPooling\u805a\u5408\u6240\u6709\u5411\u91cf
            semantic_weight: MVP fusion weight (alpha), only applies when use_visual_prototypes=True
        """
        # t = 100
        t = self.model.logit_scale
        # t = self.t
        N = visual_features[1].shape[0]

        if task == 'seg':
            # ############################################## local tokens scores ############################
            # token_features = self.cross_attention(visual_features[1])
            token_features = visual_features[1]
            
            if aggregation == 'query_conditioned':
                # 🆕 Query-conditioned anomaly construction
                normal_reps = self.normal_text_features_all  # [K, D]
                abnormal_directions = self.abnormal_text_features_all  # [M, D]
                
                # Select local normal reference for each query token
                sim_to_normals = token_features @ normal_reps.T  # [N, num_tokens, K]
                i_star = sim_to_normals.argmax(dim=-1)  # [N, num_tokens]
                n_star = normal_reps[i_star]  # [N, num_tokens, D]
                
                # Construct anomaly candidates: A = normalize(n_star + λ * Δ_j)
                A = n_star.unsqueeze(2) + lambda_scale * abnormal_directions.unsqueeze(0).unsqueeze(0)
                A = F.normalize(A, dim=-1)  # [N, num_tokens, M, D]
                
                # Scoring with logsumexp
                logits_normal = t * (token_features @ normal_reps.T)  # [N, num_tokens, K]
                s_N = torch.logsumexp(logits_normal, dim=-1)  # [N, num_tokens]
                
                logits_abnormal = t * torch.einsum('ntd,ntmd->ntm', token_features, A)
                s_A = torch.logsumexp(logits_abnormal, dim=-1)  # [N, num_tokens]
                
                local_abnormality_score = F.relu(s_A - s_N - margin)  # [N, num_tokens]
                
                # Dynamically get grid size from actual feature shape  
                num_patches = visual_features[1].shape[1]
                grid_h = grid_w = int(num_patches ** 0.5)
                
                local_abnormality_score = torch.zeros((N, num_patches)) + local_abnormality_score.cpu()
                local_abnormality_score = local_abnormality_score.reshape((N, grid_h, grid_w)).unsqueeze(1)
                
                if return_logits:
                    return local_abnormality_score.detach(), None
                return local_abnormality_score.detach()
                
            elif aggregation == 'maxpooling':
                # \ud83c\udd95 MaxPooling\u805a\u5408\uff1a\u4e0e\u6240\u6709\u5411\u91cf\u7684\u6700\u5927\u76f8\u4f3c\u5ea6
                sim_normal = token_features @ self.normal_text_features_all.T  # [N, num_tokens, n_pro]
                sim_abnormal = token_features @ self.abnormal_text_features_all.T  # [N, num_tokens, n_ab]
                
                score_normal = sim_normal.max(dim=-1)[0]  # [N, num_tokens]
                score_abnormal = sim_abnormal.max(dim=-1)[0]  # [N, num_tokens]
                
                local_logits = torch.stack([score_normal, score_abnormal], dim=-1) * t  # [N, num_tokens, 2]
            else:
                # \u539f\u59cb\u5e73\u5747\u9501\u70b9\u65b9\u6cd5
                local_logits = t * token_features @ self.text_features.T
            
            local_normality_and_abnormality_score = local_logits.softmax(dim=-1)
            local_abnormality_score = local_normality_and_abnormality_score[:, :, 1]

            # Dynamically get grid size from actual feature shape  
            num_patches = visual_features[1].shape[1]
            grid_h = grid_w = int(num_patches ** 0.5)
            
            local_abnormality_score = torch.zeros((N, num_patches)) + local_abnormality_score.cpu()
            local_abnormality_score = local_abnormality_score.reshape((N, grid_h, grid_w)).unsqueeze(1)

            if return_logits:
                if aggregation == 'query_conditioned':
                    return local_abnormality_score.detach(), None
                return local_abnormality_score.detach(), local_logits.detach()
            return local_abnormality_score.detach()

        elif task == 'cls':
            # ################################################ global cls token scores ##########################
            # global_feature = self.cross_attention(visual_features[0].unsqueeze(dim=1)).squeeze(dim=1)
            global_feature = visual_features[0]  # [N, dim]
            
            # 🔥 n(q) MVP: Hard selection of nearest normal prototype (inference-only)
            n_q_alignment = None
            E_sem = None
            
            if semantic_weight > 0 and self.use_visual_prototypes:
                # Check if training_cls_tokens exists and is valid
                has_training_tokens = (
                    hasattr(self, 'training_cls_tokens') and 
                    self.training_cls_tokens.shape[0] > 1  # Not just placeholder
                )
                
                if has_training_tokens and hasattr(self, 'abnormal_text_features_all'):
                    # Get normal_reps (support set cls embeddings)
                    normal_reps = self.training_cls_tokens  # [K, D]
                    
                    # Get MAP features (delta_map)
                    n_map = getattr(self.prompt_learner, 'n_map', len(self.abnormal_text_features_all))
                    delta_map = self.abnormal_text_features_all[:n_map]  # [N_map, D]
                    
                    # Ensure dtype compatibility
                    gf = global_feature.to(normal_reps.dtype) if global_feature.dtype != normal_reps.dtype else global_feature
                    
                    # Step 1: n(q) - Hard selection of nearest normal prototype
                    # i_star = argmax(global_feature @ normal_reps.T)
                    sim_to_normals = gf @ normal_reps.T  # [N, K]
                    i_star = sim_to_normals.argmax(dim=-1)  # [N] - index of nearest prototype
                    n_q = normal_reps[i_star]  # [N, D] - nearest normal prototype for each query
                    
                    # Step 2: Compute n(q) alignment
                    n_q_alignment = (gf * n_q).sum(dim=-1)  # [N] - q · n(q)
                    
                    # Step 3: Compute E_sem
                    # E_sem = logsumexp(q @ delta_map.T) - (q @ n(q))
                    logits_map = gf @ delta_map.T  # [N, N_map]
                    map_response = torch.logsumexp(logits_map, dim=-1)  # [N] - abnormal response
                    E_sem = map_response - n_q_alignment  # [N] - semantic deviation
                    
                    # Convert to numpy for consistency
                    n_q_alignment = n_q_alignment.cpu().numpy()
                    E_sem = E_sem.cpu().numpy()
                else:
                    # Fallback: training_cls_tokens not available
                    print("[Warning] semantic_weight > 0 but training_cls_tokens not available. Ignoring semantic fusion.")
            
            if aggregation == 'query_conditioned':
                # 🆕 Query-conditioned anomaly construction
                # global_feature: [N, D] - query CLS tokens
                # normal_reps: [K, D] - multiple normal representatives from support images
                # abnormal_directions: [M, D] - anomaly offsets Δ_j (from MAP/LAP)
                
                normal_reps = self.normal_text_features_all  # [K, D]
                abnormal_directions = self.abnormal_text_features_all  # [M, D]
                
                # Step 1: Select local normal reference n*(q) using hard selection
                sim_to_normals = global_feature @ normal_reps.T  # [N, K]
                i_star = sim_to_normals.argmax(dim=-1)  # [N]
                n_star = normal_reps[i_star]  # [N, D]
                
                # Step 2: Construct anomaly candidates conditioned on query
                # A = normalize(n_star + λ * Δ_j)
                A = n_star.unsqueeze(1) + lambda_scale * abnormal_directions.unsqueeze(0)  # [N, M, D]
                A = F.normalize(A, dim=-1)  # L2 normalize each anomaly candidate
                
                # Step 3: Scoring with logsumexp
                # Normal evidence: s_N = logsumexp(q @ normal_reps.T)
                logits_normal = t * (global_feature @ normal_reps.T)  # [N, K]
                s_N = torch.logsumexp(logits_normal, dim=-1)  # [N]
                
                # Anomaly evidence: s_A = logsumexp(q @ A.T)
                logits_abnormal = t * torch.einsum('nd,nmd->nm', global_feature, A)  # [N, M]
                s_A = torch.logsumexp(logits_abnormal, dim=-1)  # [N]
                
                # Step 4: Final anomaly score with margin
                # score = relu(s_A - s_N - margin)
                global_abnormality_score = F.relu(s_A - s_N - margin)  # [N]
                global_abnormality_score = global_abnormality_score.cpu()
                
                if return_logits:
                    # Return both evidence scores for analysis
                    evidence_scores = torch.stack([s_N, s_A], dim=-1).cpu().detach().numpy()  # [N, 2]
                    return global_abnormality_score.detach().numpy(), evidence_scores
                return global_abnormality_score.detach().numpy()
                
            elif aggregation == 'maxpooling':
                # \ud83c\udd95 MaxPooling\u805a\u5408\uff1a\u4e0e\u6240\u6709\u5411\u91cf\u7684\u6700\u5927\u76f8\u4f3c\u5ea6
                sim_normal = global_feature @ self.normal_text_features_all.T  # [N, n_pro]
                sim_abnormal = global_feature @ self.abnormal_text_features_all.T  # [N, n_ab]
                
                score_normal = sim_normal.max(dim=-1)[0]  # [N,]
                score_abnormal = sim_abnormal.max(dim=-1)[0]  # [N,]
                
                global_logits = torch.stack([score_normal, score_abnormal], dim=-1) * t  # [N, 2]
            else:
                # \u539f\u59cb\u5e73\u5747\u951a\u70b9\u65b9\u6cd5
                # Ensure dtype compatibility
                if global_feature.dtype != self.text_features.dtype:
                    global_feature = global_feature.to(self.text_features.dtype)
                global_logits = t * global_feature @ self.text_features.T  # [N, 2]: [s_normal, s_abnormal]
            
            global_normality_and_abnormality_score = global_logits.softmax(dim=-1)
            global_abnormality_score = global_normality_and_abnormality_score[:, 1]
            global_abnormality_score = global_abnormality_score.cpu()
            
            # 🔥 n(q) MVP: Semantic fusion with hard-selected normal prototype
            if E_sem is not None and n_q_alignment is not None:
                # New fusion formula: score = -(q · n(q)) + alpha * E_sem
                # where E_sem = logsumexp(q @ delta_map.T) - (q · n(q))
                
                # Baseline score: -(q · n(q))
                # This represents "distance from nearest normal prototype"
                baseline_score = -n_q_alignment  # [N] - higher means further from normal
                
                # Final score with semantic contribution
                final_score = baseline_score + semantic_weight * E_sem
                
                if return_logits:
                    # Return fused score and diagnostic info
                    # [s_normal, s_abnormal, n_q_alignment, E_sem, baseline_score]
                    logits_with_sem = np.concatenate([
                        global_logits.cpu().detach().numpy(),  # [N, 2]: [s_n, s_a]
                        n_q_alignment[:, None],                # [N, 1]: q · n(q)
                        E_sem[:, None],                        # [N, 1]: E_sem
                        baseline_score[:, None],               # [N, 1]: -(q · n(q))
                    ], axis=1)  # [N, 5]
                    return final_score, logits_with_sem
                return final_score
            elif E_sem is not None:
                # Fallback to old MVP if n_q_alignment not available
                E_geom = global_abnormality_score.detach().numpy()  # Original geometric score
                final_score = E_geom + semantic_weight * E_sem  # Simple additive fusion
                
                if return_logits:
                    # Return fused score and original logits + semantic info
                    # [s_n, s_a, E_sem, E_geom]
                    logits_with_sem = np.concatenate([global_logits.cpu().detach().numpy(), 
                                                      E_sem[:, None], 
                                                      E_geom[:, None]], axis=1)  # [N, 4]: [s_n, s_a, E_sem, E_geom]
                    return final_score, logits_with_sem
                return final_score

            if return_logits:
                return global_abnormality_score.detach().numpy(), global_logits.cpu().detach().numpy()
            return global_abnormality_score.detach().numpy()

        else:
            assert 'task error'
    
    @torch.no_grad()
    def calculate_margin_and_logits(self, visual_features):
        """
        Calculate margin (geometric distance in logit space) for baseline analysis.
        
        Returns:
            margins: np.array, shape [N], margin = s_normal - s_abnormal
            logits: np.array, shape [N, 2], [s_normal, s_abnormal]
        """
        t = self.model.logit_scale
        global_feature = visual_features[0]  # CLS token
        logits = (t * global_feature @ self.text_features.T).cpu().numpy()  # [N, 2]
        margins = logits[:, 0] - logits[:, 1]  # s_normal - s_abnormal
        return margins, logits

    def calculate_visual_anomaly_score(self, visual_features):
        N = visual_features[1].shape[0]

        score1, _ = (1.0 - visual_features[2] @ self.feature_gallery1.t()).min(dim=-1)
        score1 /= 2.0

        score2, _ = (1.0 - visual_features[3] @ self.feature_gallery2.t()).min(dim=-1)
        score2 /= 2.0

        # Dynamically get grid size from actual feature shape
        num_patches = visual_features[2].shape[1]
        grid_h = grid_w = int(num_patches ** 0.5)

        score = torch.zeros((N, num_patches)) + 0.5 * (score1 + score2).cpu()

        return score.reshape((N, grid_h, grid_w)).unsqueeze(1)

    def calculate_memory_image_score(self, visual_features):
        """Calculate image-level anomaly score from memory branch (visual features)"""
        import numpy as np
        visual_anomaly_map = self.calculate_visual_anomaly_score(visual_features)
        anomaly_map = F.interpolate(visual_anomaly_map, size=(self.out_size_h, self.out_size_w), mode='bilinear',
                                    align_corners=False)
        am_pix = anomaly_map.squeeze(1).numpy()
        # Take max over spatial dimensions for image-level score
        memory_img_scores = am_pix.reshape(am_pix.shape[0], -1).max(axis=1)
        return memory_img_scores
    
    def compute_semantic_confidence(self, visual_features, tau=0.05, k=20.0):
        """
        Compute semantic confidence (gating strength) based on logit margin.
        
        Args:
            visual_features: visual features from encoder
            tau: margin collapse threshold (default 0.05)
            k: gating slope parameter (default 20.0)
        
        Returns:
            alpha: np.array [N], semantic gating strength (0 = suppress, large = enhance)
            margins: np.array [N], logit margins for analysis
        """
        import numpy as np
        
        # Compute logit margin: m(x) = s_normal - s_abnormal
        margins, logits = self.calculate_margin_and_logits(visual_features)
        
        # Gating function: α(x) = σ(k · (|m(x)| - τ))
        # Use sigmoid to ensure smooth transition
        margin_abs = np.abs(margins)
        alpha = 1.0 / (1.0 + np.exp(-k * (margin_abs - tau)))
        
        return alpha, margins
    
    def adaptive_harmonic_fusion(self, semantic_scores, memory_scores, alpha, epsilon=1e-8):
        """
        Memory-safeguarded adaptive harmonic fusion.
        
        **Corrected Formula**: s_final = max(s_mem, harmonic_fusion_when_semantic_agrees)
        
        Key design principle:
        - Memory is the baseline/floor (cannot be pulled down by semantic)
        - Semantic can ONLY enhance anomaly detection when:
          1. It has high confidence (large |margin|)
          2. It agrees with memory that sample is anomalous
        - When semantic disagrees or is uncertain → fall back to memory
        
        Implementation:
        1. Compute gated harmonic: h = 1 / (1/s_mem + α/s_sem)
        2. Take max: s_final = max(s_mem, h)
        
        This ensures:
        - α = 0 → s_final = s_mem (pure memory)
        - α large AND s_sem high → s_final may exceed s_mem (anomaly enhancement)
        - α large BUT s_sem low → s_final = s_mem (semantic suppressed)
        
        Args:
            semantic_scores: np.array [N], semantic anomaly scores
            memory_scores: np.array [N], memory anomaly scores
            alpha: np.array [N], gating strength per sample
            epsilon: small constant to avoid division by zero
        
        Returns:
            fusion_scores: np.array [N], adaptively fused scores
        """
        import numpy as np
        
        # Clamp scores to avoid numerical issues
        semantic_scores = np.clip(semantic_scores, epsilon, 1.0)
        memory_scores = np.clip(memory_scores, epsilon, 1.0)
        
        # Compute gated harmonic fusion
        # h = 1 / (1/s_mem + α/s_sem)
        harmonic_fusion = 1.0 / (1.0 / memory_scores + alpha / semantic_scores)
        
        # Memory safeguard: fusion cannot be lower than memory baseline
        # This prevents semantic from "pulling down" reliable memory predictions
        fusion_scores = np.maximum(memory_scores, harmonic_fusion)
        
        return fusion_scores

    def forward(self, images, task):

        visual_features = self.encode_image(images)
        if task == 'seg':
            textual_anomaly_map = self.calculate_textual_anomaly_score(visual_features, 'seg')

            visual_anomaly_map = self.calculate_visual_anomaly_score(visual_features)
            #
            anomaly_map = torch.maximum(textual_anomaly_map, visual_anomaly_map)
            # anomaly_map = 0.5 * (textual_anomaly_map + visual_anomaly_map)
            # anomaly_map = visual_anomaly_map
            # anomaly_map = textual_anomaly_map

            anomaly_map = F.interpolate(anomaly_map, size=(self.out_size_h, self.out_size_w), mode='bilinear', align_corners=False)

            am_pix = anomaly_map.squeeze(1).numpy()

            am_pix_list = []

            for i in range(am_pix.shape[0]):
                am_pix[i] = gaussian_filter(am_pix[i], sigma=4)
                am_pix_list.append(am_pix[i])

            return am_pix_list

        elif task == 'cls':
            # Calculate semantic branch score (textual)
            import numpy as np
            textual_anomaly = self.calculate_textual_anomaly_score(visual_features, 'cls')
            semantic_img_scores = textual_anomaly  # Already image-level

            # Calculate memory branch score (visual)
            memory_img_scores = self.calculate_memory_image_score(visual_features)

            # Pixel-level maps for compatibility
            visual_anomaly_map = self.calculate_visual_anomaly_score(visual_features)
            anomaly_map = F.interpolate(visual_anomaly_map, size=(self.out_size_h, self.out_size_w), mode='bilinear',
                                        align_corners=False)
            am_pix = anomaly_map.squeeze(1).numpy()
            am_pix_list = [am_pix[i] for i in range(am_pix.shape[0])]

            # Return: semantic_scores, memory_scores, pixel_maps (no fusion in model)
            return (list(semantic_img_scores), list(memory_img_scores), am_pix_list)
        
        elif task == 'cls_gated':
            # Semantic-gated fusion (inference-only adaptive mechanism)
            import numpy as np
            
            # Calculate semantic branch score
            textual_anomaly = self.calculate_textual_anomaly_score(visual_features, 'cls')
            semantic_img_scores = textual_anomaly
            
            # Calculate memory branch score
            memory_img_scores = self.calculate_memory_image_score(visual_features)
            
            # Compute semantic confidence (gating strength)
            # Get gating parameters from model attributes if available
            tau = getattr(self, 'gating_tau', 0.05)
            k = getattr(self, 'gating_k', 20.0)
            alpha, margins = self.compute_semantic_confidence(visual_features, tau=tau, k=k)
            
            # Adaptive harmonic fusion with memory safeguard
            fusion_img_scores = self.adaptive_harmonic_fusion(
                semantic_img_scores, memory_img_scores, alpha
            )
            
            # Pixel-level maps for compatibility
            visual_anomaly_map = self.calculate_visual_anomaly_score(visual_features)
            anomaly_map = F.interpolate(visual_anomaly_map, size=(self.out_size_h, self.out_size_w), mode='bilinear',
                                        align_corners=False)
            am_pix = anomaly_map.squeeze(1).numpy()
            am_pix_list = [am_pix[i] for i in range(am_pix.shape[0])]
            
            # Return: semantic_scores, memory_scores, fusion_scores (gated), pixel_maps
            # Also return alpha and margins for analysis
            return (list(semantic_img_scores), list(memory_img_scores), 
                    list(fusion_img_scores), am_pix_list, list(alpha), list(margins))
        
        elif task == 'cls_detailed':
            # For detailed baseline analysis: return margins, logits, and scores
            import numpy as np
            
            # Calculate margins and logits
            margins, logits = self.calculate_margin_and_logits(visual_features)
            
            # Calculate semantic score (from softmax)
            textual_anomaly = self.calculate_textual_anomaly_score(visual_features, 'cls')
            semantic_img_scores = textual_anomaly
            
            # Calculate memory branch score
            memory_img_scores = self.calculate_memory_image_score(visual_features)
            
            # Calculate fusion score
            fusion_img_scores = np.maximum(semantic_img_scores, memory_img_scores)
            
            # Return detailed info for analysis
            return {
                'margins': margins,
                'logits': logits,  # [N, 2]: [s_normal, s_abnormal]
                'semantic_scores': semantic_img_scores,
                'memory_scores': memory_img_scores,
                'fusion_scores': fusion_img_scores
            }
        else:
            assert 'task error'

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()
