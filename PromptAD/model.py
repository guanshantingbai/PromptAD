import torch
import random
import torch.nn as nn
import numpy as np
from . import CLIPAD
from torch.nn import functional as F
from .ad_prompts import state_anomaly, class_state_abnormal, class_mapping
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
    def __init__(self, n_ctx, n_pro, n_ctx_ab, n_pro_ab, classname, clip_model, pre):
        super().__init__()

        if pre == 'fp16':
            dtype = torch.float16
        else:
            dtype = torch.float32

        # Use template prompts (same as training)
        template_prompts = state_anomaly + class_state_abnormal[classname]
        
        # Apply class mapping for display names
        display_name = class_mapping.get(classname, classname)
        
        print(f"\n[Expanded Prompts] Class: {classname}")
        print(f"  Total prompts: {len(template_prompts)}")
        print(f"  Sample prompts: {[p.format(classname) for p in template_prompts[:3]]}")

        ctx_dim = clip_model.ln_final.weight.shape[0]

        # random initialization
        normal_ctx_vectors = torch.empty(n_pro, n_ctx, ctx_dim, dtype=dtype)
        abnormal_ctx_vectors = torch.empty(n_pro_ab, n_ctx_ab, ctx_dim, dtype=dtype)

        nn.init.normal_(normal_ctx_vectors, std=0.02)
        nn.init.normal_(abnormal_ctx_vectors, std=0.02)

        normal_prompt_prefix = " ".join(["N"] * n_ctx)
        abnormal_prompt_prefix = " ".join(["A"] * n_ctx_ab)

        self.normal_ctx = nn.Parameter(normal_ctx_vectors)  # to be optimized
        self.abnormal_ctx = nn.Parameter(abnormal_ctx_vectors)  # to be optimized

        # normal prompt (use display name for readability)
        normal_prompts = [normal_prompt_prefix + " " + display_name + "." for _ in range(n_pro)]

        # abnormal prompt - format templates with original classname (file name)
        self.n_ab_handle = len(template_prompts)
        print(f"\n{'='*60}")
        print(f"[Prompt Configuration] Class: {display_name}")
        print(f"  - Mode: Template Prompts (LAP + MAP, Purge3)")
        print(f"  - Total prompts: {self.n_ab_handle}")
        print(f"{'='*60}\n")
        
        # Build abnormal prompt strings - format templates with display_name (mapped classname)
        abnormal_prompts_handle = [normal_prompt_prefix + " " + tmpl.format(display_name) + "." for tmpl in template_prompts for _ in range(n_pro)]
        abnormal_prompts_learned = [normal_prompt_prefix + " " + abnormal_prompt_prefix + " " + display_name + "." for _ in range(n_pro_ab) for _ in range(n_pro)]


        # abnormal_prompts = abnormal_prompts_learned + abnormal_prompts_handle

        tokenized_normal_prompts = CLIPAD.tokenize(normal_prompts)
        tokenized_abnormal_prompts_handle = torch.cat([CLIPAD.tokenize(p) for p in abnormal_prompts_handle])
        tokenized_abnormal_prompts_learned = torch.cat([CLIPAD.tokenize(p) for p in abnormal_prompts_learned])

        with torch.no_grad():
            normal_embedding = clip_model.token_embedding(tokenized_normal_prompts).type(dtype)
            abnormal_embedding_handle = clip_model.token_embedding(tokenized_abnormal_prompts_handle).type(dtype)
            abnormal_embedding_learned = clip_model.token_embedding(tokenized_abnormal_prompts_learned).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("normal_token_prefix", normal_embedding[:, :1, :])  # SOS
        self.register_buffer("normal_token_suffix", normal_embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.register_buffer("abnormal_token_prefix_handle", abnormal_embedding_handle[:, :1, :])  # SOS
        self.register_buffer("abnormal_token_suffix_handle", abnormal_embedding_handle[:, 1 + n_ctx:, :])  # CLS, EOS

        self.register_buffer("abnormal_token_prefix_learned", abnormal_embedding_learned[:, :1, :])  # SOS
        self.register_buffer("abnormal_token_suffix_learned", abnormal_embedding_learned[:, 1 + n_ctx + n_ctx_ab:, :])  # CLS, EOS

        self.n_pro = n_pro
        self.n_ctx = n_ctx
        self.n_pro_ab = n_pro_ab
        self.n_ctx_ab = n_ctx_ab
        self.tokenized_normal_prompts = tokenized_normal_prompts  # torch.Tensor
        self.tokenized_abnormal_prompts_handle = tokenized_abnormal_prompts_handle  # torch.Tensor
        self.tokenized_abnormal_prompts_learned = tokenized_abnormal_prompts_learned  # torch.Tensor
        # self.tokenized_abnormal_prompts = torch.cat([tokenized_abnormal_prompts_handle, tokenized_abnormal_prompts_learned], dim=0)
        # self.tokenized_abnormal_prompts = tokenized_abnormal_prompts_handle
        # self.name_lens = name_lens

    def forward(self):

        # learned normal prompt
        normal_ctx = self.normal_ctx

        normal_prefix = self.normal_token_prefix
        normal_suffix = self.normal_token_suffix

        normal_prompts = torch.cat(
            [
                normal_prefix,  # (n_pro, 1, dim)
                normal_ctx,     # (n_pro, n_ctx, dim)
                normal_suffix,  # (n_pro, *, dim)
            ],
            dim=1,
        )

        # handle abnormal prompt
        n_ab_handle = self.n_ab_handle

        n_pro, n_ctx, dim = normal_ctx.shape
        normal_ctx1 = normal_ctx.unsqueeze(0).expand(n_ab_handle, -1, -1, -1).reshape(-1, n_ctx, dim)

        abnormal_prefix_handle = self.abnormal_token_prefix_handle
        abnormal_suffix_handle = self.abnormal_token_suffix_handle

        abnormal_prompts_handle = torch.cat(
            [
                abnormal_prefix_handle,     # (n_pro * n_ab_handle, 1, dim)
                normal_ctx1,                # (n_pro * n_ab_handle, n_ctx, dim)
                abnormal_suffix_handle,     # (n_pro * n_ab_handle, *, dim)
            ],
            dim=1,
        )

        # learned abnormal prompt
        abnormal_prefix_learned = self.abnormal_token_prefix_learned
        abnormal_suffix_learned = self.abnormal_token_suffix_learned
        abnormal_ctx = self.abnormal_ctx
        n_pro_ad, n_ctx_ad, dim_ad = abnormal_ctx.shape
        normal_ctx2 = normal_ctx.unsqueeze(0).expand(self.n_pro_ab, -1, -1, -1).reshape(-1, n_ctx, dim)
        abnormal_ctx = abnormal_ctx.unsqueeze(0).expand(self.n_pro, -1, -1, -1).reshape(-1, n_ctx_ad, dim_ad)

        abnormal_prompts_learned = torch.cat(
            [
                abnormal_prefix_learned,        # (n_pro * n_pro_ab, 1, dim)
                normal_ctx2,                    # (n_pro * n_pro_ab, n_ctx, dim)
                abnormal_ctx,                   # (n_pro * n_pro_ab, n_ctx_ab, dim)
                abnormal_suffix_learned,        # (n_pro * n_pro_ab, *, dim)
            ],
            dim=1,
        )

        # abnormal_prompts = torch.cat([abnormal_prompts_handle, abnormal_prompts_learned], dim=0)
        # abnormal_prompts = abnormal_prompts_handle

        return normal_prompts, abnormal_prompts_handle, abnormal_prompts_learned


class PromptAD(torch.nn.Module):
    def __init__(self, out_size_h, out_size_w, device, backbone, pretrained_dataset, n_ctx, n_pro, n_ctx_ab, n_pro_ab, class_name,  precision='fp16', **kwargs):
        '''

        :param out_size_h:
        :param out_size_w:
        :param device:
        :param backbone:
        :param pretrained_dataset:
        '''
        super(PromptAD, self).__init__()

        self.shot = kwargs['k_shot']

        self.out_size_h = out_size_h
        self.out_size_w = out_size_w
        self.precision = 'fp16' #precision  -40% GPU memory (2.8G->1.6G) with slight performance drop

        self.device = device
        self.get_model(n_ctx, n_pro, n_ctx_ab, n_pro_ab, class_name, backbone, pretrained_dataset)
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
        
        # LSE aggregation settings (deprecated, kept for compatibility)
        self.aggregation = kwargs.get('aggregation', 'average')  # 'average', 'maxpooling', 'lse'
        self.lse_tau = kwargs.get('lse_tau', 1.0)
        
        # Multi-Abnormal Prototypes inference settings
        self.topk_abnormal = kwargs.get('topk_abnormal', None)  # None=mean, k=1→max, k>1→top-k mean
        
        # Normal-aware correction for abnormal aggregation
        self.alpha_normal_aware = kwargs.get('alpha_normal_aware', 1.0)  # Correction strength

    def get_model(self, n_ctx, n_pro, n_ctx_ab, n_pro_ab, class_name, backbone, pretrained_dataset):

        assert backbone in valid_backbones
        assert pretrained_dataset in valid_pretrained_datasets

        model, _, _ = CLIPAD.create_model_and_transforms(
            model_name=backbone, 
            pretrained=pretrained_dataset,
            precision=self.precision
        )
        tokenizer = CLIPAD.get_tokenizer(backbone)
        model.eval()

        self.prompt_learner = PromptLearner(n_ctx, n_pro, n_ctx_ab, n_pro_ab, class_name, model, self.precision)
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
        abnormal_features_all = torch.zeros((n_pro_ab * n_pro + self.prompt_learner.n_ab_handle * n_pro, 
                                            self.model.visual.output_dim))
        self.register_buffer("normal_text_features_all", normal_features_all)
        self.register_buffer("abnormal_text_features_all", abnormal_features_all)

        if self.precision == 'fp16':
            self.feature_gallery1  = self.feature_gallery1.half()
            self.feature_gallery2  = self.feature_gallery2.half()
            self.text_features  = text_features.half()
            self.normal_text_features_all = self.normal_text_features_all.half()
            self.abnormal_text_features_all = self.abnormal_text_features_all.half()

        # # for testing
        # p1, p2 = self.prompt_learner()
        self.tokenized_normal_prompts = self.prompt_learner.tokenized_normal_prompts
        self.tokenized_abnormal_prompts_handle = self.prompt_learner.tokenized_abnormal_prompts_handle
        self.tokenized_abnormal_prompts_learned = self.prompt_learner.tokenized_abnormal_prompts_learned
        self.tokenized_abnormal_prompts = torch.cat([self.tokenized_abnormal_prompts_handle, self.tokenized_abnormal_prompts_learned], dim=0)

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
        normal_text_embeddings, abnormal_text_embeddings_handle, abnormal_text_embeddings_learned = self.prompt_learner()
        abnormal_text_embeddings = torch.cat([abnormal_text_embeddings_handle, abnormal_text_embeddings_learned], dim=0)

        if self.version == "V1":
            normal_text_features = self.encode_text_embedding(normal_text_embeddings, self.tokenized_normal_prompts)
            abnormal_text_features = self.encode_text_embedding(abnormal_text_embeddings, self.tokenized_abnormal_prompts)
            
            # Store individual prompts for aggregation in inference
            self.normal_text_features_all = normal_text_features
            self.abnormal_text_features_all = abnormal_text_features
        elif self.version == "V2":
            normal_text_features = []
            for phrase_id in range(normal_text_embeddings.size()[0]):
                normal_text_feature = self.encode_text_embedding(normal_text_embeddings[phrase_id].unsqueeze(0), self.tokenized_normal_prompts)
                normal_text_feature = normal_text_feature/normal_text_feature.norm(dim=-1, keepdim=True)
                normal_text_features.append(normal_text_feature)
            normal_text_features = torch.cat(normal_text_features, 0).half()
            abnormal_text_features = []
            for phrase_id in range(abnormal_text_embeddings.size()[0]):
                abnormal_text_feature = self.encode_text_embedding(abnormal_text_embeddings[phrase_id].unsqueeze(0), self.tokenized_abnormal_prompts)
                abnormal_text_feature = abnormal_text_feature/abnormal_text_feature.norm(dim=-1, keepdim=True)
                abnormal_text_features.append(abnormal_text_feature)
            abnormal_text_features = torch.cat(abnormal_text_features, 0).half()
            
            # Store individual prompts for aggregation
            self.normal_text_features_all = normal_text_features
            self.abnormal_text_features_all = abnormal_text_features
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


    def calculate_textual_anomaly_score(self, visual_features, task, return_logits=False, topk_abnormal=None):
        """
        计算异常分数
        
        Args:
            visual_features: 视觉特征
            task: 'cls' 或 'seg'
            return_logits: 是否返回logits
            topk_abnormal: Top-k聚合异常原型（默认None=使用mean，k=1对应max）
        """
        t = self.model.logit_scale
        N = visual_features[1].shape[0]

        if task == 'seg':
            token_features = visual_features[1]  # [N, H*W, D]
            
            # Multi-Abnormal Prototypes Inference with Top-k aggregation
            if topk_abnormal is not None and hasattr(self, 'abnormal_text_features_all'):
                # Use multi-abnormal prototypes with top-k aggregation
                # Compute similarities for each pixel
                sim_normal = token_features @ self.text_features[0].unsqueeze(0).T  # [N, H*W, 1]
                sim_abnormals = token_features @ self.abnormal_text_features_all.T  # [N, H*W, K]
                
                # 🆕 Normal-aware correction for patch-level
                sim_abnormals_corrected = sim_abnormals - self.alpha_normal_aware * sim_normal  # [N, H*W, K]
                
                # Top-k aggregation for abnormal similarities (on corrected values)
                if topk_abnormal == 1:
                    # k=1: max pooling
                    aggregated_abnormal = sim_abnormals_corrected.max(dim=-1)[0]  # [N, H*W]
                else:
                    # k>1: top-k mean
                    topk_sims, _ = torch.topk(sim_abnormals_corrected, k=min(topk_abnormal, sim_abnormals_corrected.shape[-1]), dim=-1)  # [N, H*W, k]
                    aggregated_abnormal = topk_sims.mean(dim=-1)  # [N, H*W]
                
                # Build logits
                sim_normal_scalar = sim_normal.squeeze(-1)  # [N, H*W]
                local_logits = torch.stack([sim_normal_scalar, aggregated_abnormal], dim=-1) * t  # [N, H*W, 2]
            else:
                # Fallback: use mean anchors (original baseline)
                local_logits = t * token_features @ self.text_features.T  # [N, H*W, 2]
            
            local_normality_and_abnormality_score = local_logits.softmax(dim=-1)
            local_abnormality_score = local_normality_and_abnormality_score[:, :, 1]

            # Dynamically get grid size from actual feature shape  
            num_patches = visual_features[1].shape[1]
            grid_h = grid_w = int(num_patches ** 0.5)
            
            local_abnormality_score = torch.zeros((N, num_patches)) + local_abnormality_score.cpu()
            local_abnormality_score = local_abnormality_score.reshape((N, grid_h, grid_w)).unsqueeze(1)

            if return_logits:
                return local_abnormality_score.detach(), local_logits.detach()
            return local_abnormality_score.detach()

        elif task == 'cls':
            global_feature = visual_features[0]  # [N, D]
            
            # Multi-Abnormal Prototypes Inference with Top-k aggregation
            if topk_abnormal is not None and hasattr(self, 'abnormal_text_features_all'):
                # Use multi-abnormal prototypes with top-k aggregation
                # Compute similarities
                sim_normal = global_feature @ self.text_features[0].unsqueeze(0).T  # [N, 1]
                sim_abnormals = global_feature @ self.abnormal_text_features_all.T  # [N, K]
                
                # 🆕 Normal-aware correction: penalize abnormal prototypes that are too similar to normal
                sim_abnormals_corrected = sim_abnormals - self.alpha_normal_aware * sim_normal  # [N, K]
                
                # Top-k aggregation for abnormal similarities (on corrected values)
                if topk_abnormal == 1:
                    # k=1: max pooling
                    aggregated_abnormal = sim_abnormals_corrected.max(dim=-1)[0]  # [N]
                else:
                    # k>1: top-k mean
                    topk_sims, _ = torch.topk(sim_abnormals_corrected, k=min(topk_abnormal, sim_abnormals_corrected.shape[-1]), dim=-1)  # [N, k]
                    aggregated_abnormal = topk_sims.mean(dim=-1)  # [N]
                
                # Build logits
                sim_normal_scalar = sim_normal.squeeze(-1)  # [N]
                global_logits = torch.stack([sim_normal_scalar, aggregated_abnormal], dim=-1) * t  # [N, 2]
            else:
                # Fallback: use mean anchors (original baseline)
                global_logits = t * global_feature @ self.text_features.T  # [N, 2]
            
            global_normality_and_abnormality_score = global_logits.softmax(dim=-1)  # [N, 2]
            global_abnormality_score = global_normality_and_abnormality_score[:, 1]  # [N]
            global_abnormality_score = global_abnormality_score.cpu()

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

    def forward(self, images, task, aggregation='average', lse_tau=1.0):

        visual_features = self.encode_image(images)
        if task == 'seg':
            # Use topk_abnormal if available (for multi-abnormal prototypes)
            topk_abnormal = getattr(self, 'topk_abnormal', None)
            textual_anomaly_map = self.calculate_textual_anomaly_score(
                visual_features, 'seg', 
                topk_abnormal=topk_abnormal
            )

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
            # Use topk_abnormal if available (for multi-abnormal prototypes)
            topk_abnormal = getattr(self, 'topk_abnormal', None)
            textual_anomaly = self.calculate_textual_anomaly_score(
                visual_features, 'cls',
                topk_abnormal=topk_abnormal
            )
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
