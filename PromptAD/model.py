import torch
import random
import numpy as np
import torch.nn as nn
from . import CLIPAD
from torch.nn import functional as F
from .ad_prompts import *
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

        state_anomaly1 = state_anomaly + class_state_abnormal[classname]

        if classname in class_mapping:
            classname = class_mapping[classname]

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

        # normal prompt
        normal_prompts = [normal_prompt_prefix + " " + classname + "." for _ in range(n_pro)]

        # abnormal prompt
        self.n_ab_handle = len(state_anomaly1)
        abnormal_prompts_handle = [normal_prompt_prefix + " " + state.format(classname) + "." for state in state_anomaly1 for _ in range(n_pro)]
        abnormal_prompts_learned = [normal_prompt_prefix + " " + abnormal_prompt_prefix + " " + classname + "." for _ in range(n_pro_ab) for _ in range(n_pro)]

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
        
        # Store multiple prototypes for normal (K prototypes) and abnormal (M prototypes)
        # Will be dynamically sized in build_text_feature_gallery
        self.register_buffer("normal_prototypes", torch.zeros((1, self.model.visual.output_dim)))
        self.register_buffer("abnormal_prototypes", torch.zeros((1, self.model.visual.output_dim)))
        
        # Memory bank for visual features (1-NN approach)
        # Use embed_dim (896) not output_dim (640) - matches baseline implementation
        visual_gallery1 = torch.zeros((self.shot*self.grid_size[0]*self.grid_size[1], self.model.visual.embed_dim))
        self.register_buffer("feature_gallery1", visual_gallery1)
        
        visual_gallery2 = torch.zeros((self.shot*self.grid_size[0]*self.grid_size[1], self.model.visual.embed_dim))
        self.register_buffer("feature_gallery2", visual_gallery2)

        if self.precision == 'fp16':
            self.normal_prototypes = self.normal_prototypes.half()
            self.abnormal_prototypes = self.abnormal_prototypes.half()
            self.feature_gallery1 = self.feature_gallery1.half()
            self.feature_gallery2 = self.feature_gallery2.half()

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
        else:
            raise NotImplementedError

        # Store all prototypes instead of averaging (multi-prototype approach)
        # Normalize each prototype individually
        normal_text_features = normal_text_features / normal_text_features.norm(dim=-1, keepdim=True)
        abnormal_text_features = abnormal_text_features / abnormal_text_features.norm(dim=-1, keepdim=True)
        
        # Dynamically resize buffers if needed
        if self.normal_prototypes.shape[0] != normal_text_features.shape[0]:
            self.normal_prototypes = torch.zeros_like(normal_text_features)
            if self.precision == 'fp16':
                self.normal_prototypes = self.normal_prototypes.half()
        
        if self.abnormal_prototypes.shape[0] != abnormal_text_features.shape[0]:
            self.abnormal_prototypes = torch.zeros_like(abnormal_text_features)
            if self.precision == 'fp16':
                self.abnormal_prototypes = self.abnormal_prototypes.half()
        
        self.normal_prototypes.copy_(normal_text_features)
        self.abnormal_prototypes.copy_(abnormal_text_features)

    @torch.no_grad()
    def build_image_feature_gallery(self, features1, features2):
        """
        Build image feature gallery from training data features.
        This matches the baseline implementation.
        
        Args:
            features1: [B, N, D] or [B*N, D] tensor from visual layer 1
            features2: [B, N, D] or [B*N, D] tensor from visual layer 2
        """
        # Handle both 3D [B, N, D] and 2D [B*N, D] inputs
        if features1.dim() == 3:
            b1, n1, d1 = features1.shape
            features1 = features1.reshape(-1, d1)
        
        if features2.dim() == 3:
            b2, n2, d2 = features2.shape
            features2 = features2.reshape(-1, d2)
        
        # Copy normalized features to buffers (baseline implementation)
        self.feature_gallery1.copy_(F.normalize(features1, dim=-1))
        self.feature_gallery2.copy_(F.normalize(features2, dim=-1))

    def calculate_textual_anomaly_score(self, visual_features, task):
        # t = 100
        t = self.model.logit_scale
        # t = self.t
        N = visual_features[1].shape[0]

        if task == 'seg':
            # Multi-prototype approach for local patches
            token_features = visual_features[1]  # [N, num_patches, dim]
            
            # Compute similarity to all normal prototypes and take max (multi-modal normal manifold)
            normal_sim = t * token_features @ self.normal_prototypes.T  # [N, num_patches, K_normal]
            max_normal_sim = normal_sim.max(dim=-1)[0]  # [N, num_patches]
            
            # Compute similarity to all abnormal prototypes and take max (structured abnormal directions)
            abnormal_sim = t * token_features @ self.abnormal_prototypes.T  # [N, num_patches, M_abnormal]
            max_abnormal_sim = abnormal_sim.max(dim=-1)[0]  # [N, num_patches]
            
            # Softmax between best normal and best abnormal
            logits = torch.stack([max_normal_sim, max_abnormal_sim], dim=-1)  # [N, num_patches, 2]
            prob = logits.softmax(dim=-1)
            local_abnormality_score = prob[:, :, 1]  # [N, num_patches]
            
            # Dynamically get grid size from actual feature shape  
            num_patches = visual_features[1].shape[1]
            grid_h = grid_w = int(num_patches ** 0.5)
            
            # Keep on same device as input
            local_abnormality_score = local_abnormality_score.reshape((N, grid_h, grid_w)).unsqueeze(1)

            return local_abnormality_score

        elif task == 'cls':
            # Multi-prototype approach for global cls token
            global_feature = visual_features[0]  # [N, dim]
            
            # Compute similarity to all normal prototypes and take max
            normal_sim = t * global_feature @ self.normal_prototypes.T  # [N, K_normal]
            max_normal_sim = normal_sim.max(dim=-1)[0]  # [N]
            
            # Compute similarity to all abnormal prototypes and take max
            abnormal_sim = t * global_feature @ self.abnormal_prototypes.T  # [N, M_abnormal]
            max_abnormal_sim = abnormal_sim.max(dim=-1)[0]  # [N]
            
            # Softmax between best normal and best abnormal
            logits = torch.stack([max_normal_sim, max_abnormal_sim], dim=-1)  # [N, 2]
            prob = logits.softmax(dim=-1)
            global_abnormality_score = prob[:, 1]  # [N]

            global_abnormality_score = global_abnormality_score.cpu()

            return global_abnormality_score.detach().numpy()

        else:
            assert 'task error'

    def calculate_visual_anomaly_score(self, visual_features):
        """
        Calculate patch-level anomaly score based on 1-NN distance to memory bank.
        Author's implementation - no task parameter, always returns patch-level map.
        
        Args:
            visual_features: List of [cls, patches, mid_feat1, mid_feat2]
        
        Returns:
            Anomaly score map [N, 1, grid_h, grid_w]
        """
        N = visual_features[1].shape[0]
        
        # Compute 1-NN distance for features1 (gallery1)
        score1, _ = (1.0 - visual_features[2] @ self.feature_gallery1.t()).min(dim=-1)
        score1 /= 2.0
        
        # Compute 1-NN distance for features2 (gallery2)
        score2, _ = (1.0 - visual_features[3] @ self.feature_gallery2.t()).min(dim=-1)
        score2 /= 2.0
        
        # Average the two scores (keep on same device as input for fusion)
        score = 0.5 * (score1 + score2)
        
        # Reshape to spatial grid and add channel dimension
        return score.reshape((N, self.grid_size[0], self.grid_size[1])).unsqueeze(1)

    def forward(self, images, task):
        """
        Forward pass.
        
        For SEG task: Returns fused pixel-level anomaly maps (harmonic mean fusion done in model).
        For CLS task: Returns independent semantic and visual branches (fusion done in metric_cal_img).
        
        This matches baseline's implementation where CLS fusion happens in utils/metrics.py.
        """
        visual_features = self.encode_image(images)
        
        if task == 'seg':
            # Compute both semantic and visual anomaly scores
            textual_anomaly_map = self.calculate_textual_anomaly_score(visual_features, 'seg')
            visual_anomaly_map = self.calculate_visual_anomaly_score(visual_features)
            
            # Harmonic mean fusion with numerator = 1
            # score = 1 / (1/semantic + 1/visual)
            # Add small epsilon to avoid division by zero
            eps = 1e-10
            textual_anomaly_map = textual_anomaly_map.clamp(min=eps)
            visual_anomaly_map = visual_anomaly_map.clamp(min=eps)
            
            anomaly_map = 1.0 / (1.0/textual_anomaly_map + 1.0/visual_anomaly_map)

            anomaly_map = F.interpolate(anomaly_map, size=(self.out_size_h, self.out_size_w), mode='bilinear', align_corners=False)

            am_pix = anomaly_map.squeeze(1).detach().cpu().numpy()

            am_pix_list = []

            for i in range(am_pix.shape[0]):
                # Convert to float32 for gaussian_filter (doesn't support float16)
                am_pix_f32 = am_pix[i].astype(np.float32) if am_pix[i].dtype == np.float16 else am_pix[i]
                am_pix[i] = gaussian_filter(am_pix_f32, sigma=4)
                am_pix_list.append(am_pix[i])

            return am_pix_list

        elif task == 'cls':
            # CLS task: Return independent semantic and visual branches
            # Fusion will be done in metric_cal_img (utils/metrics.py)
            # This matches baseline's implementation
            
            # 1. Compute semantic branch score (multi-prototype)
            textual_anomaly = self.calculate_textual_anomaly_score(visual_features, 'cls')
            
            # 2. Compute visual branch pixel-level map (for fusion in metric_cal_img)
            visual_anomaly_map = self.calculate_visual_anomaly_score(visual_features)
            
            # Interpolate to output resolution
            anomaly_map = F.interpolate(visual_anomaly_map, 
                                       size=(self.out_size_h, self.out_size_w), 
                                       mode='bilinear', 
                                       align_corners=False)
            
            am_pix = anomaly_map.squeeze(1).detach().cpu().numpy()
            am_pix_list = []
            for i in range(am_pix.shape[0]):
                am_pix_list.append(am_pix[i])
            
            # 3. Return semantic score as image-level score (NOT fused!)
            # metric_cal_img will do: fusion = 1/(1/semantic + 1/visual_map.max())
            am_img_list = []
            for i in range(textual_anomaly.shape[0]):
                am_img_list.append(textual_anomaly[i])
            
            return am_img_list, am_pix_list
        else:
            raise ValueError(f"Unknown task: {task}")

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()

