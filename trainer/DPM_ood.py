import os
import os.path as osp
from collections import OrderedDict
import torch
import torchvision.transforms as transforms
from torch.nn import functional as F
from timm.models.layers import trunc_normal_
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.data import DataManager
from dassl.metrics import compute_accuracy
from dassl.evaluation import build_evaluator
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data.transforms.transforms import build_transform
import torch.nn as nn
from copy import deepcopy
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

import sklearn.metrics as sk
from sklearn.metrics import pairwise_distances
import numpy as np
from tqdm import tqdm
from utils.utils import CosineClassifier
import torch.nn.functional as F

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def randomcut_ood(images, scale_factor=3.5):
    """
    先将图像放大，然后随机裁剪回原始尺寸，实现局部放大效果。
    
    参数:
        images (torch.Tensor): 输入图片张量，形状为 (B, C, H, W)，其中 H=W=224。
        scale_factor (float): 放大倍数，默认为 3.5。
    
    返回:
        torch.Tensor: 处理后的图片张量，形状与输入相同。
    """
    batch_size, channels, height, width = images.size()
    assert height == 224 and width == 224, "输入图片尺寸必须为 224×224"
    
    # 计算放大后的尺寸
    enlarged_size = int(height * scale_factor)
    
    # 初始化结果张量
    result_images = torch.zeros_like(images)
    
    for i in range(batch_size):
        # 放大图像
        enlarged = torch.nn.functional.interpolate(
            images[i:i+1], 
            size=(enlarged_size, enlarged_size), 
            mode='bilinear', 
            align_corners=False
        )
        
        # 计算可裁剪的最大起始位置
        max_start = enlarged_size - height
        
        # 随机选择裁剪位置
        top = torch.randint(0, max_start + 1, (1,)).item()
        left = torch.randint(0, max_start + 1, (1,)).item()
        
        # 裁剪回原始尺寸
        crop = enlarged[:, :, top:top+height, left:left+width]
        
        # 保存结果
        result_images[i] = crop[0]
    
    return result_images

# def randomcut_ood(images):
#     """
#     从每张 224×224 的图像中随机裁剪出一块 64×64 的区域，然后调整回 224×224 大小。
    
#     参数:
#         images (torch.Tensor): 输入图片张量，形状为 (B, C, H, W)，其中 H=W=224。
    
#     返回:
#         torch.Tensor: 处理后的图片张量，形状与输入相同。
#     """
#     batch_size, channels, height, width = images.size()
#     assert height == 224 and width == 224, "输入图片尺寸必须为 224×224"
    
#     # 设置裁剪尺寸
#     crop_size = 64
    
#     # 初始化结果张量
#     result_images = torch.zeros_like(images)
    
#     for i in range(batch_size):
#         # 随机选择裁剪的起始位置
#         top = torch.randint(0, height - crop_size + 1, (1,)).item()
#         left = torch.randint(0, width - crop_size + 1, (1,)).item()
        
#         # 裁剪图像
#         crop = images[i:i+1, :, top:top+crop_size, left:left+crop_size]
        
#         # 使用插值调整大小回原尺寸
#         resized_crop = torch.nn.functional.interpolate(
#             crop, 
#             size=(height, width), 
#             mode='bilinear', 
#             align_corners=False
#         )
        
#         # 保存结果
#         result_images[i] = resized_crop[0]
    
#     return result_images

def softmax(x):
    exp_values = np.exp(x)
    probabilities = exp_values / np.sum(exp_values)
    return probabilities
_tokenizer = _Tokenizer()

CUSTOM_TEMPLATES = {
    "ImageNet": "a photo of a {}.",
    "cifar100": "a photo of a {}.",
    # "ImageNet": "nice photo of {}.",
    # "cifar100": "nice photo of {}.",
}


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps  # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)  # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))  # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def kl(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def get_kl(id_sim_mean, logits):
    _score = []
    scores = pairwise_distances(logits, id_sim_mean, metric=kl)  # Batch * 100
    train_images_targets = np.eye(id_sim_mean.shape[0])  # id_sim_lables
    kl_logits = (-scores) @ train_images_targets

    return kl_logits

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    model_path = cfg.model_path

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cuda").train()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cuda")

    # model = clip.build_model(state_dict or model.state_dict())
    model = clip.build_model(state_dict or model.state_dict()).to("cpu")

    return model


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, drop_path=0.):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, drop_path_rate=0.):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]  # stochastic depth decay rule
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask, dpr[i]) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        assert k.shape == v.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads)

        attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale

        attn = attn.softmax(dim=-1)

        x = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(
            self,
            d_model,  # 256
            nhead,  # 4
            dropout=0.1,
    ):
        super().__init__()
        self.self_attn = Attention(d_model, nhead, proj_drop=dropout)
        self.cross_attn = Attention(d_model, nhead, proj_drop=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, mem):  # men [B,L,256] x[B, 49, 256]
        q = k = v = self.norm1(x)  # [B, L, 256]
        x = x + self.self_attn(q, k, v)  # [B, L, 256]
        q = self.norm2(x)  # [B, L, 256]
        x = x + self.cross_attn(q, mem, mem)  # [B, L, 256]
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x  # [B, 49, 256]

class PositionWiseFeedForward(nn.Module):
    def __init__(self, embed_dim, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, d_ff)
        self.fc2 = nn.Linear(d_ff, embed_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class AdapterLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, d_ff, dropout):
        super(AdapterLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.feed_forward = PositionWiseFeedForward(embed_dim, d_ff)
        self.norm1 = nn.LayerNorm(embed_dim)
        # self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
         
    def forward(self, x):
        self_attn_output, _ = self.self_attn(x, x, x)
        # cross_attn_output, _ = self.cross_attn(x, y, y)
        # x = self.norm1(x + self.dropout(self_attn_output) + self.dropout(cross_attn_output))
        x = self.norm1(x + self.dropout(self_attn_output))
        # x = self.norm2(x + self.dropout(cross_attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class CustomAdapter(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, d_ff=2048, dropout=0.1):
        super(CustomAdapter, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.image_adapter = AdapterLayer(self.embed_dim, self.num_heads, self.d_ff, self.dropout)
        self.text_adapter = AdapterLayer(self.embed_dim, self.num_heads, self.d_ff, self.dropout)
     
    def forward(self, x, y):
        return self.image_adapter(x), self.text_adapter(y)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class DPM_Block(nn.Module):
    def __init__(self, 
                 text_features,
                 input_dim, 
                 num_classes):  
        super().__init__()
        self.softmax = nn.Softmax(-1)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.pre_project_s = Proj2()
        self.pre_project_t = Proj1()
        self.pre_project_vv = Proj1()
        self.scale = input_dim ** -0.5
        self.logit_scale = nn.Parameter(torch.ones([]) * 30., requires_grad=False)
        # self.CosClassifier = CosineClassifier(num_classes, input_dim)
        self.vis_gamma_p = nn.Parameter(torch.ones([]) * 0.99)  # 1e-3)  # for updating visual embedding diff
        self.vis_gamma_n = nn.Parameter(torch.ones([]) * 0.99)  # 1e-3)  # for updating visual embedding diff
        self.visual_prototype = nn.Parameter(text_features.clone().detach())  # , requires_grad=False)

    def forward(self, Fs, Ft, Fv, label):
        L, D = Ft.shape
        B, _, C = Fs.shape
        Fs = self.pre_project_s(Fs)
        A_weight = F.conv1d(Fs.permute(0, 2, 1), Ft[:, :, None])  
        A_weight1 = F.softmax(A_weight, dim=-1)  
        feat_v_a = A_weight1 @ Fs  
        Fv = self.pre_project_vv(Fv)
        Fv = Fv.unsqueeze(1)
        Fv = Fv.expand(-1, L, -1)
        feat_v = self.vis_gamma_p * feat_v_a + Fv  
        A_weightv = F.conv1d(Fs.permute(0, 2, 1), self.visual_prototype[:, :, None])  
        A_weight1v = F.softmax(A_weightv, dim=-1)  
        feat_v_av = A_weight1v @ Fs  
        feat_vv = self.vis_gamma_n * feat_v_av + Fv  
        Ft = F.normalize(Ft, dim=-1, p=2)  
        Fv = F.normalize(Fv, dim=-1, p=2)  
        feat_v = F.normalize(feat_v, dim=-1, p=2)  
        feat_vv = F.normalize(feat_vv, dim=-1, p=2)  
        visual_prototype = F.normalize(self.visual_prototype, dim=-1, p=2)  
        logits1 = torch.mul(Fv, Ft).sum(-1)
        logits2 = torch.mul(feat_v, Ft).sum(-1)
        logits3 = torch.mul(feat_vv, visual_prototype).sum(-1)
        with torch.no_grad():
            class_count = torch.bincount(label, minlength=L)
            class_sum = Fv[:, 0, :].new_zeros(L, C)
            class_sum.index_add_(0, label, Fv[:, 0, :])
            safe_class_count = class_count.float().unsqueeze(1).clamp_min(1e-8)
            class_mean = class_sum / safe_class_count
            mask = class_count > 0
            new_visual_prototype = 0.99 * self.visual_prototype + 0.01 * class_mean
            updated_visual_prototype = self.visual_prototype.clone()
            updated_visual_prototype[mask] = new_visual_prototype[mask]
            self.visual_prototype.data = updated_visual_prototype
        return logits1, logits2, logits3

    def evaluate(self, Fs, Ft, Fv):
        L, D = Ft.shape
        B, _, C = Fs.shape
        Fs = self.pre_project_s(Fs)
        A_weight = F.conv1d(Fs.permute(0, 2, 1), Ft[:, :, None])  
        A_weight1 = F.softmax(A_weight, dim=-1)  
        feat_v_a = A_weight1 @ Fs  
        Fv = self.pre_project_vv(Fv)
        Fv = Fv.unsqueeze(1)
        Fv = Fv.expand(-1, L, -1)
        feat_v = self.vis_gamma_p * feat_v_a + Fv  
        A_weightv = F.conv1d(Fs.permute(0, 2, 1), self.visual_prototype[:, :, None])  
        A_weight1v = F.softmax(A_weightv, dim=-1)  
        feat_v_av = A_weight1v @ Fs  
        feat_vv = self.vis_gamma_n * feat_v_av + Fv  
        visual_prototype = F.normalize(self.visual_prototype, dim=-1, p=2)  
        Ft = F.normalize(Ft, dim=-1, p=2)  
        Fv = F.normalize(Fv, dim=-1, p=2)  
        feat_v = F.normalize(feat_v, dim=-1, p=2)  
        feat_vv = F.normalize(feat_vv, dim=-1, p=2)  
        logits1 = self.logit_scale * torch.mul(Fv, Ft).sum(-1)
        logits2 = self.logit_scale * torch.mul(feat_v, Ft).sum(-1)
        logits3 = self.logit_scale * torch.mul(feat_vv, visual_prototype).sum(-1)
        return logits1, logits2, logits3

    def get_feature(self, Fs, Ft, Fv):
        L, D = Ft.shape
        B, _, C = Fs.shape
        Fs = self.pre_project_s(Fs)
        A_weight = F.conv1d(Fs.permute(0, 2, 1), Ft[:, :, None])  
        A_weight1 = F.softmax(A_weight, dim=-1)  
        feat_v_a = A_weight1 @ Fs  
        Fv = self.pre_project_vv(Fv)
        Fv = Fv.unsqueeze(1)
        Fv = Fv.expand(-1, L, -1)
        feat_v = self.vis_gamma_p * feat_v_a + Fv  
        A_weightv = F.conv1d(Fs.permute(0, 2, 1), self.visual_prototype[:, :, None])  
        A_weight1v = F.softmax(A_weightv, dim=-1)  
        feat_v_av = A_weight1v @ Fs  
        feat_vv = self.vis_gamma_n * feat_v_av + Fv  
        visual_prototype = F.normalize(self.visual_prototype, dim=-1, p=2)  
        Ft = F.normalize(Ft, dim=-1, p=2)
        Fv = F.normalize(Fv, dim=-1, p=2)
        feat_v = F.normalize(feat_v, dim=-1, p=2)
        feat_vv = F.normalize(feat_vv, dim=-1, p=2)  
       
        return Fv, Fs, feat_v, feat_vv

    def get_heatmap(self, Fs, Ft, Fv):
        L, D = Ft.shape
        B, _, C = Fs.shape
        Fs = self.pre_project_s(Fs)
        A_weight = F.conv1d(Fs.permute(0, 2, 1), Ft[:, :, None])
        A_weight1 = F.softmax(A_weight, dim=-1)
        feat_v_a = A_weight1 @ Fs
        Fv = self.pre_project_vv(Fv)
        Fv = Fv.unsqueeze(1)
        Fv = Fv.expand(-1, L, -1)
        feat_v = self.vis_gamma_p * feat_v_a + Fv 
        A_weightv = F.conv1d(Fs.permute(0, 2, 1), self.visual_prototype[:, :, None])
        A_weight1v = F.softmax(A_weightv, dim=-1) 
        feat_v_av = A_weight1v @ Fs
        feat_vv = self.vis_gamma_n * feat_v_av + Fv
        
        return A_weight1, A_weight1v

class DPM_T(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.classnames = classnames
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        for _, param in clip_model.named_parameters():
            param.requires_grad = False
        with torch.no_grad():
            temp = CUSTOM_TEMPLATES['cifar100']
            prompts = [temp.format(c.replace("_", " ")) for c in self.classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts])
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.dpmt = DPM_Block(text_features, 512, len(classnames))

    def forward(self, image, label):
        image_features, local_features = self.image_encoder(image.type(self.dtype))  
        prompts, tokenized_prompts = self.prompt_learner()  
        text_features = self.text_encoder(prompts, tokenized_prompts)  
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  
        local_features = local_features / local_features.norm(dim=-1, keepdim=True)  
        logits1, logits2, logits3 = self.dpmt(Fs=local_features, 
                                              Ft=text_features, 
                                              Fv=image_features,
                                              label=label)  
        return logits1, logits2, logits3

    def evaluate(self, image):
        image_features, local_features = self.image_encoder(image.type(self.dtype)) 
        prompts, tokenized_prompts = self.prompt_learner() 
        text_features = self.text_encoder(prompts, tokenized_prompts)  
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  
        local_features = local_features / local_features.norm(dim=-1, keepdim=True)  
        logits1, logits2, logits3 = self.dpmt.evaluate(Fs=local_features, 
                                                       Ft=text_features,
                                                       Fv=image_features)  
        return logits1, logits2, logits3

    def get_feature(self, image, cls_id=None):
        image_features, local_features = self.image_encoder(image.type(self.dtype)) 
        prompts, tokenized_prompts = self.prompt_learner()  
        text_features = self.text_encoder(prompts, tokenized_prompts)  
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  
        local_features = local_features / local_features.norm(dim=-1, keepdim=True)  
        Fv, Fs, feat_v, feat_vv = self.dpmt.get_feature(Fs=local_features, 
                                                        Ft=text_features,
                                                        Fv=image_features)  
        return Fv, Fs, feat_v, feat_vv, image_features

    def get_heatmap(self, image, cls_id=None):
        image_features, local_features = self.image_encoder(image.type(self.dtype))
        prompts, tokenized_prompts = self.prompt_learner()  
        text_features = self.text_encoder(prompts, tokenized_prompts) 
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)  
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  
        local_features = local_features / local_features.norm(dim=-1, keepdim=True)  
        A_weight1, A_weight1v = self.dpmt.get_heatmap(Fs=local_features,
                                                      Ft=text_features,
                                                      Fv=image_features)
        return A_weight1, A_weight1v

    def get_prototype(self):
        prompts, tokenized_prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)  
        textual_prototype = text_features / text_features.norm(dim=-1, keepdim=True) 
        visual_prototype = self.dpmt.visual_prototype
        visual_prototype = F.normalize(visual_prototype, dim=-1, p=2)
        return textual_prototype, visual_prototype


@TRAINER_REGISTRY.register()
class DPM_OOD(TrainerX):  # need to be trained

    def check_cfg(self, cfg):
        assert cfg.TRAINER.DPM.PREC in ["fp16", "fp32"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.num_classes = self.dm.num_classes
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.DPM.PREC == "fp32":
            clip_model.float()
        print("Building custom CLIP")
        self.model = DPM_T(cfg, classnames, clip_model)  # (cfg, classnames, clip_model)
        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "dpmt" not in name:
                if "prompt_learner" not in name:
                    param.requires_grad_(False)
        self.model.to(self.device)
        self.optim = build_optimizer(self.model.dpmt, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("dpmt", self.model.dpmt, self.optim, self.sched)
        self.optim_prompt = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched_prompt = build_lr_scheduler(self.optim_prompt, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim_prompt, self.sched_prompt)
        self.scaler = None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model.image_encoder = nn.DataParallel(self.model.image_encoder)
            self.model.text_encoder = nn.DataParallel(self.model.text_encoder)
        self.celoss = nn.CrossEntropyLoss()
        self.celoss.to(self.device)
        self.img_match = torch.zeros(len(classnames), len(classnames)).to(self.device)
        self.loss1 = cfg.loss1
        self.loss2 = cfg.loss2
        self.loss3 = cfg.loss3
        self.loss1_ood = cfg.loss1_ood
        self.loss2_ood = cfg.loss2_ood
        self.loss3_ood = cfg.loss3_ood
        self.evaluator = None
        self.evaluator1 = build_evaluator(cfg, lab2cname=self.lab2cname)
        self.evaluator2 = build_evaluator(cfg, lab2cname=self.lab2cname)
        self.evaluator3 = build_evaluator(cfg, lab2cname=self.lab2cname)
        self.model_path = cfg.model_path

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        custom_tfm_train += [tfm_train]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}
        self.dm = dm

    def save_ood_images(self, image, label):
        """
        保存 OOD 图片到指定目录，并以标签命名。
        Args:
            image (torch.Tensor): OOD 图片张量，形状为 [C, H, W]。
            label (int): 图片的标签。
        """
        # 定义归一化的均值和标准差（根据你的数据集设置）
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        def denormalize(image, mean, std):
            """
            对归一化的图像进行反归一化。
            Args:
                image (torch.Tensor): 归一化的图像张量，形状为 [C, H, W]。
                mean (list): 归一化时使用的均值。
                std (list): 归一化时使用的标准差。
            Returns:
                torch.Tensor: 反归一化后的图像张量。
            """
            mean = torch.tensor(mean).view(-1, 1, 1).to(image.device)
            std = torch.tensor(std).view(-1, 1, 1).to(image.device)
            return image * std + mean

        # 对图像进行反归一化
        image = denormalize(image, mean, std)

        # 将张量裁剪到 [0, 1] 范围内，避免超出有效像素值范围
        image = torch.clamp(image, 0, 1)

        # 将张量转换为 PIL 图像
        transform = transforms.ToPILImage()
        image = transform(image.cpu())

        # 构造保存路径
        save_path = os.path.join(self.output_dir, f"label_{label}.png")

        # 保存图片
        image.save(save_path)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        ood_images = randomcut_ood(image)  # [B, C, H, W]
        logits1, logits2, logits3 = self.model(image, label)  
        logits1_ood, logits2_ood, logits3_ood = self.model.evaluate(ood_images)
        # 筛选机制：根据 logits2_ood 的最大值排序，取后 10%
        max_logits2_ood = torch.max(torch.softmax(logits2_ood, dim=-1), dim=-1).values  # 获取 logits2_ood 的最大值
        num_samples = int(ood_images.size(0) * 0.1)  # 计算后 10% 的样本数量
        _, selected_indices = torch.topk(-max_logits2_ood, num_samples)  # 取最大值最小的样本索引（负号用于升序排序）
        ood_images = ood_images[selected_indices]  # 筛选出最不像的 OOD 图像
        ood_label = label[selected_indices]  # 筛选出对应的标签

        # self.save_ood_images(ood_images[0], ood_label[0].item())  # 保存 OOD 图像

        logits1_ood, logits2_ood, logits3_ood = logits1_ood[selected_indices], logits2_ood[selected_indices], logits3_ood[selected_indices]

        predictions_ood1 = torch.softmax(logits1_ood, dim=-1)
        predictions_ood2 = torch.softmax(logits2_ood, dim=-1)
        predictions_ood3 = torch.softmax(logits3_ood, dim=-1)

        loss_1 = self.celoss(20 * logits1, label)  
        loss_2 = self.celoss(20 * logits2, label)  
        loss_3 = self.celoss(20 * logits3, label)

        loss_1_ood = -torch.mean(torch.sum(predictions_ood1 * torch.log(predictions_ood1 + 1e-6), dim=-1))  
        loss_2_ood = -torch.mean(torch.sum(predictions_ood2 * torch.log(predictions_ood2 + 1e-6), dim=-1))
        loss_3_ood = -torch.mean(torch.sum(predictions_ood3 * torch.log(predictions_ood3 + 1e-6), dim=-1))

        loss = self.loss1*loss_1 + self.loss2*loss_2 + self.loss3*loss_3 \
                - self.loss1_ood*loss_1_ood - self.loss2_ood*loss_2_ood - self.loss3_ood*loss_3_ood
        self.model_backward_and_update(loss)
        loss_summary = {
            "loss": loss.item(),
            'loss_1': self.loss1*loss_1.item(),
            'loss_2': self.loss2*loss_2.item(),
            'loss_3': self.loss3*loss_3.item(),
            'loss_ood1': -self.loss1_ood*loss_1_ood.item(),
            'loss_ood2': -self.loss2_ood*loss_2_ood.item(),
            'loss_ood3': -self.loss3_ood*loss_3_ood.item(),
            "logit1_acc": compute_accuracy(logits1, label)[0].item(),
            "logit2_acc": compute_accuracy(logits2, label)[0].item(),
            "logit3_acc": compute_accuracy(logits3, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        return input.to(self.device), label.to(self.device)

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        # self.set_model_mode("eval")
        self.model.eval()
        self.evaluator1.reset()
        self.evaluator2.reset()
        self.evaluator3.reset()
        logit1_top1_correct = 0
        logit2_top1_correct = 0
        logit3_top1_correct = 0
        total_samples = 0
        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input, label)
            logit1_output = output[0]
            logit2_output = output[1]
            logit3_output = output[2]
            self.evaluator1.process(logit1_output, label)
            self.evaluator2.process(logit2_output, label)
            self.evaluator3.process(logit3_output, label)
            # 计算 logit1_output 的准确率
            _, logit1_predicted = torch.max(logit1_output, 1)
            logit1_top1_correct += (logit1_predicted == label).sum().item()
            # 计算 logit2_output 的准确率
            _, logit2_predicted = torch.max(logit2_output, 1)
            logit2_top1_correct += (logit2_predicted == label).sum().item()

            # 计算 logit3_output 的准确率
            _, logit3_predicted = torch.max(logit3_output, 1)
            logit3_top1_correct += (logit3_predicted == label).sum().item()
            
            total_samples += label.size(0)

        print("Evaluation with logit1")
        logit1_acc = logit1_top1_correct / total_samples * 100
        print('logit1_acc',logit1_acc)

        print("Evaluation with logit2")     
        logit2_acc = logit2_top1_correct / total_samples * 100
        print('logit2_acc',logit2_acc)

        print("Evaluation with logit3")     
        logit3_acc = logit3_top1_correct / total_samples * 100
        print('logit3_acc',logit3_acc)
        
        results1 = self.evaluator1.evaluate()
        results2 = self.evaluator2.evaluate()
        results3 = self.evaluator3.evaluate()

        for k, v in results1.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)
        for k, v in results2.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)
        for k, v in results3.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)
        # 合并所有的 results
        all_results = {
            "logit1_acc": logit1_acc,
            "logit2_acc": logit2_acc,
            "logit3_acc": logit3_acc,
            **results1,
            **results2,
            **results3
        }

        return all_results

    def model_inference(self, input, label=None):
        return self.model.evaluate(input)

    def get_measures(self, _pos, _neg, recall_level=0.95):
        pos = np.array(_pos[:]).reshape((-1, 1))
        neg = np.array(_neg[:]).reshape((-1, 1))
        examples = np.squeeze(np.vstack((pos, neg)))
        labels = np.zeros(len(examples), dtype=np.int32)
        labels[:len(pos)] += 1

        auroc = sk.roc_auc_score(labels, examples)
        aupr = sk.average_precision_score(labels, examples)
        fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

        return auroc, aupr, fpr

    def test_ood(self, train_loader, id_loader, ood_loader_list, out_datasets, t=100):
        self.model.eval()
        print('getting ID train cls_sim_mean......')
        total_logit1, total_logit2, total_logit3, labels = [], [], [], []

        # 获取 ID 数据的 logits 和标签
        with torch.no_grad():
            for batch_idx, (images, label) in tqdm(enumerate(train_loader)):
                images = images.cuda()
                logits1, logits2, logits3 = self.model.evaluate(images)
                labels.append(label)
                total_logit1.append(logits1)
                total_logit2.append(logits2)
                total_logit3.append(logits3)

        total_logit1 = F.softmax(torch.cat(total_logit1, dim=0).cpu() / 5, dim=-1).detach().numpy()
        total_logit2 = F.softmax(torch.cat(total_logit2, dim=0).cpu() / 5, dim=-1).detach().numpy()
        total_logit3 = F.softmax(torch.cat(total_logit3, dim=0).cpu() / 5, dim=-1).detach().numpy()
        labels = torch.cat(labels, dim=0).cpu()
        num_classes = self.num_classes

        # 计算类均值
        with torch.no_grad():
            class_sim_mean1 = np.array([total_logit1[(labels == i)].mean(axis=0) for i in range(num_classes)])
            class_sim_mean2 = np.array([total_logit2[(labels == i)].mean(axis=0) for i in range(num_classes)])
            class_sim_mean3 = np.array([total_logit3[(labels == i)].mean(axis=0) for i in range(num_classes)])

        # 获取 ID 特征
        in_fea1_mls, in_fea2_mls, in_fea3_mls, in_kl1, in_kl2, in_kl3 = self.get_feature(class_sim_mean1, class_sim_mean2, class_sim_mean3, id_loader)
        in_fea1_mcm = np.array([softmax(item / t) for item in in_fea1_mls])
        in_fea2_mcm = np.array([softmax(item / t) for item in in_fea2_mls])
        in_fea3_mcm = np.array([softmax(item / t) for item in in_fea3_mls])

        # 定义辅助函数
        def scale_features(kl_id_norm, kl_ood_norm, in_fea, out_fea):
            x_max, x_min = max(kl_id_norm.max(), kl_ood_norm.max()), min(kl_id_norm.min(), kl_ood_norm.min())
            target_max, target_min = max(in_fea.max(), out_fea.max()), min(in_fea.min(), out_fea.min())
            kl_id_norm_scaled = (kl_id_norm - x_min) / (x_max - x_min) * (target_max - target_min) + target_min
            kl_ood_norm_scaled = (kl_ood_norm - x_min) / (x_max - x_min) * (target_max - target_min) + target_min
            return kl_id_norm_scaled, kl_ood_norm_scaled

        def compute_scores(alpha, beta, kl_id_norm, kl_ood_norm, in_fea, out_fea):
            id_score = beta * kl_id_norm + alpha * np.max(in_fea, axis=1)
            ood_score = beta * kl_ood_norm + alpha * np.max(out_fea, axis=1)
            aur, _, fpr = self.get_measures(id_score, ood_score)
            return aur, fpr, aur - fpr

        # 初始化统计变量
        tm_scores = {fea: {ood_name: [] for ood_name in out_datasets} for fea in ['fea1_mls', 'fea2_mls', 'fea3_mls', 'fea1_mcm', 'fea2_mcm', 'fea3_mcm']}
        vm_scores = {fea: {ood_name: [] for ood_name in out_datasets} for fea in ['fea1_mls', 'fea2_mls', 'fea3_mls', 'fea1_mcm', 'fea2_mcm', 'fea3_mcm']}
        dpm_best_scores = {fea: {ood_name: {'alpha': None, 'beta': None, 'aur': -float('inf'), 'fpr': float('inf')} for ood_name in out_datasets} for fea in ['fea1_mls', 'fea2_mls', 'fea3_mls', 'fea1_mcm', 'fea2_mcm', 'fea3_mcm']}
        dpm_global_scores = {fea: {ood_name: [] for ood_name in out_datasets} for fea in ['fea1_mls', 'fea2_mls', 'fea3_mls', 'fea1_mcm', 'fea2_mcm', 'fea3_mcm']}
        dpm_global_best_scores = {fea: {'alpha': None, 'beta': None, 'aur': -float('inf'), 'fpr': float('inf')} for fea in ['fea1_mls', 'fea2_mls', 'fea3_mls', 'fea1_mcm', 'fea2_mcm', 'fea3_mcm']}

        # 遍历 OOD 数据集
        for i, ood_name in enumerate(out_datasets):
            print(f'******* {ood_name} *******')
            ood_loader = ood_loader_list[i]
            out_fea1_mls, out_fea2_mls, out_fea3_mls, out_kl1, out_kl2, out_kl3 = self.get_feature(class_sim_mean1, class_sim_mean2, class_sim_mean3, ood_loader)
            out_fea1_mcm = np.array([softmax(item / t) for item in out_fea1_mls])
            out_fea2_mcm = np.array([softmax(item / t) for item in out_fea2_mls])
            out_fea3_mcm = np.array([softmax(item / t) for item in out_fea3_mls])

            # 缩放特征
            kl_id_norm_mls1, kl_ood_norm_mls1 = scale_features(np.min(in_kl1, axis=1), np.min(out_kl1, axis=1), in_fea1_mls, out_fea1_mls)
            kl_id_norm_mcm1, kl_ood_norm_mcm1 = scale_features(np.min(in_kl1, axis=1), np.min(out_kl1, axis=1), in_fea1_mcm, out_fea1_mcm)
            kl_id_norm_mls2, kl_ood_norm_mls2 = scale_features(np.min(in_kl2, axis=1), np.min(out_kl2, axis=1), in_fea2_mls, out_fea2_mls)
            kl_id_norm_mcm2, kl_ood_norm_mcm2 = scale_features(np.min(in_kl2, axis=1), np.min(out_kl2, axis=1), in_fea2_mcm, out_fea2_mcm)
            kl_id_norm_mls3, kl_ood_norm_mls3 = scale_features(np.min(in_kl3, axis=1), np.min(out_kl3, axis=1), in_fea3_mls, out_fea3_mls)
            kl_id_norm_mcm3, kl_ood_norm_mcm3 = scale_features(np.min(in_kl3, axis=1), np.min(out_kl3, axis=1), in_fea3_mcm, out_fea3_mcm)

            # 遍历 alpha 和 beta
            computed_ratios = set()
            for alpha in tqdm(range(0, 101, 1), desc="Alpha Loop"):
                alpha = round(alpha / 10, 1)
                for beta in tqdm(range(0, 101, 1), desc="Beta Loop", leave=False):
                    beta = round(beta / 10 - 10, 1)
                    if alpha == 0 and beta != -10:
                        break
                    ratio = alpha / beta if beta != 0 else float('inf')
                    if ratio in computed_ratios:
                        continue
                    computed_ratios.add(ratio)

                    # 计算各个特征类型的分数
                    for fea, kl_id_norm, kl_ood_norm, in_fea, out_fea in [
                        ('fea1_mls', kl_id_norm_mls1, kl_ood_norm_mls1, in_fea1_mls, out_fea1_mls),
                        ('fea2_mls', kl_id_norm_mls2, kl_ood_norm_mls2, in_fea2_mls, out_fea2_mls),
                        ('fea3_mls', kl_id_norm_mls3, kl_ood_norm_mls3, in_fea3_mls, out_fea3_mls),
                        ('fea1_mcm', kl_id_norm_mcm1, kl_ood_norm_mcm1, in_fea1_mcm, out_fea1_mcm),
                        ('fea2_mcm', kl_id_norm_mcm2, kl_ood_norm_mcm2, in_fea2_mcm, out_fea2_mcm),
                        ('fea3_mcm', kl_id_norm_mcm3, kl_ood_norm_mcm3, in_fea3_mcm, out_fea3_mcm)
                    ]:
                        aur, fpr, score = compute_scores(alpha, beta, kl_id_norm, kl_ood_norm, in_fea, out_fea)

                        # TM 和 VM 的性能
                        if alpha == 0.1 and beta == 0:
                            tm_scores[fea][ood_name] = {'aur': aur, 'fpr': fpr}
                        if alpha == 0 and beta == -10:
                            vm_scores[fea][ood_name] = {'aur': aur, 'fpr': fpr}
                        
                        # 记录DPM性能
                        dpm_global_scores[fea][ood_name].append({'alpha': alpha, 'beta': beta, 'aur': aur, 'fpr': fpr})
                        if score > dpm_best_scores[fea][ood_name].get('aur', -float('inf')) - dpm_best_scores[fea][ood_name].get('fpr', float('inf')):
                            dpm_best_scores[fea][ood_name] = {'alpha': alpha, 'beta': beta, 'aur': aur, 'fpr': fpr}
        
        # 遍历所有特征
        for fea in ['fea1_mls', 'fea2_mls', 'fea3_mls', 'fea1_mcm', 'fea2_mcm', 'fea3_mcm']:
            # 遍历所有参数组合 (alpha, beta)
            for alpha, beta in {(result['alpha'], result['beta']) for ood_name in out_datasets for result in dpm_global_scores[fea][ood_name]}:
                # 计算DPM的平均性能
                # 筛选符合条件的记录
                filtered_records = [
                    r for ood_name in out_datasets for r in dpm_global_scores[fea][ood_name]
                    if r['alpha'] == alpha and r['beta'] == beta
                ]

                # 如果没有符合条件的记录，跳过当前组合
                if not filtered_records:
                    continue

                # 计算平均性能
                avg_aur = np.mean([r['aur'] for r in filtered_records])
                avg_fpr = np.mean([r['fpr'] for r in filtered_records])

                # 更新全局最优参数组合
                if avg_aur - avg_fpr > dpm_global_best_scores[fea]['aur'] - dpm_global_best_scores[fea]['fpr']:
                    dpm_global_best_scores[fea] = {'alpha': alpha, 'beta': beta, 'aur': avg_aur, 'fpr': avg_fpr}
            
            # 记录每个数据集的 Fixed 性能
            for ood_name in out_datasets:
                # 使用全局最优参数
                best_alpha = dpm_global_best_scores[fea]['alpha']
                best_beta = dpm_global_best_scores[fea]['beta']
                fixed_aur = next((r['aur'] for r in dpm_global_scores[fea][ood_name] if r['alpha'] == best_alpha and r['beta'] == best_beta))
                fixed_fpr = next((r['fpr'] for r in dpm_global_scores[fea][ood_name] if r['alpha'] == best_alpha and r['beta'] == best_beta))
                dpm_best_scores[fea][ood_name]['fixed'] = {'aur': fixed_aur, 'fpr': fixed_fpr}
            
        # 打印单个数据集的结果
        for ood_name in out_datasets:
            print(f"\n******* Results for {ood_name} *******")
            for fea in ['fea1_mls', 'fea2_mls', 'fea3_mls', 'fea1_mcm', 'fea2_mcm', 'fea3_mcm']:
                print(f"\nFeature: {fea}")
                # 打印 TM 和 VM 的性能
                print(f"TM: AUR = {tm_scores[fea][ood_name]['aur']:.4f}, FPR = {tm_scores[fea][ood_name]['fpr']:.4f}")
                print(f"VM: AUR = {vm_scores[fea][ood_name]['aur']:.4f}, FPR = {vm_scores[fea][ood_name]['fpr']:.4f}")
                # 打印 DPM 的 Unfixed 和 Fixed 性能
                print(f"DPM (Unfixed): alpha = {dpm_best_scores[fea][ood_name]['alpha']:.1f}, beta = {dpm_best_scores[fea][ood_name]['beta']:.1f}, AUR = {dpm_best_scores[fea][ood_name]['aur']:.4f}, FPR = {dpm_best_scores[fea][ood_name]['fpr']:.4f}")
                print(f"DPM (Fixed): alpha = {dpm_global_best_scores[fea]['alpha']:.1f}, beta = {dpm_global_best_scores[fea]['beta']:.1f}, AUR = {dpm_best_scores[fea][ood_name]['fixed']['aur']:.4f}, FPR = {dpm_best_scores[fea][ood_name]['fixed']['fpr']:.4f}")

        # 打印全局最佳结果
        print("\n******* Global Best Results Across All Datasets *******")
        for fea in ['fea1_mls', 'fea2_mls', 'fea3_mls', 'fea1_mcm', 'fea2_mcm', 'fea3_mcm']:
            print(f"\nFeature: {fea}")
            # 打印 TM 和 VM 的平均性能
            avg_tm_aur = np.mean([tm_scores[fea][ood_name]['aur'] for ood_name in out_datasets])
            avg_tm_fpr = np.mean([tm_scores[fea][ood_name]['fpr'] for ood_name in out_datasets])
            avg_vm_aur = np.mean([vm_scores[fea][ood_name]['aur'] for ood_name in out_datasets])
            avg_vm_fpr = np.mean([vm_scores[fea][ood_name]['fpr'] for ood_name in out_datasets])
            print(f"TM (Average): AUR = {avg_tm_aur:.4f}, FPR = {avg_tm_fpr:.4f}")
            print(f"VM (Average): AUR = {avg_vm_aur:.4f}, FPR = {avg_vm_fpr:.4f}")
            
            # 打印 DPM 的 Unfixed 平均性能
            avg_dpm_aur = np.mean([dpm_best_scores[fea][ood_name]['aur'] for ood_name in out_datasets])
            avg_dpm_fpr = np.mean([dpm_best_scores[fea][ood_name]['fpr'] for ood_name in out_datasets])
            print(f"DPM (Unfixed, Average): AUR = {avg_dpm_aur:.4f}, FPR = {avg_dpm_fpr:.4f}")
            
            # 打印 DPM 的 Fixed 性能
            print(f"DPM (Fixed): alpha = {dpm_global_best_scores[fea]['alpha']:.1f}, beta = {dpm_global_best_scores[fea]['beta']:.1f}, AUR = {dpm_global_best_scores[fea]['aur']:.4f}, FPR = {dpm_global_best_scores[fea]['fpr']:.4f}")

    def get_feature(self, class_sim_mean1, class_sim_mean2, class_sim_mean3, loader):
        total_logit1 = []
        total_logit2 = []
        total_logit3 = []
        kl_div1 = []
        kl_div2 = []
        kl_div3 = []
        bs = 200
        print('getting testset logits......')
        with torch.no_grad():
            for batch_idx, (images, label) in tqdm(enumerate(loader)):
                images = images.cuda()
                logits1, logits2, logits3 = self.model.evaluate(images)  # self.model.module.evaluate(images)
                total_logit1.append(logits1)
                total_logit2.append(logits2)
                total_logit3.append(logits3)
        total_logit1 = torch.cat(total_logit1, dim=0)
        total_sim1 = F.softmax(torch.tensor(total_logit1 / 5).float(), dim=-1)
        # total_sim1 = F.softmax(torch.tensor(total_logit1 / 10).float(), dim=-1)
        total_logit2 = torch.cat(total_logit2, dim=0)
        total_sim2 = F.softmax(torch.tensor(total_logit2 / 5).float(), dim=-1)
        # total_sim2 = F.softmax(torch.tensor(total_logit2 / 10).float(), dim=-1)
        total_logit3 = torch.cat(total_logit3, dim=0)
        total_sim3 = F.softmax(torch.tensor(total_logit3 / 5).float(), dim=-1)
        # total_sim3 = F.softmax(torch.tensor(total_logit3 / 10).float(), dim=-1)
        with torch.no_grad():
            print('computing kl.......',)
            for i in tqdm(range(total_sim2.shape[0] // bs)):
                cur_logits1 = total_sim1[i * bs: (i + 1) * bs]
                cur_logits2 = total_sim2[i * bs: (i + 1) * bs]
                cur_logits3 = total_sim3[i * bs: (i + 1) * bs]
                if i == total_sim2.shape[0] // bs - 1:
                    cur_logits1 = total_sim1[i * bs:]
                    cur_logits2 = total_sim2[i * bs:]
                    cur_logits3 = total_sim3[i * bs:]
                output1 = cur_logits1.data.cpu().numpy()
                output2 = cur_logits2.data.cpu().numpy()
                output3 = cur_logits3.data.cpu().numpy()
                kl_div1.append(get_kl(class_sim_mean1, output1))
                kl_div2.append(get_kl(class_sim_mean2, output2))
                kl_div3.append(get_kl(class_sim_mean3, output3))
        kl_div1 = np.concatenate(kl_div1, axis=0)
        kl_div2 = np.concatenate(kl_div2, axis=0)
        kl_div3 = np.concatenate(kl_div3, axis=0)
        return total_logit1.detach().cpu().numpy(), total_logit2.detach().cpu().numpy(), total_logit3.detach().cpu().numpy(), kl_div1, kl_div2, kl_div3

    def draw_tsne(self, id_loader, ood_loader_list, out_datasets, output_dir='./tsne_results', 
      perplexity=30, n_iter=1000, random_state=1555):
        """
        为ID和OOD数据创建t-SNE可视化，同时绘制feat_v和clip_feature两种特征的图
        
        参数:
            id_loader: ID数据的DataLoader
            ood_loader_list: OOD数据加载器列表
            out_datasets: OOD数据集名称列表，与ood_loader_list对应
            output_dir: 输出目录
            perplexity: t-SNE的perplexity参数
            n_iter: t-SNE的迭代次数
            random_state: 随机种子
        """
        self.model.eval()
        os.makedirs(output_dir, exist_ok=True)
        
        # 定义自定义颜色列表
        custom_colors = ['#3498DB', '#E74C3C', '#2ECC71', '#9B59B6', '#F39C12']
        
        # 初始化特征字典
        features_dict = {
            'feat_v': {'id': [], 'ood': []},
            'clip_feature': {'id': [], 'ood': []}
        }
        
        # 1. 收集ID数据的特征
        print("获取ID数据特征...")
        
        with torch.no_grad():
            for batch_idx, (images, labels) in tqdm(enumerate(id_loader)):
                images = images.cuda()
                Fv, Fs, feat_v, feat_vv, clip_feature = self.model.get_feature(images)
                # 存储不同类型的特征
                features_dict['feat_v']['id'].append(feat_v.mean(dim=1).cpu())
                features_dict['clip_feature']['id'].append(clip_feature.cpu())
        
        # 处理收集的ID特征
        for feat_type in features_dict:
            features_dict[feat_type]['id'] = torch.cat(features_dict[feat_type]['id'], dim=0).numpy()
        
        # 2. 收集所有OOD数据集的特征和数据源信息
        all_ood_sources = []  # 记录每个OOD样本来自哪个数据集
        ood_sample_counts = []  # 记录每个OOD数据集的样本数量
        
        for i, (ood_name, ood_loader) in enumerate(zip(out_datasets, ood_loader_list)):
            print(f"处理OOD数据集 {ood_name} ({i+1}/{len(out_datasets)})...")
            temp_features = {'feat_v': [], 'clip_feature': []}
            
            with torch.no_grad():
                for batch_idx, (images, _) in tqdm(enumerate(ood_loader)):
                    images = images.cuda()
                    Fv, Fs, feat_v, feat_vv, clip_feature = self.model.get_feature(images)
                    temp_features['feat_v'].append(feat_v.mean(dim=1).cpu())
                    temp_features['clip_feature'].append(clip_feature.cpu())
            
            # 处理收集的OOD特征
            for feat_type in features_dict:
                if temp_features[feat_type]:
                    feat_concat = torch.cat(temp_features[feat_type], dim=0).numpy()
                    features_dict[feat_type]['ood'].append(feat_concat)
                    if feat_type == 'feat_v':  # 只在一种特征类型中记录来源
                        ood_sample_counts.append(feat_concat.shape[0])
                        all_ood_sources.extend([i] * feat_concat.shape[0])
        
        # 3. 为每种特征类型创建t-SNE可视化
        for feat_type in features_dict:
            all_ood_feats = features_dict[feat_type]['ood']
            
            if all_ood_feats:
                all_ood_feats_concat = np.vstack(all_ood_feats)
                id_feats = features_dict[feat_type]['id']
                
                # 合并ID和所有OOD数据
                combined_feats = np.vstack([id_feats, all_ood_feats_concat])
                
                # 应用t-SNE降维
                print(f"对{feat_type}特征应用t-SNE降维...")
                tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_state)
                embeddings_2d = tsne.fit_transform(combined_feats)
                
                # 分离ID和OOD的嵌入
                id_count = id_feats.shape[0]
                id_embeddings = embeddings_2d[:id_count]
                ood_embeddings = embeddings_2d[id_count:]
                
                # 绘制t-SNE图
                plt.figure(figsize=(14, 12))
                
                # 将所有ID数据绘制为一个整体
                plt.scatter(
                    id_embeddings[:, 0], 
                    id_embeddings[:, 1], 
                    c=custom_colors[0],  
                    label='ID', 
                    alpha=0.7,
                    s=30, 
                    marker='o'
                )
                
                # 为每个OOD数据集绘制数据点
                start_idx = 0
                for j, ood_name in enumerate(out_datasets):
                    # 使用之前记录的样本数量确定当前数据集的范围
                    sample_count = ood_sample_counts[j]
                    end_idx = start_idx + sample_count
                    
                    plt.scatter(
                        ood_embeddings[start_idx:end_idx, 0], 
                        ood_embeddings[start_idx:end_idx, 1], 
                        c=custom_colors[j + 1],  
                        label=f'{ood_name}',
                        alpha=0.6, 
                        s=30, 
                        marker='x'
                    )
                    
                    start_idx = end_idx
                
                # 添加图例并放置在右上角
                plt.legend(loc='upper right', fontsize=12)
                
                # 移除坐标轴刻度
                plt.xticks([])
                plt.yticks([])
                
                plt.tight_layout()
                
                # 保存图像
                file_path = os.path.join(output_dir, f"tsne_{feat_type}.png")
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"t-SNE图像已保存至: {file_path}")
                
            else:
                print(f"警告: 没有OOD数据, 无法创建{feat_type}的可视化")
        
        print("t-SNE可视化完成!")

    def draw_tsne_prototype(self, output_dir='./tsne_results', perplexity=30, n_iter=1000, random_state=1555):
        """
        使用t-SNE可视化DPM模型中的文本原型和视觉原型
        
        参数:
            output_dir: 可视化结果保存的目录
            perplexity: t-SNE的perplexity参数
            n_iter: t-SNE的迭代次数
            random_state: 随机种子，用于复现结果
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        
        self.model.eval()
        os.makedirs(output_dir, exist_ok=True)

        # 获取模型中的文本原型和视觉原型
        with torch.no_grad():
            textual_prototype, visual_prototype = self.model.get_prototype()
        
        # 转换为numpy数组以便处理
        text_features_np = textual_prototype.cpu().numpy()
        visual_prototype_np = visual_prototype.cpu().numpy()
        
        # 将两种类型的原型合并在一起，以便应用t-SNE
        combined_features = np.vstack([text_features_np, visual_prototype_np])
        
        # 创建标签向量，0表示文本原型，1表示视觉原型
        n_classes = len(self.dm.dataset.classnames)
        prototype_types = np.array([0] * n_classes + [1] * n_classes)
        
        # 应用t-SNE降维
        print("应用t-SNE降维...")
        tsne = TSNE(n_components=2, perplexity=min(perplexity, n_classes-1), n_iter=n_iter, random_state=random_state)
        embeddings_2d = tsne.fit_transform(combined_features)
        
        # 分离文本和视觉原型的嵌入
        text_embeddings = embeddings_2d[:n_classes]
        visual_embeddings = embeddings_2d[n_classes:]
        
        # 绘制t-SNE图
        plt.figure(figsize=(14, 10))
        
        # 文本原型（蓝色圆形）
        plt.scatter(
            text_embeddings[:, 0], 
            text_embeddings[:, 1],
            color='#3498DB',  # 蓝色
            marker='o',
            s=80,
            alpha=0.7,
            label='Textual Prototype'
        )
        
        # 视觉原型（红色星形）
        plt.scatter(
            visual_embeddings[:, 0], 
            visual_embeddings[:, 1],
            color='#E74C3C',  # 红色
            marker='*',
            s=120,
            alpha=0.7,
            label='Visual Prototype'
        )
        
        # 绘制连接线，连接相同类别的两种原型
        for i in range(n_classes):
            plt.plot(
                [text_embeddings[i, 0], visual_embeddings[i, 0]],
                [text_embeddings[i, 1], visual_embeddings[i, 1]],
                color='black',
                linestyle='--',
                alpha=0.7,
                linewidth=1.0
            )
        # 添加连接线的图例条目
        plt.plot([], [], color='black', linestyle='--', alpha=0.7, label='Prototype Pair')
        plt.legend(fontsize=14)
        plt.grid(alpha=0.3)
        plt.xticks([])
        plt.yticks([])
        
        # 保存图像
        file_path = os.path.join(output_dir, "prototype_tsne.png")
        plt.tight_layout()
        plt.savefig(file_path, dpi=300)
        plt.close()
        
        print(f"原型t-SNE可视化已保存到: {file_path}")
        
        # 计算文本原型和视觉原型之间的距离
        distances = np.sqrt(np.sum((text_features_np - visual_prototype_np)**2, axis=1))
        
        # 绘制距离直方图
        plt.figure(figsize=(12, 6))
        plt.hist(distances, bins=20, color='#3498DB', alpha=0.7)
        # plt.axvline(x=np.mean(distances), color='r', linestyle='--', 
        #             label=f'average distance: {np.mean(distances):.4f}')
        plt.axvline(x=np.median(distances), color='g', linestyle='--', 
            label=f'median distance: {np.median(distances):.4f}')
        plt.xlabel("distance", fontsize=12)
        plt.ylabel("frequncy", fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend()
        
        # 保存距离直方图
        hist_path = os.path.join(output_dir, "prototype_distance_histogram.png")
        plt.tight_layout()
        plt.savefig(hist_path, dpi=300)
        plt.close()
        
        print(f"距离直方图已保存到: {hist_path}")

        # 保存每个类别的原型距离到CSV文件
        classnames = self.dm.dataset.classnames
        class_distances = []
        for i, (cls_name, dist) in enumerate(zip(classnames, distances)):
            class_distances.append({
                "class_idx": i, 
                "class_name": cls_name, 
                "distance": dist
            })

        # 按距离排序
        sorted_class_distances = sorted(class_distances, key=lambda x: x["distance"], reverse=True)

        # 保存为更易读的文本文件
        txt_path = os.path.join(output_dir, "prototype_distances_by_class.txt")
        with open(txt_path, 'w') as f:
            f.write("Class Prototype Distances (sorted by distance, descending)\n")
            f.write("="*60 + "\n")
            f.write(f"{'Index':<8}{'Class Name':<30}{'Distance':<12}\n")
            f.write("-"*60 + "\n")
            for item in sorted_class_distances:
                f.write(f"{item['class_idx']:<8}{item['class_name']:<30}{item['distance']:.6f}\n")
            f.write("\n")
            f.write(f"Average distance: {np.mean(distances):.6f}\n")
            f.write(f"Maximum distance: {np.max(distances):.6f}\n")
            f.write(f"Minimum distance: {np.min(distances):.6f}\n")
            f.write(f"Standard deviation: {np.std(distances):.6f}\n")

    def draw_tsne_combined(self, id_loader, ood_loader_list, out_datasets, output_dir='./tsne_combined', 
                      perplexity=30, n_iter=1000, random_state=42):
        """
        创建一个综合t-SNE可视化，将原型和数据样本结合在同一个图中
        
        参数:
            id_loader: ID数据的DataLoader
            ood_loader_list: OOD数据加载器列表
            out_datasets: OOD数据集名称列表，与ood_loader_list对应
            output_dir: 输出目录
            perplexity: t-SNE的perplexity参数
            n_iter: t-SNE的迭代次数
            random_state: 随机种子
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from sklearn.manifold import TSNE
        from collections import defaultdict
        
        self.model.eval()
        os.makedirs(output_dir, exist_ok=True)
        
        # 定义颜色和标记
        colors = {
            'ID': '#3498DB',          # 蓝色
            'Textual Prototype': '#9B59B6',  # 紫色
            'Visual Prototype': '#E74C3C',  # 红色
        }
        markers = {
            'ID': 'o',
            'Textual Prototype': '*',
            'Visual Prototype': '*',
        }  
        
        # 为OOD数据集指定颜色和标记
        ood_colors = ['#2ECC71', '#F39C12', '#FF1493']  # 绿色，橙色，品红色
        for i, dataset in enumerate(out_datasets):
            colors[dataset] = ood_colors[i % len(ood_colors)]
            markers[dataset] = 'x'
        
        # 收集特征
        features = defaultdict(list)
        labels = defaultdict(list)
        
        # 1. 获取原型
        print("获取模型原型...")
        with torch.no_grad():
            textual_prototype, visual_prototype = self.model.get_prototype()
            textual_features_np = textual_prototype.cpu().numpy()
            visual_prototype_np = visual_prototype.cpu().numpy()
        
        features['Textual Prototype'] = textual_features_np
        features['Visual Prototype'] = visual_prototype_np
        
        # 2. 收集ID数据特征
        print("获取ID数据特征...")
        id_samples = 0
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(id_loader):
                images = images.cuda()
                Fv, Fs, feat_v, feat_vv, clip_feature = self.model.get_feature(images)
                features['ID'].append(feat_v.mean(dim=1).cpu().numpy())

        if features['ID']:
            features['ID'] = np.vstack(features['ID'])
        
        # 3. 收集OOD数据特征
        for i, (ood_name, ood_loader) in enumerate(zip(out_datasets, ood_loader_list)):
            print(f"获取{ood_name}数据特征...")
            ood_samples = 0
            with torch.no_grad():
                for batch_idx, (images, _) in enumerate(ood_loader):
                    images = images.cuda()
                    Fv, Fs, feat_v, feat_vv, clip_feature = self.model.get_feature(images)
                    features[ood_name].append(feat_v.mean(dim=1).cpu().numpy())
            
            if features[ood_name]:
                features[ood_name] = np.vstack(features[ood_name])
        
        # 合并所有特征用于t-SNE
        print("合并特征并应用t-SNE...")
        all_features = []
        feature_types = []
        
        for feature_type, feature_list in features.items():
            if len(feature_list) > 0:  # 确保有特征
                all_features.append(feature_list)
                feature_types.extend([feature_type] * len(feature_list))
        
        all_features = np.vstack(all_features)
        
        # 应用t-SNE降维
        tsne = TSNE(n_components=2, perplexity=min(perplexity, len(all_features)-1), 
                    n_iter=n_iter, random_state=random_state)
        embeddings_2d = tsne.fit_transform(all_features)
        
        # 创建可视化
        plt.figure(figsize=(16, 12))
        
        # 跟踪已绘制的类别，以避免在图例中重复
        plotted_types = set()
        
        # 首先绘制所有普通数据点
        start_idx = 0
        for feature_type, feature_list in features.items():
            if len(feature_list) > 0 and feature_type not in ['Textual Prototype', 'Visual Prototype']:
                end_idx = start_idx + len(feature_list)
                type_embeddings = embeddings_2d[start_idx:end_idx]
                
                plt.scatter(
                    type_embeddings[:, 0], 
                    type_embeddings[:, 1],
                    c=colors[feature_type],
                    marker=markers[feature_type],
                    s=30,
                    alpha=0.6,
                    label=None if feature_type in plotted_types else feature_type,
                    zorder=1  # 较低的zorder值
                )
                
                plotted_types.add(feature_type)
                start_idx = end_idx

        # 然后绘制原型点，确保它们在最上层
        start_idx = 0
        for feature_type, feature_list in features.items():
            if len(feature_list) > 0 and feature_type in ['Textual Prototype', 'Visual Prototype']:
                end_idx = start_idx + len(feature_list)
                type_embeddings = embeddings_2d[start_idx:end_idx]
                
                plt.scatter(
                    type_embeddings[:, 0], 
                    type_embeddings[:, 1],
                    c=colors[feature_type],
                    marker=markers[feature_type],
                    s=100,
                    alpha=0.9,
                    label=None if feature_type in plotted_types else feature_type,
                    zorder=10  # 较高的zorder值
                )
                
                if feature_type == 'Textual Prototype':
                    text_embeddings = type_embeddings
                elif feature_type == 'Visual Prototype':
                    visual_embeddings = type_embeddings
                
                plotted_types.add(feature_type)
                start_idx = end_idx
        
        # 连接相同类别的文本和视觉原型
        if 'Textual Prototype' in features and 'Visual Prototype' in features:
            n_prototypes = min(len(text_embeddings), len(visual_embeddings))
            for i in range(n_prototypes):
                plt.plot(
                    [text_embeddings[i, 0], visual_embeddings[i, 0]],
                    [text_embeddings[i, 1], visual_embeddings[i, 1]],
                    color='black',
                    linestyle='--',
                    alpha=0.7,
                    linewidth=1.0,
                    zorder=20
                )
            
            # 添加连接线的图例条目
            plt.plot([], [], color='black', linestyle='--', alpha=0.7, label='Prototype Pair')
        
        plt.axis('off')  # 隐藏坐标轴
        plt.grid(False)
        plt.legend(fontsize=14, loc='best', markerscale=2)
        
        # 保存图像
        file_path = os.path.join(output_dir, "combined_tsne.png")
        plt.tight_layout()
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"结合原型和数据的t-SNE可视化已保存到: {file_path}")

    def _plot_distribution(self, output_dir, id_scores, ood_scores, file_name, fpr):
        """
        自定义绘图函数，根据新的命名约定保存图像。
        
        参数:
            output_dir: 输出目录路径
            id_scores: ID数据的分数
            ood_scores: OOD数据的分数
            file_name: 文件名（不含扩展名）
        """
        sns.set(style="white", palette="muted")
        palette = ['#A8BAE3', '#55AB83']
        
        id_scores = id_scores.flatten()
        ood_scores = ood_scores.flatten()
        # 计算ID分数的5%分位线（使得95%的ID分数大于该阈值）
        threshold = np.percentile(id_scores, 5)
        
        data = {
            "ID": [id_score for id_score in id_scores],
            "OOD": [ood_score for ood_score in ood_scores]
        }
        
        # 创建画布和图像
        plt.figure(figsize=(10, 6))
        
        # 使用displot并保存返回的FacetGrid对象
        g = sns.displot(data, label="id", kind="kde", palette=palette, fill=True, alpha=0.8)
        
        for ax in g.axes.flat:
            # 计算数据范围
            x_min, x_max = ax.get_xlim()
            data_range = x_max - x_min
            # 添加垂直线和文本
            ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2)
            # 使用相对位置计算文本位置 - 数据范围的5%
            text_offset = data_range * 0.05
            text_x = threshold + text_offset
            ax.text(text_x, ax.get_ylim()[1]*0.9, 
                    f'FPR@95%: {fpr:.4f}', 
                    bbox=dict(facecolor='white', alpha=0))

        # 设置横轴和纵轴的标签
        g.set_axis_labels("Scores", "Density")  # 可以根据需要修改为其他标签
        
        # 调整图表布局，确保标题不会被截断
        g.fig.tight_layout()
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存图像，使用新的命名格式
        file_path = os.path.join(output_dir, f"{file_name}.png")
        print(f"保存图像到: {file_path}")
        g.savefig(file_path, bbox_inches='tight')
        plt.close()

    def visualize(self, image_paths, output_dirs, batch_size=512):
        """
        批量可视化局部响应 (适配ViT-B/32模型，7×7 patch)
        
        参数:
            image_paths: 图像路径列表
            output_dirs: 对应的输出目录列表
            batch_size: 批处理大小
        """
        import os
        from PIL import Image
        import numpy as np
        import torch
        from tqdm import tqdm
        import clip
        
        self.model.eval()
        
        # 加载预处理函数
        _, preprocess = clip.load(self.model_path)
        
        # 将图像列表分批处理
        num_images = len(image_paths)
        num_batches = (num_images + batch_size - 1) // batch_size
        
        # 使用tqdm创建进度条
        pbar = tqdm(total=num_images, desc="处理图像", unit="img")
        
        # 记录当前处理的文件夹
        current_folder = None
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_images)
            batch_paths = image_paths[start_idx:end_idx]
            batch_dirs = output_dirs[start_idx:end_idx]
            
            # 检查当前正在处理的文件夹
            for path in batch_paths:
                folder_name = os.path.basename(os.path.dirname(path))
                if folder_name != current_folder:
                    current_folder = folder_name
                    pbar.set_description(f"处理 {folder_name}")
            
            # 加载和预处理批次中的所有图像
            batch_images = []
            valid_indices = []
            original_images = []
            valid_paths = []
            valid_dirs = []
            
            for i, img_path in enumerate(batch_paths):
                try:
                    img = Image.open(img_path).convert('RGB')
                    original_images.append(np.array(img.resize((224, 224))))
                    batch_images.append(preprocess(img))
                    valid_indices.append(i)
                    valid_paths.append(img_path)
                    valid_dirs.append(batch_dirs[i])
                except Exception as e:
                    pbar.write(f"无法处理图像 {img_path}: {e}")
            
            if not batch_images:
                continue
            
            # 堆叠为批次并移到GPU
            batch_tensor = torch.stack(batch_images).to('cuda')
            
            # 批量获取热力图
            with torch.no_grad():
                heatmap_textual, heatmap_visual = self.model.get_heatmap(batch_tensor)
                heatmap_textual = heatmap_textual.permute(0, 2, 1)
                heatmap_visual = heatmap_visual.permute(0, 2, 1)
            
            # 处理每个图像的热力图
            for i, idx in enumerate(valid_indices):
                try:
                    # 获取当前图像的热力图
                    curr_heatmap = heatmap_textual[i:i+1]
                    
                    # 计算每个patch位置上的最大响应值
                    max_tensor, _ = torch.max(curr_heatmap, dim=2)  # 形状为 [1, 49]
                    flattened_tensor = max_tensor.view(-1)  # 形状为 [49]
                    
                    # 找到前7个最大响应区域
                    top_k = min(7, flattened_tensor.size(0))
                    top_k_values, top_k_indices = torch.topk(flattened_tensor, k=top_k)
                    
                    # 获取原始图像
                    image_array = original_images[i]
                    image_name = os.path.basename(valid_paths[i])
                    output_dir = valid_dirs[i]
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # 绘制热点区域 - 使用7x7网格，每个区域32x32像素
                    grid_size = 7
                    region_size = 32
                    
                    for k, index in enumerate(top_k_indices):
                        score = top_k_values[k].item() * 80  
                        # print(f"响应区域 {k+1}: 索引 {index.item()}, 分数 {score:.4f}")
                        
                        # 颜色映射 - 根据响应值的大小选择颜色强度
                        if score > 18:
                            color = '#3f007d'  # 深紫色 - 最高响应
                        elif score > 16:
                            color = '#54278f'
                        elif score > 14:
                            color = '#6a51a3'
                        elif score > 12:
                            color = '#807dba'
                        elif score > 10:
                            color = '#9e9ac8'
                        elif score > 8:
                            color = '#bcbddc'
                        elif score > 6:
                            color = '#dadaeb'
                        else:
                            color = '#efedf5'  # 浅紫色 - 最低响应
                        
                        # 将十六进制颜色转换为RGB
                        r = int(color[1:3], 16)
                        g = int(color[3:5], 16)
                        b = int(color[5:7], 16)
                        rgb_array = np.array([r, g, b])
                        
                        # 计算区域位置 - 使用7x7网格
                        index_val = index.item()
                        row = (index_val // grid_size) * region_size
                        col = (index_val % grid_size) * region_size
                        
                        # 确保不越界
                        row_end = min(row + region_size, image_array.shape[0])
                        col_end = min(col + region_size, image_array.shape[1])
                        
                        # 填充区域颜色
                        image_array[row:row_end, col:col_end] = rgb_array
                    
                    # 保存结果
                    result_image = Image.fromarray(image_array.astype('uint8'))
                    save_path = os.path.join(output_dir, image_name)
                    result_image.save(save_path)
                    
                except Exception as e:
                    pbar.write(f"处理热力图时出错 {valid_paths[i]}: {e}")
                
                # 更新进度条
                pbar.update(1)
        
        # 关闭进度条
        pbar.close()

    def _visualize_heatmap(self, image_path, heatmap, output_dir, suffix="", region_size=32):
        """
        可视化热力图
            
        参数:
            image_path: 输入图像路径
            heatmap: 热力图张量
            output_dir: 输出目录
            suffix: 输出文件名后缀
            region_size: 区域大小
        """
        from PIL import Image
        import numpy as np
        import os
        
        # 获取图像名称
        image_name = os.path.basename(image_path)
        base_name, ext = os.path.splitext(image_name)
        
        # 调整图像大小
        new_size = (224, 224)
        image_array = self.load_resize_image(image_path, new_size)
        
        # 确保图像是3通道
        if len(image_array.shape) == 2 or image_array.shape[2] == 1:
            image_array = np.stack((image_array,) * 3, axis=-1)
        
        # 展平热力图为一维
        flattened_tensor = heatmap.view(-1)
        
        # 找到最大的10个响应值及其索引
        top_k_values, top_k_indices = torch.topk(flattened_tensor, k=10)
        
        # 可视化响应区域
        for id, idx in enumerate(top_k_indices):
            idx = idx.item()  # 确保idx是Python整数
            score = 100*top_k_values[id].item()

            # print(f"响应区域 {id+1}: 索引 {idx}, 分数 {score:.4f}")
            
            # 根据score选择颜色
            if score > 12:
                color = '#3f007d'
            elif score > 11:
                color = '#54278f'
            elif score > 10:
                color = '#6a51a3'
            elif score > 9:
                color = '#807dba'
            elif score > 8:
                color = '#9e9ac8'
            elif score > 7:
                color = '#bcbddc'
            elif score > 6:
                color = '#dadaeb'
            else:
                color = '#efedf5'
            
            # 将HEX颜色转换为RGB
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            rgb_array = np.array([r, g, b])
            
            # 计算2D坐标
            row = (idx // (224 // region_size)) * region_size
            col = (idx % (224 // region_size)) * region_size
            
            # 确保边界不超出图像
            row_end = min(row + region_size, 224)
            col_end = min(col + region_size, 224)
            
            # 将区域着色
            image_array[row:row_end, col:col_end] = rgb_array
        
        # 保存结果
        output_path = os.path.join(output_dir, f"{base_name}{suffix}{ext}")
        image = Image.fromarray(image_array.astype('uint8'))
        image.save(output_path)
        print(f"保存热力图到: {output_path}")

    def load_resize_image(self, image_path, new_size):
        """
        加载并调整图像大小
        
        参数:
            image_path: 图像路径
            new_size: 新的图像大小 (width, height)
        """
        from PIL import Image
        import numpy as np
        
        img = Image.open(image_path)
        img = img.resize(new_size)
        return np.array(img)

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        tokenized_prompts = self.tokenized_prompts
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts, tokenized_prompts 
        

class Proj1(nn.Module):
    def __init__(self,
                 visual_dim=512,
                 token_embed_dim=512,
                 **kwargs
                 ):
        super(Proj1, self).__init__()

        self.prompt_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, visual_dim),
            nn.ReLU(),
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, token_embed_dim)
        )

        self.apply(self._init_weights)

    def forward(self, x):
        '''
        x: (B, D) 512
        '''
        x = self.prompt_proj(x.float())
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class Proj2(nn.Module):
    def __init__(self,
                 visual_dim=512,
                 token_embed_dim=512,
                 **kwargs
                 ):
        super(Proj2, self).__init__()

        self.prompt_proj = nn.Sequential(
            nn.GroupNorm(1, visual_dim),  # Use GroupNorm instead of LayerNorm
            nn.Conv1d(visual_dim, visual_dim, 1),
            nn.ReLU(),
            nn.GroupNorm(1, visual_dim),  # Use GroupNorm instead of LayerNorm
            nn.Conv1d(visual_dim, token_embed_dim, 1)
        )

        self.apply(self._init_weights)

    def forward(self, x):
        '''
        x: (B, 49, D)
        '''
        x = x.permute(0, 2, 1)  # Change the order of dimensions to (B, D, 49)
        x = self.prompt_proj(x.float())
        x = x.permute(0, 2, 1)  # Change the order of dimensions to (B, 49, D)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


