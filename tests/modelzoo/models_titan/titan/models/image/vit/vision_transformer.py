""" Copyright 2020, Ross Wightman

A warpper for Vision Transformer (ViT) from timm,
which supports output with both feature and attn_map.
Features_only mode is not supported for Vision Transformer.

The official code is at 
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    see tags:v0.5.4.

"""

import math
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from timm.models.layers import (
    Mlp,
    DropPath,
    trunc_normal_,
    lecun_normal_
)
from timm.models.helpers import named_apply, adapt_input_conv
from timm.models.layers.helpers import to_2tuple
from titan.utils.registry import register_model
from titan.utils.helper import (
    download_weights,
    get_configs
)
from titan.utils.logger import logger

__all__ = [
    'vit_tiny',
    'vit_small',
    'vit_base',
    'vit_large',
    'vit_huge',
    'vit_giant',
    'vit_gigantic',
    'vit_base_sam',
    'vit_small_dino',
    'vit_base_dino',
    'vit_base_miil'
]


try:
    from torch import _assert
except ImportError:
    def _assert(condition: bool, message: str):
        assert condition, message


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            attn_drop=0.,
            proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x, attn_map = self.attn(self.norm1(x))
        x = x + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn_map


class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """
    __model_type__ = 'vit'

    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=1000,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            representation_size=None,
            distilled=False,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            embed_layer=PatchEmbed,
            norm_layer=None,
            act_layer=None,
            weight_init='',
            use_attn_map=False,
            only_cls_token=True,
            name=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
            use_attn_map (bool): whether to output attn_map of the last block
            only_cls_token (bool): only output cls token as the features, need num_classes=0
            name: model name
        """
        super().__init__()
        self.name = name
        self.only_cls_token = only_cls_token
        self.num_classes = num_classes
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(
            1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(
            1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.last_out_channels = self.num_features
        self.head = nn.Linear(
            self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(
                self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Whether to output attn_map
        self.use_attn_map = use_attn_map

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights,
                        head_bias=head_bias, jax_impl=True), self)
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(
                self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        # stole cls_tokens impl from Phil Wang, thanks
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(
                x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for _, block in enumerate(self.blocks):
            x, attn_map = block(x)
        x = self.norm(x)
        if self.dist_token is None:
            if self.only_cls_token:
                return self.pre_logits(x[:, 0]), attn_map
            else:
                return self.pre_logits(x), attn_map
        else:
            if self.only_cls_token:
                return x[:, 0], attn_map
            else:
                return x, attn_map

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            attn_map = x[2]
            x, x_dist = self.head(x[0]), self.head_dist(
                x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                if self.use_attn_map and self.training:
                    return x, x_dist, attn_map
                else:
                    return x, x_dist
            else:
                if self.use_attn_map and self.training:
                    return (x + x_dist) / 2, attn_map
                else:
                    return (x + x_dist) / 2
        else:
            attn_map = x[1]
            x = self.head(x[0])
            if self.use_attn_map and self.training:
                return x, attn_map
            else:
                return x


def _init_vit_weights(
        module: nn.Module,
        name: str = '',
        head_bias: float = 0.,
        jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(
            stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(
                            block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(
                            block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(
                            block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(
                            _n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(
                            _n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(
                            _n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(
        w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
        model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
        model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(
            _n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(
                _n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(
                _n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    logger.info('Resized position embedding: %s to %s',
                posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:,
                                         :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    logger.info('Position embedding grid-size from %s to %s',
                [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(
        1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(
        posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(
        0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


@register_model
def vit_tiny(pretrained=False, **kwargs):
    """ ViT-Tiny (Vit-Ti/16)
    NOTE: Pretrained models from ImageNet-21k have valid 21k
        classifier head and no representation (pre-logits) layer
    """
    patch_size = kwargs.pop('patch_size', 16)
    model_name = 'vit_tiny'
    pretrained_config, model_config = get_configs(**kwargs)
    if model_config['features_only']:
        raise ValueError(
            "Vision transformer does not support features_only.")
    model_config.pop('features_only')

    model_kwargs = dict(
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        name=model_name,
        **model_config)
    model = VisionTransformer(**model_kwargs)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        model.load_pretrained(weight_path)
        logger.info("Pre-trained model loading done.")
    return model


@register_model
def vit_small(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16 or 32)
    Pretrained weights from source
        https://github.com/google-research/vision_transformer.
    NOTE: Pretrained models from ImageNet-21k have valid 21k
        classifier head and no representation (pre-logits) layer
    """
    patch_size = kwargs.pop('patch_size', 16)
    model_name = 'vit_small'
    pretrained_config, model_config = get_configs(**kwargs)
    if model_config['features_only']:
        raise ValueError(
            "Vision transformer does not support features_only.")
    model_config.pop('features_only')

    model_kwargs = dict(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        name=model_name,
        **model_config)
    model = VisionTransformer(**model_kwargs)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        model.load_pretrained(weight_path)
        logger.info("Pre-trained model loading done.")
    return model


@register_model
def vit_base(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/8 or 16 or 32) from original paper
            (https://arxiv.org/abs/2010.11929).
    Pretrained weights from source
        https://github.com/google-research/vision_transformer.
    NOTE: Pretrained models from ImageNet-21k have valid 21k
        classifier head and no representation (pre-logits) layer
    """
    patch_size = kwargs.pop('patch_size', 16)
    model_name = 'vit_base'
    pretrained_config, model_config = get_configs(**kwargs)
    if model_config['features_only']:
        raise ValueError(
            "Vision transformer does not support features_only.")
    model_config.pop('features_only')

    model_kwargs = dict(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        name=model_name,
        **model_config)
    model = VisionTransformer(**model_kwargs)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        model.load_pretrained(weight_path)
        logger.info("Pre-trained model loading done.")
    return model


@register_model
def vit_large(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16 or 32) from original paper
            (https://arxiv.org/abs/2010.11929).
    Pretrained weights from source
        https://github.com/google-research/vision_transformer.
    NOTE: pretrained models from ImageNet-21k have valid 21k
        classifier head and no representation (pre-logits) layer
    """
    patch_size = kwargs.pop('patch_size', 16)
    model_name = 'vit_large'
    pretrained_config, model_config = get_configs(**kwargs)
    if model_config['features_only']:
        raise ValueError(
            "Vision transformer does not support features_only.")
    model_config.pop('features_only')

    model_kwargs = dict(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        name=model_name,
        **model_config)
    model = VisionTransformer(**model_kwargs)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        model.load_pretrained(weight_path)
        logger.info("Pre-trained model loading done.")
    return model


@register_model
def vit_huge(pretrained=False, **kwargs):
    """ ViT-Huge model (ViT-H/14) from original paper
            (https://arxiv.org/abs/2010.11929).
    Pretrained weights from source
        https://github.com/google-research/vision_transformer.
    NOTE: this model has a representation layer but the 21k
        classifier head is zero'd out in original weights
    """
    patch_size = kwargs.pop('patch_size', 14)
    model_name = 'vit_huge'
    pretrained_config, model_config = get_configs(**kwargs)
    if model_config['features_only']:
        raise ValueError(
            "Vision transformer does not support features_only.")
    model_config.pop('features_only')

    model_kwargs = dict(
        patch_size=patch_size,
        embed_dim=1280,
        depth=24,
        num_heads=16,
        name=model_name,
        **model_config)
    model = VisionTransformer(**model_kwargs)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        model.load_pretrained(weight_path)
        logger.info("Pre-trained model loading done.")
    return model


@register_model
def vit_giant(pretrained=False, **kwargs):
    """ ViT-Giant model (ViT-g/14) from `Scaling Vision Transformers`
            - https://arxiv.org/abs/2106.04560
    """
    patch_size = kwargs.pop('patch_size', 14)
    model_name = 'vit_giant'
    pretrained_config, model_config = get_configs(**kwargs)
    if model_config['features_only']:
        raise ValueError(
            "Vision transformer does not support features_only.")
    model_config.pop('features_only')

    model_kwargs = dict(
        patch_size=patch_size,
        embed_dim=1408,
        mlp_ratio=48/11,
        depth=40,
        num_heads=16,
        name=model_name,
        **model_config)
    model = VisionTransformer(**model_kwargs)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        model.load_pretrained(weight_path)
        logger.info("Pre-trained model loading done.")
    return model


@register_model
def vit_gigantic(pretrained=False, **kwargs):
    """ ViT-Gigantic model (ViT-G/14) from `Scaling Vision
            Transformers` - https://arxiv.org/abs/2106.04560
    """
    patch_size = kwargs.pop('patch_size', 14)
    model_name = 'vit_gigantic'
    pretrained_config, model_config = get_configs(**kwargs)
    if model_config['features_only']:
        raise ValueError(
            "Vision transformer does not support features_only.")
    model_config.pop('features_only')

    model_kwargs = dict(
        patch_size=patch_size,
        embed_dim=1664,
        mlp_ratio=64/13,
        depth=48,
        num_heads=16,
        name=model_name,
        **model_config)
    model = VisionTransformer(**model_kwargs)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        model.load_pretrained(weight_path)
        logger.info("Pre-trained model loading done.")
    return model


@register_model
def vit_base_sam(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/32) w/ SAM pretrained weights.
            Paper: https://arxiv.org/abs/2106.01548
    """
    patch_size = kwargs.pop('patch_size', 16)
    model_name = 'vit_base_sam'
    pretrained_config, model_config = get_configs(**kwargs)
    if model_config['features_only']:
        raise ValueError(
            "Vision transformer does not support features_only.")
    model_config.pop('features_only')

    model_kwargs = dict(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        name=model_name,
        **model_config)
    model = VisionTransformer(**model_kwargs)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        model.load_pretrained(weight_path)
        logger.info("Pre-trained model loading done.")
    return model


@register_model
def vit_small_dino(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/8 or 16) w/ DINO pretrained weights (no head)
            - https://arxiv.org/abs/2104.14294
    """
    patch_size = kwargs.pop('patch_size', 16)
    model_name = 'vit_small_dino'
    pretrained_config, model_config = get_configs(**kwargs)
    if model_config['features_only']:
        raise ValueError(
            "Vision transformer does not support features_only.")
    model_config.pop('features_only')

    model_kwargs = dict(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        name=model_name,
        **model_config)
    model = VisionTransformer(**model_kwargs)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        model.load_pretrained(weight_path)
        logger.info("Pre-trained model loading done.")
    return model


@register_model
def vit_base_dino(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/8 or 16) w/ DINO pretrained weights (no head)
            - https://arxiv.org/abs/2104.14294
    """
    patch_size = kwargs.pop('patch_size', 16)
    model_name = 'vit_base_dino'
    pretrained_config, model_config = get_configs(**kwargs)
    if model_config['features_only']:
        raise ValueError(
            "Vision transformer does not support features_only.")
    model_config.pop('features_only')

    model_kwargs = dict(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        name=model_name,
        **model_config)
    model = VisionTransformer(**model_kwargs)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        model.load_pretrained(weight_path)
        logger.info("Pre-trained model loading done.")
    return model


@register_model
def vit_base_miil(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper
            (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    patch_size = kwargs.pop('patch_size', 16)
    model_name = 'vit_base_miil'
    pretrained_config, model_config = get_configs(**kwargs)
    if model_config['features_only']:
        raise ValueError(
            "Vision transformer does not support features_only.")
    model_config.pop('features_only')

    model_kwargs = dict(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_bias=False,
        name=model_name,
        **model_config)
    model = VisionTransformer(**model_kwargs)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        model.load_pretrained(weight_path)
        logger.info("Pre-trained model loading done.")
    return model