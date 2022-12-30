# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
T2T-ViT
"""
import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_
import numpy as np
from collections import OrderedDict

from titan.models.components import TokenTransformer, TokenPerformer, \
    TransformerBlock, get_sinusoid_encoding
from titan.utils.registry import register_model
from titan.utils.helper import (
    download_weights,
    load_pretrained_model_weights,
    get_configs
)
from titan.utils.logger import logger

__all__ = [
    't2t_vit_7', 't2t_vit_10', 't2t_vit_12', 't2t_vit_14', 't2t_vit_19',
    't2t_vit_24', 't2t_vit_t_14', 't2t_vit_t_19', 't2t_vit_t_24',
    't2t_vit_14_resnext', 't2t_vit_14_wide'
]


class T2T_module(nn.Module):
    """
    Tokens-to-Token encoding module
    """

    def __init__(self,
                 img_size=224,
                 tokens_type='performer',
                 in_chans=3,
                 embed_dim=768,
                 token_dim=64):
        super().__init__()

        if tokens_type == 'transformer':
            logger.info('adopt transformer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7),
                                         stride=(4, 4),
                                         padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3),
                                         stride=(2, 2),
                                         padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3),
                                         stride=(2, 2),
                                         padding=(1, 1))

            self.attention1 = TokenTransformer(dim=in_chans * 7 * 7,
                                               in_dim=token_dim,
                                               num_heads=1,
                                               mlp_ratio=1.0)
            self.attention2 = TokenTransformer(dim=token_dim * 3 * 3,
                                               in_dim=token_dim,
                                               num_heads=1,
                                               mlp_ratio=1.0)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'performer':
            logger.info('adopt performer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7),
                                         stride=(4, 4),
                                         padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3),
                                         stride=(2, 2),
                                         padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3),
                                         stride=(2, 2),
                                         padding=(1, 1))

            self.attention1 = TokenPerformer(dim=in_chans * 7 * 7,
                                             in_dim=token_dim,
                                             kernel_ratio=0.5)
            self.attention2 = TokenPerformer(dim=token_dim * 3 * 3,
                                             in_dim=token_dim,
                                             kernel_ratio=0.5)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        # just for comparison with convolution, not our model
        elif tokens_type == 'convolution':
            # for this tokens type, you need change forward as three
            # convolution operation
            logger.info('adopt convolution layers for tokens-to-token')
            self.soft_split0 = nn.Conv2d(3,
                                         token_dim,
                                         kernel_size=(7, 7),
                                         stride=(4, 4),
                                         padding=(2, 2))  # the 1st convolution
            self.soft_split1 = nn.Conv2d(token_dim,
                                         token_dim,
                                         kernel_size=(3, 3),
                                         stride=(2, 2),
                                         padding=(1, 1))  # the 2nd convolution
            self.project = nn.Conv2d(token_dim,
                                     embed_dim,
                                     kernel_size=(3, 3),
                                     stride=(2, 2),
                                     padding=(1, 1))  # the 3rd convolution

        # there are 3 sfot split, stride are 4,2,2 seperately
        self.num_patches = (img_size // (4 * 2 * 2)) * \
            (img_size // (4 * 2 * 2))
        # this is for torchscript converting
        self.sqrt_shape = [56, 28]

    def forward(self, x):
        # step0: soft split
        x = self.soft_split0(x).transpose(1, 2)

        # iteration1: re-structurization/reconstruction
        x = self.attention1(x)
        x_shape = x.size()
        x = x.transpose(1, 2).reshape(x_shape[0], x_shape[2], self.sqrt_shape[0], self.sqrt_shape[0])
        # iteration1: soft split
        x = self.soft_split1(x).transpose(1, 2)

        # iteration2: re-structurization/reconstruction
        x = self.attention2(x)
        x_shape = x.size()
        x = x.transpose(1, 2).reshape(x_shape[0], x_shape[2], self.sqrt_shape[1], self.sqrt_shape[1])
        # iteration2: soft split
        x = self.soft_split2(x).transpose(1, 2)

        # final tokens
        x = self.project(x)

        return x


class T2T_ViT(nn.Module):
    __model_type__ = 'vit'

    def __init__(self,
                 img_size=224,
                 tokens_type='performer',
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 token_dim=64,
                 use_attn_map=False,
                 name=None):
        super().__init__()
        self.name = name
        self.num_classes = num_classes
        # last_out_channels for consistency with other models
        self.last_out_channels = self.embed_dim = embed_dim

        self.tokens_to_token = T2T_module(img_size=img_size,
                                          tokens_type=tokens_type,
                                          in_chans=in_chans,
                                          embed_dim=embed_dim,
                                          token_dim=token_dim)
        num_patches = self.tokens_to_token.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(
            n_position=num_patches + 1, d_hid=embed_dim),
            requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim,
                             num_heads=num_heads,
                             mlp_ratio=mlp_ratio,
                             qkv_bias=qkv_bias,
                             qk_scale=qk_scale,
                             drop=drop_rate,
                             attn_drop=attn_drop_rate,
                             drop_path=dpr[i],
                             norm_layer=norm_layer) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        if self.num_classes > 0:
            self.head = nn.Linear(embed_dim, num_classes)

        # Whether to output attn_map
        self.use_attn_map = use_attn_map

        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        if self.num_classes > 0:
            self.head = nn.Linear(self.embed_dim, num_classes)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.tokens_to_token(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for _, blk in enumerate(self.blocks):
            x, attn_map = blk(x)

        x = self.norm(x)
        return x[:, 0], attn_map

    def forward(self, x):
        x, attn_map = self.forward_features(x)
        if self.num_classes > 0:
            x = self.head(x)
        if self.use_attn_map and self.training:
            return x, attn_map
        return x


@register_model
def t2t_vit_7(pretrained=False,
              **kwargs):  # adopt performer for tokens to token
    model_name = 't2t_vit_7'
    pretrained_config, model_config = get_configs(**kwargs)
    if model_config['features_only']:
        raise ValueError(
            "Token-to-Token ViT does not support features_only.")
    model_config.pop('features_only')

    model = T2T_ViT(tokens_type='performer',
                    qk_scale=256**-0.5,
                    embed_dim=256,
                    depth=7,
                    num_heads=4,
                    mlp_ratio=2.,
                    name=model_name,
                    **model_config)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model loading done.")
    return model


@register_model
def t2t_vit_10(pretrained=False,
               **kwargs):  # adopt performer for tokens to token
    model_name = 't2t_vit_10'
    pretrained_config, model_config = get_configs(**kwargs)
    if model_config['features_only']:
        raise ValueError(
            "Token-to-Token ViT does not support features_only.")
    model_config.pop('features_only')

    model = T2T_ViT(tokens_type='performer',
                    qk_scale=256**-0.5,
                    embed_dim=256,
                    depth=10,
                    num_heads=4,
                    mlp_ratio=2.,
                    name=model_name,
                    **model_config)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model loading done.")
    return model


@register_model
def t2t_vit_12(pretrained=False,
               **kwargs):  # adopt performer for tokens to token
    model_name = 't2t_vit_12'
    pretrained_config, model_config = get_configs(**kwargs)
    if model_config['features_only']:
        raise ValueError(
            "Token-to-Token ViT does not support features_only.")
    model_config.pop('features_only')

    model = T2T_ViT(tokens_type='performer',
                    qk_scale=256**-0.5,
                    embed_dim=256,
                    depth=12,
                    num_heads=4,
                    mlp_ratio=2.,
                    name=model_name,
                    **model_config)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model loading done.")
    return model


@register_model
def t2t_vit_14(pretrained=False,
               **kwargs):  # adopt performer for tokens to token
    model_name = 't2t_vit_14'
    pretrained_config, model_config = get_configs(**kwargs)
    if model_config['features_only']:
        raise ValueError(
            "Token-to-Token ViT does not support features_only.")
    model_config.pop('features_only')

    model = T2T_ViT(tokens_type='performer',
                    qk_scale=384**-0.5,
                    embed_dim=384,
                    depth=14,
                    num_heads=6,
                    mlp_ratio=3.,
                    name=model_name,
                    **model_config)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model loading done.")
    return model


@register_model
def t2t_vit_19(pretrained=False,
               **kwargs):  # adopt performer for tokens to token
    model_name = 't2t_vit_19'
    pretrained_config, model_config = get_configs(**kwargs)
    if model_config['features_only']:
        raise ValueError(
            "Token-to-Token ViT does not support features_only.")
    model_config.pop('features_only')

    model = T2T_ViT(tokens_type='performer',
                    qk_scale=448**-0.5,
                    embed_dim=448,
                    depth=19,
                    num_heads=7,
                    mlp_ratio=3.,
                    name=model_name,
                    **model_config)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model loading done.")
    return model


@register_model
def t2t_vit_24(pretrained=False,
               **kwargs):  # adopt performer for tokens to token
    model_name = 't2t_vit_24'
    pretrained_config, model_config = get_configs(**kwargs)
    if model_config['features_only']:
        raise ValueError(
            "Token-to-Token ViT does not support features_only.")
    model_config.pop('features_only')

    model = T2T_ViT(tokens_type='performer',
                    qk_scale=512**-0.5,
                    embed_dim=512,
                    depth=24,
                    num_heads=8,
                    mlp_ratio=3.,
                    name=model_name,
                    **model_config)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model loading done.")
    return model


@register_model
def t2t_vit_t_14(pretrained=False,
                 **kwargs):  # adopt transformers for tokens to token
    model_name = 't2t_vit_t_14'
    pretrained_config, model_config = get_configs(**kwargs)
    if model_config['features_only']:
        raise ValueError(
            "Token-to-Token ViT does not support features_only.")
    model_config.pop('features_only')

    model = T2T_ViT(tokens_type='transformer',
                    qk_scale=384**-0.5,
                    embed_dim=384,
                    depth=14,
                    num_heads=6,
                    mlp_ratio=3.,
                    name=model_name,
                    **model_config)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model loading done.")
    return model


@register_model
def t2t_vit_t_19(pretrained=False,
                 **kwargs):  # adopt transformers for tokens to token
    model_name = 't2t_vit_t_19'
    pretrained_config, model_config = get_configs(**kwargs)
    if model_config['features_only']:
        raise ValueError(
            "Token-to-Token ViT does not support features_only.")
    model_config.pop('features_only')

    model = T2T_ViT(tokens_type='transformer',
                    qk_scale=448**-0.5,
                    embed_dim=448,
                    depth=19,
                    num_heads=7,
                    mlp_ratio=3.,
                    name=model_name,
                    **model_config)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model loading done.")
    return model


@register_model
def t2t_vit_t_24(pretrained=False,
                 **kwargs):  # adopt transformers for tokens to token
    model_name = 't2t_vit_t_24'
    pretrained_config, model_config = get_configs(**kwargs)
    if model_config['features_only']:
        raise ValueError(
            "Token-to-Token ViT does not support features_only.")
    model_config.pop('features_only')

    model = T2T_ViT(tokens_type='transformer',
                    qk_scale=512**-0.5,
                    embed_dim=512,
                    depth=24,
                    num_heads=8,
                    mlp_ratio=3.,
                    name=model_name,
                    **model_config)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model loading done.")
    return model


# rexnext and wide structure
@register_model
def t2t_vit_14_resnext(pretrained=False, **kwargs):
    model_name = 't2t_vit_14_resnext'
    pretrained_config, model_config = get_configs(**kwargs)
    if model_config['features_only']:
        raise ValueError(
            "Token-to-Token ViT does not support features_only.")
    model_config.pop('features_only')

    model = T2T_ViT(tokens_type='performer',
                    qk_scale=384**-0.5,
                    embed_dim=384,
                    depth=14,
                    num_heads=32,
                    mlp_ratio=3.,
                    name=model_name,
                    **model_config)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model loading done.")
    return model


@register_model
def t2t_vit_14_wide(pretrained=False, **kwargs):
    model_name = 't2t_vit_14_wide'
    pretrained_config, model_config = get_configs(**kwargs)
    if model_config['features_only']:
        raise ValueError(
            "Token-to-Token ViT does not support features_only.")
    model_config.pop('features_only')

    model = T2T_ViT(tokens_type='performer',
                    qk_scale=512**-0.5,
                    embed_dim=768,
                    depth=4,
                    num_heads=12,
                    mlp_ratio=3.,
                    name=model_name,
                    **model_config)
    if pretrained:
        weight_path = download_weights(
            model_name,
            **pretrained_config)
        load_pretrained_model_weights(model, weight_path)
        logger.info("Pre-trained model loading done.")
    return model
