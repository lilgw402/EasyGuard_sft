#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Description: copy and edited from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
'''

import os
from typing import Type, Any, Callable, Union, List, Optional, Dict, Tuple
from collections import OrderedDict

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm

from fex import _logger as log
from fex.utils.load import load_pretrained_state_dict
from fex.utils.torch_io import load as torch_io_load

from .with_ptx import TransformerEncoderLayer, USE_PTX_TRANSFORMER_CONF_VIT


__all__ = [
    'VisualTransformer',
]


LayerNorm = nn.LayerNorm


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, dropout=0., layernorm_eps=1.0e-5):
        super().__init__()
        self.width = width
        self.layers = layers
        default_config = dict(
            n_layers=layers,
            dim=width,
            n_heads=heads,
            dim_ff=width * 4,
            act='gelu',
            p_drop_hidden=dropout,
            p_drop_attn=dropout,
            return_layers=[],
            clamp_inf_nan=False,
            layer_norm_eps=layernorm_eps,
            fp16=True,
            layernorm_fp16=True,
            fuse_qkv_projs=False,
            omit_other_attn_output=True,
            layer_grad_checkpoint=False,
            use_pre_layernorm=True,  # <= pre-layernorm
        )
        ft_config = dict(
            layernorm_type='ft',
            use_ft_layernorm=True,
            use_ft_softmax=True,
            use_ft_linear_in_attn=True,
            use_ft_transpose_in_attn=True,
            use_ft_mm_in_attn=True,
            use_ft_linear_in_attn_out=True,
            use_ft_linear_in_ffn=True,
            mha_acts_unite_d01=True,
            dropout_in_ffn=True,  # To be the same as before
            use_ft_ffn_linear_fusion=True,
            use_ffn_output_dropout=True,
            use_ft_attn_out_proj_dropout_fusion=True,
            pad_seq_len_even=True,
        )
        default_config.update(ft_config)  # ptx trace的时候注释掉这句
        default_config.update(USE_PTX_TRANSFORMER_CONF_VIT)
        self.resblocks = nn.ModuleList([TransformerEncoderLayer(default_config) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        attn_mask = torch.zeros((x.size(0), 1, x.size(1), x.size(1)), device=x.device, dtype=x.dtype)
        for block in self.resblocks:
            x = block(x, attn_mask)
            x = x[0]  # `ptx.ops.transformer.TransformerEncoderLayerOutput`
        return x


class VisualTransformer(nn.Module):
    def __init__(self, patch_length: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, dropout=0., emb_dropout=0., layernorm_eps=1.0e-5):
        super().__init__()
        self.patch_length = patch_length
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(patch_length + 1, width))
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, dropout=dropout, layernorm_eps=layernorm_eps)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, output_last_layer=False, return_dict=False):
        output_last_layer = return_dict  # output_last_layer deprecrated
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        bsz, cur_patch_length = x.shape[:2]
        # pos emb，如果不够的用最后一个，比较hacky，先这么用着
        pos_emb = self.positional_embedding[-1].repeat(cur_patch_length, 1)
        pos_emb[:self.patch_length + 1] = self.positional_embedding[:cur_patch_length]
        x = x + pos_emb
        x = self.emb_dropout(x)
        x = self.ln_pre(x)

        # x = x.permute(1, 0, 2)  # NLD -> LND  # only for torch native mha
        x = self.transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD  # only for torch native mha

        cls_out = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            cls_out = cls_out @ self.proj
        # TODO: 这样的判断可能会影响性能？看看怎么优雅的改
        if output_last_layer:
            return {'feature_map': x, 'pooled_out': cls_out}
        else:
            return cls_out


def visual_transformer_B32(output_dim, dropout=0.1, emb_dropout=0.1, layernorm_eps=1.0e-5, patch_length=49):
    """
    patch length: 是vit将图片分块的数量。默认是224x224大小的图片，按32像素分块，共分成7x7=49块。
    """
    model = VisualTransformer(patch_length, 32, 768, 12, 12, output_dim, dropout=dropout, emb_dropout=emb_dropout, layernorm_eps=layernorm_eps)
    return model


def visual_transformer_B16(output_dim, dropout=0.1, emb_dropout=0.1, layernorm_eps=1.0e-5, patch_length=196):
    """
    patch length: 是vit将图片分块的数量。默认是224x224大小的图片，按16像素分块，共分成14x14=196块。
    """
    model = VisualTransformer(patch_length, 16, 768, 12, 12, output_dim, dropout=dropout, emb_dropout=emb_dropout, layernorm_eps=layernorm_eps)
    return model
