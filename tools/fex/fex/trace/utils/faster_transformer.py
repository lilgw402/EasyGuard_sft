# -*- coding: utf-8 -*-
'''
Created on Jan-12-21 18:13
faster_transformer.py
@author: liuzhen.nlp
Description:
'''

from typing import List, Dict, Tuple, Union
from copy import deepcopy
import os
import sys
from fex import _logger as logger

import torch

from fex.nn.backbone.albert import Transformer
from fex.nn.backbone.albert_v2 import Transformer as TransformerV2

FT_VERSION = int(os.getenv('FT_VERSION', '3'))
FT_MAP = {
    'torch_1.5_ft_v2': 'libths_fastertransformer_op.so',
    'torch_1.5_ft_v3': 'libths_fastertransformer_op_py15_ftv3.so',
    'torch_1.6_ft_v2': 'libths_fastertransformer_op_py16.so',
    'torch_1.6_ft_v3': 'libths_fastertransformer_op_py16_ftv3.so'
}

lib_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                       "libs/ftops")

if torch.__version__.startswith("1.5"):
    soname = 'torch_1.5'
elif torch.__version__.startswith("1.6.0"):
    soname = 'torch_1.6'
elif torch.__version__.startswith("1.3.1"):
    soname = 'torch.1.5'
elif torch.__version__.startswith("1.8"):
    import xperf.ops
    xperf.ops.load_ft_torch()
    soname = None
else:
    print("Only support 1.3.1 && 1.5.0 && 1.6.0 && 1.8.1, \
                but the current torch_version: {}"                                                                                                    .format(torch.__version__))
    soname = ""

if soname:
    soname = '%s_ft_v%d' % (soname, FT_VERSION)
    sopath = FT_MAP[soname]
    logger.info("[Fast Transformer] use [%s] %s" % (soname, sopath))
    so = os.path.join(lib_dir, sopath)
    torch.ops.load_library(so)


class BertEncoderWeights:
    """[summary]
    """

    def __init__(
        self,
        n_layers: int,
        weights: Union[Dict[str, torch.Tensor], Transformer, TransformerV2],
    ):
        self.n_layers = n_layers
        self.w = [[] for _ in range(n_layers)]

        if isinstance(weights, dict):
            # When `weights` is a state dict
            for i in range(n_layers):
                p = 'blocks.%s.' % i
                self.w[i].append(weights[p + 'attn.proj_q.weight'].transpose(
                    -1, -2).contiguous())
                self.w[i].append(weights[p + 'attn.proj_k.weight'].transpose(
                    -1, -2).contiguous())
                self.w[i].append(weights[p + 'attn.proj_v.weight'].transpose(
                    -1, -2).contiguous())
                self.w[i].append(weights[p + 'attn.proj_q.bias'])
                self.w[i].append(weights[p + 'attn.proj_k.bias'])
                self.w[i].append(weights[p + 'attn.proj_v.bias'])
                self.w[i].append(weights[p + 'proj.weight'].transpose(
                    -1, -2).contiguous())
                self.w[i].append(weights[p + 'proj.bias'])
                self.w[i].append(weights[p + 'norm1.gamma'])
                self.w[i].append(weights[p + 'norm1.beta'])
                self.w[i].append(weights[p + 'pwff.fc1.weight'].transpose(
                    -1, -2).contiguous())
                self.w[i].append(weights[p + 'pwff.fc1.bias'])
                self.w[i].append(weights[p + 'pwff.fc2.weight'].transpose(
                    -1, -2).contiguous())
                self.w[i].append(weights[p + 'pwff.fc2.bias'])
                self.w[i].append(weights[p + 'norm2.gamma'])
                self.w[i].append(weights[p + 'norm2.beta'])

        elif isinstance(weights, (Transformer, TransformerV2)):
            # When `weights` is a Transformer instance
            for i in range(n_layers):
                self.w[i].append(
                    weights.blocks[i].attn.proj_q.weight.data.transpose(
                        -1, -2).contiguous())
                self.w[i].append(
                    weights.blocks[i].attn.proj_k.weight.data.transpose(
                        -1, -2).contiguous())
                self.w[i].append(
                    weights.blocks[i].attn.proj_v.weight.data.transpose(
                        -1, -2).contiguous())
                self.w[i].append(weights.blocks[i].attn.proj_q.bias.data)
                self.w[i].append(weights.blocks[i].attn.proj_k.bias.data)
                self.w[i].append(weights.blocks[i].attn.proj_v.bias.data)
                self.w[i].append(weights.blocks[i].proj.weight.data.transpose(
                    -1, -2).contiguous())
                self.w[i].append(weights.blocks[i].proj.bias.data)
                self.w[i].append(weights.blocks[i].norm1.weight.data)
                self.w[i].append(weights.blocks[i].norm1.bias.data)
                self.w[i].append(
                    weights.blocks[i].pwff.fc1.weight.data.transpose(
                        -1, -2).contiguous())
                self.w[i].append(weights.blocks[i].pwff.fc1.bias.data)
                self.w[i].append(
                    weights.blocks[i].pwff.fc2.weight.data.transpose(
                        -1, -2).contiguous())
                self.w[i].append(weights.blocks[i].pwff.fc2.bias.data)
                self.w[i].append(weights.blocks[i].norm2.weight.data)
                self.w[i].append(weights.blocks[i].norm2.bias.data)

    def to_cuda(self):
        for i in range(self.n_layers):
            for j in range(len(self.w[i])):
                self.w[i][j] = self.w[i][j].cuda()

    def to_half(self):
        for i in range(self.n_layers):
            for j in range(len(self.w[i])):
                self.w[i][j] = self.w[i][j].half()


class FTEncoderLayer(torch.nn.Module):
    def __init__(self,
                 layer_num,
                 head_num,
                 head_size,
                 weights,
                 use_fp16=False,
                 not_cast_layer_norm=False,
                 is_remove_padding=False):
        super().__init__()
        self.layer_num = int(layer_num)
        self.head_num = int(head_num)
        self.head_size = int(head_size)
        self.w = torch.nn.ParameterList(
            [torch.nn.Parameter(w) for w in weights])
        self.not_cast_layer_norm = not_cast_layer_norm
        self.is_remove_padding = is_remove_padding
        if use_fp16:
            for i in range(16):
                if not_cast_layer_norm and i in [8, 9, 14, 15]:  # layer norm
                    continue
                self.w[i].data = self.w[i].data.half()

    def forward(self, hidden_states, attention_mask):
        if torch.__version__.startswith("1.3.1"):
            return torch.ops.fastertransformer.encoder(self.head_num,
                                                       self.head_size, *self.w,
                                                       hidden_states,
                                                       attention_mask)
        if torch.__version__.startswith(
                "1.5") or torch.__version__.startswith(
                    "1.6.0") or torch.__version__.startswith('1.8'):
            return torch.ops.FasterTransformer.BertTransformer(
                self.head_num, self.head_size, *self.w, hidden_states,
                attention_mask, self.is_remove_padding)

        raise Exception("unsupport torch version for fastertransformer")


class FasterTransformerEncoder(torch.nn.Module):
    def __init__(self,
                 layer_num: int,
                 head_num: int,
                 hidden_dim: int,
                 weights: BertEncoderWeights,
                 use_fp16: bool = False,
                 return_all_hidden_states: bool = False,
                 is_remove_padding: bool = False):
        super().__init__()
        self.layer_num = layer_num
        self.head_num = head_num
        self.hidden_dim = hidden_dim
        self.head_size = int(hidden_dim / head_num)
        self.use_fp16 = use_fp16
        self.return_all_hidden_states = return_all_hidden_states
        self.is_remove_padding = is_remove_padding
        assert len(weights.w) == self.layer_num

        self.layers = torch.nn.ModuleList()
        for i in range(self.layer_num):
            layer = FTEncoderLayer(self.layer_num,
                                   self.head_num,
                                   self.head_size,
                                   weights.w[i],
                                   use_fp16=use_fp16,
                                   is_remove_padding=self.is_remove_padding)
            self.layers.append(deepcopy(layer))

    def forward(
            self,
            # (batch_size, seq_len, hidden_dim)
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,  # (batch_size, seq_len, seq_len)
    ):
        if self.use_fp16 and hidden_states.dtype == torch.float32:
            hidden_states = hidden_states.half()

        attention_mask = attention_mask.to(dtype=hidden_states.dtype)
        if attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, :].repeat(
                1,
                hidden_states.size()[1], 1)

        all_encoder_layers: List[torch.Tensor] = []
        for i in range(self.layer_num):
            hidden_states = self.layers[i].forward(hidden_states,
                                                   attention_mask)
            if self.return_all_hidden_states:
                all_encoder_layers.append(hidden_states)

        if not self.return_all_hidden_states:
            all_encoder_layers.append(hidden_states)

        return all_encoder_layers

    @classmethod
    def from_transformer(cls,
                         trans: Union[Transformer, TransformerV2],
                         fp16: bool = False,
                         return_all_hidden_states: bool = False,
                         is_remove_padding: bool = False):
        n_layers = len(trans.blocks)
        n_heads = trans.blocks[0].attn.n_heads
        hidden_dim = trans.blocks[0].proj.in_features
        weights = BertEncoderWeights(n_layers, trans)

        return cls(n_layers,
                   n_heads,
                   hidden_dim,
                   weights,
                   use_fp16=fp16,
                   return_all_hidden_states=return_all_hidden_states,
                   is_remove_padding=is_remove_padding)


def cast_to_fast_transformer(model: torch.nn.Module,
                             **kwargs) -> torch.nn.Module:
    if isinstance(model, (Transformer, TransformerV2)):
        return FasterTransformerEncoder.from_transformer(model, **kwargs)
    submodules = [kv for kv in model._modules.items()]
    for name, module in submodules:
        setattr(model, name, cast_to_fast_transformer(module, **kwargs))
    return model


class FTCustomMHA(torch.nn.Module):
    def __init__(self,
                 n_heads: int,
                 hidden_dim: int,
                 weights: List[torch.Tensor],
                 fp16: bool = False):
        super().__init__()

        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.head_size = int(hidden_dim / n_heads)
        self.weights = weights
        self.fp16 = fp16
        self.op = torch.ops.FasterTransformer.MultiHeadAttention

    def forward(self, query_states, key_states, value_states,
                key_padding_mask):
        """
        Args:
            query_states: [from_seq_len, batch_size, hidden_dim]
            key_states: [to_seq_len, batch_size, hidden_dim]
            attention_mask: [batch_size, to_seq_len]
        """
        query_states = self.op(self.hidden_dim, self.n_heads, *self.weights,
                               query_states.contiguous(),
                               key_states.contiguous(),
                               key_padding_mask.to(dtype=query_states.dtype))
        return (query_states, None)

    @classmethod
    def from_torch_mha(cls,
                       a: torch.nn.MultiheadAttention,
                       fp16: bool = False):
        assert isinstance(
            a, torch.nn.MultiheadAttention
        ), 'Unsupported MultiheadAttention implementation for FT conversion.'
        n_heads = a.num_heads
        hidden_dim = a.embed_dim
        if a._qkv_same_embed_dim is False:
            weights = [
                a.q_proj_weight.transpose(-1, -2).contiguous(),
                a.k_proj_weight.transpose(-1, -2).contiguous(),
                a.v_proj_weight.transpose(-1, -2).contiguous(),
                a.in_proj_bias,
                a.bias_k,
                a.bias_v,
                a.out_proj.weight.transpose(-1, -2).contiguous(),
                a.out_proj.bias,
            ]
        else:
            weights = [
                a.in_proj_weight[:a.embed_dim].transpose(-1, -2).contiguous(),
                a.in_proj_weight[a.embed_dim:2 * a.embed_dim].transpose(-1, -2).contiguous(),
                a.in_proj_weight[2 * a.embed_dim:3 * a.embed_dim].transpose(-1, -2).contiguous(),
                a.in_proj_bias[:a.embed_dim],
                a.in_proj_bias[a.embed_dim:2 * a.embed_dim],
                a.in_proj_bias[2 * a.embed_dim:3 * a.embed_dim],
                a.out_proj.weight.transpose(-1, -2).contiguous(),
                a.out_proj.bias,
            ]

        weights = [w.detach().cuda() for w in weights]
        if fp16:
            weights = [w.half() for w in weights]
        return cls(n_heads, hidden_dim, weights, fp16=fp16)


def cast_to_fast_mha(model: torch.nn.Module, **kwargs) -> torch.nn.Module:
    if not torch.__version__.startswith('1.8'):
        logger.warning("fast mha is only support for torch-1.8, skip convert")
        return model

    logger.warning('mha is not statable, skip it')
    return model

    if isinstance(model, torch.nn.MultiheadAttention):
        model = FTCustomMHA.from_torch_mha(model, **kwargs)
        return model
    submodules = [kv for kv in model._modules.items()]
    for name, module in submodules:
        setattr(model, name, cast_to_fast_mha(module, **kwargs))
    return model
