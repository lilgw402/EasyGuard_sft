#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: Huang Wenguan (huangwenguan@bytedance.com)
@Date: 2020-05-13 16:07:07
LastEditTime: 2020-11-16 19:53:55
LastEditors: Huang Wenguan
@Description: ALBert的一版实现，加载archer预训练好的tf模型用。
链路是archer->ptx->our_module
实现上尽可能参考ptx，减少参数名映射
'''

from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptx.ops.transformer import TransformerEncoder as _PTXTransformerEncoder

from .with_ptx import TransformerEncoderLayer, USE_PTX_TRANSFORMER_CONF_ALBERT


class ALBert(nn.Module):
    """ ALBert Backbone，其实是Bert的超集，比Bert多了embedding projection
        但和传统意义的albert不一样，没有实现layer共享
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.embedding = BertEmbedding(config, padding_index=2)
        self.encoder = Transformer(config)
        if self.config.with_pooler:
            self.pooler = BertPooler(config)

        # init weights
        self.apply(self.init_weights)

        if self.config.word_embedding_frozen:
            for p in self.word_embeddings.parameters():
                p.requires_grad = False

        if config.frozen_layers is not None and config.frozen_layers >= 1:
            self.frozen_parameters(config.frozen_layers)

    def init_weights(self, module):
        """ Initialize the weights. # TODO: 需要吗
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, input_segment_ids, input_mask, *args, **kwargs):
        embeddings = self.embedding(input_ids=input_ids, token_type_ids=input_segment_ids, *args, **kwargs)
        out = self.encoder(embeddings, input_mask)
        if isinstance(out, tuple):
            encoded_layers, attention_probs = out
        else:
            encoded_layers = out
            attention_probs = None
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output) if self.config.with_pooler else None

        return {'encoded_layers': encoded_layers,
                'pooled_output': pooled_output,
                'attention_probs': attention_probs}


class BertEmbedding(nn.Module):
    """
    https://code.byted.org/nlp/ptx/blob/master/ptx/core/bert.py#L340
    Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config, padding_index=2):
        super().__init__()

        self.project_embedding_first = config.project_embedding_first
        dim = config.hidden_size if self.project_embedding_first else config.embedding_size
        self.token_embedder_tokens = torch.nn.Embedding(
            config.vocab_size, config.embedding_size, padding_idx=padding_index)
        self.token_embedder_positions = torch.nn.Embedding(config.max_position_embeddings, dim)
        self.token_embedder_segments = torch.nn.Embedding(config.type_vocab_size, dim)

        self.norm = nn.LayerNorm(dim, eps=config.layernorm_eps)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        if config.embedding_size != config.hidden_size:
            self.proj_embedding_hidden = torch.nn.Linear(config.embedding_size, config.hidden_size)
        else:
            self.proj_embedding_hidden = None

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        # self.register_buffer('position_ids', torch.arange(max_len).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        """
        支持传inputs_embeds，来代替token-embedding，这个不错
        """
        if inputs_embeds is None:
            inputs_embeds = self.token_embedder_tokens(input_ids)

        # 先 project
        if self.project_embedding_first and self.proj_embedding_hidden:
            inputs_embeds = self.proj_embedding_hidden(inputs_embeds)

        bsz, length = inputs_embeds.size()[:2]
        if position_ids is None:
            position_ids = torch.arange(0, length, dtype=torch.long, device=input_ids.device).expand(bsz, length)

        position_embeddings = self.token_embedder_positions(position_ids)
        token_type_embeddings = self.token_embedder_segments(token_type_ids)
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)

        # 后 project
        if not self.project_embedding_first and self.proj_embedding_hidden:
            embeddings = self.proj_embedding_hidden(embeddings)

        return embeddings


class BertPooler(nn.Module):
    """
    https://code.byted.org/nlp/ptx/blob/master/ptx/core/model.py#L110
    """

    def __init__(self, config):
        """
        Args:
            option:
                dim:
        """
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None, mask_value: Optional[int] = None):
    """
    Expands attn_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    if mask_value is None:
        mask_value = torch.finfo(dtype).min
    return inverted_mask.masked_fill(inverted_mask.bool(), mask_value)


class Transformer(_PTXTransformerEncoder):
    def __init__(self, config):
        default_config = dict(
            n_layers=config.num_hidden_layers,
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            dim_ff=config.intermediate_size,
            act='gelu',
            p_drop_hidden=config.hidden_dropout_prob,
            p_drop_attn=config.hidden_dropout_prob,
            return_layers=[i+1 for i in range(config.num_hidden_layers)],
            clamp_inf_nan=False,
            layer_norm_eps=config.layernorm_eps,
            max_batch_size=128,
            max_seq_length=512,
            fp16=True,
            layernorm_fp16=True,
            fuse_qkv_projs=False,
            omit_other_attn_output=True,
            layer_grad_checkpoint=False,
            use_pre_layernorm=False,
            layernorm_type='default',
            pos_emb_type=config.get('pos_emb_type', ''),
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
            dropout_in_ffn=False,
            use_ft_ffn_linear_fusion=True,
            use_ffn_output_dropout=True,
            use_ft_attn_out_proj_dropout_fusion=True,
            pad_seq_len_even=True,
        )
        default_config.update(ft_config) # ptx trace的时候注释掉这句
        default_config.update(USE_PTX_TRANSFORMER_CONF_ALBERT)
        super().__init__(default_config)
        self.attn_weights = None

    def get_output_dim(self):
        return self.dim

    def forward(self, h, mask):
        assert mask.dim() == 2
        mask = _expand_mask(mask, h.dtype)
        super_output = super().forward(h, mask)
        all_layer_outputs = super_output.all_hidden_states
        return all_layer_outputs, None


class _Transformer(nn.Module):
    """ Transformer with Self-Attentive Blocks"""

    def __init__(self, config):
        """
        """
        super().__init__()
        self.blocks = nn.ModuleList([self._create_layer(config) for _ in range(config.num_hidden_layers)])
        self.dim = config.hidden_size
        self.attn_weights = None

    def get_output_dim(self):
        return self.dim

    def _create_layer(self, config):
        default_config = dict(
            n_layers=config.num_hidden_layers,
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            dim_ff=config.intermediate_size,
            act='gelu',
            p_drop_hidden=config.hidden_dropout_prob,
            p_drop_attn=config.hidden_dropout_prob,
            return_layers=[],
            clamp_inf_nan=False,
            layer_norm_eps=config.layernorm_eps,
            max_batch_size=128,
            max_seq_length=512,
            fp16=True,
            layernorm_fp16=True,
            fuse_qkv_projs=False,
            omit_other_attn_output=True,
            layer_grad_checkpoint=False,
            layernorm_type='ft',
            use_pre_layernorm=False,
            use_ft_layernorm=True,
            use_ft_softmax=True,
            use_ft_linear_in_attn=True,
            use_ft_transpose_in_attn=True,
            use_ft_mm_in_attn=True,
            use_ft_linear_in_attn_out=True,
            use_ft_linear_in_ffn=True,
            mha_acts_unite_d01=True,
            dropout_in_ffn=False,
            use_ft_ffn_linear_fusion=True,
            use_ffn_output_dropout=True,
            use_ft_attn_out_proj_dropout_fusion=True,
            pad_seq_len_even=True,
        )
        default_config.update(USE_PTX_TRANSFORMER_CONF_ALBERT)
        return TransformerEncoderLayer(default_config)

    def forward(self, h, mask):
        # seq_len = h.size(1)
        # if seq_len % 2 != 0:
        #     raise ValueError(f'Sequence length is {seq_len}, which must be an even number (%2==0) instead.')
        assert mask.dim() == 2
        mask = _expand_mask(mask, h.dtype)
        all_layer_outputs = []
        for block in self.blocks:
            h = block(h, mask)
            h = h[0]  # `ptx.ops.transformer.TransformerEncoderLayerOutput`
            all_layer_outputs.append(h)
        return all_layer_outputs, None


class AlbertLMPredictionHead(torch.nn.Module):
    """ albert的预测head """

    def __init__(self, config, embedding_weights):
        """
        Args:
            option:
                dim:
                embedding_dim:
                layer_norm_eps:
                vocab_size:
        """
        super().__init__()
        self.dense = torch.nn.Linear(config.hidden_size, config.embedding_size)
        self.activation = gelu_new
        self.layer_norm = nn.LayerNorm(config.embedding_size, eps=config.layernorm_eps)

        #self.decoder = torch.nn.Linear(config.embedding_size, config.vocab_size, bias=False)
        self.decoder = nn.Linear(embedding_weights.size(1),
                                 embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = embedding_weights

        self.bias = torch.nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.activation(self.dense(hidden_states))
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    # return 0.5 * x * (1 + torch.tanh(math.sqrt(math.pi / 2) * (x + 0.044715 * x ** 3)))
