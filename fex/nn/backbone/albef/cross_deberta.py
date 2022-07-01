#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
有 cross attention 版本的 deberta
"""
from typing import Optional, List
import torch

from ptx.model.deberta.model import DebertaEncoder, DebertaEncoderLayer
from ptx.model.deberta.disentangled_attention import DisentangledMHA  # , FastDisentangledMHA, FTDisentangledMHA
from ptx.model.deberta.disentangled_attention import DAConfig, build_relative_position
from ptx.ops.transformer import _expand_mask, MultiHeadAttention
from fex.nn.module.xbert import BertAttention, BertConfig, BertSelfAttention


class CrossDebertaEncoderLayer(DebertaEncoderLayer):
    attention_class = DisentangledMHA

    def __init__(self, config: DAConfig, **kwargs):
        if isinstance(config, dict):
            config = DAConfig(**config)
        super().__init__(config)
        self.n_heads = config.n_heads
        cross_config = BertConfig.from_dict({'hidden_size': config.dim, 'num_attention_heads': config.n_heads, 'encoder_width': config.dim})
        self.cross_attn = BertAttention(cross_config, is_cross_attention=True)
        # self.if_encoder_empty = config.get('if_encoder_empty', 'default') # [default, skip, self]
        self.if_encoder_empty = 'default'

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor,
        attn_mask_expanded: torch.Tensor,
        relative_pos: torch.Tensor,
        relative_pos_embed: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attn_mask: torch.Tensor,
        q_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        # 1. self attention
        # residual = hidden_states
        residual = hidden_states if q_state is None else q_state
        self_attn_output = self.attn(
            hidden_states, attn_mask=attn_mask_expanded,
            # The only diff from original Transformer encoder layer
            relative_pos=relative_pos, relative_pos_embed=relative_pos_embed,
            q_state=q_state,
        )
        if not self.config.obey_other_attn_output:
            hidden_states = self.proj(self_attn_output)
        else:
            hidden_states = self.proj(self_attn_output.attn_outputs)
        hidden_states = self.dropout1(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.norm1(hidden_states)

        # 2. cross attention
        if self.if_encoder_empty == 'default':
            cross_attention_outputs = self.cross_attn(
                hidden_states=hidden_states,
                attention_mask=attn_mask,
                head_mask=None,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attn_mask,
                output_attentions=True,
                # cosine_facter=self.cosine_facter,
            )
            hidden_states = cross_attention_outputs[0]
            cross_attn_ewights = cross_attention_outputs[1]
        elif self.if_encoder_empty == 'skip':
            is_encoder_mask = encoder_attn_mask[:, 0, 0, 0] < 0.  # [bsz], 1 表示被mask，用原始的
            cross_attention_outputs = self.cross_attn(
                hidden_states=hidden_states,
                attention_mask=attn_mask,
                head_mask=None,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attn_mask,
                output_attentions=True,
                # cosine_facter=self.cosine_facter,
            )
            hidden_states = hidden_states * is_encoder_mask + cross_attention_outputs[0] * (~is_encoder_mask)
            cross_attn_ewights = cross_attention_outputs[1]

        # 3. Position-wise Feed-Forward Networks
        residual = hidden_states
        hidden_states = self.pwff(hidden_states)
        hidden_states = self.dropout2(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.norm2(hidden_states)

        return hidden_states, cross_attn_ewights


class CrossDebertaEncoder(DebertaEncoder):
    layer_class = CrossDebertaEncoderLayer

    def __init__(self, config: DAConfig):
        super().__init__(config)

    def forward(self, hidden_states, encoder_hidden_states, attn_mask=None, encoder_attn_mask=None, output_attentions=False) -> List[torch.Tensor]:

        attn_mask_expanded = _expand_mask(attn_mask, hidden_states.dtype) if attn_mask.dim() != 4 else attn_mask

        relative_pos = build_relative_position(
            hidden_states.size(-2),
            hidden_states.size(-2),
            bucket_size=self.position_buckets,
            max_position=self.max_relative_positions,
            device=hidden_states.device,
        )

        all_hidden_states = []
        if 0 in self.config.return_layers:
            all_hidden_states.append(hidden_states)
        all_attn_weights = []
        # all_attn_probs = []
        # all_q_states = []
        # all_k_states = []
        # all_v_states = []

        for idx, block in enumerate(self.blocks):
            hidden_states, attn_weights = block(
                hidden_states, attn_mask, attn_mask_expanded,
                relative_pos, self.rel_embeddings.weight,
                encoder_hidden_states, encoder_attn_mask,
            )
            if (idx + 1) in self.config.return_layers:
                all_hidden_states.append(hidden_states)
                all_attn_weights.append(attn_weights)

        return hidden_states, all_hidden_states, relative_pos, all_attn_weights
