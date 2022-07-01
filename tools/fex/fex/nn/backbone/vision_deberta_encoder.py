#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
vision + deberta
https://arxiv.org/abs/2006.03654
"""

from typing import Dict, List, Optional
from functools import partial

import torch
import torch.nn as nn

from fex import _logger as log
from fex.nn.visual_tokenizer import create_visual_tokenizer
from fex.config import CfgNode

try:
    from ptx.model.deberta.model import DebertaBare, DebertaEncoder, FastDebertaEncoder
    from ptx.model.bert import BertPooler, init_weights
    from ptx.model.deberta.disentangled_attention import DAConfig, build_relative_position
    from ptx.model.deberta.disentangled_attention import DisentangledMHA  # , FastDisentangledMHA, FTDisentangledMHA
except Exception as e:
    log.warning('ptx is not installed!')


class VisionDeberta(nn.Module):
    """
    只涉及前向，可以用来作为下游finetune时的编码的backbone
    https://code.byted.org/nlp/ptx/blob/master/ptx/model/deberta/model.py#L158
    """

    def __init__(self,
                 config: CfgNode = None,
                 vocab_size: int = 1000,
                 dim: int = 768,
                 dim_ff: int = 3072,
                 n_segments: int = 16,
                 p_drop_hidden: float = 0.1,
                 n_heads: int = 12,
                 n_layers: int = 12,
                 initializer_range: float = 0.02,
                 layernorm_type: str = 'default',
                 layernorm_fp16: bool = False,
                 layer_norm_eps: float = 1e-12,
                 p_drop_attn: float = 0.1,
                 embedding_dim: int = None,
                 embedding_dropout: float = None,
                 act: str = 'gelu',
                 pool: bool = True,
                 padding_index: int = 0,
                 attention_clamp_inf: bool = False,
                 max_relative_positions: int = 512,
                 extra_da_transformer_config: Optional[Dict] = None,
                 omit_other_output: bool = False,
                 use_fast: bool = False,
                 visual_front: bool = False,
                 visual_type: str = 'VitB32',
                 visual_dim: int = 512,
                 middle_size: int = 128,
                 visual_config: dict = None,
                 **kwargs
                 ):
        """
        Args:
            option: See `DebertaEncoder` and `BertEmbedding`
            middle_size: 视觉部分压缩到的embedding 维度，默认为128
        """
        super().__init__()
        if config is not None:
            vocab_size, dim, dim_ff, n_segments, p_drop_hidden, n_heads, n_layers, \
                initializer_range, layernorm_type, layernorm_fp16, layer_norm_eps, p_drop_attn, \
                embedding_dim, embedding_dropout, act, pool, padding_index, attention_clamp_inf, \
                max_relative_positions, extra_da_transformer_config, omit_other_output, use_fast, \
                visual_front, visual_type, visual_dim, middle_size, visual_config = self.load_from_config(config)
        self.visual_front = visual_front

        self.embedding = VEmbedding(dim=dim,
                                    vocab_size=vocab_size,
                                    n_segments=n_segments,
                                    max_len=0,  # 0 是因为没有绝对位置编码
                                    p_drop_hidden=p_drop_hidden if embedding_dropout is None else embedding_dropout,
                                    layer_norm_eps=layer_norm_eps,
                                    layernorm_type=layernorm_type,
                                    embedding_dim=embedding_dim,
                                    visual_front=visual_front
                                    )
        if layernorm_fp16:
            self.embedding.norm._simply_cast = True

        self.proj_embedding_hidden = None
        if embedding_dim != dim:
            self.proj_embedding_hidden = torch.nn.Linear(embedding_dim, dim)

        self._omit_other_output = omit_other_output
        self.da_config = DAConfig(
            n_layers, dim, n_heads, dim_ff,
            act=act,
            layernorm_type=layernorm_type,
            p_drop_hidden=p_drop_hidden,
            p_drop_attn=p_drop_attn,
            return_layers=list(range(n_layers + 1)) if not self._omit_other_output else [],
            clamp_inf_nan=attention_clamp_inf,
            layer_norm_eps=layer_norm_eps,
            max_relative_positions=max_relative_positions,
            mha_acts_unite_d01=False,
            **(extra_da_transformer_config or {}),
        )

        self.encoder = (DebertaEncoder if not use_fast else FastDebertaEncoder)(self.da_config)

        self.pooler = BertPooler(dict(dim=dim)) if pool else None

        self.apply(partial(init_weights, initializer_range=initializer_range))

        self.padding_index = padding_index

        # visual auto-encoder
        self.visual = create_visual_tokenizer(visual_type, **(visual_config))
        # 对visual embedding 有一个映射，主要作用是用来做压缩存储。
        self.middle_size = middle_size
        self.v_projector = torch.nn.Sequential(
            torch.nn.Linear(visual_dim, middle_size),
            torch.nn.Tanh(),
            torch.nn.Linear(middle_size, embedding_dim)
        )

    def forward(self, mode='tv', *args, **kwargs):
        """ mode 来判断需要什么foward
        mode = tv: 视觉文本一同编码
        mode = v: 只编码视觉
        mode = t: 只编码文本
        """
        if mode == 'tv':
            return self.text_visual_forward(*args, **kwargs)
        elif mode == 't':
            return self.text_only_forward(*args, **kwargs)
        elif mode == 'v':
            return self.visual_only_forward(*args, **kwargs)

    def text_visual_forward(self, input_ids, input_segment_ids, input_mask,
                            frames=None, frames_mask=None,  # deprecated
                            image=None, image_mask=None,
                            visual_embeds=None, *args, **kwargs):
        """
        先两个模态一起拼接过 encoder
        如果 visual_embs 不为空，就直接用，否则会用frames 来现算
        注意：这里没有做frames为空的检查。其实不太好。
        """
        frames = image if image is not None else frames
        frames_mask = image_mask if image_mask is not None else frames_mask
        if visual_embeds is None:
            visual_embeds = self.encode_frames(frames)
        if visual_embeds.shape[-1] != self.middle_size:
            visual_embeds = self.v_projector[0](visual_embeds)
        frames_emb = self.project_frames_to_emb_size(visual_embeds)
        embeddings, m_input_mask = self.embedding(input_ids=input_ids,
                                                  token_type_ids=input_segment_ids,
                                                  input_mask=input_mask,
                                                  visual_embeds=frames_emb,
                                                  visual_mask=frames_mask,
                                                  mode='tv')

        if self.proj_embedding_hidden is not None:
            embeddings = self.proj_embedding_hidden(embeddings)

        sequence_output, encoder_all_hidden_states, relative_pos = self.encoder(embeddings, m_input_mask)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        if self.visual_front:
            visual_length = frames_mask.shape[1]  # frame_num
            visual_final_out = sequence_output[:, 1:visual_length]  # 第0个位置是cls，不要
            text_final_out = torch.cat([sequence_output[:, :1], sequence_output[:, visual_length + 1:]], dim=1)  # +1 是因为cls
        else:
            text_length = input_mask.shape[1]
            text_final_out = sequence_output[:, :text_length]
            visual_final_out = sequence_output[:, text_length:]

        return {'encoded_layers': encoder_all_hidden_states,
                'pooled_output': pooled_output,
                'text_final_output': text_final_out,     # [CLS], t1, ..., tm, [SEP]
                'visual_final_output': visual_final_out,  # f1, ... fn
                'visual_tower_output': visual_embeds,
                'embeddings': embeddings,
                'embedding_masks': m_input_mask,
                'relative_pos': relative_pos}

    def visual_only_forward(self,
                            frames: torch.Tensor,
                            frames_mask: torch.Tensor,
                            visual_embeds: torch.Tensor = None,
                            *args, **kwargs
                            ):
        if visual_embeds is None:
            visual_embeds = self.encode_frames(frames)
        if visual_embeds.shape[-1] != self.middle_size:
            visual_embeds = self.v_projector[0](visual_embeds)
        frames_emb = self.project_frames_to_emb_size(visual_embeds)
        embeddings, input_mask = self.embedding(visual_embeds=frames_emb,
                                                visual_mask=frames_mask,
                                                mode='v')

        if self.proj_embedding_hidden is not None:
            embeddings = self.proj_embedding_hidden(embeddings)

        sequence_output, encoder_all_hidden_states, relative_pos = self.encoder(embeddings, input_mask)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        return {'tower_output': visual_embeds,
                'frames_emb': frames_emb,
                'encoded_layers': encoder_all_hidden_states,
                'pooled_output': pooled_output,
                'embeddings': embeddings,
                'embedding_masks': input_mask,
                'relative_pos': relative_pos}

    def text_only_forward(
        self,
        input_ids=None,
        input_segment_ids=None,
        input_mask=None,
        output_pooled=True,
        output_rel_pos=False,
        position_ids=None,  # Useless, for api compat
        *args, **kwargs
    ):
        """ 只有文本输入的forward """
        if input_mask is None:
            input_mask = (input_ids != self.padding_index).to(dtype=self.embedding.token_embedder_tokens.weight.dtype)

        embeddings, input_mask = self.embedding(
            input_ids=input_ids,
            token_type_ids=input_segment_ids,
            # position_ids=position_ids, # 注意这里不传position ids
            input_mask=input_mask,
            mode='t'
        )

        if self.proj_embedding_hidden is not None:
            embeddings = self.proj_embedding_hidden(embeddings)

        sequence_output, encoder_all_hidden_states, relative_pos = self.encoder(embeddings, input_mask)
        pooled_output = None
        if (self.pooler is not None) and output_pooled:
            pooled_output = self.pooler(sequence_output)

        if self._omit_other_output:
            ret = {'encoded_layers': sequence_output}
            if output_pooled:
                ret['pooled_output'] = pooled_output
            else:
                ret['pooled_output'] = None
            if output_rel_pos:
                if relative_pos is None:  # In case of FT, which does not return rel pos
                    # print('Recreating relative_pos')
                    relative_pos = build_relative_position(
                        embeddings.size(-2),
                        embeddings.size(-2),
                        bucket_size=self.encoder.position_buckets,
                        max_position=self.encoder.max_relative_positions,
                        device=embeddings.device,
                    )
                ret['relative_pos'] = relative_pos
            return ret
        else:
            return {
                'encoded_layers': encoder_all_hidden_states,
                'pooled_output': pooled_output,
                'relative_pos': relative_pos,
                'embeddings': embeddings,
                'embedding_masks': input_mask,
                'embedding': embeddings,
                'hidden': encoder_all_hidden_states,
                'attention': None,
            }

    def encode_frames(self, frames):
        """ encode 到 128 维度 """
        N, F, C, H, W = frames.shape
        frames = torch.reshape(frames, [N * F, C, H, W])
        emb_itm = self.visual(frames)
        emb_itm = emb_itm.reshape([N, F, -1])  # [N, F, dim]
        emb = self.v_projector[0](emb_itm)
        # print(emb.shape, emb, torch.max(emb), torch.min(emb), 'emb_itm')
        return emb

    def project_frames_to_emb_size(self, emb):
        """
        把 128 的 frame embedding 映射到 word embedding size """
        emb = self.v_projector[1](emb)  # 有点危险
        emb = self.v_projector[2](emb)
        return emb

    def load_from_config(self, config):
        """ 从config加载参数 """
        params = ['vocab_size', 'dim', 'dim_ff', 'n_segments', 'p_drop_hidden', 'n_heads',
                  'n_layers', 'initializer_range', 'layernorm_type', 'layernorm_fp16',
                  'layer_norm_eps', 'p_drop_attn', 'embedding_dim', 'embedding_dropout',
                  'act', 'pool', 'padding_index', 'attention_clamp_inf', 'max_relative_positions',
                  'extra_da_transformer_config', 'omit_other_output', 'use_fast', 'visual_front',
                  'visual_type', 'visual_dim', 'middle_size', 'visual_config']
        return [getattr(config.BERT, p, None) for p in params]


class VEmbedding(nn.Module):
    """
    https://code.byted.org/nlp/ptx/blob/master/ptx/core/bert.py#L340
    Construct the embeddings from word, position and token_type embeddings.
    注意这个和 falbert 下的 VEmbedding有两个区别：
    1. 视觉不再加 [IMG]
    2. 如果传入的position id 是 None，不再加位置编码
    """

    def __init__(self,
                 dim: int = 768,
                 vocab_size: int = 1000,
                 n_segments: int = 2,
                 max_len: int = 8,
                 p_drop_hidden: float = None,
                 layer_norm_eps: float = 1e-12,
                 layernorm_type: str = 'default',
                 embedding_dim: int = 256,
                 padding_index: int = 2,
                 project_embedding_first: bool = False,
                 n_frames: int = 16,
                 need_visual_ln: bool = True,
                 visual_front: bool = False):
        super().__init__()

        self.project_embedding_first = project_embedding_first
        self.token_embedder_tokens = torch.nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_index)
        # self.token_embedder_positions = torch.nn.Embedding(max_len, p_dim) # TODO: fix it
        self.token_embedder_segments = torch.nn.Embedding(n_segments, embedding_dim)

        self.norm = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)
        self.dropout = torch.nn.Dropout(p_drop_hidden)

        self.need_visual_ln = need_visual_ln
        if need_visual_ln:
            self.visual_ln = nn.LayerNorm(embedding_dim, eps=layer_norm_eps)

        self.v_segment_embeddings = torch.nn.Embedding(1, embedding_dim)
        # self.v_token_embedder_positions = torch.nn.Embedding(n_frames, p_dim)
        # TODO: 是否要用上面的做初始化
        # self.v_token_embedder_positions.weight = self.token_embedder_positions.weight[:config.max_frame_num]

        self.is_visual_front = visual_front

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, input_mask=None, visual_embeds=None, visual_mask=None, mode='tv'):
        """
        embedding 构造。
        文本端：token embedding + position embedding + segment embedding
        视觉端：visual embedding + position embedding + segment embedding

        几个点:
        1. 两个模态的position embedding和 segment embedding是分开的
        """
        if mode == 't':
            embeddings = self.text_forward(input_ids, token_type_ids, position_ids)
        elif mode == 'v':
            embeddings, input_mask = self.visual_forward(visual_embeds, visual_mask)
        elif mode == 'tv':
            # 文本
            embeddings = self.text_forward(input_ids, token_type_ids, position_ids)
            # 视觉
            v_embeddings, v_input_mask = self.visual_forward(visual_embeds, visual_mask)

            if self.is_visual_front:
                # [cls] img_1, ..., img_n, [SEP], text ... [SEP]
                embeddings = torch.cat([embeddings[:, 0, :].unsqueeze(1), v_embeddings, embeddings[:, 1:, :]], dim=1)
                input_mask = torch.cat([input_mask[:, 0].unsqueeze(1), v_input_mask, input_mask[:, 1:]], dim=1)
            else:
                embeddings = torch.cat([embeddings, v_embeddings], dim=1)
                input_mask = torch.cat([input_mask, v_input_mask], dim=1)
        else:
            raise ValueError('Unknown mode [%s] in VEmbedding forward' % mode)

        # 后处理
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, input_mask

    def text_forward(self, input_ids, token_type_ids, position_ids=None):
        inputs_embeds = self.token_embedder_tokens(input_ids)
        # position
        bsz, length = inputs_embeds.size()[:2]
        if position_ids is None:
            position_embeddings = 0.
        else:
            position_embeddings = self.token_embedder_positions(position_ids)
        # segment
        token_type_embeddings = self.token_embedder_segments(token_type_ids)
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        return embeddings

    def visual_forward(self, visual_embeds, visual_mask, position_ids=None, *args, **kwargs):
        # 1. token
        if self.need_visual_ln:
            visual_embeds = self.visual_ln(visual_embeds)
        bsz, visual_length = visual_embeds.size()[:2]
        inputs_embeds = visual_embeds
        # 5. position embedding
        if position_ids is None:
            position_embeddings = 0.
        else:
            position_embeddings = self.v_token_embedder_positions(position_ids)  # fix
        #position_embeddings = self.token_embedder_positions(position_ids)
        # 6. segment embedding
        segment_embeddings = self.v_segment_embeddings(torch.zeros_like(visual_mask, device=visual_mask.device, dtype=torch.long))
        # 7. 后处理
        embeddings = inputs_embeds + position_embeddings + segment_embeddings
        return embeddings, visual_mask
