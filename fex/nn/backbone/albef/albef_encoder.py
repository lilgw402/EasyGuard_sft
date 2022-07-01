#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 albef encode 的部分抽象出来，好处是下游任务只需要依赖这个backbone即可
"""
from typing import Tuple

import os
import torch
import torch.nn.functional as F
from torch import nn

from fex.config import CfgNode
from fex.nn.visual_tokenizer import create_visual_tokenizer
from fex.nn.backbone.albert import ALBert, Transformer

from fex.nn.module.xbert import BertConfig, TransformerCrossEncoder, BertPooler
from .cross_deberta import CrossDebertaEncoder

from ptx.model.deberta.disentangled_attention import DAConfig
from ptx.model.deberta.model import DebertaBare


class ALBEF(nn.Module):
    """
    albef
    模型参数：可以分成3部分，
    1. visual encoder。是一个纯纯的visual encoder，只负责将图片（视频）编码成 feature map
    2. text encoder。  是一个纯纯的text encoder，譬如一个bert，只负责将文本编码成 text embedding
    3. cross encoder。 负责建模图像和文本之间的交互。

    encode:
    几种情况：
    1. visual only：只做visual encoder
    2. frames：适配视频的情况
    3. text only：只过text encoder
    4. cross：交互

    """

    def __init__(self, config: CfgNode):
        super().__init__()
        self.PAD_IDX = 2
        self.dtype = torch.float16  # TODO: hack

        # 定义参数
        self.visual_type = config.network.visual_type
        feature_map_dim = config.network.get('feature_map_dim', 768)
        self.visual = create_visual_tokenizer(self.visual_type, **config.get('network.visual_config', {}))
        self.visual_emb_proj = nn.Linear(feature_map_dim, config.BERT.hidden_size) if feature_map_dim != config.BERT.hidden_size else None

        self.text_backbone_mode = config.network.get('text_type', 'albert_concat')
        if self.text_backbone_mode == 'albert_concat':
            config.BERT.with_pooler = False
            self.text = ALBert(config.BERT)
            self.cross = Transformer(config.network.cross_config)
        elif self.text_backbone_mode == 'albert_cross':
            self.text = ALBert(config.BERT)
            cross_config = BertConfig.from_dict(config.network.cross_config)
            self.cross = TransformerCrossEncoder(cross_config)
        elif self.text_backbone_mode == 'deberta_cross':
            text_config = config.BERT
            self.text = DebertaBare(vocab_size=text_config.vocab_size,
                                    dim=text_config.hidden_size,
                                    dim_ff=text_config.intermediate_size,
                                    n_segments=text_config.type_vocab_size,
                                    p_drop_hidden=text_config.hidden_dropout_prob,
                                    n_heads=text_config.num_attention_heads,
                                    n_layers=text_config.num_hidden_layers,
                                    initializer_range=text_config.initializer_range,
                                    layernorm_type='default',
                                    layernorm_fp16=False,
                                    layer_norm_eps=text_config.layernorm_eps,
                                    p_drop_attn=0.1,
                                    embedding_dim=text_config.embedding_size,
                                    act='gelu',
                                    pool=False,
                                    padding_index=self.PAD_IDX,
                                    attention_clamp_inf=False,
                                    max_relative_positions=text_config.max_position_embeddings,
                                    use_fast=False,
                                    extra_da_transformer_config={'mha_acts_unite_d01': False},
                                    )
            cross_config = config.network.cross_config
            self.cross_config = DAConfig(
                n_layers=cross_config.num_hidden_layers,
                dim=cross_config.hidden_size,
                n_heads=cross_config.num_attention_heads,
                dim_ff=cross_config.intermediate_size,
                act='gelu',
                layernorm_type='default',
                p_drop_hidden=cross_config.hidden_dropout_prob,
                p_drop_attn=cross_config.get('p_drop_attn', 0.1),
                return_layers=list(range(cross_config.num_hidden_layers + 1)),
                clamp_inf_nan=False,
                layer_norm_eps=cross_config.layernorm_eps,
                max_relative_positions=cross_config.max_position_embeddings,
                mha_acts_unite_d01=False,
            )
            self.cross = CrossDebertaEncoder(self.cross_config)

        self.pooler = BertPooler(config.BERT)

    def forward(self, mode='tv', *args, **kwargs):
        """ mode 来判断需要什么foward
        mode = tv: 视觉文本一同编码
        mode = v: 只编码视觉
        mode = t: 只编码文本
        """
        if mode == 'tv':
            return self.encode(*args, **kwargs)
        elif mode == 't':
            return self.text_encode(*args, **kwargs)
        elif mode == 'v':
            return self.visual_encode(*args, **kwargs)

    def encode(self, input_ids, input_segment_ids, input_mask,
               image, image_mask, mode='tv', *args, **kwargs):
        """ mode 来判断需要什么 encode
        mode = tv: 视觉文本一同编码
        mode = v: 只编码视觉
        mode = t: 只编码文本
        TODO: 支持其他mode
        这里有一个问题：如果是没有图片的情况下，纯文本的模型，过cross encoder应该怎么做：
        1. 忽略cross_attention 模块
        2. 把 cross 的
        更优雅的办法是让cross encoder 变成拼接的，这样比较好兼容
        """

        # image
        is_frames = image.dim() == 5
        if is_frames:  # 多图的情况
            bsz, frame_num, c, h, w = image.shape
            image = image.reshape([bsz * frame_num, c, h, w])
            image_mask = image_mask.reshape(-1)

        image_embeds, image_final_token = self.visual_encode(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        image_atts = image_atts * image_mask.unsqueeze(-1)

        if is_frames:
            _, seq_len, dim = image_embeds.shape
            image_embeds = image_embeds.reshape([bsz, frame_num * seq_len, dim])
            image_atts = image_atts.reshape([bsz, -1])

        # text
        text_output = self.text_encode(input_ids=input_ids, input_segment_ids=input_segment_ids, input_mask=input_mask)
        text_embeds = text_output['encoded_layers'][-1]
        # cross
        cross_out = self.cross_encoder_encode(text_embeds, input_mask, image_embeds, image_atts, return_dict=True)
        pooled_output = self.pooler(cross_out['sequence_output'])
        return {'pooled_output': pooled_output,
                'embedding_masks': text_output['embedding_masks'],
                'encoded_layers': cross_out['encoded_layers'],
                }

    def visual_encode(self, image, return_dict=False):
        visual_out = self.visual(image, return_dict=True)
        if self.visual_type == 'SwinVidtBW7':
            image_embeds = visual_out.pop('det_tgt')
        else:
            image_embeds = visual_out.pop('feature_map')
        if self.visual_emb_proj is not None:
            image_embeds = self.visual_emb_proj(image_embeds)
        visual_out['feature_map'] = image_embeds
        if return_dict:
            return visual_out
        else:
            return visual_out['feature_map'], visual_out['pooled_out']

    def text_encode(self, input_ids, input_segment_ids, input_mask):
        if self.text_backbone_mode.startswith('albert'):
            text_output = self.text(input_ids=input_ids, input_segment_ids=input_segment_ids, input_mask=input_mask)
        elif self.text_backbone_mode.startswith('deberta'):
            text_output = self.text(input_ids=input_ids, segment_ids=input_segment_ids, attention_mask=input_mask)
            text_output = {
                'pooled_output': text_output['pooled_output'],
                'encoded_layers': text_output['hidden'],
                'embedding_masks': text_output['mask'],
                'embedding': text_output['embedding']
            }

        return text_output

    def cross_encoder_encode(self, text_embeds, input_mask, image_embeds, image_atts, return_dict=False):
        """
        返回最后一层的embedding
        """
        if self.text_backbone_mode == 'albert_concat':
            concat_embeds, concat_mask = self.concat(text_embeds, image_embeds, input_mask, image_atts)
            encoded_layers, _ = self.cross(h=concat_embeds, mask=concat_mask)
            sequence_output = encoded_layers[-1]
            return sequence_output
        elif self.text_backbone_mode == 'albert_cross':
            if len(input_mask.shape) == 2:
                attention_mask = self.get_extended_attention_mask(input_mask, input_mask.size(),
                                                                  input_mask.device, False)
            else:
                attention_mask = input_mask
            if len(image_atts.shape) == 2:
                image_atts = self.invert_attention_mask(image_atts)
            output = self.cross(hidden_states=text_embeds,
                                attention_mask=attention_mask,
                                encoder_hidden_states=image_embeds,
                                encoder_attention_mask=image_atts,
                                return_dict=True,
                                )
            return output.last_hidden_state
        elif self.text_backbone_mode == 'deberta_cross':
            if len(input_mask.shape) == 2:
                attention_mask = self.get_extended_attention_mask(input_mask, input_mask.size(),
                                                                  input_mask.device, False)
            else:
                attention_mask = input_mask
            if len(image_atts.shape) == 2:
                image_atts = self.invert_attention_mask(image_atts)
            sequence_output, encoder_all_hidden_states, relative_pos, all_attn_weights = self.cross(text_embeds, image_embeds, attention_mask, image_atts, output_attentions=True)
            #cross_output = self.cross(text_embeds, image_embeds, attention_mask, image_atts, output_attentions=True)
            if return_dict:
                return {'sequence_output': sequence_output,
                        'relative_pos': relative_pos,
                        'encoded_layers': encoder_all_hidden_states,
                        'attn_weights': all_attn_weights
                        }
            else:
                return sequence_output

    def concat(self, te, ie, tm, im):
        return torch.cat([te, ie], dim=1), torch.cat([tm, im], dim=-1)

    def get_extended_attention_mask(self, attention_mask: torch.Tensor, input_shape: Tuple[int], device, is_decoder: bool) -> torch.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                # in case past_key_values are used we need to add a prefix ones mask to the causal mask
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)

                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                    causal_mask = torch.cat(
                        [
                            torch.ones(
                                (batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype
                            ),
                            causal_mask,
                        ],
                        axis=-1,
                    )

                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def invert_attention_mask(self, encoder_attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Invert an attention mask (e.g., switches 0. and 1.).
        Args:
            encoder_attention_mask (`torch.Tensor`): An attention mask.
        Returns:
            `torch.Tensor`: The inverted attention mask.
        """
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility

        if self.dtype == torch.float16:
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e4
        elif self.dtype in [torch.bfloat16, torch.float32]:
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -1e9
        else:
            raise ValueError(
                f"{self.dtype} not recognized. `dtype` should be set to either `torch.float32` or `torch.float16`"
            )

        return encoder_extended_attention_mask
