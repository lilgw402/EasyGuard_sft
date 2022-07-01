#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Huang Wenguan (huangwenguan@bytedance.com)
Date: 2020-09-02 22:34:57
LastEditTime: 2020-11-16 20:09:56
LastEditors: Huang Wenguan
Description: albert test
'''

import unittest
from unittest import TestCase
import torch

from fex.config import cfg, reset_cfg
from fex.utils.load import load_from_pretrain
from fex.utils.torch_io import load as torch_io_load
from fex.model.vision_deberta import VisionDebertaPretraining


def gen_fake_input():
    input_ids = torch.stack([torch.Tensor([0, 422, 951, 3]).long() for _ in range(8)], dim=0)
    segment_ids = torch.stack([torch.Tensor([0, 0, 0, 0]).long() for _ in range(8)], dim=0)
    position_ids = torch.stack([torch.Tensor([0, 1, 2, 3]).long() for _ in range(8)], dim=0)
    masked_lm_positions = torch.stack([torch.Tensor([1, 2]).long() for _ in range(8)], dim=0)
    masked_lm_ids = torch.stack([torch.Tensor([421, 952]).long() for _ in range(8)], dim=0)
    input_mask = input_ids != 2
    return {'input_ids': input_ids,
            'input_segment_ids': segment_ids,
            'position_ids': position_ids,
            'input_mask': input_mask,
            'masked_lm_positions': masked_lm_positions,
            'masked_lm_ids': masked_lm_ids,
            'masked_tokens': masked_lm_ids
            }


class TestVisionDebertaModel(TestCase):
    """ test VisionDebertaModel """

    @unittest.skip(reason='doas not support in CI')
    def test_vitb32(self):
        cfg = reset_cfg()
        cfg.update_cfg('hdfs://haruna/home/byte_search_nlp_lq/multimodal/confighub/deberta_qt.yaml')
        model = VisionDebertaPretraining(
            max_len=512,
            abs_pos_embedding=True,
            ignore_index=-1,
            calc_mlm_accuracy=True,
            tie_embedding=True,
            use_emd=True,
            num_emd_groups=1,
            emd_group_repeat=2,
            head_layernorm_type='default',

            vocab_size=145608,
            dim=768,
            dim_ff=3072,
            n_segments=16,
            p_drop_hidden=0.1,
            n_heads=12,
            n_layers=6,
            initializer_range=0.02,
            layernorm_type='default',
            layernorm_fp16=False,
            layer_norm_eps=1e-6,
            p_drop_attn=0.1,
            embedding_dim=256,
            embedding_dropout=0.1,
            act='gelu',
            pool=True,
            padding_index=2,
            attention_clamp_inf=False,
            max_relative_positions=512,
            extra_da_transformer_config=None,
            omit_other_output=False,
            use_fast=False,
            visual_type='VitB32',
            visual_dim=512,
            middle_size=128,
            visual_config={
                'output_dim': 512,
                'vit_dropout': 0.1,
                'vit_emb_dropout': 0.0,
                'patch_length': 49
            }
        )
        model.eval()
        if torch.cuda.is_available():
            model.to('cuda')

        torch.manual_seed(42)
        with torch.no_grad():
            fake_input = gen_fake_input()

            fake_input['frames'] = torch.randn([8, 14, 3, 224, 224])
            fake_input['frames_mask'] = torch.ones([8, 14]).long()
            if torch.cuda.is_available():
                for k, v in fake_input.items():
                    print(k, v.shape)
                    fake_input[k] = v.to('cuda')

            output = model(**fake_input)
            for k, v in output.items():
                print(k, type(v))
                # if k == 'loss':
                #    print(k, v)
                # else:
                #    print(k, v.shape)

    # def test_dino_vits16(self):
    #     cfg = reset_cfg()
    #     cfg.update_cfg('/data00/huangwenguan/vlws/fuxi/fuxi/config/vlp/vlp_l6.yaml')
    #     cfg.BERT.visual_type = 'DINO/ViTS16'
    #     cfg.BERT.visual_dim = 384
    #     model = FrameALBert(cfg)
    #     model.eval()
    #     if torch.cuda.is_available():
    #         model.to('cuda')

    #     torch.manual_seed(42)
    #     with torch.no_grad():
    #         fake_input = gen_fake_input()
    #         fake_input['frames'] = torch.randn([8, 14, 3, 224, 224])
    #         fake_input['frames_mask'] = torch.ones([8, 14]).long()
    #         if torch.cuda.is_available():
    #             for k, v in fake_input.items():
    #                 print(k, v.shape)
    #                 fake_input[k] = v.to('cuda')
    #         output = model(**fake_input)
    #         for k, v in output.items():
    #             print(k, len(v), v[0].shape)

    # def test_dino_vitb8(self):
    #     cfg = reset_cfg()
    #     cfg.update_cfg('/data00/huangwenguan/vlws/fuxi/fuxi/config/vlp/vlp_l6.yaml')
    #     cfg.BERT.visual_type = 'DINO/ViTB8'
    #     cfg.BERT.visual_dim = 768
    #     model = FrameALBert(cfg)
    #     model.eval()
    #     if torch.cuda.is_available():
    #         model.to('cuda')

    #     torch.manual_seed(42)
    #     with torch.no_grad():
    #         fake_input = gen_fake_input()
    #         fake_input['frames'] = torch.randn([8, 14, 3, 224, 224])
    #         fake_input['frames_mask'] = torch.ones([8, 14]).long()
    #         if torch.cuda.is_available():
    #             for k, v in fake_input.items():
    #                 print(k, v.shape)
    #                 fake_input[k] = v.to('cuda')
    #         output = model(**fake_input)
    #         for k, v in output.items():
    #             print(k, len(v), v[0].shape)

    # def test_text_only_random(self):
    #     torch.manual_seed(42)
    #     with torch.no_grad():
    #         fake_input = gen_fake_input()
    #         if torch.cuda.is_available():
    #             for k, v in fake_input.items():
    #                 fake_input[k] = v.to('cuda')
    #         fake_input['is_text_visual'] = False
    #         output = self.model(**fake_input)
    #         for k, v in output.items():
    #             print(k, len(v), v[0].shape)


if __name__ == '__main__':
    unittest.main()
