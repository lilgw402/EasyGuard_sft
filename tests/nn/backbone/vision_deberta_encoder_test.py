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
from fex.nn.backbone.vision_deberta_encoder import VisionDeberta


def gen_fake_input():
    input_ids = torch.stack([torch.Tensor([0, 422, 951, 3]).long() for _ in range(8)], dim=0)
    segment_ids = torch.stack([torch.Tensor([0, 0, 0, 0]).long() for _ in range(8)], dim=0)
    input_mask = input_ids != 2
    return {'input_ids': input_ids,
            'input_segment_ids': segment_ids,
            'input_mask': input_mask}


class TestVisionDeberta(TestCase):
    """ test VisionDeberta """

    @unittest.skip(reason='doas not support in CI')
    def test_vitb32(self):
        cfg = reset_cfg()
        cfg.update_cfg('hdfs://haruna/home/byte_search_nlp_lq/multimodal/confighub/deberta_qt.yaml')
        model = VisionDeberta(
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

            print('t: ')
            fake_input['mode'] = 't'
            output = model(**fake_input)
            for k, v in output.items():
                if v is not None:
                    print(k, len(v), v[0].shape)

            print('v: ')
            fake_input['mode'] = 'v'
            output = model(**fake_input)
            for k, v in output.items():
                if v is not None:
                    print(k, len(v), v[0].shape)

            print('tv: ')
            fake_input['mode'] = 't'
            output = model(**fake_input)
            for k, v in output.items():
                if v is not None:
                    print(k, len(v), v[0].shape)

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
