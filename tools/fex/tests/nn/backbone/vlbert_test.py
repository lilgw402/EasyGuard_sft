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

from fex.nn import VisualLinguisticBert
from fex.config import cfg
from fex.utils.load import load_from_pretrain
from fex.utils.torch_io import load as torch_io_load


def gen_fake_input():
    input_ids = torch.stack([torch.Tensor([0, 422, 951, 3]).long() for _ in range(8)], dim=0)
    segment_ids = torch.stack([torch.Tensor([0, 0, 0, 0]).long() for _ in range(8)], dim=0)
    input_mask = input_ids != 2
    return {'input_ids': input_ids,
            'input_segment_ids': segment_ids,
            'input_mask': input_mask}


def params_mapping(ckpt, path):
    mapping = {
        'vlbert.embedding_LayerNorm': 'vlbert.embedding.norm',
        'vlbert.word_embeddings': 'vlbert.embedding.token_embedder_tokens',
        'vlbert.position_embeddings': 'vlbert.embedding.token_embedder_positions',
        'vlbert.token_type_embeddings': 'vlbert.embedding.token_embedder_segments',
        'vlbert.visual_ln': 'vlbert.embedding.visual_ln',
        'image_feature_extractor.visual_backbone': 'vlbert.visual_tokenizer.resnet',
        'image_feature_extractor.visual_du_sample': 'vlbert.visual_tokenizer.du_sample',
        'image_feature_extractor.img_emb_encoder': 'vlbert.visual_tokenizer.image_encoder',
        'img_emb_decoder': 'vlbert.visual_tokenizer.image_decoder',

    }
    for i in range(6):
        mapping.update({
            'vlbert.encoder.layer.%s.attention.self.query' % i: 'vlbert.encoder.blocks.%s.attn.proj_q' % i,
            'vlbert.encoder.layer.%s.attention.self.key' % i: 'vlbert.encoder.blocks.%s.attn.proj_k' % i,
            'vlbert.encoder.layer.%s.attention.self.value' % i: 'vlbert.encoder.blocks.%s.attn.proj_v' % i,
            'vlbert.encoder.layer.%s.attention.output.dense' % i: 'vlbert.encoder.blocks.%s.proj' % i,
            'vlbert.encoder.layer.%s.attention.output.LayerNorm' % i: 'vlbert.encoder.blocks.%s.norm1' % i,
            'vlbert.encoder.layer.%s.intermediate.dense' % i: 'vlbert.encoder.blocks.%s.pwff.fc1' % i,
            'vlbert.encoder.layer.%s.output.dense' % i: 'vlbert.encoder.blocks.%s.pwff.fc2' % i,
            'vlbert.encoder.layer.%s.output.LayerNorm' % i: 'vlbert.encoder.blocks.%s.norm2' % i
        })

    params = torch.load(ckpt)
    pretrain_state_dict_parsed = {}
    for k, v in params.items():
        if k.startswith('module.'):
            k = k.replace('module.', '')
        if 'image_feature_extractor.du_sample' in k:
            continue
        no_match = True
        for pretrain_prefix, new_prefix in mapping.items():
            if k.startswith(pretrain_prefix):
                k = new_prefix + k[len(pretrain_prefix):]
                pretrain_state_dict_parsed[k] = v
                no_match = False
                break
        if no_match:
            pretrain_state_dict_parsed[k] = v

    torch.save(pretrain_state_dict_parsed, path)


class TestVLBert(TestCase):
    """ test albert """

    @unittest.skip("config and ckpt not exist, it will be fix by@huangwenguan")
    def setUp(self):
        cfg.update_cfg(
            'config/pretrain/vlbert.yaml')
        vlbert_ckpt = '/mnt/nlp-lq/huangwenguan/modelhub/dy_14b_filter_nonsp_ernie_novl/model_state_epoch_95543.th'
        vlbert_ckpt_fex = '/mnt/nlp-lq/huangwenguan/modelhub/dy_14b_filter_nonsp_ernie_novl/model_state_epoch_95543_fex.th'
        self.model = VisualLinguisticBert(cfg)
        params_mapping(vlbert_ckpt, vlbert_ckpt_fex)
        load_from_pretrain(self.model,
                           pretrain_paths=vlbert_ckpt_fex,
                           prefix_changes=['vlbert.->'])
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to('cuda')

        # TODO: 增加一个验证正确性的test
    @unittest.skip("skip")
    def test_resnet_random(self):
        torch.manual_seed(42)
        with torch.no_grad():
            fake_input = gen_fake_input()
            fake_input['image'] = torch.randn([8, 3, 256, 256])
            fake_input['image_mask'] = torch.ones([8, 5]).long()
            if torch.cuda.is_available():
                for k, v in fake_input.items():
                    print(k, v.shape)
                    fake_input[k] = v.to('cuda')
            output = self.model(**fake_input)
            for k, v in output.items():
                print(k, len(v), v[0].shape)

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

    # def test_validation(self):
    #     pass


if __name__ == '__main__':
    unittest.main()
