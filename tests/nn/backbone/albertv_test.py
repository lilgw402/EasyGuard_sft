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

from fex.nn import ALBertV
from fex.config import cfg, reset_cfg
from fex.utils.load import load_from_pretrain
from fex.utils.torch_io import load as torch_io_load


def gen_fake_input():
    input_ids = torch.stack([torch.Tensor([0, 422, 951, 3]).long() for _ in range(8)], dim=0)
    segment_ids = torch.stack([torch.Tensor([0, 0, 0, 0]).long() for _ in range(8)], dim=0)
    input_mask = input_ids != 2
    return {'input_ids': input_ids,
            'input_segment_ids': segment_ids,
            'input_mask': input_mask}


class TestALBertV(TestCase):
    """ test albert """

    @unittest.skip(reason='doas not support in CI')
    def test_resnet_random(self):
        cfg = reset_cfg()
        cfg.update_cfg('./ci/test_data/test_config/tuso_mlm.yaml')
        albert_ckpt = 'hdfs://haruna/home/byte_search_nlp_lq/user/huangwenguan/modelhub/albert_6l_zh_mix_oldcut_20200921/fex/model_t_fp32_ln.th'
        resnet_ckpt = 'hdfs://haruna/home/byte_search_nlp_lq/user/huangwenguan/vl/model/pretrained_model/resnet50-19c8e357-t3.pth'
        self.model = ALBertV(cfg)
        load_from_pretrain(self.model,
                           pretrain_paths=[
                               albert_ckpt,
                               resnet_ckpt
                           ],
                           prefix_changes=['resnet->visual_tokenizer.resnet',
                                           'v_proj->visual_tokenizer.v_proj',
                                           ])
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to('cuda')

        torch.manual_seed(42)
        with torch.no_grad():
            fake_input = gen_fake_input()
            fake_input['image'] = torch.randn([8, 14, 3, 256, 256])
            fake_input['image_mask'] = torch.ones([8, 14]).long()
            if torch.cuda.is_available():
                for k, v in fake_input.items():
                    print(k, v.shape)
                    fake_input[k] = v.to('cuda')
            output = self.model(**fake_input)
            for k, v in output.items():
                print(k, len(v), v[0].shape)

    @unittest.skip(reason='doas not support in CI')
    def test_text_only_random(self):
        cfg = reset_cfg()
        cfg.update_cfg('./ci/test_data/test_config/tuso_mlm.yaml')
        albert_ckpt = 'hdfs://haruna/home/byte_search_nlp_lq/user/huangwenguan/modelhub/albert_6l_zh_mix_oldcut_20200921/fex/model_t_fp32_ln.th'
        resnet_ckpt = 'hdfs://haruna/home/byte_search_nlp_lq/user/huangwenguan/vl/model/pretrained_model/resnet50-19c8e357-t3.pth'
        self.model = ALBertV(cfg)
        load_from_pretrain(self.model,
                           pretrain_paths=[
                               albert_ckpt,
                               resnet_ckpt
                           ],
                           prefix_changes=['resnet->visual_tokenizer.resnet',
                                           'v_proj->visual_tokenizer.v_proj',
                                           ])
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to('cuda')

        torch.manual_seed(42)
        with torch.no_grad():
            fake_input = gen_fake_input()
            if torch.cuda.is_available():
                for k, v in fake_input.items():
                    fake_input[k] = v.to('cuda')
            fake_input['is_text_visual'] = False
            output = self.model(**fake_input)
            for k, v in output.items():
                print(k, len(v), v[0].shape)


if __name__ == '__main__':
    unittest.main()
