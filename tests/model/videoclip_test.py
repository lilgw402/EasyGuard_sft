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

from fex.model.videoclip import VideoCLIP
from fex.config import cfg, reset_cfg
from fex.utils.load import load_from_pretrain
from fex.utils.torch_io import load as torch_io_load


def gen_fake_input():
    input_ids = torch.stack([torch.Tensor([0, 422, 951, 3]).long() for _ in range(16)], dim=0)
    segment_ids = torch.stack([torch.Tensor([0, 0, 0, 0]).long() for _ in range(16)], dim=0)
    input_mask = input_ids != 2
    return {'input_ids': input_ids,
            'input_segment_ids': segment_ids,
            'input_mask': input_mask,
            }


class TestVideoCLIP(TestCase):
    """ test VideoCLIP """

    @unittest.skip(reason='doas not support in CI')
    def test_vitb32(self):
        cfg = reset_cfg()
        cfg.update_cfg('hdfs://haruna/home/byte_search_nlp_lq/multimodal/confighub/train_a_videoclip.yaml')
        model = VideoCLIP(cfg)
        model.eval()
        if torch.cuda.is_available():
            model.to('cuda')

        torch.manual_seed(42)
        with torch.no_grad():
            fake_input = gen_fake_input()

            fake_input['images'] = torch.randn([16, 8, 3, 224, 224])
            fake_input['images_mask'] = torch.ones([16, 8])
            if torch.cuda.is_available():
                for k, v in fake_input.items():
                    print(k, v.shape)
                    fake_input[k] = v.to('cuda')

            output = model(**fake_input)
            for k, v in output.items():
                if k == 'loss':
                    print(k, v)
                else:
                    print(k, v.shape)


if __name__ == '__main__':
    unittest.main()
