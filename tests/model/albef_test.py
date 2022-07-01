#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Description: albef test
'''

import unittest
from unittest import TestCase
import torch

from fex.config import cfg, reset_cfg
from fex.utils.load import load_from_pretrain
from fex.utils.torch_io import load as torch_io_load
from fex.model.albef import ALBEF


def gen_fake_input():
    input_ids = torch.stack([torch.Tensor([0, 422, 951, 3]).long() for _ in range(8)], dim=0)
    segment_ids = torch.stack([torch.Tensor([0, 0, 0, 0]).long() for _ in range(8)], dim=0)
    #position_ids = torch.stack([torch.Tensor([0, 1, 2, 3]).long() for _ in range(8)], dim=0)
    #masked_lm_positions = torch.stack([torch.Tensor([1, 2]).long() for _ in range(8)], dim=0)
    #masked_lm_ids = torch.stack([torch.Tensor([421, 952]).long() for _ in range(8)], dim=0)
    input_mask = input_ids != 2
    return {'input_ids': input_ids,
            'input_segment_ids': segment_ids,
            'input_mask': input_mask,
            'image': torch.randn([8, 3, 224, 224]),
            'image_mask': torch.ones([8], dtype=torch.int64)
            # 'position_ids': position_ids,
            # 'masked_lm_positions': masked_lm_positions,
            # 'masked_lm_ids': masked_lm_ids,
            # 'masked_tokens': masked_lm_ids
            }


class TestALBEFPretraining(TestCase):
    """
    test ALBEFPretraining
    一共测试3种模式：
    1. albert cross
    2. albert concat
    3. deberta cross

    然后再基于 deberta cross，测试下面几个visual encoder
    1. deit
    2. swin
    3. clip-vit
    """

    @unittest.skip(reason='doas not support in CI')
    def test_albert_cross(self):
        ""
        cfg = reset_cfg()
        cfg.update_cfg('/home/tiger/ws/fuxi3/fuxi/config/cross/v2/albef.yaml')
        model = ALBEF(cfg)
        model.eval()
        if torch.cuda.is_available():
            model.to('cuda')

        torch.manual_seed(42)
        with torch.no_grad():
            fake_input = gen_fake_input()

            if torch.cuda.is_available():
                for k, v in fake_input.items():
                    print(k, v.shape)
                    fake_input[k] = v.to('cuda')

            output = model(**fake_input)
            for k, v in output.items():
                print(k, type(v), v)

    @unittest.skip(reason='doas not support in CI')
    def test_albert_concat(self):
        cfg = reset_cfg()
        cfg.update_cfg('/home/tiger/ws/fuxi3/fuxi/config/cross/v2/albef_r.yaml')
        model = ALBEF(cfg)
        model.eval()
        if torch.cuda.is_available():
            model.to('cuda')

        torch.manual_seed(42)
        with torch.no_grad():
            fake_input = gen_fake_input()

            if torch.cuda.is_available():
                for k, v in fake_input.items():
                    print(k, v.shape)
                    fake_input[k] = v.to('cuda')

            output = model(**fake_input)
            for k, v in output.items():
                print(k, type(v), v)

    @unittest.skip(reason='doas not support in CI')
    def test_deberta_cross(self):
        cfg = reset_cfg()
        cfg.update_cfg('/home/tiger/ws/fuxi3/fuxi/config/cross/v2/albef_deberta.yaml')
        model = ALBEF(cfg)
        model.eval()
        if torch.cuda.is_available():
            model.to('cuda')

        torch.manual_seed(42)
        with torch.no_grad():
            fake_input = gen_fake_input()

            if torch.cuda.is_available():
                for k, v in fake_input.items():
                    print(k, v.shape)
                    fake_input[k] = v.to('cuda')

            output = model(**fake_input)
            for k, v in output.items():
                print(k, type(v), v)

    @unittest.skip(reason='doas not support in CI')
    def test_deberta_cross_swin(self):
        cfg = reset_cfg()
        cfg.update_cfg('/home/tiger/ws/fuxi3/fuxi/config/cross/v2/albef_deberta_swin.yaml')
        model = ALBEF(cfg)
        model.eval()
        if torch.cuda.is_available():
            model.to('cuda')

        torch.manual_seed(42)
        with torch.no_grad():
            fake_input = gen_fake_input()

            if torch.cuda.is_available():
                for k, v in fake_input.items():
                    print(k, v.shape)
                    fake_input[k] = v.to('cuda')

            output = model(**fake_input)
            for k, v in output.items():
                print(k, type(v), v)

    @unittest.skip(reason='doas not support in CI')
    def test_deberta_cross_clipvit(self):
        cfg = reset_cfg()
        cfg.update_cfg('/home/tiger/ws/fuxi3/fuxi/config/cross/v2/albef_deberta_clip.yaml')
        model = ALBEF(cfg)
        model.eval()
        if torch.cuda.is_available():
            model.to('cuda')

        torch.manual_seed(42)
        with torch.no_grad():
            fake_input = gen_fake_input()

            if torch.cuda.is_available():
                for k, v in fake_input.items():
                    print(k, v.shape)
                    fake_input[k] = v.to('cuda')

            output = model(**fake_input)
            for k, v in output.items():
                print(k, type(v), v)

    @unittest.skip(reason='doas not support in CI')
    def test_empty_image(self):
        cfg = reset_cfg()
        cfg.update_cfg('/home/tiger/ws/fuxi3/fuxi/config/cross/v2/albef_deberta_clip.yaml')
        model = ALBEF(cfg)
        model.eval()
        if torch.cuda.is_available():
            model.to('cuda')

        torch.manual_seed(42)
        with torch.no_grad():
            fake_input = gen_fake_input()
            fake_input['image_mask'] = torch.zeros([8], dtype=torch.int64)

            if torch.cuda.is_available():
                for k, v in fake_input.items():
                    print(k, v.shape)
                    fake_input[k] = v.to('cuda')

            output = model(**fake_input)
            for k, v in output.items():
                print(k, type(v), v)


if __name__ == '__main__':
    unittest.main()
