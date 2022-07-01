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

from fex.nn import FrameALBert
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


class TestFALBert(TestCase):
    """ test Frame Arch """

    def setUp(self):
        torch.manual_seed(42)
        cfg = reset_cfg()
        cfg.update_cfg('hdfs://haruna/home/byte_search_nlp_lq/multimodal/confighub/falbert_resnet.yaml')
        self.model = FrameALBert(cfg)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to('cuda')

    @unittest.skip(reason='doas not support in CI')
    def test_resnet_random(self):

        with torch.no_grad():
            fake_input = gen_fake_input()
            fake_input['frames'] = torch.randn([8, 14, 3, 256, 256])
            fake_input['frames_mask'] = torch.ones([8, 14]).long()
            if torch.cuda.is_available():
                for k, v in fake_input.items():
                    fake_input[k] = v.to('cuda')
            output = self.model(**fake_input)
            self.assertEqual(len(output['encoded_layers']), 6)
            self.assertEqual(list(output['encoded_layers'][-1].shape), [8, 19, 768])
            self.assertEqual(list(output['text_final_output'].shape), [8, 4, 768])
            self.assertEqual(list(output['visual_final_output'].shape), [8, 15, 768])
            self.assertEqual(list(output['visual_tower_output'].shape), [8, 14, 128])
            self.assertEqual(list(output['embedding_masks'].shape), [8, 19])
            self.assertEqual(list(output['embeddings'].shape), [8, 19, 768])
            self.assertEqual(len(output['attention_probs']), 6)
            self.assertEqual(list(output['attention_probs'][-1].shape), [8, 12, 19, 19])


if __name__ == '__main__':
    unittest.main()
