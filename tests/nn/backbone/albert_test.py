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

from fex.nn import ALBert
from fex.config import cfg
from fex.utils.load import smart_load_pretrained_state_dict
from fex.utils.torch_io import load as torch_io_load


def gen_fake_input():
    input_ids = torch.Tensor([0, 422, 951, 3]).long().unsqueeze(0)
    segment_ids = torch.Tensor([0, 0, 0, 0]).long().unsqueeze(0)
    input_mask = input_ids != 2
    return {'input_ids': input_ids,
            'input_segment_ids': segment_ids,
            'input_mask': input_mask}


class TestALBert(TestCase):
    """ test albert """

    def setUp(self):
        torch.manual_seed(42)
        cfg.update_cfg(
            'hdfs://haruna/home/byte_search_nlp_lq/multimodal/confighub/albert.yaml')
        ckpt = 'hdfs://haruna/home/byte_search_nlp_lq/multimodal/modelhub/bvr_all_add_data_manual/fex/model.th'
        self.model = ALBert(cfg.BERT)
        smart_load_pretrained_state_dict(self.model, torch_io_load(ckpt))
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to('cuda')

        # TODO: 增加一个验证正确性的test

    @unittest.skip(reason='doas not support in CI')
    def test_format(self):
        with torch.no_grad():
            fake_input = gen_fake_input()
            if torch.cuda.is_available():
                for k, v in fake_input.items():
                    fake_input[k] = v.to('cuda')
            output = self.model(**fake_input)
            self.assertEqual(len(output['encoded_layers']), 6)
            self.assertEqual(len(output['attention_probs']), 6)
            self.assertEqual(list(output['pooled_output'].shape), [1, 768])
            self.assertEqual(list(output['encoded_layers'][-1].shape), [1, 4, 768])
            self.assertEqual(list(output['attention_probs'][-1].shape), [1, 12, 4, 4])


if __name__ == '__main__':
    unittest.main()
