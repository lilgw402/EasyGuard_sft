#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest
from unittest import TestCase

import torch
from PIL import Image

from fex.nn.visual_tokenizer import create_visual_tokenizer

BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
TEST_DIR = os.path.join(BASE_DIR, "../ci/test_data")


class TesTDETR(TestCase):

    def test_not_all_patch_detr(self):
        detr_config = {
            'num_hidden_layers': 6,
            'num_attention_heads': 4,
            'hidden_size': 128,
            'intermediate_size': 512,
            'hidden_dropout_prob': 0.1,
            'encoder_width': 128,
            'layernorm_eps': 1.0e-6,
            'initializer_range': 0.02,
            'cross_attention_config': {
                'hidden_size': 128,
                'num_attention_heads': 4,
                'encoder_width': 128}
        }
        model = create_visual_tokenizer('DETRDINO/ViTB16',
                                        keep_token=8,
                                        mode='dec',
                                        detr=detr_config)
        model.cuda()
        torch.manual_seed(42)

        with torch.no_grad():
            fake_input = torch.randn([16, 3, 224, 224])
            if torch.cuda.is_available():
                fake_input = fake_input.to('cuda')
            output = model(fake_input)
            # for k, v in output.items():
            #     print(k, v.shape)


if __name__ == '__main__':
    unittest.main()
