#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Huang Wenguan (huangwenguan@bytedance.com)
Date: 2020-11-12 13:27:17
LastEditTime: 2020-11-18 14:42:16
LastEditors: Huang Wenguan
Description: test resnet
'''

import os
import unittest
from unittest import TestCase

import torch
from PIL import Image
from torchvision import transforms

from fex.nn.backbone.swin_transformer_3d import swin_transformer_3d_B244


class TestSwinTransformer3D(TestCase):
    """ test swin """

    @unittest.skip(reason='doas not support in CI')
    def test_swin_base_random_input(self):
        torch.manual_seed(42)
        pretrained = 'hdfs://haruna/home/byte_search_nlp_lq/multimodal/modelhub/swinb224_dycover_20211206/model.th'
        model = swin_transformer_3d_B244(pretrained)
        if torch.cuda.is_available():
            model.cuda()
        model.eval()

        with torch.no_grad():
            fake_input = torch.randn([16, 3, 8, 224, 224])
            if torch.cuda.is_available():
                fake_input = fake_input.to('cuda')
            output = model(fake_input)
            print(output.shape, 'output')


if __name__ == '__main__':
    unittest.main()
