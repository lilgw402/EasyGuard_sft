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
import pytest

import torch
from PIL import Image
from torchvision import transforms

from fex.nn import swin_transformer_tiny, swin_transformer_small, swin_transformer_base


class TestSwinTransformer(TestCase):
    """ test swin """

    @unittest.skip(reason='doas not support in CI')
    def test_swin_tiny_random_input(self):
        torch.manual_seed(42)
        model = swin_transformer_tiny(512)
        if torch.cuda.is_available():
            model.cuda()
        model.eval()

        with torch.no_grad():
            fake_input = torch.randn([16, 3, 224, 224])
            if torch.cuda.is_available():
                fake_input = fake_input.to('cuda')
            output = model(fake_input)
            self.assertEqual(list(output.shape), [16, 512])
            print(output.shape, 'output')

    @unittest.skip(reason='doas not support in CI')
    def test_swin_base_random_input(self):
        torch.manual_seed(42)
        model = swin_transformer_base(512)
        if torch.cuda.is_available():
            model.cuda()
        model.eval()

        with torch.no_grad():
            fake_input = torch.randn([16, 3, 224, 224])
            if torch.cuda.is_available():
                fake_input = fake_input.to('cuda')
            output = model(fake_input)
            self.assertEqual(list(output.shape), [16, 512])
            print(output.shape, 'output')

    """
    def test_swin_tiny_random_input_img384(self):
        torch.manual_seed(42)
        model = swin_transformer_tiny(512, img_size=384)
        if torch.cuda.is_available():
            model.cuda()
        model.eval()

        with torch.no_grad():
            fake_input = torch.randn([16, 3, 384, 384])
            if torch.cuda.is_available():
                fake_input = fake_input.to('cuda')
            output = model(fake_input)
            self.assertEqual(list(output.shape), [16, 512])
            print(output.shape, 'output')
    """


if __name__ == '__main__':
    unittest.main()
