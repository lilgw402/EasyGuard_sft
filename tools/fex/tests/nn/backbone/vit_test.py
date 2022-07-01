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

from fex.nn import visual_transformer_B32


class TestVisionTransformer(TestCase):
    """ test resnet """

    @unittest.skip(reason='doas not support in CI')
    def test_vit_random_input(self):
        torch.manual_seed(42)
        self.model = visual_transformer_B32(512)
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

        with torch.no_grad():
            fake_input = torch.randn([1, 3, 384, 216])
            if torch.cuda.is_available():
                fake_input = fake_input.to('cuda')
            output = self.model(fake_input)
            print(output.shape, 'output')


if __name__ == '__main__':
    unittest.main()
