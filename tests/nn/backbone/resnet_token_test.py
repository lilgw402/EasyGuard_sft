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

from fex.nn.backbone.resnet_token import ResnetTokenizer
from fex.config import cfg, reset_cfg

BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
TEST_DIR = os.path.join(BASE_DIR, "../.codebase/pipelines/test_data")


class TestResnetToken(TestCase):
    """ test resnet """

    @unittest.skip(reason="skip")
    def setUp(self):
        input_image = Image.open(os.path.join(TEST_DIR, "dog.jpg"))
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        # create a mini-batch as expected by the model
        self.input_batch = input_tensor.unsqueeze(0)

        cfg = reset_cfg()
        cfg.NETWORK.activ_and_nobias = False
        self.model = ResnetTokenizer(cfg)
        if int(os.environ.get('ISCI', 0)) != 1:
            print('loading pretrain model')
        self.model.eval()
        if torch.cuda.is_available():
            self.input_batch = self.input_batch.to('cuda')
            self.model.to('cuda')

    @unittest.skip(reason="skip")
    def test_resnet_random(self):
        torch.manual_seed(42)
        with torch.no_grad():
            fake_input = torch.randn([1, 3, 224, 224])
            if torch.cuda.is_available():
                fake_input = fake_input.to('cuda')
            visual_embs = self.model(fake_input)
            print(visual_embs.shape)

    @unittest.skip(reason="skip")
    def test_frame_random(self):
        torch.manual_seed(42)
        with torch.no_grad():
            fake_input = torch.randn([8, 16, 3, 224, 224])
            if torch.cuda.is_available():
                fake_input = fake_input.to('cuda')
            visual_embs = self.model(fake_input)
            print(visual_embs.shape)

    @unittest.skip(reason="skip")
    @pytest.mark.skipif(int(os.environ.get('ISCI', 0)) == 1, reason="CI environment not support")
    def test_resnet_dog(self):
        torch.manual_seed(42)
        with torch.no_grad():
            visual_embs = self.model(self.input_batch)
            print(visual_embs.shape)


if __name__ == '__main__':
    unittest.main()
