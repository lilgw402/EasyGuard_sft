#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
dino test
'''

import os
import unittest
from unittest import TestCase
import pytest

import torch
from PIL import Image
from torchvision import transforms

from fex.nn.backbone.dino import dino_vit_s16, dino_vit_s8, dino_vit_b8, dino_vit_b16, dino_vit_b32
from fex.utils.load import load_from_pretrain

BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
TEST_DIR = os.path.join(BASE_DIR, "../ci/test_data")


class TestDino(TestCase):
    """ test dino """

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
        self.input_batch = input_tensor.unsqueeze(0).cuda()

    @unittest.skip(reason='doas not support in CI')
    def test_dino_vit_s16(self):
        torch.manual_seed(42)
        model = dino_vit_s16()
        model.cuda()
        ckpt = 'hdfs://haruna/home/byte_search_nlp_lq/user/huangwenguan/modelhub/dino_official/dino_deitsmall16_pretrain.pth'
        load_from_pretrain(model, ckpt)
        with torch.no_grad():
            fake_input = torch.randn([8, 3, 224, 224]).cuda()
            output = model(fake_input)
            print(output.shape, 'output')

            output = model(self.input_batch)
            print(output.shape, 'output dog')

    @unittest.skip(reason='doas not support in CI')
    def test_dino_vit_s8(self):
        torch.manual_seed(42)
        model = dino_vit_s8()
        model.cuda()
        ckpt = 'hdfs://haruna/home/byte_search_nlp_lq/user/huangwenguan/modelhub/dino_official/dino_deitsmall8_pretrain.pth'
        load_from_pretrain(model, ckpt)
        with torch.no_grad():
            fake_input = torch.randn([8, 3, 224, 224]).cuda()
            output = model(fake_input)
            print(output.shape, 'output')

            output = model(self.input_batch)
            print(output.shape, 'output dog')

    @unittest.skip(reason='doas not support in CI')
    def test_dino_vit_b16(self):
        torch.manual_seed(42)
        model = dino_vit_b16()
        model.cuda()
        ckpt = 'hdfs://haruna/home/byte_search_nlp_lq/user/huangwenguan/modelhub/dino_official/dino_vitbase16_pretrain.pth'
        load_from_pretrain(model, ckpt)
        with torch.no_grad():
            fake_input = torch.randn([8, 3, 224, 224]).cuda()
            output = model(fake_input)
            print(output.shape, 'output')

            output = model(self.input_batch)
            print(output.shape, 'output dog')

    @unittest.skip(reason='doas not support in CI')
    def test_dino_vit_b8(self):
        torch.manual_seed(42)
        model = dino_vit_b8()
        model.cuda()
        ckpt = 'hdfs://haruna/home/byte_search_nlp_lq/user/huangwenguan/modelhub/dino_official/dino_vitbase8_pretrain.pth'
        load_from_pretrain(model, ckpt)
        with torch.no_grad():
            fake_input = torch.randn([8, 3, 224, 224]).cuda()
            output = model(fake_input)
            print(output.shape, 'output')

            output = model(self.input_batch)
            print(output.shape, 'output dog')

    # def test_dino_vit_b32(self):
    #     torch.manual_seed(42)
    #     model = dino_vit_b32()
    #     model.cuda()
    #     ckpt = '/mnt/nlp-lq/huangwenguan/modelhub/dino_official/dino_vitbase8_pretrain.pth'
    #     load_from_pretrain(model, ckpt)
    #     with torch.no_grad():
    #         fake_input = torch.randn([8, 3, 224, 224]).cuda()
    #         output = model(fake_input)
    #         print(output.shape, 'output')

    #         output = model(self.input_batch)
    #         print(output.shape, 'output dog')


if __name__ == '__main__':
    unittest.main()
