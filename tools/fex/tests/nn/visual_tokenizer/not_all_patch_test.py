#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
test not all patch
'''

import os
import unittest
from unittest import TestCase

import torch
from PIL import Image

from fex.nn.visual_tokenizer import create_visual_tokenizer

BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
TEST_DIR = os.path.join(BASE_DIR, "../ci/test_data")


class TestNotAllPatch(TestCase):
    """ test NotAllPatch """

    def te1st_not_all_patch_cls(self):

        model = create_visual_tokenizer('NAPClipB16S224', keep_token=8)
        model.cuda()
        torch.manual_seed(42)
        print(model)

        with torch.no_grad():
            fake_input = torch.randn([8, 3, 224, 224])
            if torch.cuda.is_available():
                fake_input = fake_input.to('cuda')
            output = model(fake_input)
            for k, v in output.items():
                print(k, v.shape)

    def te1st_not_all_patch_attn(self):

        model = create_visual_tokenizer('NAPClipB16S224', keep_token=8, mode='attn')
        model.cuda()
        torch.manual_seed(42)
        print(model)

        with torch.no_grad():
            fake_input = torch.randn([8, 3, 224, 224])
            if torch.cuda.is_available():
                fake_input = fake_input.to('cuda')
            output = model(fake_input)
            for k, v in output.items():
                print(k, v.shape)

    def te1st_not_all_patch_cluster(self):

        model = create_visual_tokenizer('NAPClipB16S224', keep_token=8, mode='cluster')
        model.cuda()
        torch.manual_seed(42)
        print(model)

        with torch.no_grad():
            fake_input = torch.randn([8, 3, 224, 224])
            if torch.cuda.is_available():
                fake_input = fake_input.to('cuda')
            output = model(fake_input)
            for k, v in output.items():
                print(k, v.shape)


if __name__ == '__main__':
    unittest.main()
