#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
swin vidt test
'''

import os
import unittest
from unittest import TestCase

import torch
from PIL import Image
from torchvision import transforms

from fex.nn import swin_base_win7_vidt
from fex.utils.load import load_from_pretrain


class TestSwinVidt(TestCase):
    """ test swin """

    @unittest.skip(reason='doas not support in CI')
    def test_swin_base_random_input(self):
        torch.manual_seed(42)
        model = swin_base_win7_vidt(det_token_num=8)
        load_from_pretrain(model, '/mnt/bd/hwg-bd-lq/modelhub/detr/vidt_plus_base_det300.th',
                           [
                               'backbone.det_token->det_token:truncate1(8)',
                               'backbone.det_pos_embed->det_pos_embed:truncate1(8)',
                               'backbone.->'])
        if torch.cuda.is_available():
            model.cuda()
        model.eval()

        with torch.no_grad():
            fake_input = torch.randn([16, 3, 224, 224])
            fake_mask = torch.zeros([16, 224, 224])
            if torch.cuda.is_available():
                fake_input = fake_input.to('cuda')
                fake_mask = fake_mask.to('cuda')
            output = model(fake_input, fake_mask)
            print(output['feature_map'].shape)
            print(output['det_tgt'].shape)
            print(output['pooled_out'].shape)
            for k in output['encoded_layers']:
                print(k.shape)


if __name__ == '__main__':
    unittest.main()
