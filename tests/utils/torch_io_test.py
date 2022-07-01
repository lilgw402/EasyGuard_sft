#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Huang Wenguan (huangwenguan@bytedance.com)
Date: 2020-11-09 21:28:19
LastEditTime: 2020-11-12 14:24:51
LastEditors: Huang Wenguan
Description: test of torch model hdfs io
'''

import unittest
from unittest import TestCase
import random
import torch

from fex.utils.torch_io import load


class TestTorchHdfsIO(TestCase):
    """ test of hdfs io api """

    @unittest.skip("need GDPR")
    def test_load_resnet_model(self):
        """ test resnet model from pytorch and hdfs """

        params_from_hdfs = load(
            'hdfs://haruna/home/byte_search_nlp_lq/user/huangwenguan/vl/model/pretrained_model/resnet50-19c8e357-t3.pth'
        )

        params_from_torchvision = torch.hub.load_state_dict_from_url(
            'https://download.pytorch.org/models/resnet50-19c8e357.pth', progress=False)

        self.assertEqual(len(params_from_hdfs.keys()), len(params_from_torchvision.keys()))
        self.assertEqual(set(params_from_hdfs.keys()), set(params_from_torchvision.keys()))

        keys = list(params_from_hdfs.keys())

        for _ in range(5):
            key = random.choice(keys)
            assert torch.allclose(params_from_hdfs[key], params_from_torchvision[key])


if __name__ == "__main__":
    unittest.main()
