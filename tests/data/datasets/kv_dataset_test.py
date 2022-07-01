#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Huang Wenguan (huangwenguan@bytedance.com)
Date: 2020-11-12 15:09:54
LastEditTime: 2020-11-16 17:23:58
LastEditors: Huang Wenguan
Description: test of ResNetClassifier
'''

import unittest
from unittest import TestCase
import io
from PIL import Image
import torch

from fex.data import KVDataset, KVSampler, worker_init_fn


class TestKVDataset(TestCase):
    """ test kvdataset """

    @unittest.skip(reason='doas not support in CI')
    def test_dataset(self):
        data_path = "hdfs://haruna/home/byte_search_nlp_lq/user/huangwenguan/imagenet/val"
        num_readers = 32
        batch_size = 32
        my_dataset = KVDataset(data_path, num_readers)
        my_sampler = KVSampler(my_dataset, batch_size=batch_size,
                               num_replicas=1,
                               rank=0,
                               shuffle=False)
        my_loader = torch.utils.data.DataLoader(my_dataset,
                                                batch_size=None,  # 这里要 None, 因为batching是在Dataset里做
                                                sampler=my_sampler,
                                                num_workers=8,
                                                worker_init_fn=worker_init_fn)

        for idx, (key, value) in enumerate(my_loader):
            self.assertEqual(len(key), len(value))
            for v in value:
                img = Image.open(io.BytesIO(v))
                self.assertEqual(len(img.size), 2)
            if idx > 10:
                break


if __name__ == '__main__':
    unittest.main()
