#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Huang Wenguan (huangwenguan@bytedance.com)
Date: 2020-11-12 15:09:54
LastEditTime: 2020-11-18 14:39:48
LastEditors: Huang Wenguan
Description: test of ResNetClassifier
'''

import unittest
from unittest import TestCase
import json
import torch

from fex.data import DistLineReadingDataset


class TestTFRecordDataset(TestCase):
    """ test tfrecord dataset """

    @unittest.skip("Data Path not exist")
    def test_dataset(self):
        # data_path不存在了
        data_path = "hdfs://haruna/home/byte_search_nlp_lq/user/huangwenguan/data/label/d0825/text/train_pair_nvis_re"
        batch_size = 32
        my_dataset = DistLineReadingDataset(data_path, shuffle=True, rank=0, world_size=1,
                                            repeat=False)
        my_loader = torch.utils.data.DataLoader(my_dataset,
                                                batch_size=batch_size,
                                                num_workers=8)

        for idx, data in enumerate(my_loader):
            self.assertEqual(len(data), batch_size)
            for line in data:
                json.loads(line.strip())
            if idx > 10:
                break

    # TODO: 测试多file的dataset，split_shard、shuffle等的测试


if __name__ == '__main__':
    unittest.main()
