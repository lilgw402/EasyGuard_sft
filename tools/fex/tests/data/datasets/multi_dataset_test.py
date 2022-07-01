#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Descripttion:
version:
Author: suiguobin
Date: 2021-08-02 20:16:29
LastEditors: suiguobin
LastEditTime: 2021-08-26 16:39:59
'''

import unittest
from unittest import TestCase
import json
import torch

from fex.data import DistLineReadingDataset
from fex.data import MultiDataset
from fex.data.datasets.iteration_strategies import SizeProportionalIterationStrategy, RoundRobinIterationStrategy
import time


class TestMultiDataset(TestCase):
    """ test Multi dataset """

    @unittest.skip(reason='doas not support in CI')
    def test_dataset(self):
        data_path = "hdfs://haruna/home/byte_search_nlp_lq/user/suiguobin/image/train_test/image_title"
        batch_size = 32
        dataset1 = DistLineReadingDataset(data_path, shuffle=True, rank=0, world_size=1,
                                          repeat=False)

        data_path2 = "hdfs://haruna/home/byte_search_nlp_lq/user/suiguobin/image/train_test/image_image"
        dataset2 = DistLineReadingDataset(data_path2, shuffle=True, rank=0, world_size=1,
                                          repeat=False)

        sizes = [222750, 17580]
        datasets = {
            'image_title': dataset1,
            'image_image': dataset2
        }
        # strategy = SizeProportionalIterationStrategy(datasets, sizes)
        strategy = RoundRobinIterationStrategy(datasets)
        multidataset = MultiDataset(datasets, iteration_strategy=strategy, repeat=False)

        my_loader = torch.utils.data.DataLoader(multidataset,
                                                batch_size=batch_size,
                                                num_workers=8)
        st_time = time.time()
        for idx, data in enumerate(my_loader):
            self.assertEqual(len(data), batch_size)
            for line in data:
                json.loads(line.strip())
            if idx > 10:
                break
        ed_time = time.time()
        print('cost time:', (ed_time - st_time) / 1000.0)
    # TODO: 测试多file的dataset，split_shard、shuffle等的测试


if __name__ == '__main__':
    unittest.main()
