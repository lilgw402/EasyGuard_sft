#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Huang Wenguan (huangwenguan@bytedance.com)
Date: 2020-11-12 15:09:54
LastEditTime: 2020-11-16 17:20:22
LastEditors: Huang Wenguan
Description: test of ResNetClassifier
'''

import unittest
from unittest import TestCase
import torch
import numpy as np

from fex.data import TFRecordDataset


description = {'input_ids': 'int',
               'input_mask': 'int',
               'input_segment_ids': 'int',
               'label_ids': 'int'}


def input_fn(example):
    features = {}
    for key, typename in description.items():
        tf_typename = {
            "byte": "bytes_list",
            "float": "float_list",
            "int": "int64_list"
        }[typename]

        if key not in example.features.feature:
            raise ValueError("Key {} doesn't exist.".format(key))
        value = getattr(example.features.feature[key], tf_typename).value
        if typename == "byte":
            value = np.frombuffer(value[0], dtype=np.uint8)
        elif typename == "float":
            value = np.array(value, dtype=np.float32)
        elif typename == "int":
            value = np.array(value, dtype=np.int32)
        features[key] = value
    return features


class TestTFRecordDataset(TestCase):
    """ test tfrecord dataset """

    @unittest.skip(reason='doas not support in CI')
    def test_dataset(self):
        data_path = "hdfs://haruna/home/byte_search_nlp_lq/user/zhouxincheng/clue/cmnli/eval.tfrecord"
        batch_size = 32
        my_dataset = TFRecordDataset(data_path, shuffle=True, rank=0, world_size=1,
                                     repeat=False, input_fn=input_fn)
        my_loader = torch.utils.data.DataLoader(my_dataset,
                                                batch_size=batch_size,
                                                num_workers=1)

        for idx, data in enumerate(my_loader):
            self.assertEqual(list(data['input_ids'].shape), [batch_size, 128])
            self.assertEqual(list(data['input_mask'].shape), [batch_size, 128])
            self.assertEqual(list(data['input_segment_ids'].shape), [batch_size, 128])
            self.assertEqual(list(data['label_ids'].shape), [batch_size, 1])
            if idx > 10:
                break

    # TODO: 测试多file的dataset，split_shard、shuffle等的测试


if __name__ == '__main__':
    unittest.main()
