#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Huang Wenguan (huangwenguan@bytedance.com)
Date: 2020-11-12 15:09:54
LastEditTime: 2020-11-18 18:59:12
LastEditors: Huang Wenguan
Description: test of ResNetClassifier
'''

import copy
import json
import unittest
from unittest import TestCase
import torch

from fex.data import DistLineReadingDataset
from fex.data.pipe import MLMPipe
from fex.config import cfg, reset_cfg


class TestMLMPipe(TestCase):
    """ test mlm pipe """

    def setUp(self):
        data_path = "hdfs://haruna/home/byte_search_nlp_lq/user/huangwenguan/data/tusou/tusou_14b_cut_rn_flt_val"
        batch_size = 4
        my_dataset = DistLineReadingDataset(data_path, shuffle=True, rank=0, world_size=1,
                                            repeat=False)
        self.my_loader = torch.utils.data.DataLoader(my_dataset,
                                                     batch_size=batch_size,
                                                     num_workers=8)
        cfg = reset_cfg()
        cfg.update_cfg('./ci/test_data/test_config/tuso_mlm.yaml')
        self.cfg = cfg

    @unittest.skip(reason='doas not support in CI')
    def test_mlm_pipe(self):
        my_cfg = copy.deepcopy(self.cfg)
        mlm_pipe = MLMPipe(my_cfg)

        for idx, data in enumerate(self.my_loader):
            for line in data:
                raw_data = json.loads(line.strip())
                example = mlm_pipe(raw_data)
                self.assertEqual(set(example.keys()), set(['relationship_label', 'mlm_labels', 'input_ids', 'segment_ids']))
                self.assertEqual(len(example['input_ids']), len(example['mlm_labels']), len(example['segment_ids']))
                self.assertLess(len(example['input_ids']), my_cfg.DATASET.SEQ_LEN)
            if idx > 10:
                break

    @unittest.skip(reason='doas not support in CI')
    def test_rate(self):
        my_cfg = copy.deepcopy(self.cfg)
        my_cfg.defrost()
        my_cfg.NETWORK.WITH_REL_LOSS = True
        mlm_pipe = MLMPipe(my_cfg)

        labels = []
        for idx, data in enumerate(self.my_loader):
            for line in data:
                raw_data = json.loads(line.strip())
                example = mlm_pipe(raw_data)
                labels.append(example['relationship_label'])
            if idx > 100:
                break

        rate = sum(labels) / len(labels)
        self.assertTrue(0.25 < rate < 0.75)

    @unittest.skip(reason="not_mask_rate variable not meeting expectation, with be fix by@huangwenguan")
    def test_mlm_rate(self):
        my_cfg = copy.deepcopy(self.cfg)
        mlm_pipe = MLMPipe(my_cfg)

        token_num = 0
        mlm_labels_cnt = 0
        all_zero_cnt = 0
        data_num = 0
        for idx, data in enumerate(self.my_loader):
            for line in data:
                raw_data = json.loads(line.strip())
                example = mlm_pipe(raw_data)
                token_num += len(example['mlm_labels'])
                mlm_labels_cnt += sum([1 if i > 0 else 0 for i in example['mlm_labels']])
                data_num += 1
                all_zero_cnt += 1 if sum([1 if i > 0 else 0 for i in example['mlm_labels']]) == 0 else 0
            if idx > 100:
                break

        mlm_rate = mlm_labels_cnt * 1.0 / token_num
        not_mask_rate = all_zero_cnt * 1.0 / data_num
        self.assertTrue(0.1 < mlm_rate < 0.2)
        self.assertTrue(not_mask_rate < 0.08)

    @unittest.skip(reason='doas not support in CI')
    def test_origin(self):
        my_cfg = copy.deepcopy(self.cfg)
        my_cfg.defrost()
        my_cfg.DATASET.MASK_STYLE = 'origin'
        mlm_pipe = MLMPipe(my_cfg)

        for idx, data in enumerate(self.my_loader):
            for line in data:
                raw_data = json.loads(line.strip())
                example = mlm_pipe(raw_data)
                self.assertEqual(set(example.keys()), set(['relationship_label', 'mlm_labels', 'input_ids', 'segment_ids']))
                self.assertEqual(len(example['input_ids']), len(example['mlm_labels']), len(example['segment_ids']))
                self.assertLess(len(example['input_ids']), my_cfg.DATASET.SEQ_LEN)
            if idx > 10:
                break


if __name__ == '__main__':
    unittest.main()
