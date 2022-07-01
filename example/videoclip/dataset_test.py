# -*- coding: utf-8 -*-

import os
import unittest
from unittest import TestCase
import torch

from fex.config import cfg
from example.videoclip.dataset import VideoDataset
from example.videoclip.dataset_matx import VideoDatasetMatx, worker_init_fn


class VideoDatasetTest(TestCase):
    def setUp(self):
        self.config_path = os.path.join(
            os.path.dirname(__file__), "config.yaml")
        self.batch_size = 2
        cfg.update_cfg(self.config_path)

        self.dataset = VideoDataset(config=cfg,
                                    data_path=cfg.dataset.train_path,
                                    rank=int(os.environ.get('RANK') or 0),
                                    world_size=int(
                                        os.environ.get('WORLD_SIZE') or 1),
                                    shuffle=False,
                                    repeat=False,
                                    pick_query=False)
        self.loader = torch.utils.data.DataLoader(dataset=self.dataset,
                                                  batch_size=self.batch_size,
                                                  num_workers=1,
                                                  pin_memory=True,
                                                  collate_fn=self.dataset.collect_fn)

        self.dataset_matx = VideoDatasetMatx(config=cfg,
                                             data_path=cfg.dataset.train_path,
                                             rank=int(
                                                 os.environ.get('RANK') or 0),
                                             world_size=int(
                                                 os.environ.get('WORLD_SIZE') or 1),
                                             shuffle=False,
                                             repeat=False,
                                             pick_query=False)
        self.matx_loader = torch.utils.data.DataLoader(dataset=self.dataset_matx,
                                                       batch_size=self.batch_size,
                                                       num_workers=1,
                                                       pin_memory=True,
                                                       collate_fn=self.dataset_matx.collect_fn,
                                                       worker_init_fn=worker_init_fn)

    def test_loader(self):
        dataset_iter = iter(self.loader)
        dataset_matx_iter = iter(self.matx_loader)

        for i in range(10):
            res = next(dataset_iter)
            matx_res = next(dataset_matx_iter)

            print('--' * 10)
            print(res["input_ids"])
            print(matx_res["input_ids"])
            print('--' * 10)
            print(res["input_segment_ids"])
            print(matx_res["input_segment_ids"])
            print('--' * 10)
            print(res["input_mask"])
            print(matx_res["input_mask"])


if __name__ == "__main__":
    unittest.main()
