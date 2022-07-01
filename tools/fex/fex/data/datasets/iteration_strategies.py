'''
Descripttion:
version:
Author: suiguobin
Date: 2021-08-02 20:16:29
LastEditors: suiguobin
LastEditTime: 2021-08-20 14:44:10
'''
# Copyright (c) Facebook, Inc. and its affiliates.

import logging
import warnings
from typing import Dict

import numpy as np
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


class RoundRobinIterationStrategy():
    """
    Samples datasets one by one in round robin fashion.

    Start index can be specified in config as `start_idx`.

    Also defaults to size proportional sampling as roundrobin
    doesn't make sense with validation and testing splits
    as they need to finish one complete epoch.
    """

    def __init__(
        self, dataloaders, start_idx=0, name="round_robin", *args, **kwargs
    ):
        self.dataloaders = dataloaders
        self._current_idx = start_idx

    def __call__(self, *args, **kwargs):
        nxt = self._current_idx
        self._current_idx = (self._current_idx + 1) % len(self.dataloaders)
        return nxt


class SizeProportionalIterationStrategy():
    """
    Samples index based on size of each dataset. Bigger datasets
    are sampled more and this strategy requires completing
    all iterators before starting new ones. Default in MMF.
    """

    def __init__(
        self, dataloaders, data_sizes, name="size_proportional", *args, **kwargs
    ):
        self.dataloaders = dataloaders
        self._per_dataset_lengths = []
        self._total_length = 0
        for dataset_instance_length in data_sizes:
            self._per_dataset_lengths.append(dataset_instance_length)
            self._total_length += dataset_instance_length

        self._dataset_probabilities = self._per_dataset_lengths[:]
        self._dataset_probabilities = [
            prob / self._total_length for prob in self._dataset_probabilities
        ]
        print("self._dataset_probabilities:", self._dataset_probabilities)

    def __call__(self, *args, **kwargs):
        choice = np.random.choice(
            len(self.dataloaders), 1, p=self._dataset_probabilities
        )[0]
        return choice
