#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Descripttion: multi dataset
version:
Author: suiguobin
Date: 2021-08-02 20:16:29
LastEditors: suiguobin
LastEditTime: 2021-08-20 14:51:47
'''

from typing import List, Any, Dict
import warnings
import random
from itertools import cycle
import torch
from torch.utils.data import IterableDataset

from fex.utils.hdfs_io import hopen, hlist_files
from fex import _logger as log
from fex.data.datasets import iteration_strategies


class MultiDataset(IterableDataset):  # pylint: disable=W0223
    """
    iterate several folders.
    """

    def __init__(self,
                 loaders,
                 iteration_strategy=None,
                 repeat=True):
        super().__init__()
        if loaders is None or len(loaders) == 0:
            warnings.warn(
                "Empty loaders passed into MultiDataLoader. This can have "
                "unintended consequences."
            )

        if iteration_strategy is None:
            iteration_strategy = iteration_strategies.RoundRobinIterationStrategy(
                loaders
            )
        self.loaders_ = loaders
        self._iteration_strategy = iteration_strategy
        self._loaders = {}
        self._num_datasets = len(loaders)
        self.dataset_list = list(loaders.keys())
        self._iterators = {}
        self._finished_iterators = {}

        self.current_index = 0
        self.repeat = repeat

    @property
    def iteration_strategy(self):
        return self._iteration_strategy

    @property
    def num_datasets(self):
        return self._num_datasets

    @property
    def loaders(self):
        return self._loaders

    @property
    def current_dataset_name(self):
        return self.dataset_list[self.current_index]

    def __iter__(self):
        """

        """
        self._finished_iterators = {}
        for key, value in self.loaders_.items():
            self._loaders[key] = iter(value)
        while True:
            try:
                res = next(self.loaders[self.current_dataset_name])
                self.change_dataloader()
                yield res

            except StopIteration:
                self._finished_iterators[self.current_dataset_name] = 1
                if len(self._finished_iterators) == self.num_datasets:
                    if not self.repeat:
                        break
                    self._finished_iterators = {}
                    for key, value in self.loaders_.items():
                        self._loaders[key] = iter(value)
                else:
                    self.change_dataloader()

    def change_dataloader(self):
        choice = 0

        choice = self.iteration_strategy()

        while self.dataset_list[choice] in self._finished_iterators:
            choice = self.iteration_strategy()

        self.current_index = choice
