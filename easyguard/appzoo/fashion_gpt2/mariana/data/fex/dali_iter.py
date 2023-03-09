# -*- coding: utf-8 -*-
'''
Created on Feb-04-21 16:15
pytorch_dali_pipeline.py
Description:
'''

from typing import Dict
import random
import os
import numpy as np
import torch
try:
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    DALIGenericIterator = None
    # print("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")


class PytorchDaliIter:
    def __init__(self, dali_pipeline, size, *args, **kargs):
        if DALIGenericIterator is None:
            raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")
        self.dali_pipeline = dali_pipeline
        self.dali_pipeline.build()
        self.queue = self.dali_pipeline.queue
        self._size = size  # don't pass size to dali iter as it may conflict with the last batch policy
        self.dali_iter = DALIGenericIterator(pipelines=dali_pipeline, *args, **kargs)

    def __len__(self):
        return self._size

    def __iter__(self):
        return self

    def __next__(self):
        dali_iter_next = self.dali_iter.__next__()
        extra_data = self.queue.get()
        extra_data.update(dali_iter_next[0])
        return extra_data
