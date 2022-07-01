#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Huang Wenguan (huangwenguan@bytedance.com)
Date: 2020-11-16 15:25:34
LastEditTime: 2020-11-16 20:37:32
LastEditors: Huang Wenguan
Description: tfrecord dataset, copy from https://code.byted.org/wushiwei/workbench/blob/master/data_process/datasets.py
'''

from typing import Iterable, List, Any, Callable, IO
import struct
import gc
import warnings
import random
from itertools import cycle
import torch
from torch.utils.data import IterableDataset
from tfrecord import example_pb2

from fex.utils.hdfs_io import hopen, hlist_files
from fex import _logger as log


class TFRecordDataset(IterableDataset):  # pylint: disable=W0223
    """
    iterate a set of folders.
    """

    def __init__(self,
                 data_path,
                 rank: int = 0,
                 world_size: int = 1,
                 shuffle=False,
                 repeat=False,
                 input_fn: Callable = lambda x: x):
        super().__init__()
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size

        self.files = hlist_files(data_path.split(','))
        self.is_hdfs = data_path.startswith('hdfs')

        self.input_fn = input_fn
        self.repeat = repeat
        log.info('[DATA]--all dataset containing {} files.'.format(len(self.files)))

        if len(self.files) % self.world_size != 0:
            warnings.warn('[DATA]--Whole dataset file num %s cannot split to worldsize %s ' %
                          (len(self.files), self.world_size))

    def generate(self):
        """
        在dataloader里调用，dataloader会启num_worker个进程来遍历dataset。
        这个函数一开始做了两次split_shard，对文件进行划分。
        self.files是总的数据集的文件数，
        第一次split_shard是基于rank和world_size来分的，是gpu节点维度的；
        第二次split_shard是基于worker_info的，是一个gpu节点的dataloader内，给不同的worker划分文件。

        """
        if self.world_size == 1 or len(self.files) == 1:
            cur_dataloader_files = self.files
        else:
            cur_dataloader_files = split_shard(self.files, self.rank, self.world_size)
            # log.info("[DataLoader]--rank:{} size:{} are processing file: {}".format(
            #     self.rank, self.world_size, cur_dataloader_files))

        worker_info = torch.utils.data.get_worker_info()
        if len(cur_dataloader_files) % worker_info.num_workers != 0:
            warnings.warn('[DATA]--current dataloader %s file num %s cannot split to worker_num %s ' %
                          (self.rank, len(cur_dataloader_files), worker_info.num_workers))

        if worker_info is not None:
            cur_worker_files = split_shard(cur_dataloader_files, worker_info.id, worker_info.num_workers)
            log.info("[DataLoader] --> rank:{} worker:{} size:{} are processing files: {} ..".format(
                self.rank, worker_info.id, worker_info.num_workers, cur_worker_files[0]))

        if self.shuffle:
            random.shuffle(cur_worker_files)
        for filepath in cur_worker_files:
            if self.is_hdfs:
                print("processing file:", filepath)
                with hopen(filepath, 'rb') as reader:
                    for example in tfrecord_iterator(reader):
                        yield self.input_fn(example)
                continue
            with open(filepath, 'rb') as reader:
                for example in tfrecord_iterator(reader):
                    yield self.input_fn(example)
            gc.collect()

    def __iter__(self):
        if not self.repeat:
            return self.generate()
        return cycle(self.generate())


def split_shard(data: List[Any], shard_idx, shard_size):
    num = len(data)
    if num < shard_size:
        raise RuntimeError("num:{} < shard size:{}".format(num, shard_size))
    start_idx = (num * shard_idx) // shard_size
    end_idx = (num * (shard_idx + 1)) // shard_size
    return data[start_idx: end_idx]


def tfrecord_iterator(stream: IO[Any]) -> Iterable:
    buffer_size = 1024 * 1024
    datum_bytes = bytearray(buffer_size)

    def read_records():
        while True:
            length_bytes = stream.read(8)
            if len(length_bytes) != 8:
                #raise RuntimeError("Failed to read the record size.")
                log.info("Reading length failed, stop reading.")
                return
            crc_bytes = stream.read(4)
            if len(crc_bytes) != 4:
                raise RuntimeError("Failed to read the start token.")
            length, = struct.unpack("<Q", length_bytes)
            if length > len(datum_bytes):
                datum_bytes.zfill(int(length * 1.5))
            datum_bytes_view = memoryview(datum_bytes)[:length]
            datum_bytes_view = stream.read(length)
            if len(datum_bytes_view) != length:
                raise RuntimeError("Failed to read the record.")
            crc_bytes = stream.read(4)
            if len(crc_bytes) != 4:
                raise RuntimeError("Failed to read the end token.")
            yield datum_bytes_view

    for record in read_records():
        example = example_pb2.Example()
        example.ParseFromString(record)
        yield example
