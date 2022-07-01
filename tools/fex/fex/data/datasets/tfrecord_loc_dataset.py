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
import warnings
import random
import torch
from torch.utils.data import Dataset
from tfrecord import example_pb2

from fex.utils.hdfs_io import hopen, hlist_files
from fex import _logger as log


class TFRecordLocalDataset(Dataset):  # pylint: disable=W0223
    """
    iterate a set of folders.
    """

    def __init__(self,
                 data_path,
                 shuffle=False,
                 input_fn: Callable = lambda x: x):
        super().__init__()
        self.shuffle = shuffle
        self.files = hlist_files(data_path.split(','))
        self.is_hdfs = data_path.startswith('hdfs')
        self.input_fn = input_fn
        self.pool = []
        self.load_all_set()
        log.info('[DATA]--all dataset containing {} files.'.format(len(self.files)))


    def load_all_set(self):
        for filepath in self.files:
            if self.is_hdfs:
                print("processing file:", filepath)
                with hopen(filepath, 'rb') as reader:
                    for example in tfrecord_iterator(reader):
                        self.pool.append(self.input_fn(example))
            else:
                with open(filepath, 'rb') as reader:
                    for example in tfrecord_iterator(reader):
                        self.pool.append(self.input_fn(example))

    def __len__(self):
        return len(self.pool)

    def __getitem__(self, index):
        return self.pool[index]


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
