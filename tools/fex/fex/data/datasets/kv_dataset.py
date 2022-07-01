#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Huang Wenguan (huangwenguan@bytedance.com)
Date: 2020-11-12 20:11:56
LastEditTime: 2020-11-16 17:25:29
LastEditors: Huang Wenguan
Description: based on kv loader, see https://bytedance.feishu.cn/docs/doccnqFaodw8Tp03v15zVUcwfLf#MNNgBI
'''

import os
import time
import random
import multiprocessing as mp
import torch
try:
    from dataloader import KVReader
except:
    print('dataloader is not installed, you can install it from https://bytedance.feishu.cn/wiki/wikcnlBYX2qZKX9MCsms8ianpmc')


class KVDataset(torch.utils.data.Dataset):
    """
    KVDataset, key value的形式，一次读多条数据，他是继承torch.Dataset的，需要配合Sampler使用

    """

    def __init__(self, path: str, num_readers: int):
        """
        Args:
            path (str): kv-reader的数据的路径
            num_readers (int): 进程数。进程越多性能越快，不过内存等也会占用更多，可以适量调节。一些建议值：8, 16等都可。
        Notes:
            为了性能考虑，他不是一条条拿数据，所以不支持`collect_fn`那种形式，需要在dataset里就做好batching的工作
        """
        super().__init__()
        self.path = path
        self.num_readers = num_readers
        # fix KVReader hangs
        local_rank = int(os.getenv('LOCAL_RANK', '0'))
        time.sleep(local_rank * 8 + random.random() * 4)
        # Use another process to avoid libhdfs.so fork issue
        with mp.Pool(1) as p:
            self.keys = p.map(get_keys, [(path, num_readers)])[0]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if isinstance(index, list):
            if len(index) == 0:
                return [], []
            index = [self.keys[i] for i in index]
            data = self.reader.read_many(index)
            return index, data
        else:
            raise LookupError('Unsupported index: list and int are supported')


class KVSampler(torch.utils.data.distributed.DistributedSampler):
    """ kv sampler """

    def __init__(self, dataset, batch_size, num_replicas, rank, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.batch_size = batch_size

    def __iter__(self):
        iterable = super().__iter__()
        return chunk(iterable, self.batch_size)


def get_keys(args):
    return KVReader(*args).list_keys()


def chunk(iterable, chunk_size):
    """
    将 index iterable 转化为 batch index iterable
    如 chunk([4, 2, 3, 1], 2) ==> [[4, 2], [3, 1]]
    """
    ret = []
    for record in iterable:
        ret.append(record)
        if len(ret) == chunk_size:
            yield ret
            ret = []
    if ret:
        yield ret


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    # Avoid "cannot pickle KVReader object" error
    dataset.reader = KVReader(dataset.path, dataset.num_readers)
