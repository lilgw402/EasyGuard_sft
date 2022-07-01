#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
DistLineReadingDatasetV2

V2 相比V1 支持resume、queue shuffling 的功能
'''

import os
from typing import List, Any
import warnings
import random
from itertools import cycle
import torch
from torch.utils.data import IterableDataset

from fex.config import CfgNode
from fex.utils.hdfs_io import hopen, hlist_files
from fex.utils.torch_io import load as torch_io_load
from fex.utils.distributed import local_rank_zero_only
from fex import _logger as log


class DistLineReadingDatasetV2(IterableDataset):  # pylint: disable=W0223
    """
    iterate a set of folders.
    """

    def __init__(self,
                 data_path: str,
                 rank: int = 0,
                 world_size: int = 1,
                 shuffle: bool = False,
                 repeat: bool = False,
                 verbose: bool = True,
                 buffer_size: int = -1,
                 meta_data_path: str = None,
                 state_path: str = None
                 ):
        """
        data_path: 数据文件夹路径，会list出这个文件夹下面的所有file。支持多个文件夹，用 `,` 隔开
        rank: 在多机多卡的情况下，需要根据rank来划分
        world_size: 在多机多卡的情况下，需要根据world_size来划分
        repeat: 是否重复，如果重复的话，在遍历完一遍数据后，会继续重新遍历
        shuffle: 是否shuffle，按file shuffle；以及如果有buffer的话，对buffer shuffle
        verbose: 是否打印一些log
        buffer_size: 是否构造一个buffer 来预存一些数据。这个的好处是配合shuffle可以做到一定程度的打散
        meta_data_path: 记录数据meta 信息的config 路径，主要用来load 每个文件的行数
        state_path: 记录 data offset，对于resume 有用
        """
        super().__init__()
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size

        self.files = hlist_files(data_path.split(','))
        self.files = [f for f in self.files if f.find('_SUCCESS') < 0]
        self.files.sort()

        self.is_hdfs = data_path.startswith('hdfs')
        self.repeat = repeat
        log.info(
            '[DATA]--all dataset containing {} files.'.format(len(self.files)))

        if len(self.files) % self.world_size != 0:
            log.info('[DATA]--Whole dataset file num %s cannot split to worldsize %s ' %
                     (len(self.files), self.world_size))
        self.verbose = verbose

        # 记录和判断 data offset 的，用于训练resume 用的
        self.file_sizes = self._load_file_size_meta(meta_data_path)  # 只有在有data_offset 的情况会被用
        self.load_data_offsets(state_path)

        # local data buffer
        self.buffer = []
        self.buffer_size = buffer_size

    def _load_file_size_meta(self, meta_config_path):
        """
        meta_config_path 是一个yaml
        譬如：
        train_meta:
          hdfs://haruna/home/byte_search_nlp_lq/user/suiguobin/image/clip/tusou_datas/20210627_tbase_1.3b/part-00000.snappy: 222511
          hdfs://haruna/home/byte_search_nlp_lq/user/suiguobin/image/clip/tusou_datas/20210627_tbase_1.3b/part-00001.snappy: 222300
          ...
        """
        if meta_config_path:
            train_meta_cfg = CfgNode()
            train_meta_cfg.update_cfg(meta_config_path)
            return dict(train_meta_cfg.get('train_meta'))
        else:
            return {}

    def load_data_offsets(self, training_state_path=None):
        """ 加载 data offset """
        self.data_offsets = {}
        if training_state_path is not None:
            training_state = torch_io_load(training_state_path, map_location='cpu')
            self.data_offsets = training_state['data_offsets']
            data_offsets_basename = {os.path.basename(k): v for k, v in self.data_offsets.items()}
            local_rank_zero_only(log.info)(f'[Resuming] data offsets: {data_offsets_basename}')

    def generate(self, seed=42):
        """
        # TODO: 加cache，加prefetch
        在dataloader里调用，dataloader会启num_worker个进程来遍历dataset。
        这个函数一开始做了两次split_shard，对文件进行划分。
        self.files是总的数据集的文件数，
        第一次split_shard是基于rank和world_size来分的，是gpu节点维度的；
        第二次split_shard是基于worker_info的，是一个gpu节点的dataloader内，给不同的worker划分文件。

        """

        # 第一次 split: 按 rank 划分。
        # 先对files做一次sort和（seed）shuffle，每个rank拿到的seed都是一样的。这样得到的list是一样的，保证split时不重复。
        # TODO: 这里的seed实际一直一样，后面可以想办法，让seed 从trainer 传过来，让每个epoch里，每个rank拿到的file不一样，更好的做shuffle。
        if self.shuffle:
            self.files = self.sort_and_shuffle(self.files, seed)
        else:
            self.files.sort()
        if self.world_size == 1 or len(self.files) == 1:
            cur_dataloader_files = self.files
        else:
            cur_dataloader_files = split_shard(
                self.files, self.rank, self.world_size)

        # 第二次 split：各个rank内部，将 cur_dataloader_files 按 num_workers 分。注意每个worker都会执行。
        # 每个rank的每个 worker 拿到的都是这个：cur_dataloader_files，是一样的
        while True:
            worker_info = torch.utils.data.get_worker_info()

            if worker_info is not None:
                if len(cur_dataloader_files) % worker_info.num_workers != 0 and self.verbose:
                    log.info('[DATA]--current dataloader [%s] file num %s cannot split to worker_num %s ' %
                             (self.rank, len(cur_dataloader_files), worker_info.num_workers))
                # 这里是真正做第二次split的地方， cur_worker_files 是每个worker 拿到的
                cur_worker_files = split_shard(
                    cur_dataloader_files, worker_info.id, worker_info.num_workers)
            else:
                # num_worker=0，只有主进程的情况
                cur_worker_files = cur_dataloader_files

            if self.shuffle:  # 每个epoch下，虽然每个rank-每个worker对应的files是一样的，但会这里shuffle一下，读的顺序按file有打乱。
                random.shuffle(cur_worker_files)
            # cur_worker_files 是每个worker拿到的结果
            if self.verbose:
                log.info(
                    f"[DataLoader] --> Rank:[{self.rank}]  Workers:[{worker_info.id if worker_info else 0}] process file: {len(cur_worker_files)} :{self.get_surfix(cur_worker_files[:3])}  ..."
                )

            open_fn = hopen if self.is_hdfs else open
            # 在有 data_offsets 的情况下，将 dataoffs 里存着的之前用过的数据都剔除掉
            for filepath in cur_worker_files:
                # TODO: 放这好吗
                prev_offset = self.data_offsets.get(filepath, 0)
                if prev_offset + 1 >= self.file_sizes.get(filepath, float('inf')):  # 这个 file 在 data_offsets 里记录用完了，结束
                    continue
                with open_fn(filepath, 'rb') as reader:
                    for line_id, line in enumerate(reader):
                        if line_id < prev_offset:  # 小于 offset 的行数都 continue
                            continue
                        if self.buffer_size <= 0:  # 如果没有buffer 直接返回
                            yield line.decode(), filepath, line_id
                        elif len(self.buffer) < self.buffer_size:  # 如果有buffer_size 且没塞完，先塞进去
                            self.buffer.append([line.decode(), filepath, line_id])
                        else:
                            yield self.enq_deq_buffer(line.decode(), filepath, line_id)

            # 最后如果buffer里有东西，就pop一下
            if self.buffer:
                if self.shuffle:
                    random.shuffle(self.buffer)
                for line, filepath, line_id in self.buffer:
                    yield line, filepath, line_id

            # 如果不repeat，就一直循环
            if not self.repeat:
                break

    def enq_deq_buffer(self, line, filepath, line_id):
        """
        在有buffer的情况下的处理，从buffer里pop一条出来，再把当前条塞进去
        """
        pop_line, pop_filepath, pop_line_id = self.buffer.pop(random.randrange(self.buffer_size)) if self.shuffle else self.buffer.pop(0)  # 注意是空的情况
        self.buffer.append([line, filepath, line_id])
        return pop_line, pop_filepath, pop_line_id

    def __iter__(self):
        return self.generate()

    def reset(self, seed):
        del self.buffer
        self.buffer = []
        return self.generate(seed)

    def sort_and_shuffle(self, data, seed):
        data.sort()
        random.Random(seed).shuffle(data)
        return data

    def get_surfix(self, name_list):
        return [n.split('/')[-1] for n in name_list]


def split_shard(data: List[Any], shard_idx: int, shard_size: int):
    num = len(data)
    if num < shard_size:
        raise RuntimeError("num:{} < shard size:{}".format(num, shard_size))
    start_idx = (num * shard_idx) // shard_size
    end_idx = (num * (shard_idx + 1)) // shard_size
    return data[start_idx: end_idx]
