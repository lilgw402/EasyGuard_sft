# -*- coding: utf-8 -*-
'''
image dataset json format, from hdfa
'''

from base64 import b64decode
from collections import defaultdict
import io
import random
import json
from PIL import Image
import traceback
import os

import numpy as np
import torch

from fex.config import CfgNode
from fex import _logger as log
from fex.utils.distributed import local_rank_zero_only
from fex.utils.torch_io import load as torch_io_load
from fex.utils.hdfs_io import hopen
from fex.data.datasets.dist_dataset import split_shard
from example.clip.dataset import ImageDataset


def load_data_offsets(training_state_path=None):
    data_offsets = None
    if training_state_path is not None:
        training_state = torch_io_load(training_state_path, map_location='cpu')
        data_offsets = training_state['data_offsets']
        data_offsets_basename = {os.path.basename(k): v for k, v in data_offsets.items()}
        local_rank_zero_only(log.info)(f'[Resuming] data offsets: {data_offsets_basename}')
    return data_offsets


class ElasticImageDataset(ImageDataset):
    """
    image dataset
    """

    def __init__(self,
                 config: CfgNode,
                 data_path: str,
                 rank: int = 0,
                 world_size: int = 1,
                 shuffle: bool = True,
                 repeat: bool = False,
                 preprocess_mode: str = 'torchvision',
                 training_state_path: dict = None,
                 *args, **kwargs):
        super().__init__(config, data_path, rank, world_size, shuffle, repeat,
                         preprocess_mode, *args, **kwargs)
        self.data_offsets = load_data_offsets(training_state_path)
        # load num records of each file
        train_meta_path = config.get('dataset.train_meta_path')
        assert train_meta_path is not None, 'must specify train meta'
        self.train_file_sizes = {}
        for path in train_meta_path.split(','):
            train_meta_cfg = CfgNode()
            train_meta_cfg.update_cfg(path)
            self.train_file_sizes.update(dict(train_meta_cfg.get('train_meta')))

    def __iter__(self):
        for example, filepath, line_id in self.generate():
            try:
                data_item = json.loads(example)

                # 1. 文本处理：根据text_field 来取文本的字段
                text = ''
                try:
                    for field in self.text_field:
                        cur_field = data_item[field]
                        if cur_field:
                            text += cur_field + ' '
                    text = text.strip()
                except Exception as e:
                    log.error(e, data_item.keys())

                tokens = self.tokenizer.tokenize(text)
                if len(tokens) == 0:
                    # log.error('empty input: %s, will skip, %s/%s=%s' % (text, self.emp, self.tot, round(self.emp/self.tot, 4)))
                    continue
                tokens = ['[CLS]'] + tokens[:self.max_len - 2] + ['[SEP]']
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

                # 2. image 处理 兼容抖音封面数据和图搜数据
                try:
                    image_binary = data_item[self.image_field]
                    image_str = b64decode(image_binary)
                except Exception as e:
                    log.error(e, data_item.keys())
                if self.preprocess_mode == 'torchvision':
                    # 如果是 torchvision，用pillow 来读，现场做transform
                    image = Image.open(io.BytesIO(image_str)).convert("RGB")
                    image = self.transform(image)
                elif self.preprocess_mode == 'dali':
                    # 如果是 dali，基于np 来读
                    image = np.array(np.frombuffer(image_str, dtype=np.uint8))
                elif self.preprocess_mode == 'bytedvision':
                    # 如果是byted vision 直接当做str就行
                    image = image_str

                res = {'image': image,
                       'input_ids': token_ids,
                       'filepath': filepath,
                       'line_id': line_id}

                yield res

            except Exception as e:
                log.error(traceback.format_exc())
                log.error('encounter broken data: %s' % e)
                log.error(data_item.keys())

    def collect_fn(self, data):
        images = []
        input_ids = []
        input_mask = []
        input_segment_ids = []
        #max_len = max([len(b['input_ids']) for b in data])
        max_len = self.max_len
        data_offsets = defaultdict(int)

        for i, ibatch in enumerate(data):
            images.append(ibatch["image"])
            input_ids.append(ibatch['input_ids'][:max_len] + [self.PAD] * (max_len - len(ibatch['input_ids'])))
            input_mask.append([1] * len(ibatch['input_ids'][:max_len]) + [0] * (max_len - len(ibatch['input_ids'])))
            input_segment_ids.append([0] * max_len)
            filepath = ibatch['filepath']
            line_id = ibatch['line_id']
            data_offsets[filepath] = max(data_offsets[filepath], line_id)

        if self.preprocess_mode == 'torchvision':
            images = torch.stack(images, dim=0)
        input_ids = torch.tensor(input_ids)
        input_mask = torch.tensor(input_mask)
        input_segment_ids = torch.tensor(input_segment_ids)
        res = {"image": images,
               "input_ids": input_ids,
               "input_mask": input_mask,
               "input_segment_ids": input_segment_ids,
               "data_offsets": dict(data_offsets)
               }
        return res

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
                if len(cur_dataloader_files) % worker_info.num_workers != 0:
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
            log.info(f"[DataLoader] --> Rank:[{self.rank}]  Workers:[{worker_info.id}] process file: {len(cur_worker_files)} :{self.get_surfix(cur_worker_files)}  ...")

            open_fn = hopen if self.is_hdfs else open
            if self.data_offsets is not None:
                # shoud resume progress
                log.info(f'[DataLoader] Rank [{self.rank}] Worker [{worker_info.id}] is resuming data offsets ...')
                for filepath in cur_worker_files:
                    prev_offset = self.data_offsets.get(filepath, 0)
                    if prev_offset + 1 < self.train_file_sizes[filepath]:
                        # file is not finished, continue reading
                        with open_fn(filepath, 'rb') as reader:
                            for line_id, line in enumerate(reader):
                                if line_id >= prev_offset:
                                    yield line.decode(), filepath, line_id
                # clear previous progress
                self.data_offsets = None
            else:
                for filepath in cur_worker_files:
                    with open_fn(filepath, 'rb') as reader:
                        for line_id, line in enumerate(reader):
                            yield line.decode(), filepath, line_id

            if not self.repeat:
                break
