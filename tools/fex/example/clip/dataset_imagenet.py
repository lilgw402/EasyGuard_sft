#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Huang Wenguan (huangwenguan@bytedance.com)
Date: 2020-11-12 21:20:12
LastEditTime: 2020-11-18 11:14:19
LastEditors: Huang Wenguan
Description: imagnet kv dataset
'''

from typing import List
import functools
import io
import re
import json
import traceback
from PIL import Image
import numpy as np

import torch

from fex.data import KVDataset
from fex import _logger as log
from fex.data import BertTokenizer
from fex.utils.hdfs_io import hopen


class ImageNetKVDataset(KVDataset):
    """
    如果log很多，加这个试试
    export LIBHDFS_OPTS=-Dhadoop.root.logger=${HADOOP_ROOT_LOGGER:-ERROR,console}
    """

    def __init__(self, config, path, num_readers, transform=None, *args, **kwargs):
        super().__init__(path, num_readers)
        self.transform = transform  # 如果dali的预处理，这里是 None，dataset不做预处理，只解析数据返回，在后面预处理。
        self.tag2label = load_labelmapping()

        # 是否需要标签的文本表示
        self.need_text = config.get('dataset.need_text', True)
        self.use_english = config.get('dataset.use_english', False)
        if self.need_text:
            vocab_file = config.network.vocab_file
            self.tokenizer = BertTokenizer(vocab_file, do_lower_case=False)
            self.seq_len = config.dataset.max_len

    def __getitem__(self, index):
        index, images = super().__getitem__(index)

        transformed_images = []
        transformed_labels = []
        name_ids = []
        names = []
        for i, iindex in zip(images, index):
            try:
                if self.transform:
                    img = Image.open(io.BytesIO(i)).convert("RGB")
                    image = self.transform(img)
                else:
                    image = np.array(np.frombuffer(i, dtype=np.uint8))
                # key 长这样： '/n01644900.tar/n01644900_8245.JPEG'
                tagname = re.match('/.*/', iindex).group().replace('.tar', '')[1:-1]
                label_idx = self.tag2label[tagname]['label_idx']
                transformed_images.append(image)
                transformed_labels.append(label_idx)
                if self.need_text:
                    if self.use_english:
                        name = self.tag2label[tagname]['english_name']  # .replace(',', '')
                    else:
                        name = self.tag2label[tagname]['chinese_name']
                    names.append(name)
                    name_ids.append(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(name)))
            except Exception as e:
                log.error('Error %s: on key %s; %s' % (e, iindex, image.shape))
                log.error(traceback.format_exc())
                break
        if transformed_images:
            if self.transform:
                transformed_images = torch.stack(transformed_images, dim=0)
                transformed_labels = torch.tensor(transformed_labels)
            output = {'image': transformed_images,
                      'label': transformed_labels,
                      'index': index,
                      'names': names}

            if self.need_text:
                input_ids, input_mask = truncate_and_pad(name_ids, self.seq_len, need_tensor=self.transform is not None)
                output.update({'input_ids': input_ids,
                               'input_mask': input_mask
                               })
            return output

    # @property
    # @functools.lru_cache()
    # def keys(self):
    #     return self.reader.list_keys()


def truncate_and_pad(name_ids: List[int], max_len: int, need_tensor=True):
    """
    need_tenors: 是否需要返回的是tensor。在正常pytorch dataloader的使用场景，需要是tensor；在dali的场景，需要是list of nd.array
    """
    # max_len = max([len(n) for n in name_ids])
    input_ids = []
    input_mask = []
    for name in name_ids:
        if need_tensor:
            input_ids.append(name[:max_len] + [0] * (max_len - len(name)))
        else:
            input_ids.append(name[:max_len] + [0] * (max_len - len(name)))
            input_mask.append([1] * len(name[:max_len]) + [0] * (max_len - len(name)))
    if need_tensor:
        input_ids = torch.Tensor(input_ids).long()
        input_mask = (input_ids != 0).int()
    return input_ids, input_mask


def load_labelmapping():
    with hopen('hdfs://haruna/home/byte_search_nlp_lq/user/huangwenguan/imagenet_label/label_mapping.json') as f:
        tag2info = {}
        for l in f:
            jl = json.loads(l)
            tag2info[jl.pop('tag')] = jl
    return tag2info
