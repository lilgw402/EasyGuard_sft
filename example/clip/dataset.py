# -*- coding: utf-8 -*-
'''
image dataset json format, from hdfa
'''

import io
import random
from PIL import Image
import json
import traceback
import numpy as np
from base64 import b64decode

import torch
import torchvision.transforms as transforms

from fex.config import CfgNode
from fex import _logger as log
from fex.utils.hdfs_io import hopen
from fex.data.datasets.dist_dataset import DistLineReadingDataset
from fex.data import BertTokenizer


class ImageDataset(DistLineReadingDataset):
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
                 *args, **kwargs):
        super().__init__(data_path, rank, world_size, shuffle, repeat)

        if preprocess_mode not in ['torchvision', 'dali', 'bytedvision']:
            raise ValueError(f"preprocess_mode {preprocess_mode} not in ['torchvision', 'dali', 'bytedvision']")

        self.preprocess_mode = preprocess_mode
        if self.preprocess_mode == 'torchvision':
            self.transform = kwargs.get('transform')  # torchvision 模式下，默认有 transform

        vocab_file = config.get('network.vocab_file')
        self.max_len = config.get('dataset.max_len', 32)
        self.text_field = config.get('dataset.fields', config.get('dataset.text_field', ['text']))
        self.image_field = config.get('dataset.image_field', 'b64_resized_binary')
        self.tokenizer = BertTokenizer(vocab_file,
                                       do_lower_case=config.get('network.do_lower_case', True),
                                       tokenize_emoji=config.get('network.tokenize_emoji', False),
                                       greedy_sharp=config.get('network.greedy_sharp', False)
                                       )
        self.PAD = self.tokenizer.vocab['[PAD]']

    def __iter__(self):
        for example in self.generate():
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
                       'input_ids': token_ids}

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

        for i, ibatch in enumerate(data):
            images.append(ibatch["image"])
            input_ids.append(ibatch['input_ids'][:max_len] + [self.PAD] * (max_len - len(ibatch['input_ids'])))
            input_mask.append([1] * len(ibatch['input_ids'][:max_len]) + [0] * (max_len - len(ibatch['input_ids'])))
            input_segment_ids.append([0] * max_len)

        if self.preprocess_mode == 'torchvision':
            images = torch.stack(images, dim=0)
        input_ids = torch.tensor(input_ids)
        input_mask = torch.tensor(input_mask)
        input_segment_ids = torch.tensor(input_segment_ids)
        res = {"image": images,
               "input_ids": input_ids,
               "input_mask": input_mask,
               "input_segment_ids": input_segment_ids,
               }
        return res


def get_transform(mode: str = "train"):
    """
    根据不同的data，返回不同的transform
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if mode == "train":
        com_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    elif mode == 'val':
        com_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])
    else:
        raise ValueError('mode [%s] is not in [train, val]' % mode)
    return com_transforms
