# -*- coding: utf-8 -*-
'''
Created on Nov-25-20 15:18
imagenet_dist_dataset.py
@author: liuzhen.nlp
Description:
'''
import io
import random
from PIL import Image
import json
import traceback
import logging
import numpy as np
import torch
from base64 import b64decode

from ..dist_dataset import DistLineReadingDataset
from ..tokenization import BertTokenizer


class DYCoverDataset(DistLineReadingDataset):
    """
    抖音封面数据集
    """

    def __init__(self, vocab_file, max_len, data_path, rank=0, world_size=1, shuffle=True, repeat=False,
                 transform=None, data_size=-1):
        super().__init__(data_path, rank, world_size, shuffle, repeat, data_size=data_size)
        self.world_size = world_size
        self.transform = transform
        self.max_len = max_len
        self.data_size = data_size
        self.tokenizer = BertTokenizer(vocab_file, do_lower_case=False)
        self.PAD = self.tokenizer.vocab['[PAD]']
        self.emp = 0
        self.tot = 1

    def __iter__(self):
        for example in self.generate():
            try:
                data_item = json.loads(example)
                # image = np.array(np.frombuffer(b64decode(data_item["cover_image_content"]), dtype=np.uint8))
                image_binary = data_item.get("custom_image", data_item.get("spf_image", data_item.get("cover_image")))
                image_str = b64decode(image_binary)
                if self.transform:
                    image = Image.open(io.BytesIO(image_str)).convert("RGB")
                    image = self.transform(image)
                else:
                    image = np.array(np.frombuffer(image_str, dtype=np.uint8))

                label_idx = 0

                query = data_item['query']
                tokens = self.tokenizer.tokenize(query)
                self.tot += 1
                if len(tokens) == 0:
                    self.emp += 1
                    if random.random() < 0.001:
                        logging.error('empty input: %s, will skip, %s/%s=%s' % (query, self.emp, self.tot, round(self.emp/self.tot, 4)))
                    continue
                tokens = ['[CLS]'] + tokens[:self.max_len-2] + ['[SEP]']
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

                res = {'image': image,
                       'input_ids': token_ids}

                yield res

            except Exception as e:
                logging.error(traceback.format_exc())
                logging.error('encounter broken data: %s' % e)
                logging.error(data_item.keys())

    def collect_fn(self, data):
        images = []
        labels = []
        input_ids = []
        input_mask = []
        input_segment_ids = []
        t_emb = []
        # max_len = max([len(b['input_ids']) for b in data])
        max_len = self.max_len

        for i, ibatch in enumerate(data):
            images.append(ibatch["image"])
            labels.append(ibatch.get("label", 0))
            input_ids.append(ibatch['input_ids'][:max_len]+ [self.PAD] * (max_len - len(ibatch['input_ids'])))
            input_mask.append([1] * len(ibatch['input_ids'][:max_len]) + [0] * (max_len - len(ibatch['input_ids'])))
            input_segment_ids.append([0] * len(ibatch['input_ids'][:max_len]) + [0] * (max_len - len(ibatch['input_ids'])))

        if self.transform:
            images = torch.stack(images, dim=0)

        labels = torch.tensor(labels)
        input_ids = torch.tensor(input_ids)
        input_mask = torch.tensor(input_mask)
        input_segment_ids = torch.tensor(input_segment_ids)
        res = {"image": images, "label": labels,
               "input_ids": input_ids, "input_mask": input_mask,
               "input_segment_ids": input_segment_ids}
        return res
