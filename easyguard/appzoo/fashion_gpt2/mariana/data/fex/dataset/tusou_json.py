# -*- coding: utf-8 -*-
'''
Created on Nov-25-20 15:18
imagenet_dist_dataset.py
@author: liuzhen.nlp
Description:
'''

import io
from PIL import Image
import json
import traceback
import logging
import numpy as np
import torch
from base64 import b64decode

from cruise.utilities.hdfs_io import hopen

from ..dist_dataset import DistLineReadingDataset
from ..tokenization import BertTokenizer


class TusouJsonDataset(DistLineReadingDataset):
    """
    图搜训练数据集
    """

    def __init__(self, vocab_file, max_len, data_path, rank=0, world_size=1, shuffle=True, repeat=False,
                 transform=None, data_size=-1, need_temb=False, is_bert_style=True, is_clip_style=False):
        super().__init__(data_path, rank, world_size, shuffle, repeat, data_size=data_size)
        self.world_size = world_size
        self.transform = transform
        self.max_len = max_len
        self.need_temb = need_temb
        self.is_bert_style = is_bert_style
        self.is_clip_style = is_clip_style
        self.data_size = data_size
        self.tokenizer = BertTokenizer(vocab_file, do_lower_case=False)
        self.PAD = self.tokenizer.vocab['[PAD]']
        self.emp = 0
        self.tot = 1
        # 如果是clip官方的形式，在最后加一个 [CLIP] 的token，表示结束
        if self.is_clip_style:
            self.tokenizer.vocab['[CLIP]'] = len(self.tokenizer.vocab)

        self.tag2chinesename = load_tag2chinesename()

    def __iter__(self):
        for example in self.generate():
            try:
                data_item = json.loads(example)

                label_idx = 0

                if 'text' in data_item:
                    text = data_item['text']
                else:
                    if 'tag' in data_item:
                        tag = data_item['tag']
                    else:
                        tag = data_item["imagenet_info"]["synset"]
                    # tag = random.choice(list(self.tag2chinesename.keys()))
                    text = self.tag2chinesename[tag]

                # 根据fields 来取数据的版本，先不用
                # text = ''
                # try:
                #     for field in self.fields:
                #         cur_field = data_item[field]
                #         if cur_field:
                #             text += cur_field + ' '
                #     text = text.strip()
                # except Exception as e:
                #     # print(e, data_item.keys())
                #     text = data_item['text']
                # text = text.replace(';', '')
                # text = text.strip()

                tokens = self.tokenizer.tokenize(text)
                self.tot += 1
                if len(tokens) == 0:
                    self.emp += 1
                    if self.emp % 1000000 == 0:
                        logging.error('empty input: %s, will skip, %s/%s=%s' % (text, self.emp, self.tot, round(self.emp/self.tot, 4)))
                    continue
                if self.is_bert_style:
                    tokens = ['[CLS]'] + tokens[:self.max_len-2] + ['[SEP]']
                elif self.is_clip_style:
                    tokens = tokens[:self.max_len-1] + ['[CLIP]']
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

                # image = np.array(np.frombuffer(b64decode(data_item["cover_image_content"]), dtype=np.uint8))
                image_str = b64decode(data_item.get("b64_resized_binary", data_item.get("b64_binary", "")))
                if self.transform:
                    image = Image.open(io.BytesIO(image_str)).convert("RGB")
                    image = self.transform(image)
                else:
                    image = np.array(np.frombuffer(image_str, dtype=np.uint8))

                res = {'image': image, 'label': label_idx,
                       'input_ids': token_ids}

                if self.need_temb and 't_emb' in data_item:
                    res['t_emb'] = data_item['t_emb']

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
            labels.append(ibatch["label"])
            input_ids.append(ibatch['input_ids'][:max_len] + [self.PAD] * (max_len - len(ibatch['input_ids'])))
            input_mask.append([1] * len(ibatch['input_ids'][:max_len]) + [0] * (max_len - len(ibatch['input_ids'])))
            input_segment_ids.append([0] * len(ibatch['input_ids'][:max_len]) + [0] * (max_len - len(ibatch['input_ids'])))
            if self.need_temb:
                t_emb.append(ibatch["t_emb"])

        if self.transform:
            images = torch.stack(images, dim=0)
        labels = torch.tensor(labels)
        input_ids = torch.tensor(input_ids)
        input_mask = torch.tensor(input_mask)
        input_segment_ids = torch.tensor(input_segment_ids)
        res = {"image": images, "label": labels,
               "input_ids": input_ids, "input_mask": input_mask,
               "input_segment_ids": input_segment_ids}
        if self.need_temb:
            res["t_emb"] = t_emb
        return res


def load_label2idx():
    with hopen('hdfs://haruna/home/byte_search_nlp_lq/user/huangwenguan/imagenet_label/idx2label.json') as f:
        idx2label = json.loads(f.read())
        label2idx = {l: int(i) for i, l in idx2label.items()}
    return label2idx, idx2label


def load_tag2label():
    tagid2label = {}
    with hopen("/home/byte_search_nlp_lq/user/huangwenguan/imagenet_label/tag2name.txt") as f:
        for l in f:
            tag, name = l.decode().strip().split(' ', 1)
            tagid2label[tag] = name.strip()
    return tagid2label

def load_tag2chinesename():
    with hopen('hdfs://haruna/home/byte_search_nlp_lq/user/huangwenguan/imagenet_label/tag2chinesename.json') as f:
        tag2chinesename = {}
        for l in f:
            jl = json.loads(l)
            tag2chinesename[jl['tag']] = jl['cut_name']
    return tag2chinesename
