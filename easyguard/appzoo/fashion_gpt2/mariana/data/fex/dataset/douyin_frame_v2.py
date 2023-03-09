"""Douyin frame dataset v2, use with dali pipeline"""
import io
import sys
import json
import logging
import traceback

import base64
from PIL import Image

import numpy as np
import torch

from ..tokenization import BertTokenizer
from ..dist_dataset import DistLineReadingDataset


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class DouyinFrameDatasetV2(DistLineReadingDataset):  # pylint: disable=abstract-method
    """
    Douyin Frame dataset

    """

    def __init__(
            self,
            vocab_file,
            data_path,
            rank=0,
            world_size=1,
            shuffle=False,
            repeat=False,
            is_bert_style=False, need_text=True, need_ocr=True, seq_len=48,
            data_size=-1,
            *args,
            **kwargs):
        super().__init__(data_path, rank, world_size, shuffle, repeat, data_size=data_size)
        self.debug = kwargs.get('debug', False)

        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        # self.transforms = transforms.Compose([
        #     transforms.Resize(288),
        #     transforms.CenterCrop(288),
        #     transforms.ToTensor(),
        #     normalize])
        self.transform = None

        self.tokenizer = BertTokenizer(vocab_file, do_lower_case=False)
        self.PAD = self.tokenizer.vocab['[PAD]']
        self.is_bert_style = is_bert_style
        self.need_text = need_text
        self.need_ocr = need_ocr
        self.max_len = seq_len
        self.frame_len = 8  # TODO: 写死了8，后面改下

    def __iter__(self):
        for example in self.generate():
            try:
                if example:
                    example = self.__transform_(example)
                    if example:
                        yield example
            except Exception as e:  # pylint: disable=broad-except
                print(e)
                logging.error('encounter broken data: %s; %s', e, traceback.format_exc())

    def __transform_(self, example):
        example = json.loads(example)
        model_input = {}

        # 1. 图片预处理
        frames = example.pop('frames')
        images = []
        for f in frames:
            if 'binary' not in f or not f['binary']:
                continue
            image_str = base64.decodebytes(f['binary'].encode('utf-8'))
            if self.transform:
                image = Image.open(io.BytesIO(image_str)).convert("RGB")
                image = self.transforms(image)
            else:
                image = np.array(np.frombuffer(image_str, dtype=np.uint8))
            images.append(image)
        if len(images) != 8:
            eprint('frames num not equal to 8, is %s, will skip' % len(images))
            return {}
        if self.transform:
            images = torch.stack(images, dim=0)
        # 1. 文本预处理
        input_ids, segment_ids, input_mask = self._parse_text(example)

        model_input = {
            'frames': images,
            'input_ids': input_ids,
            'segment_ids': segment_ids,
            'input_mask': input_mask
        }
        if self.debug:
            model_input['doc'] = {'label_info': example['label_info'], 'gid': example['gid']}
        return model_input

    def text_to_tensor(self, title, ocr=None, username=None):
        title_tokens = self.tokenizer.tokenize(title)
        tokens = ['[CLS]'] + title_tokens + ['[SEP]']
        segment_ids = [0] * (len(title_tokens) + 2)

        if username is not None:
            username_tokens = self.tokenizer.tokenize(username)
            tokens = tokens + username_tokens + ['[SEP]']
            segment_ids = segment_ids + ([1] * (len(username_tokens) + 1))
        if ocr is not None and self.need_ocr:
            ocr_tokens = self.tokenizer.tokenize(ocr)
            tokens = tokens + ocr_tokens + ['[SEP]']
            segment_ids = segment_ids + ([2] * (len(ocr_tokens) + 1))

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = input_ids[:self.max_len]
        segment_ids = segment_ids[:self.max_len]
        input_mask = [1] * len(input_ids) + [0] * (self.max_len - len(input_ids))
        input_ids = input_ids + [self.PAD] * (self.max_len - len(input_ids))
        segment_ids = segment_ids + [0] * (self.max_len - len(segment_ids))

        return input_ids, segment_ids, input_mask

    def _parse_text(self, data_item):
        title = data_item.get('title', '')
        ocr = data_item.get('ocr_content', data_item.get('ocr', ''))
        username = data_item.get('username')

        input_ids, segment_ids, input_mask = self.text_to_tensor(title, username, ocr)
        return input_ids, segment_ids, input_mask

    def collect_fn(self, batch):
        """ 实现batching的操作，主要是一些pad的逻辑 """
        docs = []
        input_ids = []
        segment_ids = []
        input_mask = []
        images = [[] for _ in range(self.frame_len)]

        for ibatch in batch:
            input_ids.append(ibatch['input_ids'])
            segment_ids.append(ibatch['segment_ids'])
            input_mask.append(ibatch['input_mask'])
            for j, frame in enumerate(ibatch['frames']):
                images[j].append(frame)
            if self.debug:
                docs.append(ibatch['doc'])
        if self.transform:
            images_bf = []
            for fs in images:
                images_bf.extend(fs)
            images = torch.stack(images_bf, dim=0)
            bf, c, h, w = images.shape
            images = images.reshape([len(input_ids), self.frame_len, c, h, w])

        result = {
            'input_ids': torch.tensor(input_ids),
            'input_segment_ids': torch.tensor(segment_ids),
            'input_mask': torch.tensor(input_mask)
        }
        for i, image_pac in enumerate(images):
            result['image%s' % i] = image_pac

        if self.debug:
            result['docs'] = docs

        return result
