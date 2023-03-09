"""Douyin frame dataset"""
import io
import json
import logging

import base64
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from cruise.utilities.hdfs_io import hdfs_open

from ..tokenization import BertTokenizer


class DouyinFrameDataset(Dataset):  # pylint: disable=abstract-method
    """
    Douyin Frame dataset

    """

    def __init__(self, vocab_file, data_path, rank=0, world_size=1,
                 is_bert_style=False, need_text=True, need_ocr=True, seq_len=48, **kwargs):
        super().__init__()
        self.debug = kwargs.get('debug', False)
        self.rank = rank
        self.world_size = world_size
        self._load_all_set(data_path)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transforms = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])

        self.tokenizer = BertTokenizer(vocab_file, do_lower_case=False)
        self.tokenizer.vocab['[CLIP]'] = len(self.tokenizer.vocab)
        self.PAD = self.tokenizer.vocab['[PAD]']
        self.is_bert_style = is_bert_style
        self.need_text = need_text
        self.need_ocr = need_ocr
        self.max_len = seq_len

    def _load_all_set(self, data_path):
        logging.info('[Rank {}/{}]loading images from {} ...'.format(
            self.rank, self.world_size, data_path))
        videos = []
        with hdfs_open(data_path) as f:
            for fl in f:
                jl = json.loads(fl)
                videos.append(jl)
        if self.rank == 0 and self.world_size == 1:  # 单机单卡的情况
            self.videos = videos
        else:
            # 否则分shard
            total_data_size = len(videos)
            data_size_per_shard = total_data_size // self.world_size
            if self.rank == self.world_size - 1:
                self.videos = videos[self.rank * data_size_per_shard:]
                logging.info('[Douyin Frame load all set]: current rank: %s, start: %s, end: %s' % (
                    self.rank, self.rank * data_size_per_shard, len(videos)))
            else:
                self.videos = videos[self.rank *
                                     data_size_per_shard:(self.rank + 1) * data_size_per_shard]
                logging.info(
                    '[Douyin Frame load all set]: current rank: %s, start: %s, end: %s' % (
                        self.rank, self.rank * data_size_per_shard, (self.rank + 1) * data_size_per_shard))

    def __len__(self):
        return len(self.videos)

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

    def __getitem__(self, index):
        data_item = self.videos[index]
        model_input = {}
        # 2. 图片预处理
        frames = data_item.pop('frames')
        input_ids, segment_ids, input_mask = self._parse_text(data_item)
        images = []
        for f in frames:
            if 'b64_binary' not in f:
                continue
            image_str = base64.b64decode(f['b64_binary'])
            image = Image.open(io.BytesIO(image_str)).convert("RGB")
            image = self.transforms(image)
            images.append(image)
        images = torch.stack(images, dim=0)
        model_input['image'] = images
        model_input['input_ids'] = input_ids
        model_input['segment_ids'] = segment_ids
        model_input['input_mask'] = input_mask
        model_input['doc'] = {'gid': data_item['gid'],
                              'query': data_item['query']}
        return model_input

    def collect_fn(self, batch):
        """ 实现batching的操作，主要是一些pad的逻辑 """
        image = []
        # image_mask = []
        docs = []
        input_ids = []
        segment_ids = []
        input_mask = []

        for ibatch in batch:
            input_ids.append(ibatch['input_ids'])
            segment_ids.append(ibatch['segment_ids'])
            input_mask.append(ibatch['input_mask'])
            image.append(ibatch['image'])
            if self.debug:
                docs.append(ibatch['doc'])
        image = torch.stack(image, dim=0)
        input_ids = torch.tensor(input_ids)
        segment_ids = torch.tensor(segment_ids)
        input_mask = torch.tensor(input_mask)
        result = {
            'image': image,
            'input_ids': input_ids,
            'input_segment_ids': segment_ids,
            'input_mask': input_mask
        }

        if self.debug:
            result['docs'] = docs

        return result
