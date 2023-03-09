"""Tusou query dataset, use with dali pipeline"""
import json
import logging
import traceback

import torch

from ..tokenization import BertTokenizer
from ..dist_dataset import DistLineReadingDataset


class TusouQueryDataset(DistLineReadingDataset):  # pylint: disable=abstract-method
    """
    图片集的 query dataset，will output query

    """
    def __init__(self, vocab_file, data_path, rank=0, world_size=1, shuffle=False, repeat=False,
                 data_size=-1, seq_len=24, **kwargs):
        super().__init__(data_path, rank, world_size, shuffle, repeat, data_size=data_size)
        self.debug = kwargs.get('debug', False)
        self.seq_len = seq_len
        self.tokenizer = BertTokenizer(vocab_file,
                                       do_lower_case=False)
        self.PAD = self.tokenizer.vocab['[PAD]']

    def __iter__(self):
        for example in self.generate():
            try:
                if example:
                    example = self.transform(example)
                    if example:
                        yield example
            except Exception as e:  # pylint: disable=broad-except
                logging.error('encounter broken data: %s; %s', e,
                              traceback.format_exc())

    def transform(self, example):
        example = json.loads(example)
        model_input = {}

        # 1. 文本预处理
        text_tokens = self.tokenizer.tokenize(
            example['query'])[:self.seq_len - 2]
        text_tokens = ['[CLS]'] + text_tokens + ['[SEP]']
        text_input = self.tokenizer.convert_tokens_to_ids(text_tokens)
        model_input['input_ids'] = text_input

        if self.debug:
            model_input['doc'] = {
                'query': example['query'],
                'gid': example['doc_image_id']
            }

        return model_input

    def collect_fn(self, batch):
        """ 实现batching的操作，主要是一些pad的逻辑 """
        max_length = max([len(data['input_ids']) for data in batch])

        input_ids = []
        segment_ids = []
        input_mask = []
        docs = []

        for ibatch in batch:
            i_input_ids = ibatch['input_ids'] + [self.PAD] * (
                max_length - len(ibatch['input_ids']))
            i_input_mask = [1] * len(ibatch['input_ids']) + [0] * (
                max_length - len(ibatch['input_ids']))
            input_ids.append(i_input_ids)
            input_mask.append(i_input_mask)
            if self.debug:
                docs.append(ibatch['doc'])
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.zeros_like(input_ids)
        input_mask = torch.tensor(input_mask, dtype=torch.long)

        result = {
            'input_ids': input_ids,
            'input_segment_ids': segment_ids,
            'input_mask': input_mask,
        }

        if self.debug:
            result['docs'] = docs

        return result
