"""Douyin query dataset"""
import json
import logging

import torch
from torch.utils.data import Dataset
from cruise.utilities.hdfs_io import hdfs_open
from cruise.utilities.distributed import DIST_ENV

from ..tokenization import BertTokenizer


def clip_pad_1d(tensor, pad_length, pad=0):
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    tensor_ret = torch.zeros((pad_length, ), dtype=tensor.dtype) + pad
    tensor_ret[:min(tensor.shape[0], pad_length)] = tensor[:min(tensor.shape[0], pad_length)]

    return tensor_ret


class DouyinQueryDataset(Dataset):  # pylint: disable=abstract-method
    """
    douyin query dataset，
    output query

    """

    def __init__(self, vocab_file, data_path, rank=0, world_size=1, seq_len=48, *args, **kwargs):
        super().__init__()
        self.data_path = data_path
        self.seq_len = seq_len
        self.debug = kwargs.get('debug', False)
        self.rank = rank
        self.world_size = world_size
        self.tokenizer = BertTokenizer(vocab_file, do_lower_case=False)
        self.tokenizer.vocab['[CLIP]'] = len(self.tokenizer.vocab)
        self.PAD = self.tokenizer.vocab['[PAD]']
        self.queries, self.q2docs = self._load_queries()
        logging.info('[Rank {}] Douyin Query: finish loaded {} queries'.format(DIST_ENV.rank, len(self.queries)))

    def __len__(self):
        return len(self.queries)

    def _load_queries(self):
        queries = []
        q2docs = {}
        gids = []
        with hdfs_open(self.data_path) as f:
            for fl in f:
                jl = json.loads(fl)
                queries.append(jl['query'])
                gids.append(list(set(jl['gids'])))

        if self.world_size != 1:
            total_data_size = len(queries)
            data_size_per_shard = total_data_size // self.world_size
            if self.rank == self.world_size - 1:
                queries = queries[self.rank * data_size_per_shard:]
                gids = gids[self.rank * data_size_per_shard:]
                logging.info('[Douyin Query load query]: current rank: %s, start: %s, end: %s' % (
                    self.rank, self.rank * data_size_per_shard, len(queries)))
            else:
                queries = queries[self.rank *
                                  data_size_per_shard:(self.rank + 1) * data_size_per_shard]
                gids = gids[self.rank * data_size_per_shard:(self.rank + 1) * data_size_per_shard]
                logging.info(
                    '[Douyin Query load query]: current rank: %s, start: %s, end: %s' % (
                        self.rank, self.rank * data_size_per_shard, (self.rank + 1) * data_size_per_shard))

        q2docs = {q: gs for q, gs in zip(queries, gids)}
        return queries, q2docs

    def __getitem__(self, index):
        input_ids = self.text_preprocess(self.queries[index])
        model_input = {'input_ids': input_ids}
        model_input['doc'] = {'query': self.queries[index],
                              'gids': self.q2docs[self.queries[index]]}
        return model_input

    def text_preprocess(self, text):
        cur_text_tokens = self.tokenizer.tokenize(text)[:self.seq_len - 2]
        cur_text_tokens = ['[CLS]'] + cur_text_tokens + ['[SEP]']
        cur_text_input = self.tokenizer.convert_tokens_to_ids(cur_text_tokens)
        return cur_text_input

    def collect_fn(self, batch):
        """ 实现batching的操作，主要是一些pad的逻辑 """
        max_length = max([len(data['input_ids']) for data in batch])

        input_ids = []
        segment_ids = []
        input_mask = []
        docs = []

        for ibatch in batch:
            i_input_ids = clip_pad_1d(ibatch['input_ids'], max_length, pad=self.PAD).to(torch.int64)
            input_ids.append(i_input_ids)
            if 'segment_ids' in ibatch:
                segment_ids.append(
                    clip_pad_1d(
                        ibatch['segment_ids'],
                        max_length,
                        pad=0).to(
                        torch.int64))
            else:
                segment_ids.append(torch.zeros_like(i_input_ids))
            if 'input_mask' in ibatch:
                input_mask.append(
                    clip_pad_1d(
                        ibatch['input_mask'],
                        max_length,
                        pad=0).to(
                        torch.int64))
            else:
                input_mask.append(i_input_ids != self.PAD)

            if self.debug:
                docs.append(ibatch['doc'])

        input_ids = torch.stack(input_ids, dim=0)
        segment_ids = torch.stack(segment_ids, dim=0)
        input_mask = torch.stack(input_mask, dim=0).to(torch.int64)

        result = {
            'input_ids': input_ids,
            'input_segment_ids': segment_ids,
            'input_mask': input_mask,
        }

        if self.debug:
            result['docs'] = docs

        return result
