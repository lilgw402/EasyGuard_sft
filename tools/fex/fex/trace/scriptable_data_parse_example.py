# -*- coding: utf-8 -*-
'''
Created on Jan-13-21 16:44
scriptable_data_parse_example.py
@author: liuzhen.nlp
Description: 
'''
from typing import Dict, List, Tuple

import torch

from fex.trace.utils import load_vocab, MultiDomainConcator, ScriptBertTokenizer, \
    tail_first_truncate, clip_pad_1d, clip_pad_2d, make_mask_1d
from fex.trace.scriptable_data_parse_engine import ScriptableDataParseEngine


class DataParseEngineExample(ScriptableDataParseEngine):
    """ 这是一个 data_parse 的example
    """

    def __init__(self, vocab_file: str, field_names, seq_len: int = 64, image_token_num: int = 8, image_feature_dim: int = 128):
        super().__init__()
        self.seq_len: int = seq_len
        self.vocab_file: str = vocab_file
        self.field_names: List[str] = field_names
        self.image_token_num: int = image_token_num
        self.image_feature_dim: int = image_feature_dim
        self.vocab: Dict[str, int] = load_vocab(self.vocab_file)
        self._PAD: int = self.vocab['[PAD]']
        self.concat_fn = torch.jit.script(MultiDomainConcator(self.vocab_file))
        self.tokenizer = torch.jit.script(ScriptBertTokenizer(self.vocab_file))

    def forward(self, queries: List[str], doc_items: List[Dict[str, str]], vision_embed: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self._fwd_list_(queries, doc_items, vision_embed)

    @torch.jit.export
    def _fwd_list_(self, queries: List[str], doclist: List[Dict[str, str]], vision_embed_list: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch_size: int = len(doclist)
        query_toks_list: List[List[str]] = []
        query_toks: List[str] = []
        bpe_toks = self.tokenizer(queries[0])[:self.seq_len]
        for toks in bpe_toks:
            query_toks.extend(toks)

        query_toks_list = [query_toks] * batch_size

        batch_dict_tensor: List[Dict[str, torch.Tensor]] = []
        for i, doc in enumerate(doclist):
            vision_embed: torch.Tensor = vision_embed_list[i]
            input_ids, segment_ids = self._fwd_instance_(
                query_toks_list[i], doc, vision_embed.size(0), skip_domain=False)

            tendict: Dict[str, torch.Tensor] = {}
            tendict["input_ids"] = torch.tensor(input_ids, dtype=torch.int64)
            tendict["segment_ids"] = torch.tensor(
                segment_ids, dtype=torch.int64)
            tendict["vision_embed"] = vision_embed
            batch_dict_tensor.append(tendict)

        max_length = self.seq_len

        input_ids: List[torch.Tensor] = []
        segment_ids: List[torch.Tensor] = []
        input_mask: List[torch.Tensor] = []

        vision_embeds: List[torch.Tensor] = []
        vision_masks: List[torch.Tensor] = []

        for i, ibatch in enumerate(batch_dict_tensor):
            input_ids.append(clip_pad_1d(
                ibatch['input_ids'], max_length, pad=self._PAD).to(torch.int64))
            segment_ids.append(clip_pad_1d(
                ibatch['segment_ids'], max_length, pad=0).to(torch.int64))
            input_mask.append((input_ids[i] != self._PAD))

            # 通过visual_embedding dim来判断是否是空embedding
            if ibatch['vision_embed'].size(1) == self.image_feature_dim:
                vision_masks.append(make_mask_1d(
                    ibatch['vision_embed'].size(0), self.image_token_num))
            else:
                vision_masks.append(make_mask_1d(0, self.image_token_num))

            vision_embeds.append(clip_pad_2d(
                ibatch['vision_embed'], (self.image_token_num, self.image_feature_dim), pad=0))

        # 将query放在batch的最后一行，当成qtv数据，在visual域上需要mask掉
        query_input_ids, query_segment_ids = self._fwd_instance_(
            query_toks_list[0], {'': ''}, 0, skip_domain=True)

        query_input_ids = torch.tensor(query_input_ids, dtype=torch.int64)
        query_segment_ids = torch.tensor(query_segment_ids, dtype=torch.int64)
        query_input_ids_pad = clip_pad_1d(
            query_input_ids, max_length, pad=self._PAD).to(torch.int64)

        input_ids.append(query_input_ids_pad)
        segment_ids.append(clip_pad_1d(query_segment_ids,
                                       max_length, pad=0).to(torch.int64))
        input_mask.append((query_input_ids_pad != self._PAD))
        vision_masks.append(make_mask_1d(0, self.image_token_num))
        vision_embeds.append(torch.zeros(
            self.image_token_num, self.image_feature_dim, dtype=vision_embeds[0].dtype))

        input_ids = torch.stack(input_ids, dim=0)
        segment_ids = torch.stack(segment_ids, dim=0)
        input_mask = torch.stack(input_mask, dim=0).to(dtype=torch.int64)
        vision_embeds = torch.stack(vision_embeds, dim=0)
        vision_masks = torch.stack(vision_masks, dim=0)

        return {"input_ids": input_ids, "segment_ids": segment_ids, "input_masks": input_mask,
                "vision_embeds": vision_embeds, "vision_masks": vision_masks}

    @torch.jit.export
    def _fwd_instance_(
            self,
            query_tok: List[str],
            domain_dict: Dict[str, str],
            vision_embed_size: int = 0,
            skip_domain: bool = False) -> Tuple[List[int], List[int]]:
        domains: List[List[str]] = []
        not_empty_domain_num: int = 0
        if not skip_domain:
            for field_name in self.field_names:
                domain = domain_dict[field_name]
                d_toks: List[str] = []
                domain_tok = self.tokenizer(domain)
                if len(domain_tok) > 0:
                    not_empty_domain_num += 1
                for toks in domain_tok:
                    d_toks.extend(toks)
                domains.append(d_toks)

            domain_length = max(1, self.seq_len - len(query_tok) -
                                vision_embed_size - 2 - not_empty_domain_num)

            domains = tail_first_truncate(domains, domain_length)

        input_ids, segment_ids = self.concat_fn(query_tok, domains)
        return input_ids, segment_ids

if __name__ == "__main__":
    import numpy as np

    field_names = ['titles', 'user_nicknames', 'challenges']
    data_parse = DataParseEngineExample(vocab_file="/mnt/nlp-lq/bert/vocabs/fine_fix.txt", field_names=field_names)
    traced_data_parse = torch.jit.script(data_parse)

    batch_size = 8

    queries = ["hello world"] * batch_size

    doc_items: List[Dict[str, str]] = []
    doc_info: Dict[str, str] = {}

    for field in field_names:
        doc_info[field] = "hello world"

    doc_items.extend([doc_info] * batch_size)

    vision_embed = torch.as_tensor(np.random.rand(8, 128)).float()
    vision_embed_list = [vision_embed for i in range(batch_size)]

    tensor_dict = traced_data_parse(queries, doc_items, vision_embed_list)
    print('-------> tensor_dict keys: ', tensor_dict.keys())
