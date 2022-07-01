#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: Huang Wenguan (huangwenguan@bytedance.com)
@Date: 2020-05-13 17:23:55
LastEditTime: 2020-09-10 21:27:46
LastEditors: Huang Wenguan
@Description: 预处理的pipe
'''

import torch
import io
import base64
import logging
import numpy as np

from fex.data import BertTokenizer


__all__ = ['MultiFieldPipe']


class MultiDomainPipe(object):
    def __init__(self, config, is_training=False, *args, **kwargs):
        self.config = config
        do_lower_case = config.NETWORK.DO_LOWER_CASE if hasattr(config.NETWORK, 'DO_LOWER_CASE') else True
        tokenize_emoji = config.NETWORK.TOKENIZE_EMOJI if hasattr(config.NETWORK, 'TOKENIZE_EMOJI') else True
        greedy_sharp = config.NETWORK.GREEDY_SHARP if hasattr(config.NETWORK, 'GREEDY_SHARP') else True
        self.tokenizer = BertTokenizer(config.NETWORK.VOCAB_FILE,
                                       do_lower_case=do_lower_case,
                                       tokenize_emoji=tokenize_emoji,
                                       greedy_sharp=greedy_sharp)
        print('tokenize config: do_lower_case: %s; tokenize_emoji: %s; greedy_sharp: %s' % (do_lower_case, tokenize_emoji, greedy_sharp))

        self.is_training = is_training
        self.fields = self.config.DATASET.FIELDS  # fields我们期待他是按重要度排序的，因为truncate的逻辑是从后往前
        self.seq_len = self.config.DATASET.SEQ_LEN
        self.debug = kwargs.pop('debug', False)
        self.PAD = self.tokenizer.vocab['[PAD]']
        self.training = is_training
        self.query_visual_only = config.DATASET.QUERY_VISUAL_ONLY  # 是在只学qv，不用title等
        self.use_tower = False # config.DATASET.USE_TOWER  # query 不拼接在doc端
        self.need_query = config.DATASET.NEED_QUERY

    def __call__(self, *args, **example):
        """
        对文本做预处理的流程
        """

        text_input = {}

        # 1. 先对query做处理
        query = example['query']
        query_tkd = self.tokenizer.tokenize(query)
        query_tkd = query_tkd[:int(self.seq_len / 2)]  # 很硬

        if self.need_query:
            query_input_tokens = ['[CLS]'] + query_tkd + ['[SEP]']
            text_input['query_input_ids'] = self.tokenizer.convert_tokens_to_ids(query_input_tokens)
            text_input['query_segment_ids'] = [0] * len(text_input['query_input_ids'])

        # 2. 对doc做处理
        if self.training:
            for pole in ['pos', 'neg']:
                domains = []
                if not self.query_visual_only:
                    for field_name_c in self.fields:
                        domain_tkd = []
                        for field_name in field_name_c.split('+'):
                            cur_domain = example['%s_%s' % (pole, field_name)].strip()
                            cur_domain_tkd = self.tokenizer.tokenize(cur_domain)
                            domain_tkd.extend(cur_domain_tkd)
                        domains.append(domain_tkd)
                if self.use_tower:
                    tokens, segment_ids = self.truncate_and_assemble([], domains, self.seq_len)
                else:
                    tokens, segment_ids = self.truncate_and_assemble(query_tkd, domains, self.seq_len)
                text_input['%s_input_ids' % pole] = self.tokenizer.convert_tokens_to_ids(tokens)
                text_input['%s_segment_ids' % pole] = segment_ids
        else:
            domains = []
            if not self.query_visual_only:
                for field_name_c in self.fields:
                    domain_tkd = []
                    for field_name in field_name_c.split('+'):
                        cur_domain = example[field_name].strip()
                        cur_domain_tkd = self.tokenizer.tokenize(cur_domain)
                        domain_tkd.extend(cur_domain_tkd)
                    domains.append(domain_tkd)
            if self.use_tower:
                tokens, segment_ids = self.truncate_and_assemble([], domains, self.seq_len)
            else:
                tokens, segment_ids = self.truncate_and_assemble(query_tkd, domains, self.seq_len)

            text_input['input_ids'] = self.tokenizer.convert_tokens_to_ids(tokens)
            text_input['segment_ids'] = segment_ids

        return text_input

    @staticmethod
    def truncate_and_assemble(query_tokens, domains, length, skip_empty=True):
        """
        对所有domain的文本，做合并，以及truncate
        截断策略是，保证越前面的域保留越多
        """
        # [CLS] query [SEP] title [SEP] username [SEP] ... [SEP]
        tokens = ['[CLS]']
        segment_ids = [0]
        if query_tokens:
            tokens += query_tokens + ['[SEP]']
            segment_ids += [0] * (len(query_tokens) + 1)
        for i, domain in enumerate(domains):
            if len(domain) == 0 and skip_empty:
                continue
            tokens.extend(domain)
            tokens.append('[SEP]')
            segment_ids.extend([i + 1] * (len(domain) + 1))  # 从1开始，domain递增，0是query
            if len(tokens) > length - 1:
                # 长度超出了就截断，然后返回
                tokens = tokens[:length - 1]
                segment_ids = segment_ids[:length - 1]
                tokens.append('[SEP]')
                segment_ids.append(i + 1)
                break

        assert len(tokens) == len(segment_ids)
        return tokens, segment_ids
