# -*- coding: utf-8 -*-

"""
文本预处理的matx text pipe
"""
from typing import List, Dict
import numpy
import os

import torch
import matx
import matx_text

from fex.matx.text_ops import BertTokenizer, BertInputsBuilder, \
    MultiDomainConcatBuilder, TaskManager
from fex.utils.hdfs_io import HADOOP_BIN


def get_local_vocab_file(file_path: str):
    if file_path.startswith("hdfs"):
        file_name = os.path.split(file_path)[-1]
        local_file_path = os.path.join(os.path.abspath(
            os.path.dirname(__file__)), file_name)
        if not os.path.exists(local_file_path):
            os.system("{} dfs -get {} {}".format(HADOOP_BIN,
                                                 file_path, local_file_path))
        return local_file_path
    return file_path


class MatxTextPipe:
    """
    matx 文本处理 pipeline。
    功能：将一个batch的原始数据，转换成tensor。

    有两个mode，一个是training mode，一个是trace mode。
    training mode下，input是raw string，output是torch.tensor
    trace mode 下，input是raw string，output是 matx ndarray
    根据 `is_trace` 参数来控制哪个mode
    """

    def __init__(self,
                 vocab_file: str,
                 max_seq_len: int,
                 add_title: bool = False,
                 do_lower_case: bool = False,
                 tokenize_emoji: bool = False,
                 greedy_sharp: bool = True,
                 is_trace: bool = False,
                 thread_num: bool = 4
                 ):
        """
        Matx 版文本 Pipeline
        Args:
            vocab_file: 词表
            max_seq_len: 最大长度
            image_token_num: 图片 token的数量，主要是用于处理 image embedding 的padding逻辑
            image_feature_dim: 图片 embedding 的dimension，作用同上
            do_lower_case: tokenzier 的配置，是否全小写
            tokenize_emoji: tokenzier 的配置，是否给 emoji 两边都加空格，作为一个单独的词看待。
                一般在其他语种生效，中文分词libcut默认或将emoji分开。外语如英语没有分词的过程，emoji容易造成UNK。
                如 "happy😺"，需要词表里包含 "##😺"，否则会UNK。这个开关在中文下基本无用，主要是英语用
            greedy_sharp: tokenizer 的配置，是否贪心的匹配 无##的词。
                如果greedy_sharp为true，即使用无##模式，如果greedy_sharp为false，即使用有##模式。等价于 not `do_wordpiece`。
                默认为true。
                在bpe的时候，对被切开的词的后半部分，必须得是 "##x" 的形式，这样会增加UNK的比例。
                默认google的tokenizer是greedy_sharp=False 的形式。
                如果greedy_sharp 是true，则会先看 "##x" 是在词表里，如果不在，会看 "x" 是否在词表里。
            is_trace: 是否在trace的模式下，matx有一些warp的逻辑
        """
        vocab_file = get_local_vocab_file(vocab_file)
        self.is_trace = is_trace
        self.add_title = add_title
        self.max_seq_len = max_seq_len
        self.max_seq_len = max_seq_len
        self.thread_num = thread_num

        # text tokenizer
        do_wordpiece = not greedy_sharp
        word_piece_tokenizer = matx_text.WordPieceTokenizerOp(location=vocab_file,
                                                              do_wordpiece=do_wordpiece,
                                                              do_lower_case=do_lower_case)

        # 如果是 trace 模式下，初始化时需要 `matx.script` 一下
        if self.is_trace:
            self.task_manager = matx.script(TaskManager)(pool_size=self.thread_num,
                                                         use_lockfree_pool=True)
            self.matx_bert_tokenizer = matx.script(BertTokenizer)(tokenizer=word_piece_tokenizer,
                                                                  do_lower_case=do_lower_case,
                                                                  tokenize_emoji=tokenize_emoji,
                                                                  task_manager=self.task_manager)
            # domain 拼接 op
            self.multi_domain_concat_builder = matx.script(
                MultiDomainConcatBuilder)(max_seq_len=max_seq_len)

            # 将 batch_inputs_tokens转为input_ids_tensor, segment_ids_tensor和mask_ids_tensor
            self.build_input_builder = matx.script(BertInputsBuilder)(
                max_seq_len=max_seq_len, vocab_file=vocab_file)
        else:
            self.matx_bert_tokenizer = BertTokenizer(tokenizer=word_piece_tokenizer,
                                                     do_lower_case=do_lower_case,
                                                     tokenize_emoji=tokenize_emoji)
            self.multi_domain_concat_builder = MultiDomainConcatBuilder(
                max_seq_len=max_seq_len)
            self.build_input_builder = BertInputsBuilder(
                max_seq_len=max_seq_len, vocab_file=vocab_file)

    def __call__(self, *args, **kwargs):
        if self.is_trace:
            return self.trace_process(*args, **kwargs)
        else:
            return self.train_process(*args, **kwargs)

    def train_process(self,
                      queries: List[str],
                      titles: List[str] = []):
        """
        训练的处理过程
        这里写死了必须有这么些个fields
        """

        batch_output_tensor: Dict[str, torch.Tensor] = {}
        # process query
        query_tokens = self.matx_bert_tokenizer(queries)
        query_input_tokens, query_segment_ids = self.multi_domain_concat_builder(
            [query_tokens], [0], [16])
        query_input_tensor, query_segment_tensor, query_mask_tensor = self.build_input_builder(
            query_input_tokens, query_segment_ids)
        batch_output_tensor["query_input_ids"] = torch.tensor(
            query_input_tensor.asnumpy())
        batch_output_tensor["query_segment_ids"] = torch.tensor(
            query_segment_tensor.asnumpy())
        batch_output_tensor["query_input_mask"] = torch.tensor(
            query_mask_tensor.asnumpy())

        if self.add_title and len(titles) > 0:
            titles_tokens = self.matx_bert_tokenizer(titles)
            titles_input_tokens, titles_segment_ids = self.multi_domain_concat_builder(
                [titles_tokens], [0], [self.max_seq_len])
            titles_input_tensor, titles_segment_tensor, titles_mask_tensor = self.build_input_builder(
                titles_input_tokens, titles_segment_ids)
            batch_output_tensor["title_input_ids"] = torch.tensor(
                titles_input_tensor.asnumpy())
            batch_output_tensor["title_segment_ids"] = torch.tensor(
                titles_segment_tensor.asnumpy())
            batch_output_tensor["title_input_mask"] = torch.tensor(
                titles_mask_tensor.asnumpy())

        return batch_output_tensor

    def trace_process(self,
                      queries: List[str],
                      titles: List[str]):
        """
        Trace的整个过程
        """

        # tokenizer
        query_tokens = self.matx_bert_tokenizer(queries)
        titles_tokens = self.matx_bert_tokenizer(titles)

        # multi domain concat
        query_input_tokens, query_segment_ids = self.multi_domain_concat_builder(
            [query_tokens], [0], [self.max_seq_len])
        query_input_tensor, query_segment_tensor, query_mask_tensor = self.build_input_builder(
            query_input_tokens, query_segment_ids)

        titles_input_tokens, titles_segment_ids = self.multi_domain_concat_builder(
            [titles_tokens], [0], [self.max_seq_len])
        titles_input_tensor, titles_segment_tensor, titles_mask_tensor = self.build_input_builder(
            titles_input_tokens, titles_segment_ids)

        return query_input_tensor, query_segment_tensor, query_mask_tensor, titles_input_tensor, titles_segment_tensor, titles_mask_tensor
