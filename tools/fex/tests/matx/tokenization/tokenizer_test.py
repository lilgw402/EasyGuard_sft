#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
test matx tokenizer
'''

import os
import json
import unittest
from unittest import TestCase

from fex.utils.hdfs_io import hopen
from fex.matx.tokenization import MatxBertTokenizer
from fex.data import BertTokenizer


BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
TEST_DIR = os.path.join(BASE_DIR, "../ci/test_data")


class TestMatxBertTokenizer(TestCase):
    """ test MatxBertTokenizer """

    @unittest.skip("Data Path not exist")
    def test_tokenizer_with_single_data(self):
        vocab_path = 'hdfs://haruna/home/byte_search_nlp_lq/multimodal/modelhub/text/electra_12l_lr1_search_rnd_20200716/archer/zh_old_cut_145607.vocab'
        tokenizer = MatxBertTokenizer(vocab_path=vocab_path, do_cut=False)
        text = '字节跳动不跳动跟我南京市长江大桥有什么关系'
        res = tokenizer(text)
        assert res == ['字节', '##跳', '##动', '##不', '##跳', '##动', '##跟', '##我', '##南', '##京', '##市', '##长江', '##大', '##桥', '##有', '##什', '##么', '##关', '##系']

    @unittest.skip("Data Path not exist")
    def test_tokenizer_with_single_data_with_cut(self):
        vocab_path = 'hdfs://haruna/home/byte_search_nlp_lq/multimodal/modelhub/text/electra_12l_lr1_search_rnd_20200716/archer/zh_old_cut_145607.vocab'
        tokenizer = MatxBertTokenizer(vocab_path=vocab_path, do_cut=True)
        text = '字节跳动不跳动跟我南京市长江大桥有什么关系'
        res = tokenizer(text)
        assert res == ['字节', '跳动', '不', '跳动', '跟', '我', '南京', '市', '长江', '大桥', '有', '什么', '关系']

    @unittest.skip("Data Path not exist")
    def test_compare_python_tokenizer_and_matx_tokenizer(self):
        vocab_path = 'hdfs://haruna/home/byte_search_nlp_lq/multimodal/modelhub/text/electra_12l_lr1_search_rnd_20200716/archer/zh_old_cut_145607.vocab'
        #vocab_path = '/home/tiger/0827_vocab.txt'
        tokenizer = MatxBertTokenizer(vocab_path=vocab_path,
                                      do_cut=False,
                                      lower_case=True,
                                      unk_token='[UNK]',
                                      wordpiece_type='bert')
        python_tokenizer = BertTokenizer(vocab_file=vocab_path,
                                         do_lower_case=True,
                                         greedy_sharp=False)

        def cmp(text):
            matx_out = tokenizer(text)
            python_out = python_tokenizer.tokenize(text)
            if matx_out == python_out:
                return True
            else:
                print(f'origin: {text}\nmatx  : {matx_out}\npython: {python_out}\n----')
                return False

        c = 1
        t = 0
        with hopen('hdfs://haruna/home/byte_search_nlp_lq/multimodal/data/video/ctr_pair_emb/part-00126.4mz') as f:
            for l in f:
                jl = json.loads(l)
                if not cmp(jl['query']):
                    t += 1
                    print(c, t, t / c)
                c += 1
        rate = t / c
        print(rate)
        assert rate < 0.0005  # 主要是一些奇异字符的归一化，有微小的diff
        # origin: hablo espa ñ ol
        # matx  : ['hab', '##lo', 'esp', '##a', 'ñ', 'ol']
        # python: ['hab', '##lo', 'esp', '##a', 'n', 'ol']

    @unittest.skip("Data Path not exist")
    def test_matx_tokenizer_caster_version(self):
        vocab_path = 'hdfs://haruna/home/byte_search_nlp_lq/multimodal/modelhub/text/electra_12l_lr1_search_rnd_20200716/archer/zh_old_cut_145607.vocab'
        #vocab_path = '/home/tiger/0827_vocab.txt'
        tokenizer = MatxBertTokenizer(vocab_path=vocab_path,
                                      do_cut=True,
                                      lower_case=True,
                                      unk_token='[HASH]',
                                      unk_kept=True,
                                      wordpiece_type='caster-bert')

        with hopen('hdfs://haruna/home/byte_search_nlp_lq/multimodal/data/video/ctr_pair_emb/part-00126.4mz') as f:
            for l in f:
                jl = json.loads(l)
                print(jl['query'], tokenizer(jl['query']))
                break


if __name__ == '__main__':
    unittest.main()
