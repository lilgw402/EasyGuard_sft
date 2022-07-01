# -*- coding: utf-8 -*-
'''
Created on May-27-21 16:57
test_tokenizer_diff.py
'''
import json
import os
import unittest
import matx
import matx_text

from fex.utils.hdfs_io import hopen
from fex.data.tokenization import BertTokenizer
from fex.matx.text_ops.tokenization import BertTokenizer as MatxBertTokenizer
from fex.matx.text_ops import TaskManager


class TestMatxTokenizer(unittest.TestCase):
    """test matx_bert_tokenizer """

    def setUp(self):
        test_path = os.path.dirname(
            os.path.abspath(os.path.expanduser(__file__)))
        self.data_path = "hdfs://haruna/home/byte_search_nlp_lq/user/suiguobin/multi_model/datasets/train_video_56w_20210112_loc_emb/data_98"
        self.vocab_file = os.path.join(test_path, "./fine_fix.txt")

    def test_max_tokenizer_diff(self):
        bert_tokenizer = BertTokenizer(
            vocab_file=self.vocab_file,
            do_lower_case=True,
            tokenize_emoji=True)

        task_manager = matx.script(TaskManager)(pool_size=8,
                                                use_lockfree_pool=True)
        word_piece_tokenizer = matx_text.WordPieceTokenizerOp(location=self.vocab_file,
                                                              do_wordpiece=False,
                                                              do_lower_case=True)
        matx_bert_tokenizer = matx.script(MatxBertTokenizer)(tokenizer=word_piece_tokenizer,
                                                             do_lower_case=True,
                                                             tokenize_emoji=True)
        matx_bert_tokenizer2 = matx.script(MatxBertTokenizer)(tokenizer=word_piece_tokenizer,
                                                              do_lower_case=True,
                                                              tokenize_emoji=True,
                                                              task_manager=task_manager)

        def gen_data():
            with hopen(self.data_path, 'r') as fr:
                for line in fr:
                    data = json.loads(line)
                    title = data['pos_title']
                    yield title

        process_num = 0
        for title in gen_data():
            if process_num >= 500:
                break

            tokenizer_ret = bert_tokenizer.tokenize(title)
            matx_tokenizer_ret = matx_bert_tokenizer([title])[0]
            matx_tokenizer_ret2 = matx_bert_tokenizer2([title])[0]

            assert tokenizer_ret == matx_tokenizer_ret == matx_tokenizer_ret2


if __name__ == "__main__":
    unittest.main()
