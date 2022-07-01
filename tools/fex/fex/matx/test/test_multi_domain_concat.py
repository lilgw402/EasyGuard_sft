# -*- coding: utf-8 -*-

from typing import List
import os
import json
from fex.utils.hdfs_io import hopen
from fex.matx.text_ops import MultiDomainConcatBuilder
from fex.matx.text_ops import BertTokenizer
from fex.matx.text_ops import BertInputsBuilder
import matx
import matx_text
import unittest


class TestMultiDomainConcat(unittest.TestCase):
    """test multi_domain_concat """

    def setUp(self):
        test_path = os.path.dirname(
            os.path.abspath(os.path.expanduser(__file__)))
        self.data_path = "hdfs://haruna/home/byte_search_nlp_lq/user/suiguobin/multi_model/datasets/train_video_56w_20210112_loc_emb/data_98"
        self.vocab_file = os.path.join(test_path, "./fine_fix.txt")

    def test_multi_domain_concat(self):
        word_piece_tokenizer = matx_text.WordPieceTokenizerOp(
            location=self.vocab_file)
        matx_bert_tokenizer = matx.script(BertTokenizer)(tokenizer=word_piece_tokenizer,
                                                         do_lower_case=True,
                                                         tokenize_emoji=True)
        multi_domain_concat_builder = matx.script(
            MultiDomainConcatBuilder)(max_seq_len=48)

        build_input_builder = matx.script(BertInputsBuilder)(
            max_seq_len=48, vocab_file=self.vocab_file)

        def process(titles: List[str], usernames: List[str], challenges: List[str], ocrs: List[str]):
            title_tokens = matx_bert_tokenizer(titles)
            username_tokens = matx_bert_tokenizer(usernames)
            challenge_tokens = matx_bert_tokenizer(challenges)
            ocr_tokens = matx_bert_tokenizer(ocrs)

            input_tokens, input_segment_ids = multi_domain_concat_builder(
                [title_tokens, username_tokens, challenge_tokens, ocr_tokens], [0, 1, 2, 3], [-1, -1, -1, -1])
            inputs_tensor, segment_tensor, mask_tensor = build_input_builder(
                input_tokens, input_segment_ids)
            return inputs_tensor, segment_tensor, mask_tensor

        def gen_data():
            with hopen(self.data_path, 'r') as fr:
                for line in fr:
                    data = json.loads(line)
                    title = data['pos_title']
                    username = data['pos_username']
                    challenge = data['pos_challenge']
                    ocr = data['pos_ocr']
                    yield title, username, challenge, ocr

        jit_module = matx.pipeline.Trace(process,
                                         ["hello world"],
                                         ["hello world"],
                                         ["hello world"],
                                         ["hello world"])

        for title, username, challenge, ocr in gen_data():
            input_ids_tensor, input_segment_tensor, mask_ids_tensor = jit_module.Run({"titles": [title], "usernames": [
                username], "challenges": [challenge], "ocrs": [ocr]})
            self.assertEqual(input_ids_tensor.shape(),
                             input_segment_tensor.shape())
            break


if __name__ == "__main__":
    unittest.main()
