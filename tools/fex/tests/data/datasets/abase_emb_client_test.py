# -*- coding: utf-8 -*-
'''
Created on Dec-08-20 17:44
abase_emb_dataset.py
@author: liuzhen.nlp
Description: embedding dataset for train with abase
'''

import unittest
from unittest import TestCase
import torch

from fex.data import AbaseEmbClient


class TestAbaseClient(TestCase):

    @unittest.skip(reason='doas not support in CI')
    def test_keys(self):

        gids = [6801348301871467779, 6896024937492958478]

        abase_client = AbaseEmbClient()
        gid2emb = abase_client.get_embeddings(gids)
        for k, v in gid2emb.items():
            self.assertEqual(list(v.shape), [8, 128])


if __name__ == '__main__':
    unittest.main()
