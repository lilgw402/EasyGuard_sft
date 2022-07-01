#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
test of checkpointer
'''

import unittest
from unittest import TestCase

import fex
from fex.utils.checkpointer import Checkpointer


class TestCheckpointer(TestCase):
    """ test of checkpointer api """

    @unittest.skip("config and ckpt not exist, it will be fix by@huangwenguan")
    def test_hlist_files(self):
        """ test restore ckpt """
        serialization_dir = "hdfs://haruna/home/byte_search_nlp_lq/multimodal/test_data/model_dir"
        checkpointer = Checkpointer(serialization_dir)
        model_path, training_state_path = checkpointer.find_latest_checkpoint()
        self.assertEqual(model_path, 'hdfs://haruna/home/byte_search_nlp_lq/multimodal/test_data/model_dir/model_state_epoch_249.th')
        self.assertEqual(training_state_path, 'hdfs://haruna/home/byte_search_nlp_lq/multimodal/test_data/model_dir/middle_training_state_249')


if __name__ == "__main__":
    unittest.main()
