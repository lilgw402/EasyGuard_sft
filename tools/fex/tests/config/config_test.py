# -*- coding: utf-8 -*-
'''
Created on Nov-11-20 10:50
config_test.py
@author: liuzhen.nlp
Description: config test
'''
import os
import unittest
from fex.config import reset_cfg

BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
TEST_DIR = os.path.join(BASE_DIR, "ci/test_data/test_config")
HDFS_CONFIG = "hdfs://haruna/home/byte_search_nlp_lq/user/liuzhen.nlp/test_default.yaml"
HDFS_CONFIG_EXTRA = "hdfs://haruna/home/byte_search_nlp_lq/user/liuzhen.nlp/test_default_b.yaml"


class TestConfig(unittest.TestCase):
    """test config """

    def test_get_cfg_defaults(self):
        """ 测试默认的配置 """
        cfg = reset_cfg()
        self.assertEqual(cfg.TRAINER.LOG_FREQUENT, 100)

    def test_update_local_config(self):
        """ 测试update配置 """
        cfg = reset_cfg()
        cfg.update_cfg(os.path.join(TEST_DIR, 'test_default.yaml'))
        self.assertEqual(cfg.DATASET.TRAIN_SIZE, 100)

    def test_add_new_attr(self):
        """ 测试向配置中添加新的key """
        cfg = reset_cfg()
        cfg.update_cfg(os.path.join(TEST_DIR, 'test_update.yaml'))
        self.assertEqual(cfg.DATASET.NEW_SET, 'new_test')
        self.assertEqual(cfg.NET, 'test')

    def test_not_exist_attr(self):
        """ 测试获取不存在的key，进行错误提示 """
        cfg = reset_cfg()
        cfg.update_cfg(os.path.join(TEST_DIR, 'test_default.yaml'))
        with self.assertRaises(AttributeError):
            cfg.GOOD

    @unittest.skip(reason='doas not support in CI')
    def test_update_hdfs_config(self):
        """ 测试hdfs更新config """
        cfg = reset_cfg()
        cfg.update_cfg(HDFS_CONFIG)
        self.assertEqual(cfg.DATASET.TRAIN_SIZE, 100)

    def test_freeze(self):
        """ 测试 freeze """
        cfg = reset_cfg()
        cfg.update_cfg(os.path.join(TEST_DIR, 'test_default.yaml'))
        cfg.freeze()
        with self.assertRaises(AttributeError):
            cfg.MODULE = "NET"

    def test_reset(self):
        """ 测试 reset 方法 """
        cfg = reset_cfg()
        cfg.update_cfg(os.path.join(TEST_DIR, 'test_default.yaml'))
        self.assertEqual(cfg.DATASET.DATASET, "FramePairDataSet")
        cfg = reset_cfg()
        self.assertEqual(cfg.TRAINER.TRAIN_BATCH_SIZE, None)

    def test_dump(self):
        """ 测试 dump方法 """
        cfg = reset_cfg()
        cfg.update_cfg(os.path.join(TEST_DIR, 'test_default.yaml'))
        cfg.dump(results_dir=TEST_DIR, config_name="new_test_default.yaml")
        self.assertEqual(os.path.exists(os.path.join(
            TEST_DIR, "new_test_default.yaml")), True)
        os.remove(os.path.join(TEST_DIR, "new_test_default.yaml"))

    def test_base_config(self):
        """ 测试 base config功能"""
        cfg = reset_cfg()
        cfg.update_cfg(os.path.join(TEST_DIR, 'test_default.yaml'))
        self.assertEqual(cfg.RNG_SEED, 567)
        self.assertEqual(cfg.TEST_DATA.DATASET, "TestDataSet")
        self.assertEqual(cfg.MOCK_DATA.DATASET, "MOCKDataSet")
        self.assertEqual(cfg.MOCK_DATA.TRAIN_SIZE, 300)

    def test_merge_from_list(self):
        """ 测试 merge_from_args_opts"""
        cfg = reset_cfg()
        cfg.update_cfg(os.path.join(TEST_DIR, 'test_default.yaml'), cfg_list=["DATASET.TRAIN_SIZE", "500", "TRAIN.BATCH_SIZE", "200"])
        self.assertEqual(cfg.DATASET.TRAIN_SIZE, 500)
        self.assertEqual(cfg.TRAIN.BATCH_SIZE, 200)

    @unittest.skip(reason='doas not support in CI')
    def test_merge_multi_config(self):
        """ 测试三重嵌套load """
        # b base on a, a base on basic config
        cfg = reset_cfg()
        cfg.update_cfg(HDFS_CONFIG_EXTRA)
        self.assertEqual(cfg.TRAIN.END_EPOCH, 30)

    def test_merge_from_str(self):
        """ 测试 merge_from_str"""
        cfg = reset_cfg()
        cfg.update_cfg(os.path.join(TEST_DIR, 'test_default.yaml'))
        cfg.merge_from_str("DATASET.TRAIN_SIZE=500;TRAIN.BATCH_SIZE=200")
        self.assertEqual(cfg.DATASET.TRAIN_SIZE, 500)
        self.assertEqual(cfg.TRAIN.BATCH_SIZE, 200)


if __name__ == '__main__':
    unittest.main()
