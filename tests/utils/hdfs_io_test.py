#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Huang Wenguan (huangwenguan@bytedance.com)
Date: 2020-11-09 21:28:19
LastEditTime: 2020-11-11 17:58:22
LastEditors: Huang Wenguan
Description: test of hdfs io
'''

import os
import unittest
from unittest import TestCase

from fex.utils import hdfs_io


class TestHdfsIO(TestCase):
    """ test of hdfs io api """

    def setUp(self):
        self.hdfs_root = 'hdfs://haruna/user/lijiahao.plus/fex/test_data'
        if hdfs_io.hexists(self.hdfs_root):
            hdfs_io.hrm(self.hdfs_root)
        hdfs_io.hmkdir(self.hdfs_root)

    @unittest.skip("need GDPR")
    def test_hlist_files(self):
        # create file a
        a_path = os.path.join(self.hdfs_root, 'a')
        with hdfs_io.hopen(a_path, 'wb'):
            pass
        self.assertEqual(
            sorted(hdfs_io.hlist_files([self.hdfs_root])),
            [a_path]
        )
        self.assertEqual(
            sorted(hdfs_io.hlist_files([a_path])),
            [a_path]
        )
        # create file b
        b_path = os.path.join(self.hdfs_root, 'b')
        with hdfs_io.hopen(b_path, 'wb'):
            pass
        self.assertEqual(
            sorted(hdfs_io.hlist_files([self.hdfs_root])),
            [a_path, b_path]
        )
        # create dir c
        c_path = os.path.join(self.hdfs_root, 'c')
        hdfs_io.hmkdir(c_path)
        self.assertEqual(
            sorted(hdfs_io.hlist_files([self.hdfs_root])),
            [a_path, b_path, c_path]
        )
        # remove file b
        hdfs_io.hrm(b_path)
        self.assertEqual(
            sorted(hdfs_io.hlist_files([self.hdfs_root])),
            [a_path, c_path]
        )
        # remove dir c
        hdfs_io.hrm(c_path)
        self.assertEqual(
            sorted(hdfs_io.hlist_files([self.hdfs_root])),
            [a_path]
        )
        # remove file a
        hdfs_io.hrm(a_path)
        self.assertEqual(
            sorted(hdfs_io.hlist_files([self.hdfs_root])),
            []
        )

    @unittest.skip("need GDPR")
    def test_hexists_hisdir_hisfile(self):
        # create file a
        a_path = os.path.join(self.hdfs_root, 'a')
        with hdfs_io.hopen(a_path, 'wb'):
            pass
        self.assertTrue(hdfs_io.hexists(a_path))
        self.assertFalse(hdfs_io.hisdir(a_path))
        # self.assertTrue(hdfs_io.hisfile(a_path))
        # create dir b
        b_path = os.path.join(self.hdfs_root, 'b')
        hdfs_io.hmkdir(b_path)
        self.assertTrue(hdfs_io.hexists(b_path))
        # self.assertTrue(hdfs_io.hisdir(b_path))   # FIXME: hisdir(b_path) should be true
        # self.assertFalse(hdfs_io.hisfile(b_path))
        # remove file a
        hdfs_io.hrm(a_path)
        self.assertFalse(hdfs_io.hexists(a_path))
        self.assertFalse(hdfs_io.hisdir(a_path))
        # self.assertFalse(hdfs_io.hisfile(a_path))
        # remove dir b
        hdfs_io.hrm(b_path)
        self.assertFalse(hdfs_io.hexists(b_path))
        self.assertFalse(hdfs_io.hisdir(b_path))
        # self.assertFalse(hdfs_io.hisfile(b_path))

    @unittest.skip("need GDPR")
    def test_hrm(self):
        hdfs_dir = os.path.join(self.hdfs_root, 'hrm')
        hdfs_io.hmkdir(hdfs_dir)
        # create hrm/abc
        with hdfs_io.hopen(os.path.join(hdfs_dir, 'abc'), 'wb') as f:
            f.write(b'hello world')
        # create hrm/sub/
        sub_dir = os.path.join(hdfs_dir, 'sub')
        hdfs_io.hmkdir(sub_dir)
        # create hrm/sub/xyz
        with hdfs_io.hopen(os.path.join(sub_dir, 'xyz'), 'wb') as f:
            f.write(b'bytedance aml')

        self.assertEqual(hdfs_io.hlist_files([self.hdfs_root]), [hdfs_dir])
        # remove hrm/
        hdfs_io.hrm(hdfs_dir)
        self.assertEqual(hdfs_io.hlist_files([self.hdfs_root]), [])


if __name__ == "__main__":
    unittest.main()
