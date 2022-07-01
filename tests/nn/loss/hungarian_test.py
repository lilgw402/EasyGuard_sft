#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
test hungarian
'''

import unittest
from unittest import TestCase
import torch

from fex.nn.loss.hungarian import HungarianMatcherCE, HungarianMatcherCOS, HungarianMatcherCOSBatch
from fex.nn.loss.set_criterion import SetCriterionCE, SetCriterionCOS


class TestHungarianMatcher(TestCase):
    """ test hungarian
    """

    def test_hungarian_ce(self):
        """
        fake_logits = tensor([[[ 1.9269,  1.4873,  0.9007, -2.1055],
                            [ 0.6784, -1.2345, -0.0431, -1.6047],
                            [ 0.3559, -0.6866, -0.4934,  0.2415]],

                            [[-1.1109,  0.0915, -2.3169, -0.2168],
                            [-0.3097, -0.3957,  0.8034, -0.6216],
                            [-0.5920, -0.0631, -0.8286,  0.3309]]])
        target = [
            [0, 2, 1],
            [2, 0]
        ]
        解是：
        (tensor([0, 1, 2]), tensor([2, 0, 1])) (tensor([1, 2]), tensor([0, 1]))
        batch 0 的第0个头预测对应第2个target（即1），获得cost 1.4873，
                第1个头预测对应第0个target（即0），获得cost 0.6784，
                第2个头预测对应第1个target（即2），获得cost -0.4934

        batch 1 的第0个头预测空结果。
                第1个头预测对应第0个target（即2），获得cost 0.8034，
                第2个头预测对应第1个target（即0），获得cost -0.5920

        以上cost实际得再softmax一下
        """

        torch.manual_seed(42)
        """ 3 分类，但是模型输出要给4个logits，因为要预留给no-object category"""
        matcher = HungarianMatcherCE()
        set_criterion = SetCriterionCE(num_classes=3, eos_coef=0.1)
        with torch.no_grad():
            targets = [torch.tensor([0, 2, 1]).long(), torch.tensor([2, 0]).long()]
            fake_logits = torch.randn([2, 3, 4])  # bsz, nqueries, nclasses+1

            # print(fake_logits)
            indexs = matcher(fake_logits, targets)
            head_idx = [i[0] for i in indexs]
            target_idx = [i[1] for i in indexs]
            # print(head_idx, target_idx) # (tensor([0, 1, 2]), tensor([2, 0, 1])) (tensor([1, 2]), tensor([0, 1]))
            head_idx = [x.tolist() for x in head_idx]
            target_idx = [x.tolist() for x in target_idx]
            self.assertEqual(head_idx, [[0, 1, 2], [1, 2]])
            self.assertEqual(target_idx, [[2, 0, 1], [0, 1]])

            loss = set_criterion(fake_logits, targets)
            self.assertAlmostEqual(loss.tolist(), 1.1865, places=4)
            # print(loss) # tensor(1.1865)

    def test_hungarian_cos(self):
        """
        target = [
            [v11],
            [v21, v22]
        ]
        fake_vec = [
            [_, _, v11],
            [_, v22, v21]
        ]
        解是：
        (tensor([2]), tensor([0])) (tensor([1, 2]), tensor([1, 0]))
        batch 0 的第2个头预测对应第0个target（即v11），获得cost -1，

        batch 1 的第0个头预测空结果。
                第1个头预测对应第1个target（即v22），获得cost -1，
                第2个头预测对应第0个target（即v21），获得cost -1

        以上cost实际得再softmax一下
        """
        torch.manual_seed(42)
        matcher = HungarianMatcherCOS()
        set_criterion = SetCriterionCOS(dim=128, eos_coef=0.1)
        with torch.no_grad():
            b1_t1 = torch.randn(1, 128)
            b2_t1 = torch.randn(1, 128)
            b2_t2 = torch.randn(1, 128)
            targets = [b1_t1, torch.cat([b2_t1, b2_t2])]
            fake_vec = torch.randn([2, 3, 128])  # bsz, nqueries, dim
            fake_vec[0, 2] = b1_t1
            fake_vec[1, 1] = b2_t2
            fake_vec[1, 2] = b2_t1

            indexs = matcher(fake_vec, targets)
            head_idx = [i[0] for i in indexs]
            target_idx = [i[1] for i in indexs]
            # print(head_idx, target_idx) # (tensor([2]), tensor([0])) (tensor([1, 2]), tensor([1, 0]))
            head_idx = [x.tolist() for x in head_idx]
            target_idx = [x.tolist() for x in target_idx]
            self.assertEqual(head_idx, [[2], [1, 2]])
            self.assertEqual(target_idx, [[0], [1, 0]])

            loss = set_criterion(fake_vec, targets)
            self.assertAlmostEqual(loss.tolist(), 0.0508, places=4)
            # print(loss) # tensor(0.0508)

    def test_hungarian_cos_batch(self):
        """
        """
        torch.manual_seed(42)
        matcher = HungarianMatcherCOSBatch()
        set_criterion = SetCriterionCOS(dim=128, eos_coef=0.1, matcher='HungarianMatcherCOSBatch')
        with torch.no_grad():
            targets = torch.randn(4, 16, 128)
            fake_vec = torch.randn(4, 8, 128)

            # b1_t1 = torch.randn(1, 128)
            # b2_t1 = torch.randn(1, 128)
            # b2_t2 = torch.randn(1, 128)
            # targets = [b1_t1, torch.cat([b2_t1, b2_t2])]
            # fake_vec = torch.randn([2, 3, 128]) # bsz, nqueries, dim
            # fake_vec[0, 2] = b1_t1
            # fake_vec[1, 1] = b2_t2
            # fake_vec[1, 2] = b2_t1

            indexs = matcher(fake_vec, targets)
            head_idx = [i[0] for i in indexs]
            target_idx = [i[1] for i in indexs]
            print(head_idx, 'head idx')  # (tensor([2]), tensor([0])) (tensor([1, 2]), tensor([1, 0]))
            print(target_idx, 'target_idx idx')  # (tensor([2]), tensor([0])) (tensor([1, 2]), tensor([1, 0]))
            head_idx = [x.tolist() for x in head_idx]
            target_idx = [x.tolist() for x in target_idx]
            # self.assertEqual(head_idx, [[2], [0]])
            # self.assertEqual(target_idx, [[1, 2], [1, 0]])

            loss = set_criterion(fake_vec, targets)
            # self.assertAlmostEqual(loss.tolist(), 0.0508, places=4)
            print(loss)  # tensor(0.0508)


if __name__ == '__main__':
    unittest.main()
