#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
test circle loss
'''

import os
import unittest
from unittest import TestCase
import pytest

import torch
from fex.nn.loss import CircleLoss


class TestCircleloss(TestCase):
    """ test create_visual_tokenizer """

    def test_circleloss_random(self):
        torch.manual_seed(42)
        criterion = CircleLoss(m=0.25, gamma=256)
        with torch.no_grad():
            feat = torch.nn.functional.normalize(torch.rand(256, 64, requires_grad=True))
            lbl = torch.randint(high=10, size=(256,))
            criterion = CircleLoss(m=0.25, gamma=256)
            circle_loss = criterion(feat, lbl)

            print(circle_loss)
            #self.assertEqual(list(logits.shape), [1000])


if __name__ == '__main__':
    unittest.main()
