#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Huang Wenguan (huangwenguan@bytedance.com)
Date: 2020-11-17 20:20:23
LastEditTime: 2020-11-17 21:28:11
LastEditors: Huang Wenguan
Description:
'''

from .anchor import MILNCELoss
from .supcon import SupConLoss
from .ntxent import NTXentLoss, LearnableNTXentLoss
from .circle import CircleLoss, convert_label_to_similarity
from .iou import IOULoss
from .hungarian import HungarianMatcherCE, HungarianMatcherCOS, HungarianMatcherCOSBatch
from .set_criterion import SetCriterionCE, SetCriterionCOS
