#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Huang Wenguan (huangwenguan@bytedance.com)
Date: 2020-11-09 21:55:46
LastEditTime: 2020-11-11 19:23:21
LastEditors: Huang Wenguan
Description: init of utils
'''

from enum import Enum


class AMPType(Enum):
    """ amp 之前都是用apex的，后面的一些版本的torch官方引入了 """
    APEX = 'apex'
    NATIVE = 'native'


class MetricType(Enum):
    SCALAR = "scalar"
    HISTOGRAM = "histogram"
    IMAGE = "image"
    IMAGES = "images"
