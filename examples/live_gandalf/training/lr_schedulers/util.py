# -*- coding:utf-8 -*-
#
# Copyright (c) 2020 Bytedance.com, Inc. All Rights Reserved
#
# Each engineer has a duty to keep the code elegant
#
"""
FileName: utils.py
Author: Ye Jinxing (yejinxing.yjx@bytedance.com)
Created Time: 2021-06-19 15:09:42
"""
import numpy as np


def with_warm_up(learning_rate, warmup_steps=0):
    def decorator(func):
        r"""
        func: python function
            args:
                - step
                    current step count.
                - total_step
                    step amount during this training.
            kw args:
                whatever, customized args.
        """

        def warm_up(*args, **kw):
            step = args[0]
            total_step = kw.get("_total_step", -1)
            if warmup_steps > 0 and step <= warmup_steps:
                return learning_rate * np.min(((step / warmup_steps), 1.0))
            else:
                total_step = np.max((0, total_step - warmup_steps))
                return func(step - warmup_steps, np.max((total_step, warmup_steps + 1)))

        return warm_up

    return decorator
