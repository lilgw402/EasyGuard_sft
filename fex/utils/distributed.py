#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Huang Wenguan (huangwenguan@bytedance.com)
Date: 2020-11-11 16:53:10
LastEditTime: 2020-11-12 11:15:55
LastEditors: Huang Wenguan
Description: 一些和分布式有关的手脚架函数
'''

import os
import warnings

import torch
import torch.distributed as dist

from functools import wraps
from fex import _logger as log
from fex import log_file


def local_rank_zero_only(fn):
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if local_rank == 0:
            return fn(*args, **kwargs)
    return wrapped_fn


def rank_zero_only(fn):
    """ 一个手脚架，用来方便只在rank 0 做一些操作 """
    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if rank_zero_only.rank == 0:
            return fn(*args, **kwargs)

    return wrapped_fn


# add the attribute to the function but don't overwrite in case Trainer has already set it
rank_zero_only.rank = getattr(rank_zero_only, 'rank', int(os.environ.get('RANK', 0)))  # type: ignore


@rank_zero_only
def rank0print(str):
    print(str)


def _warn(*args, **kwargs):
    warnings.warn(*args, **kwargs)


def _info(*args, **kwargs):
    log.info(*args, **kwargs)


def _debug(*args, **kwargs):
    log.debug(*args, **kwargs)


rank_zero_debug = rank_zero_only(_debug)
rank_zero_info = rank_zero_only(_info)
rank_zero_warn = rank_zero_only(_warn)


class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, rank, world_size, group=None):
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor, group)
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
            None,
            None
        )


class SubGroup:
    group: dist.ProcessGroup = None

    @classmethod
    def init_sub_groups(cls, rank, world_size, sub_world_size, timeout=dist.default_pg_timeout, backend=None):
        if world_size % sub_world_size != 0:
            raise RuntimeError(f'world_size({world_size}) is not multiple of sub_world_size({sub_world_size})')
        for start_rank in range(0, world_size, sub_world_size):
            sub_ranks = list(range(start_rank, start_rank + sub_world_size))
            sub_group = dist.new_group(sub_ranks, timeout, backend)
            if rank in sub_ranks:
                cls.group = sub_group
