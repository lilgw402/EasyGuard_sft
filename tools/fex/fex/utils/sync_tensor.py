# -*- coding: utf-8 -*-
'''
Created on Dec-23-20 15:16
sync_tensor.py
Description:
'''

import os
from typing import Tuple, Union, Optional, Any

import torch
from torch.distributed import ReduceOp


def sync_tensor(input_value: Union[torch.Tensor, float], group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = None, device_rank: int = None) -> torch.Tensor:
    """
    Function to reduce a tensor across worker processes during distributed training
    Args:
        input_value: the value to sync and reduce (typically tensor or number)
        group: the process group to gather results from. Defaults to all processes (world)
        reduce_op: the reduction operation. Defaults to sum.
            Can also be a string of 'avg', 'mean' to calculate the mean during reduction.
    Return:
        reduced value
    """
    if (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()):
        return input_value

    cuda_device = device_rank if device_rank is not None else int(os.environ.get('LOCAL_RANK', 0))
    divide_by_world_size = False

    if group is None:
        group = torch.distributed.group.WORLD

    # TODO: 后续根据需求添加更多的reduce_op支持
    if reduce_op is None:
        reduce_op = torch.distributed.ReduceOp.SUM
    elif isinstance(reduce_op, str) and reduce_op in ("avg", "mean"):
        reduce_op = torch.distributed.ReduceOp.SUM
        divide_by_world_size = True

    # sync all processes before reduction
    # torch.distributed.barrier(group=group)
    if isinstance(input_value, torch.Tensor):
        result_new_tensor = input_value.clone()
        if not result_new_tensor.is_cuda:
            result_new_tensor = result_new_tensor.to(torch.device(cuda_device))
    else:
        result_new_tensor = torch.tensor(input_value).to(torch.device(cuda_device))

    torch.distributed.all_reduce(
        result_new_tensor, op=reduce_op, group=group, async_op=False)

    if divide_by_world_size:
        result_new_tensor = result_new_tensor / torch.distributed.get_world_size(group)

    return result_new_tensor
