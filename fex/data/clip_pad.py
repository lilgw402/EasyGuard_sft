#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 一些图片padding的操作 """

from typing import List
import torch


def clip_pad_images(tensor, pad_shape: List[int], pad:int=0):
    """
    Clip clip_pad_images of the pad area.
    :param tensor: [c, H, W]
    :param pad_shape: [h, w]
    :return: [c, h, w]
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    H, W = tensor.shape[1:]
    h = pad_shape[1]
    w = pad_shape[2]

    tensor_ret = torch.zeros((tensor.shape[0], h, w), dtype=tensor.dtype) + pad
    tensor_ret[:, :min(h, H), :min(w, W)] = tensor[:, :min(h, H), :min(w, W)]

    return tensor_ret


def clip_pad_boxes(tensor, pad_length: int, pad:int=0):
    """
        Clip boxes of the pad area.
        :param tensor: [k, d]
        :param pad_shape: K
        :return: [K, d]
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    k = tensor.shape[0]
    d = tensor.shape[1]
    K = pad_length
    tensor_ret = torch.zeros((K, d), dtype=tensor.dtype) + pad
    tensor_ret[:min(k, K), :] = tensor[:min(k, K), :]

    return tensor_ret


def clip_pad_1d(tensor, pad_length, pad=0):
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    tensor_ret = torch.zeros((pad_length, ), dtype=tensor.dtype) + pad
    tensor_ret[:min(tensor.shape[0], pad_length)] = tensor[:min(tensor.shape[0], pad_length)]

    return tensor_ret


def clip_pad_2d(tensor, pad_shape, pad=0):
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    tensor_ret = torch.zeros(*pad_shape, dtype=tensor.dtype) + pad
    tensor_ret[:min(tensor.shape[0], pad_shape[0]), :min(tensor.shape[1], pad_shape[1])] \
        = tensor[:min(tensor.shape[0], pad_shape[0]), :min(tensor.shape[1], pad_shape[1])]

    return tensor_ret

def clip_pad_4d(tensor, pad_shape, pad=0):
    tensor_ret = torch.zeros(*pad_shape, dtype=tensor[0].dtype) + pad
    for i in range(len(tensor)):
        tensor_ret[i][:,:min(tensor[i].shape[1], tensor_ret[i].shape[1]), :min(tensor[i].shape[2], tensor_ret[i].shape[2])] \
             = tensor[i][:,:min(tensor[i].shape[1], tensor_ret[i].shape[1]), :min(tensor[i].shape[2], tensor_ret[i].shape[2])]
    return tensor_ret
