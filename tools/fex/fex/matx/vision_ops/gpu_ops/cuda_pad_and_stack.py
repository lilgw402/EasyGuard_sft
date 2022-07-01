# -*- coding: utf-8 -*-
from typing import List, Tuple
import math

import matx
from byted_vision.gpu_operators import CudaPadStackOp


class CaculatePadParams:
    def __init__(self, sizes: List[int]) -> None:
        self.dst_sizes: List[int] = sizes

    def __call__(self, image: matx.NDArray) -> Tuple[int, int, int, int]:
        dst_height, dst_width = self.dst_sizes[0], self.dst_sizes[1]
        shape: List[int] = image.shape()
        src_height: int = shape[0]
        src_width: int = shape[1]

        left_pad = 0
        right_pad = 0
        top_pad = 0
        bottom_pad = 0
        if dst_height == src_height and dst_width == src_width:
            return top_pad, bottom_pad, left_pad, right_pad

        if dst_width > src_width:
            # w没对齐, 左右两边pad
            left_pad = math.floor((dst_width - src_width) / 2)
            right_pad = dst_width - src_width - left_pad

        else:
            # h没对齐, 上下两边pad
            top_pad = math.floor((dst_height - src_height) / 2)
            bottom_pad = dst_height - src_height - top_pad
        return top_pad, bottom_pad, left_pad, right_pad


class ImagesPadAndStack:
    def __init__(self, task_manager: object, sizes: List[int], device_id: int) -> None:
        self.task_manager: object = task_manager
        self.caculate_pad_params: CaculatePadParams = CaculatePadParams(sizes)
        self.pad_and_stack: CudaPadStackOp = CudaPadStackOp(
            device_id)

    def __call__(self, images: List[matx.NDArray], stream: object, pad_type: str, pad_value: int = 0) -> matx.NDArray:
        image_size = len(images)

        tops = matx.List()
        bottoms = matx.List()
        lefts = matx.List()
        rights = matx.List()

        tops.reserve(image_size)
        bottoms.reserve(image_size)
        lefts.reserve(image_size)
        rights.reserve(image_size)

        futures = []
        for index in range(image_size):
            futures.append(self.task_manager.get_thread_pool().Submit(
                self.caculate_pad_params, images[index]))

        for f in futures:
            ret = f.get()
            tops.append(ret[0])
            bottoms.append(ret[1])
            lefts.append(ret[2])
            rights.append(ret[3])

        return self.pad_and_stack(images, tops, bottoms, lefts, rights, pad_type, pad_value, stream)
