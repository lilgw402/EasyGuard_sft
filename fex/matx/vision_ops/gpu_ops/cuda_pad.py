# -*- coding: utf-8 -*-
from typing import List
import math

import matx
from byted_vision.gpu_operators import CudaCopyMakeBorderOp, CudaBatchCopyMakeBorderOp


class ImagePad:
    def __init__(self, sizes: List[int], device_id: int, pad_values: List[int]) -> None:
        self.dst_sizes: List[int] = sizes
        self.pad_values: List[int] = pad_values
        self.opencv_padding_op: CudaCopyMakeBorderOp = CudaCopyMakeBorderOp(
            device_id)

    def __call__(self, image: matx.NDArray, pad_type: str,
                 stream: object) -> matx.NDArray:
        dst_height, dst_width = self.dst_sizes[0], self.dst_sizes[1]
        shape: List[int] = image.shape()
        src_height: int = shape[0]
        src_width: int = shape[1]

        left_pad = 0
        right_pad = 0
        top_pad = 0
        bottom_pad = 0

        if dst_height == src_height and dst_width == src_width:
            return image

        if dst_width > src_width:
            # w没对齐, 左右两边pad
            left_pad = math.floor((dst_width - src_width) / 2)
            right_pad = dst_width - src_width - left_pad

        if dst_height > src_height:
            # h没对齐, 上下两边pad
            top_pad = math.floor((dst_height - src_height) / 2)
            bottom_pad = dst_height - src_height - top_pad

        padding_image = self.opencv_padding_op(image, top_pad, bottom_pad,
                                               left_pad, right_pad, pad_type,
                                               self.pad_values, stream)
        return padding_image


class BatchImagePad:
    def __init__(self,
                 task_manager: object,
                 sizes: List[int],
                 device_id: int,
                 thread_num: int,
                 pad_values: List[int] = [0, 0, 0]) -> None:
        self.thread_num: int = thread_num
        self.task_manager: object = task_manager
        self.dst_sizes: List[int] = sizes
        self.image_pad: ImagePad = ImagePad(sizes, device_id, pad_values)

    def __call__(self, images: List[matx.NDArray], pad_type: str,
                 streams: List[object]) -> List[matx.NDArray]:
        image_size = len(images)
        results_nd = matx.List()
        results_nd.reserve(image_size)

        futures = []
        for index in range(image_size):
            futures.append(self.task_manager.get_thread_pool().Submit(
                self.image_pad, images[index], pad_type,
                streams[index % self.thread_num]))

        for f in futures:
            results_nd.append(f.get())

        return results_nd


class CalculatePadParams:
    def __init__(self, sizes: List[int], with_corner: bool) -> None:
        self.dst_sizes: List[int] = sizes
        self.with_corner: bool = with_corner

    def __call__(self, image: matx.NDArray) -> List[int]:
        params: List[int] = matx.List()
        params.reserve(4)

        dst_height, dst_width = self.dst_sizes[0], self.dst_sizes[1]
        shape: List[int] = image.shape()
        src_height: int = shape[0]
        src_width: int = shape[1]

        left_pad = 0
        right_pad = 0
        top_pad = 0
        bottom_pad = 0

        if dst_width > src_width:
            # w没对齐, 左右两边pad
            if self.with_corner:
                right_pad = dst_width - src_width
            else:
                left_pad = math.floor((dst_width - src_width) / 2)
                right_pad = dst_width - src_width - left_pad

        if dst_height > src_height:
            # h没对齐, 上下两边pad
            if self.with_corner:
                bottom_pad = dst_height - src_height
            else:
                top_pad = math.floor((dst_height - src_height) / 2)
                bottom_pad = dst_height - src_height - top_pad

        params.append(top_pad)
        params.append(bottom_pad)
        params.append(left_pad)
        params.append(right_pad)
        return params


class NativeBatchGPUImagePad:
    def __init__(self,
                 task_manager: object,
                 sizes: List[int],
                 device_id: int,
                 with_corner: bool = False,
                 pad_values: List[int] = [0, 0, 0]) -> None:
        self.task_manager: object = task_manager
        self.dst_sizes: List[int] = sizes
        self.pad_values: List[int] = pad_values
        self.cal_params: CalculatePadParams = CalculatePadParams(sizes, with_corner)
        self.batch_gpu_image_pad: CudaBatchCopyMakeBorderOp = CudaBatchCopyMakeBorderOp(
            device_id)

    def __call__(self,
                 images: List[matx.NDArray],
                 pad_type: str,
                 stream: object) -> List[matx.NDArray]:
        image_size = len(images)
        top_pads = matx.List()
        bottom_pads = matx.List()
        left_pads = matx.List()
        right_pads = matx.List()

        top_pads.reserve(image_size)
        bottom_pads.reserve(image_size)
        left_pads.reserve(image_size)
        right_pads.reserve(image_size)

        futures = []
        for index in range(image_size):
            futures.append(self.task_manager.get_thread_pool().Submit(
                self.cal_params, images[index]))

        for f in futures:
            params = f.get()
            top_pads.append(params[0])
            bottom_pads.append(params[1])
            left_pads.append(params[2])
            right_pads.append(params[3])

        return self.batch_gpu_image_pad(images, top_pads, bottom_pads, left_pads, right_pads, pad_type, self.pad_values, stream)
