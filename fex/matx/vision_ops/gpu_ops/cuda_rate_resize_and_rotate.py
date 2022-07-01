# -*- coding: utf-8 -*-

from typing import List
import math

import matx
from byted_vision.gpu_operators import CudaResizeOp, CudaRotateOp


class GPURateResizeAndRotate:
    def __init__(self,
                 sizes: List[int] = [0, 0],
                 resize_shorter: int = 0,
                 resize_longer: int = 0,
                 need_transpose: bool = False,
                 device_id: int = 0) -> None:
        self.resize_sizes: List[int] = sizes
        self.resize_shorter: int = resize_shorter
        self.resize_longer: int = resize_longer
        self.need_transpose: bool = need_transpose
        self.gpu_opencv_resize_op: CudaResizeOp = CudaResizeOp(
            device_id)
        self.gpu_opencv_rotate_op: CudaRotateOp = CudaRotateOp(
            device_id)

    def __call__(self, image: matx.NDArray, rotate_type: int, resize_type: str,
                 stream: object) -> matx.NDArray:
        is_transpose: bool = False
        shape: List[int] = image.shape()
        src_height: int = shape[0]
        src_width: int = shape[1]

        adjust_width: int = 0
        adjust_height: int = 0

        # 对图片是否做翻转
        if (self.need_transpose and src_height < src_width):
            src_height = shape[1]
            src_width = shape[0]
            is_transpose = True

        if self.resize_shorter > 0:
            if (src_width <= src_height and src_width == self.resize_shorter) or \
                    (src_height <= src_width and src_height == self.resize_shorter):
                adjust_width = src_width
                adjust_height = src_height

            if src_width < src_height:
                adjust_width = self.resize_shorter
                adjust_height = int(self.resize_shorter * src_height /
                                    src_width)
            else:
                adjust_height = self.resize_shorter
                adjust_width = int(self.resize_shorter * src_width /
                                   src_height)

        elif self.resize_longer > 0:
            if (src_width >= src_height and src_width == self.resize_longer) or \
                    (src_height >= src_width and src_height == self.resize_longer):
                adjust_height = src_height
                adjust_width = src_width

            if src_width < src_height:
                adjust_height = self.resize_longer
                adjust_width = int(self.resize_longer * src_width / src_height)
            else:
                adjust_width = self.resize_longer
                adjust_height = int(self.resize_longer * src_height /
                                    src_width)

        else:
            assert len(
                self.resize_sizes) == 2, "resize_size args is equal to 2"
            # Convention (h, w)
            size_h, size_w = self.resize_sizes[0], self.resize_sizes[1]

            ratio_src = src_width / src_height
            ratio_dst = size_w / size_h

            if ratio_src > ratio_dst:
                # 原始的w更长，按住w，放缩h
                adjust_width = size_w
                adjust_height = math.floor((size_w / src_width) * src_height)
            elif ratio_src < ratio_dst:
                # 原始的h更长，按住h，放缩w
                adjust_height = size_h
                adjust_width = math.floor((size_h / src_height) * src_width)
            else:
                adjust_height = size_h
                adjust_width = size_w

        resize_width: int = adjust_width
        resize_height: int = adjust_height

        if is_transpose:
            resize_width = adjust_height
            resize_height = adjust_width

        resize_image = self.gpu_opencv_resize_op(image,
                                                 (resize_height, resize_width),
                                                 0, 0, resize_type, stream)
        if is_transpose:
            return self.gpu_opencv_rotate_op(resize_image, rotate_type, stream)
        return resize_image


class BatchCudaRateResizeAndRotate:
    def __init__(self, task_manager: object, sizes: List[int],
                 resize_shorter: int, resize_longer: int, need_transpose: bool,
                 device_id: int, thread_num: int) -> None:
        self.thread_num: int = thread_num
        self.task_manager: object = task_manager
        self.gpu_rate_resize_and_rotate: GPURateResizeAndRotate = GPURateResizeAndRotate(
            sizes, resize_shorter, resize_longer, need_transpose, device_id)

    def __call__(self, images: List[matx.NDArray], rotate_type: int,
                 resize_type: str,
                 streams: List[object]) -> List[matx.NDArray]:
        image_size = len(images)
        results_nd = matx.List()
        results_nd.reserve(image_size)

        futures = []
        for index in range(image_size):
            futures.append(self.task_manager.get_thread_pool().Submit(
                self.gpu_rate_resize_and_rotate, images[index], rotate_type,
                resize_type, streams[index % self.thread_num]))

        for f in futures:
            results_nd.append(f.get())

        return results_nd
