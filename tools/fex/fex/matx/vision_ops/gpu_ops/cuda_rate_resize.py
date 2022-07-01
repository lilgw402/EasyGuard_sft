# -*- coding: utf-8 -*-
from typing import List, Tuple
import math

import matx
from byted_vision.gpu_operators import CudaResizeOp, CudaBatchResizeOp


class GPURateResize:
    def __init__(self,
                 sizes: List[int] = [0, 0],
                 resize_shorter: int = 0,
                 resize_longer: int = 0,
                 device_id: int = 0) -> None:
        self.resize_shorter: int = resize_shorter
        self.resize_longer: int = resize_longer
        self.resize_sizes: List[int] = sizes
        self.gpu_opencv_resize_op: CudaResizeOp = \
            CudaResizeOp(device_id)

    def __call__(self, image: matx.NDArray, resize_type: str, stream: object) -> matx.NDArray:
        shape: List[int] = image.shape()
        src_height: int = shape[0]
        src_width: int = shape[1]

        adjust_width: int = 0
        adjust_height: int = 0

        if self.resize_shorter > 0:
            if (src_width <= src_height and src_width == self.resize_shorter) or \
                    (src_height <= src_width and src_height == self.resize_shorter):
                adjust_width = src_width
                adjust_height = src_height

            if src_width < src_height:
                adjust_width = self.resize_shorter
                adjust_height = int(self.resize_shorter *
                                    src_height / src_width)
            else:
                adjust_height = self.resize_shorter
                adjust_width = int(self.resize_shorter *
                                   src_width / src_height)

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
                adjust_height = int(self.resize_longer *
                                    src_height / src_width)

        else:
            assert len(self.resize_sizes) == 2, "resize_size args is equal to 2"
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

        resize_image = self.gpu_opencv_resize_op(
            image, (adjust_height, adjust_width), 0, 0, resize_type, stream)
        return resize_image


class BatchGPURateResize:
    def __init__(self,
                 task_manager: object,
                 sizes: List[int] = [0, 0],
                 resize_shorter: int = 0,
                 resize_longer: int = 0,
                 device_id: int = 0,
                 thread_num: int = 1) -> None:
        self.thread_num: int = thread_num
        self.task_manager: object = task_manager
        self.gpu_rate_resize: GPURateResize = GPURateResize(
            sizes, resize_shorter, resize_longer, device_id)

    def __call__(self, images: List[matx.NDArray], resize_type: str, streams: List[object]) -> List[matx.NDArray]:
        image_size = len(images)
        results_nd = matx.List()
        results_nd.reserve(image_size)

        futures = []
        for index in range(image_size):
            futures.append(self.task_manager.get_thread_pool().Submit(
                self.gpu_rate_resize, images[index], resize_type, streams[index % self.thread_num]))

        for f in futures:
            results_nd.append(f.get())

        return results_nd


class CalculateResizeParams:
    def __init__(self,
                 sizes: List[int],
                 resize_shorter: int = 0,
                 resize_longer: int = 0) -> None:
        self.resize_shorter: int = resize_shorter
        self.resize_longer: int = resize_longer
        self.resize_sizes: List[int] = sizes

    def __call__(self, image: matx.NDArray) -> Tuple[int, int]:
        shape: List[int] = image.shape()
        src_height: int = shape[0]
        src_width: int = shape[1]

        adjust_width: int = 0
        adjust_height: int = 0

        if self.resize_shorter > 0:
            if (src_width <= src_height and src_width == self.resize_shorter) or \
                    (src_height <= src_width and src_height == self.resize_shorter):
                adjust_width = src_width
                adjust_height = src_height

            if src_width < src_height:
                adjust_width = self.resize_shorter
                adjust_height = int(self.resize_shorter *
                                    src_height / src_width)
            else:
                adjust_height = self.resize_shorter
                adjust_width = int(self.resize_shorter *
                                   src_width / src_height)

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
                adjust_height = int(self.resize_longer *
                                    src_height / src_width)

        else:
            assert len(self.resize_sizes) == 2, "resize_size args is equal to 2"
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

        return (adjust_height, adjust_width)


class NativeBatchGPURateResize:
    def __init__(self,
                 task_manager: object,
                 sizes: List[int] = [0, 0],
                 resize_shorter: int = 0,
                 resize_longer: int = 0,
                 device_id: int = 0) -> None:
        self.batch_gpu_rate_resize: CudaBatchResizeOp = CudaBatchResizeOp(
            device_id)
        self.calcualte_resize_params: CalculateResizeParams = CalculateResizeParams(
            sizes, resize_shorter, resize_longer)
        self.task_manager: object = task_manager

    def __call__(self, images: List[matx.NDArray], resize_type: str, stream: object) -> List[matx.NDArray]:
        image_size = len(images)
        adjust_heights = matx.List()
        adjust_widths = matx.List()
        adjust_heights.reserve(image_size)
        adjust_widths.reserve(image_size)

        futures = []
        for index in range(image_size):
            futures.append(self.task_manager.get_thread_pool().Submit(
                self.calcualte_resize_params, images[index]))

        for f in futures:
            params = f.get()
            adjust_heights.append(params[0])
            adjust_widths.append(params[1])

        return self.batch_gpu_rate_resize(images, adjust_heights, adjust_widths, resize_type, stream)
