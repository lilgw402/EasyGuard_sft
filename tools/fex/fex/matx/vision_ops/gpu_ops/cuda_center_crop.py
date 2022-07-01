# -*- coding: utf-8 -*-

from typing import List

import matx
from byted_vision.gpu_operators import CudaCustomCropOp, CudaBatchCenterCropOp


class CenterCrop:
    def __init__(self, sizes: List[int], device_id: int) -> None:
        self.crop_sizes: List[int] = sizes
        self.custom_crop_op: CudaCustomCropOp = CudaCustomCropOp(
            device_id)

    def __call__(self, image: matx.NDArray, stream: object) -> matx.NDArray:
        dst_height, dst_width = self.crop_sizes[0], self.crop_sizes[1]
        shape: List[int] = image.shape()
        src_height: int = shape[0]
        src_width: int = shape[1]

        width_begin = (src_width - dst_width) // 2
        height_begin = (src_height - dst_height) // 2

        return self.custom_crop_op(image, width_begin, height_begin, dst_width, dst_height, stream)


class BatchGPUCenterCrop:
    def __init__(self, task_manager: object, sizes: List[int], device_id: int, thread_num: int) -> None:
        self.thread_num: int = thread_num
        self.task_manager: object = task_manager
        self.center_crop: CenterCrop = CenterCrop(sizes, device_id)

    def __call__(self, images: List[matx.NDArray], streams: List[object]) -> List[matx.NDArray]:
        image_size = len(images)
        results_nd = matx.List()
        results_nd.reserve(image_size)

        futures = []
        for index in range(image_size):
            futures.append(self.task_manager.get_thread_pool().Submit(
                self.center_crop, images[index], streams[index % self.thread_num]))

        for f in futures:
            results_nd.append(f.get())

        return results_nd


class NativeBatchGPUCenterCrop:
    def __init__(self, sizes: List[int], device_id: int) -> None:
        self.sizes: List[int] = sizes
        self.batch_center_crop: CudaBatchCenterCropOp = CudaBatchCenterCropOp(
            device_id)

    def __call__(self, images: List[matx.NDArray], stream: object) -> List[matx.NDArray]:
        image_size = len(images)
        heights = matx.List([self.sizes[0]] * image_size)
        widths = matx.List([self.sizes[1]] * image_size)
        return self.batch_center_crop(images, widths, heights, stream)
