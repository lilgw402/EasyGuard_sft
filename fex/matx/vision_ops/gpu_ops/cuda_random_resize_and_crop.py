# -*- coding: utf-8 -*-

from typing import List

import matx
from byted_vision.gpu_operators import CudaRandomResizedCropOp, CudaBatchRandomResizedCropOp


class BatchCudaRandomResizeCrop:
    def __init__(self,
                 task_manager: object,
                 sizes: List[int],
                 scale: List[float],
                 ratio: List[float],
                 device_id: int,
                 thread_num: int) -> None:
        self.thread_num: int = thread_num
        self.task_manager: object = task_manager
        self.dst_sizes: List[int] = sizes
        self.random_resize_crop: CudaRandomResizedCropOp = CudaRandomResizedCropOp(
            device_id, scale, ratio)

    def __call__(self, images: List[matx.NDArray], interp_type: str, streams: List[object]) -> List[matx.NDArray]:
        image_size = len(images)
        results_nd = matx.List()
        results_nd.reserve(image_size)

        futures = []
        for index in range(image_size):
            futures.append(self.task_manager.get_thread_pool().Submit(
                self.random_resize_crop, images[index], self.dst_sizes, interp_type, streams[index % self.thread_num]))

        for f in futures:
            results_nd.append(f.get())

        return results_nd


class NativeBatchCudaRandomResizeCrop:
    def __init__(self,
                 sizes: List[int],
                 scale: List[float],
                 ratio: List[float],
                 device_id: int) -> None:
        self.dst_sizes: List[int] = sizes
        self.batch_cuda_random_resize_crop: CudaBatchRandomResizedCropOp = CudaBatchRandomResizedCropOp(
            device_id, scale, ratio)

    def __call__(self, images: List[matx.NDArray], interp_type: str, stream: object) -> List[matx.NDArray]:
        image_size = len(images)
        dsr_heights = matx.List([self.dst_sizes[0]] * image_size)
        dsr_widths = matx.List([self.dst_sizes[1]] * image_size)

        return self.batch_cuda_random_resize_crop(images, dsr_heights, dsr_widths, interp_type, stream)
