# -*- coding: utf-8 -*-

from typing import List, Any

import matx
from byted_vision.gpu_operators import CudaImdecodeOp, CudaImdecodeRandomCropOp


class BatchCudaImdecodeOp:
    def __init__(self,
                 task_manager: object,
                 device_id: int,
                 thread_num: int,
                 streams_maker: Any,
                 output_fmt: str = "RGB") -> None:
        self.thread_num: int = thread_num
        self.task_manager: object = task_manager
        self.device_id: int = device_id
        self.decode_op: CudaImdecodeOp = CudaImdecodeOp(
            device_id, streams_maker, output_fmt)

    def __call__(self, frames: List[bytes], streams: List[object]) -> List[matx.NDArray]:
        frames_size = len(frames)
        decode_nd_ls = matx.List()
        decode_nd_ls.reserve(frames_size)

        futures = []
        for index in range(frames_size):
            stream_index = index % self.thread_num
            futures.append(self.task_manager.get_thread_pool().Submit(
                self.decode_op, frames[index], streams[stream_index]))

        for f in futures:
            decode_nd_ls.append(f.get())

        return decode_nd_ls


class BatchCudaImdecodeRandomCropOp:
    def __init__(self,
                 task_manager: object,
                 device_id: int,
                 thread_num: int,
                 scale: List[float],
                 ratio: List[float],
                 streams_maker: Any,
                 output_fmt: str = "RGB") -> None:
        self.thread_num: int = thread_num
        self.task_manager: object = task_manager
        self.device_id: int = device_id
        self.decode_and_randomcrop: CudaImdecodeRandomCropOp = CudaImdecodeRandomCropOp(
            device_id, streams_maker, output_fmt, scale, ratio)

    def __call__(self, frames: List[bytes], streams: List[object]) -> List[matx.NDArray]:
        frames_size = len(frames)
        decode_nd_ls = matx.List()
        decode_nd_ls.reserve(frames_size)

        futures = []
        for index in range(frames_size):
            stream_index = index % self.thread_num
            futures.append(self.task_manager.get_thread_pool().Submit(
                self.decode_and_randomcrop, frames[index], streams[stream_index]))

        for f in futures:
            decode_nd_ls.append(f.get())

        return decode_nd_ls
