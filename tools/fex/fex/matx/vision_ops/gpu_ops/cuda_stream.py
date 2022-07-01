# -*- coding: utf-8 -*-

from typing import List

import matx


class CreateCudaStreams:
    def __init__(self, device_id: int, thread_num: int) -> None:
        self.thread_num: int = thread_num
        self.device_id: int = device_id
        self.streams: List[object] = matx.List()

        for i in range(self.thread_num):
            self.streams.append(matx.create_stream(self.device_id))

    def __call__(self) -> List[object]:
        return self.streams


class BatchCudaStreamSync:
    def __init__(self, device_id: int) -> None:
        self.device_id: int = device_id

    def __call__(self, image_nds: List[matx.NDArray], stream_ls: List[object]) -> List[matx.NDArray]:
        for stream in stream_ls:
            matx.stream_sync(stream, self.device_id)
        return image_nds


class CudaStreamSync:
    def __init__(self, device_id: int) -> None:
        self.device_id: int = device_id

    def __call__(self, image_nd: matx.NDArray, stream: object) -> matx.NDArray:
        matx.stream_sync(stream, self.device_id)
        return image_nd
