# -*- coding: utf-8 -*-

from typing import List

import matx
from byted_vision.gpu_operators import CudaCvtColorOp


class BatchCvtColor:
    def __init__(self, task_manager: object, device_id: int, thread_num: int) -> None:
        self.task_manager: object = task_manager
        self.cvt_color_op: CudaCvtColorOp = \
            CudaCvtColorOp(device_id)

    def __call__(self, images: List[matx.NDArray], color_type: str, streams: List[object]) -> List[matx.NDArray]:
        image_size = len(images)
        results_nd = matx.List()
        results_nd.reserve(image_size)

        futures = []
        for index in range(image_size):
            futures.append(self.task_manager.get_thread_pool().Submit(
                self.cvt_color_op, images[index], color_type, streams[index % self.thread_num]))

        for f in futures:
            results_nd.append(f.get())

        return results_nd
