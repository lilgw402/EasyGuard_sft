# -*- coding: utf-8 -*-

"""
Vision GPU Pipeline
"""
from typing import List, Tuple
import matx
import byted_vision
import random

from fex import _logger as log
from fex.matx import convert_matx_ndarry_to_torch_tensor
from fex.matx.vision_ops import TaskManager
#from byted_vision.gpu_operators import CudaBatchCropOp, CudaBatchFlipOp, CudaBatchRotateOp
from byted_vision import ImdecodeOp, CvtColorOp, COLOR_BGR2RGB
from fex.matx.vision_ops.gpu_ops import (BatchCudaImdecodeOp,
                                         BatchCudaImdecodeRandomCropOp,
                                         CreateCudaStreams,
                                         BatchCudaStreamSync,
                                         NativeBatchGPURateResize,
                                         NativeBatchGPUImagePad,
                                         NativeBatchGPUCenterCrop,
                                         CudaKernelNormalizeFusionWithStack)


class CPUBatchImdecode:
    def __init__(self, device_id: int, task_manager: object) -> None:
        self.task_manager: object = task_manager
        self.device_id: int = device_id
        self.cpu_decode_op: ImdecodeOp = ImdecodeOp()
        self.cv_color_op: CvtColorOp = CvtColorOp()
        self.tensor_api: matx.NativeObject = matx.make_native_object("TensorAPI")

    def __call__(self, frames: List[bytes]) -> List[matx.NDArray]:
        frames_size = len(frames)
        cpu_decode_nd_ls = matx.List()
        cpu_decode_nd_ls.reserve(frames_size)

        futures = []
        for index in range(frames_size):
            futures.append(self.task_manager.get_thread_pool().Submit(
                self.cpu_decode_op, frames[index]))

        cvt_futures = []
        for f in futures:
            cvt_futures.append(self.task_manager.get_thread_pool().Submit(
                self.cv_color_op, f.get(), COLOR_BGR2RGB))

        for f in cvt_futures:
            cpu_nd = f.get()
            cpu_decode_nd_ls.append(self.tensor_api.to_device(
                cpu_nd, "cuda:{}".format(self.device_id)))

        return cpu_decode_nd_ls


class VisionGPUPipe():
    """ matx vision GPU pipe """

    def __init__(self,
                 image_dsr_width: int,
                 image_dsr_height: int,
                 resize_longer: int,
                 mean: List[float],
                 std: List[float],
                 thread_num: int = 8,
                 device_id: int = 0,
                 ):
        self.image_dsr_width: int = image_dsr_width
        self.image_dsr_height: int = image_dsr_height
        self.resize_longer: int = resize_longer
        self.mean: List[float] = mean
        self.std: List[float] = std
        self.thread_num: int = thread_num
        self.device_id: int = device_id

        self.task_manager = matx.script(TaskManager)(
            pool_size=self.thread_num, use_lockfree_pool=True)

        self.create_cuda_streams = matx.script(CreateCudaStreams)(
            device_id=self.device_id, thread_num=self.thread_num)

        # self.batch_decode_op = matx.script(BatchCudaImdecodeOp)(
        #     task_manager=self.task_manager,
        #     device_id=self.device_id,
        #     thread_num=self.thread_num,
        #     streams_maker=self.create_cuda_streams)

        self.batch_cpu_decode_op = matx.script(CPUBatchImdecode)(device_id=device_id,
                                                                 task_manager=self.task_manager)

        self.batch_center_crop_op = matx.script(NativeBatchGPUCenterCrop)(
            sizes=[self.image_dsr_height, self.image_dsr_width], device_id=self.device_id)

        self.batch_rate_resize = matx.script(NativeBatchGPURateResize)(
            task_manager=self.task_manager, resize_longer=self.resize_longer, device_id=self.device_id)

        self.batch_image_pad = matx.script(NativeBatchGPUImagePad)(
            task_manager=self.task_manager, sizes=[
                self.image_dsr_height, self.image_dsr_width], device_id=self.device_id, pad_values=[
                255, 255, 255])

        self.batch_streams_sync = matx.script(
            BatchCudaStreamSync)(device_id=self.device_id)

        self.stack_and_normalize_op = matx.script(CudaKernelNormalizeFusionWithStack)(
            device_id=self.device_id, mean=self.mean, std=self.std)

    def __call__(self, images: List[bytes]):
        # 1. 创建cuda_stream，方便多流多线程
        streams = self.create_cuda_streams()
        # 2. 解码
        gpu_image_nds = self.batch_cpu_decode_op(images)
        # 3. stream_sync，等待所有stream处理完数据
        # sync_images = self.batch_streams_sync(gpu_image_nds, streams)
        # 4. rate_resize
        rate_resize = self.batch_rate_resize(
            gpu_image_nds, byted_vision.INTER_LINEAR, streams[0])
        # 5. 对image做pad
        pad_images = self.batch_image_pad(
            rate_resize, byted_vision.BORDER_CONSTANT, streams[0])
        # 6. 对image做center_crop
        center_crop_images = self.batch_center_crop_op(rate_resize, streams[0])
        # 7. normalize all images
        sync_nchw_image = self.stack_and_normalize_op(
            center_crop_images, streams[0])
        return sync_nchw_image
