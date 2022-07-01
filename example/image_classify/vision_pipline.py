# -*- coding: utf-8 -*-

from typing import List
from torch.utils.data import DataLoader

import matx
import byted_vision

from fex.matx import convert_matx_ndarry_to_torch_tensor
from fex.matx.vision_ops import TaskManager
from fex.matx.vision_ops.gpu_ops import BatchCudaImdecodeOp, BatchCudaImdecodeRandomCropOp, NativeBatchGPURateResize, \
    CudaKernelNormalizeWithStack, CreateCudaStreams, BatchCudaStreamSync, NativeBatchGPUCenterCrop
from fex.matx.vision_ops.vision_loader_prefetch import VisionLoaderPrefetchBase


class VisionGPUPipe():
    """ byted vision GPU pipe """

    def __init__(self,
                 image_dsr_width: int,
                 image_dsr_height: int,
                 resize_shorter: int,
                 mean: List[float],
                 std: List[float],
                 scale: List[float],
                 ratio: List[float],
                 thread_num: int = 8,
                 device_id: int = 0,
                 mode: str = "train",
                 is_trace: bool = False):
        self.image_dsr_width: int = image_dsr_width
        self.image_dsr_height: int = image_dsr_height
        self.resize_shorter: int = resize_shorter
        self.mean: List[float] = mean
        self.std: List[float] = std
        self.scale: List[float] = scale
        self.ratio: List[float] = ratio
        self.thread_num: int = thread_num
        self.device_id: int = device_id
        self.is_trace: bool = is_trace
        self.mode: str = mode

        self.task_manager = matx.script(TaskManager)(
            pool_size=self.thread_num, use_lockfree_pool=True)

        self.create_cuda_streams = matx.script(CreateCudaStreams)(
            device_id=self.device_id, thread_num=self.thread_num)

        self.batch_decode_op = matx.script(BatchCudaImdecodeOp)(task_manager=self.task_manager,
                                                                device_id=self.device_id,
                                                                thread_num=self.thread_num,
                                                                streams_maker=self.create_cuda_streams)

        self.batch_decode_random_crop_op = matx.script(BatchCudaImdecodeRandomCropOp)(task_manager=self.task_manager,
                                                                                      device_id=self.device_id,
                                                                                      thread_num=self.thread_num,
                                                                                      streams_maker=self.create_cuda_streams,
                                                                                      scale=self.scale,
                                                                                      ratio=self.ratio)

        self.native_batch_center_crop = matx.script(NativeBatchGPUCenterCrop)(sizes=[self.image_dsr_height,
                                                                                     self.image_dsr_width],
                                                                              device_id=self.device_id)

        self.native_batch_rate_resize = matx.script(NativeBatchGPURateResize)(task_manager=self.task_manager,
                                                                              resize_shorter=self.resize_shorter,
                                                                              device_id=self.device_id)
        self.batch_streams_sync = matx.script(
            BatchCudaStreamSync)(device_id=self.device_id)

        self.stack_and_normalize_op = matx.script(CudaKernelNormalizeWithStack)(device_id=self.device_id,
                                                                                mean=self.mean,
                                                                                std=self.std)

    def __call__(self, *args, **kwargs):
        if self.is_trace:
            return self.trace_process(*args, **kwargs)
        elif self.mode == "train":
            return self.train_process(*args, **kwargs)
        else:
            return self.val_process(*args, **kwargs)

    def train_process(self, frames: List[bytes]):
        # 1. 创建cuda_stream，方便多流多线程
        streams = self.create_cuda_streams()
        # 2. 解码和RandomCrop
        gpu_image_and_random_crop_nd = self.batch_decode_random_crop_op(
            frames, streams)
        # 3. stream_sync，等待所有stream处理完数据
        sync_images = self.batch_streams_sync(
            gpu_image_and_random_crop_nd, streams)
        # 4. rate resize
        rate_resize_images = self.native_batch_rate_resize(
            sync_images, byted_vision.INTER_LINEAR, streams[0])
        # 5. center crop
        crop_images = self.native_batch_center_crop(
            rate_resize_images, streams[0])
        # 6. normalize all images
        sync_nchw_image = self.stack_and_normalize_op(
            crop_images, byted_vision.CV_32F, streams[0])
        # 7. convert to torch.Tensor
        torch_tensor = convert_matx_ndarry_to_torch_tensor(sync_nchw_image)
        return torch_tensor

    def val_process(self, frames: List[bytes]):
        # 1. 创建cuda_stream，方便多流多线程
        streams = self.create_cuda_streams()
        # 2. 解码
        gpu_image_nds = self.batch_decode_op(frames, streams)
        # 3. stream_sync，等待所有stream处理完数据
        sync_images = self.batch_streams_sync(gpu_image_nds, streams)
        # 4. rate_resize
        rate_resize = self.native_batch_rate_resize(
            sync_images, byted_vision.INTER_LINEAR, streams[0])
        # 5. center_crop
        crop_images = self.native_batch_center_crop(rate_resize, streams[0])
        # 6. normalize all images
        sync_nchw_image = self.stack_and_normalize_op(
            crop_images, byted_vision.CV_32F, streams[0])
        # 7. convert to torch.Tensor
        torch_tensor = convert_matx_ndarry_to_torch_tensor(sync_nchw_image)
        return torch_tensor

    def trace_process(self, frames: List[bytes]):
        # 1. 创建cuda_stream，方便多流多线程
        streams = self.create_cuda_streams()
        # 2. 解码
        gpu_image_nds = self.batch_decode_op(frames, streams)
        # 3. stream_sync，等待所有stream处理完数据
        sync_images = self.batch_streams_sync(gpu_image_nds, streams)
        # 4. rate_resize
        rate_resize = self.native_batch_rate_resize(
            sync_images, byted_vision.INTER_LINEAR, streams[0])
        # 5. center_crop
        crop_images = self.native_batch_center_crop(rate_resize, streams[0])
        # 6. normalize all images
        sync_nchw_image = self.stack_and_normalize_op(
            crop_images, byted_vision.CV_32F, streams[0])
        return sync_nchw_image


class VisionImageLoader(VisionLoaderPrefetchBase):
    def __init__(self,
                 data_iter: DataLoader,
                 image_dsr_width: int,
                 image_dsr_height: int,
                 resize_shorter: int,
                 normalize_mean: List[float],
                 normalize_std: List[float],
                 scale: List[float],
                 ratio: List[float],
                 mode: str,
                 output_map: List[str],
                 thread_num: int = 6,
                 device_id: int = 0,
                 max_prefetch_num: int = 1):

        self.dsr_height = image_dsr_height
        self.dsr_width = image_dsr_width
        self.resize_shorter = resize_shorter
        self.mean = normalize_mean
        self.std = normalize_std
        self.scale = scale
        self.ratio = ratio
        self.mode = mode
        self.thread_num = thread_num
        self.device_id = device_id

        super().__init__(data_iter=data_iter,
                         mode=mode,
                         output_map=output_map,
                         max_prefetch_num=max_prefetch_num)

    def init_vision_pipeline(self):
        vision_pipeline = VisionGPUPipe(self.dsr_height,
                                        self.dsr_width,
                                        self.resize_shorter,
                                        self.mean,
                                        self.std,
                                        self.scale,
                                        self.ratio,
                                        self.thread_num,
                                        self.device_id,
                                        self.mode)
        return vision_pipeline
