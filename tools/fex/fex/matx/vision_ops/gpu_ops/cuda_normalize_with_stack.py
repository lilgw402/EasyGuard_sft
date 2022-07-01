# -*- coding: utf-8 -*-
from typing import List

import matx
from byted_vision.gpu_operators import CudaNormalizeOp, CudaStackOp, CudaConvertToOp, CudaReformatOp, CudaNormConvertFormatOp


class CudaKernelNormalize:
    def __init__(self,
                 device_id: int,
                 mean: List[float],
                 std: List[float]) -> None:
        self.device_id: int = device_id
        self.normalize_op: CudaNormalizeOp = CudaNormalizeOp(
            self.device_id, mean, std)
        self.convert_op: CudaConvertToOp = CudaConvertToOp(
            self.device_id)
        self.permute_op: CudaReformatOp = CudaReformatOp(
            self.device_id)

    def __call__(self, image: matx.NDArray, dtype: str, stream: object) -> matx.NDArray:
        images_float32 = self.convert_op(image, dtype, 1.0, 0.0, stream)
        normalize_image = self.normalize_op(images_float32, stream)
        nchw_image = self.permute_op(
            normalize_image, "NHWC", "NCHW", stream)
        matx.stream_sync(stream, self.device_id)
        return nchw_image


class CudaKernelNormalizeWithStack:
    def __init__(self,
                 device_id: int,
                 mean: List[float],
                 std: List[float]) -> None:
        self.device_id: int = device_id
        self.normalize_op: CudaNormalizeOp = CudaNormalizeOp(
            self.device_id, mean, std)
        self.stack_image_op: CudaStackOp = CudaStackOp(
            self.device_id)
        self.convert_op: CudaConvertToOp = CudaConvertToOp(
            self.device_id)
        self.permute_op: CudaReformatOp = CudaReformatOp(
            self.device_id)

    def __call__(self, image_ls: List[matx.NDArray], dtype: str, stream: object) -> matx.NDArray:
        stack_image = self.stack_image_op(image_ls, stream)
        images_float32 = self.convert_op(stack_image, dtype, 1.0, 0.0, stream)
        normalize_image = self.normalize_op(images_float32, stream)
        nchw_image = self.permute_op(
            normalize_image, "NHWC", "NCHW", stream)
        matx.stream_sync(stream, self.device_id)
        return nchw_image


class CudaKernelNormalizeFusionWithStack:
    def __init__(self,
                 device_id: int,
                 mean: List[float],
                 std: List[float],
                 dtype: str = "CV_32F",
                 layout: str = "NCHW") -> None:
        self.device_id: int = device_id
        self.normalize_op: CudaNormConvertFormatOp = CudaNormConvertFormatOp(
            self.device_id, mean, std, layout, dtype)
        self.stack_image_op: CudaStackOp = CudaStackOp(
            self.device_id)

    def __call__(self, image_ls: List[matx.NDArray], stream: object) -> matx.NDArray:
        stack_image = self.stack_image_op(image_ls, stream)
        normalize_image = self.normalize_op(stack_image, stream)
        matx.stream_sync(stream, self.device_id)
        return normalize_image
