#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Vision Pipeline'''

from typing import Dict, List
import base64
import torch
from fex.matx.vision_ops.vision_cpu_transform import RateResizeAndRotate, NormalizeAndTranspose, Pad, \
    CvtColorOp, ImdecodeOp, matx, byted_vision


class ImageTransforms:
    def __init__(self, mean: List[float], std: List[float], sizes: List[int], need_transpose: bool):
        self.rate_resize_and_rotate_op = RateResizeAndRotate(
            sizes, need_transpose)
        self.normalize_op = NormalizeAndTranspose(mean, std)
        self.padding_op = Pad(sizes)
        self.image_decode = ImdecodeOp()
        self.cvt_color = CvtColorOp()

    def image_process(self, image_bytes: bytes) -> matx.NDArray:
        image_nd = self.image_decode(image_bytes)
        rgb_image_nd = self.cvt_color(
            image_nd, byted_vision.COLOR_BGR2RGB)
        rate_resize_and_rotate_image = self.rate_resize_and_rotate_op(
            rgb_image_nd)
        padding_image = self.padding_op(rate_resize_and_rotate_image)
        normalize_image = self.normalize_op(padding_image)
        return normalize_image


class MatxVisionPipe():
    """ matx vision pipe """

    def __init__(self, config, *args, **kwargs):
        self.config = config
        self.min_size = self.config.DATASET.IMAGE_MIN_SIZE
        self.max_size = self.config.DATASET.IMAGE_MAX_SIZE
        self.need_transpose = config.DATASET.NEED_TRANSPOSE
        self.to_bgr255 = config.DATASET.TOBGR255
        self.mean = list(config.DATASET.PIXEL_MEANS)
        self.std = list(config.DATASET.PIXEL_STDS)
        if not self.to_bgr255:
            self.mean = [mean * 255 for mean in self.mean]
            self.std = [std * 255 for std in self.std]

        self.image_transform = ImageTransforms(self.mean, self.std,
                                               [self.min_size, self.max_size],
                                               self.need_transpose)

    def __call__(self, image_str: bytes) -> Dict:
        """
        对一张图片进行预处理。
        """
        image_nd = self.image_transform.image_process(
            self.b64_decode(image_str))
        image_tensor = torch.from_numpy(image_nd.asnumpy())
        transed_h, transed_w = image_tensor.shape[1:3]  # image_tensor: CHW

        # 预处理出来的image_tensor 默认是RGB 0-255
        # 如果是图像检测的resnet预训练模型，输入是BGR 0-255，所以需要转一下；
        # 如果是imagenet-resnet预训练模型，输入是RGB 0-1，所以需要 / 255
        if self.to_bgr255:
            image_tensor = image_tensor[[2, 1, 0]]

        return {'image': image_tensor,
                'origin_h': 0, 'origin_w': 0,
                'transformed_h': transed_h, 'transformed_w': transed_w}

    def b64_decode(self, string):
        if isinstance(string, str):
            string = string.encode()
        return base64.decodebytes(string)
