#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Vision Pipeline'''

from typing import Dict
import base64
import io
from PIL import Image
import torch
from torchvision import transforms
# from fex.lib.cvops.cvwrapper import wrap_imread_and_rate_resize
try:
    import visionops
    visionops.torch_load_op_library()
except:
    pass


class VisionPipe():
    """ vision pipe """

    def __init__(self, config, *args, **kwargs):
        self.config = config
        self.min_size = self.config.DATASET.IMAGE_MIN_SIZE
        self.max_size = self.config.DATASET.IMAGE_MAX_SIZE
        self.need_transpose = config.DATASET.NEED_TRANSPOSE
        self.to_bgr255 = config.DATASET.TOBGR255
        self.mean = torch.Tensor(config.DATASET.PIXEL_MEANS)
        self.std = torch.Tensor(config.DATASET.PIXEL_STDS)
        self.already_preprocess = self.config.DATASET.ALREADY_PREPROCESS
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image_str: bytes) -> Dict:
        """
        对一张图片预处理的核心函数，op版。
        因为前面的版本流程太冗长，容易出diff。我们直接把图片处理压成一个op，直接调。
        """
        if self.already_preprocess:
            image_tensor = self.to_tensor(self.load_image(self.b64_decode(image_str)))
            is_transposed = self.need_transpose
            h, w = -1, -1
            transed_h, transed_w = image_tensor.shape[1:3]    # image_tensor: CHW
        else:
            image_bytes = self.b64_decode(image_str)
            image_tensor = torch.ops.vis.imread_and_rate_resize(image_bytes, self.min_size, self.max_size,
                                                                self.need_transpose)
            transed_h, transed_w = image_tensor.shape[1:3]    # image_tensor: CHW

            # 预处理出来的image_tensor 默认是RGB 0-255
            # 如果是图像检测的resnet预训练模型，输入是BGR 0-255，所以需要转一下；
            # 如果是imagenet-resnet预训练模型，输入是RGB 0-1，所以需要 / 255
            if self.to_bgr255:
                image_tensor = image_tensor[[2, 1, 0]]
            else:
                image_tensor = image_tensor / 255

        # 把padding的位置置为0
        padding_place_h = image_tensor.sum([0, 2]).bool()
        padding_place_w = image_tensor.sum([0, 1]).bool()
        padding_place = padding_place_w & padding_place_h.unsqueeze(0).t()

        # 归一化
        image_tensor.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        image_tensor = image_tensor * padding_place

        return {
            'image': image_tensor,
            'origin_h': 0,
            'origin_w': 0,
            'transformed_h': transed_h,
            'transformed_w': transed_w,
            'is_transposed': False
        }

    def b64_decode(self, string):
        if isinstance(string, str):
            string = string.encode()
        return base64.decodebytes(string)

    def load_image(self, buffer):
        # return cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_COLOR)
        return Image.open(io.BytesIO(buffer))
