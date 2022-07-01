# -*- coding: utf-8 -*-
"""
BytedVision 1.6 GPU Pipeline
"""

from typing import List

from torch.utils.data import DataLoader
import matx
import byted_vision
try:
    from byted_vision import (ImdecodeOp,
                              ImdecodeRandomCropOp,
                              FlipOp,
                              CropOp,
                              CenterCropOp,
                              ResizeOp,
                              PadOp,
                              TransposeNormalizeOp)
except Exception as e:
    print(e)

from fex.matx.vision_ops.vision_loader_prefetch import VisionLoaderPrefetchBase


class VisionGPUPipeV2():
    """
    byted_vision 的预处理 pipeline
    """

    def __init__(self,
                 device_id: int = 0,
                 mode: str = "eval",
                 image_size: int = 224,
                 ):
        self.is_training = mode == 'train'
        self.mode = mode
        # 1. decode
        resized_size = image_size if self.is_training else int(image_size / 0.875)  # 如果是eval 则先resize 到256，再crop
        self.device = matx.Device("cuda:%s" % device_id)
        if self.is_training:
            self.random_crop_decode = ImdecodeRandomCropOp(device=self.device,
                                                           fmt="RGB",
                                                           scale=[0.1, 1.0],
                                                           ratio=[0.8, 1.25])
            self.resize = ResizeOp(device=self.device,
                                   size=(resized_size, resized_size),
                                   interp=byted_vision.PILLOW_INTER_LINEAR,
                                   mode=byted_vision.RESIZE_DEFAULT)
            self.random_flip = FlipOp(device=self.device,
                                      flip_code=byted_vision.HORIZONTAL_FLIP,
                                      prob=0.5)
        else:
            self.decode = ImdecodeOp(device=self.device, fmt="RGB")
            self.resize = ResizeOp(device=self.device,
                                   size=(resized_size, resized_size),
                                   interp=byted_vision.PILLOW_INTER_LINEAR,
                                   mode=byted_vision.RESIZE_NOT_SMALLER)

            self.center_crop = CenterCropOp(device=self.device,
                                            sizes=(image_size, image_size))

        # 3. normalize
        self.trnaspose_norm = TransposeNormalizeOp(device=self.device,
                                                   mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                                   std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                                                   input_layout=byted_vision.NHWC,
                                                   output_layout=byted_vision.NCHW)

    def __call__(self, image, **kwargs):
        if self.is_training:
            decode_nds = self.random_crop_decode(image)
            flip_nds = self.random_flip(decode_nds)
            resize_nds = self.resize(flip_nds)
            transpose_nd = self.trnaspose_norm(resize_nds, byted_vision.SYNC)
        else:
            decode_nds = self.decode(image)
            resize_nds = self.resize(decode_nds)
            crop_nds = self.center_crop(resize_nds)
            transpose_nd = self.trnaspose_norm(crop_nds, byted_vision.SYNC)
        if self.mode == "trace":
            return transpose_nd
        return transpose_nd.torch()


class BytedvisionLoader(VisionLoaderPrefetchBase):
    def __init__(self,
                 data_iter: DataLoader,
                 output_map: List[str],
                 device_id: int = 0,
                 max_prefetch_num: int = 1,
                 mode: str = 'eval'):

        self.mode = mode
        self.device_id = device_id
        super().__init__(data_iter=data_iter,
                         mode=mode,
                         output_map=output_map,
                         max_prefetch_num=max_prefetch_num)

    def init_vision_pipeline(self):
        vision_pipeline = VisionGPUPipeV2(device_id=self.device_id,
                                          mode=self.mode)
        return vision_pipeline
