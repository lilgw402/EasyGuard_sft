# -*- coding: utf-8 -*-
'''
Created on May-10-21 21:09
byted_vision_cpu_transform.py
Description: 实现CPU相关的vision transform
'''
import random
import math

from typing import List, Tuple

import matx
import byted_vision
from byted_vision import CenterCropOp, CustomCropOp, CvtColorOp, \
    ImdecodeOp, ResizeOp, FlipOp, RotateOp, CopyMakeBorderOp


class Pad:
    def __init__(self, sizes: List[int]) -> None:
        self.dst_sizes = sizes
        self.opencv_padding_op = CopyMakeBorderOp()

    def __call__(self, image: matx.NDArray) -> matx.NDArray:
        dst_width, dst_height = self.dst_sizes[0], self.dst_sizes[1]
        shape: List[int] = image.shape()
        src_height: int = shape[0]
        src_width: int = shape[1]

        left_pad = 0
        right_pad = 0
        top_pad = 0
        bottom_pad = 0

        if dst_height == src_height and dst_width == src_width:
            return image

        if dst_width > src_width:
            # w没对齐, 左右两边pad
            left_pad = math.floor((dst_width - src_width) / 2)
            right_pad = dst_width - src_width - left_pad

        if dst_height > src_height:
            # h没对齐, 上下两边pad
            top_pad = math.floor((dst_height - src_height) / 2)
            bottom_pad = dst_height - src_height - top_pad

        padding_image = self.opencv_padding_op(image, top_pad, bottom_pad,
                                               left_pad, right_pad,
                                               byted_vision.BORDER_CONSTANT, 0)
        return padding_image


class NormalizeAndTranspose:
    def __init__(self, mean: List[float], std: List[float]) -> None:
        self.mean: List[float] = mean
        self.std: List[float] = std

    def __call__(self, image: matx.NDArray) -> matx.NDArray:
        transpose_image = image.transpose([2, 0, 1])  # convert to CHW
        float32_image = transpose_image.as_type("float32")

        nd_mean = matx.NDArray(self.mean, [3, 1, 1], float32_image.dtype())
        nd_std = matx.NDArray(self.std, [3, 1, 1], float32_image.dtype())
        nd_image_sub = matx.nd_sub(float32_image, nd_mean)
        return matx.nd_div(nd_image_sub, nd_std)


class RateResizeAndRotate:
    def __init__(self, sizes: List[int], need_transpose: bool) -> None:
        self.resize_sizes = sizes
        self.need_transpose = need_transpose
        self.opencv_resize_op = ResizeOp()
        self.opencv_rotate_op = RotateOp()

    def __call__(self, image: matx.NDArray) -> matx.NDArray:
        shape: List[int] = image.shape()
        src_height: int = shape[0]
        src_width: int = shape[1]
        # 对图片是否做翻转
        if (self.need_transpose and src_height < src_width):
            image = self.opencv_rotate_op(
                image, byted_vision.ROTATE_90_CLOCKWISE)

        # Convention (w, h)
        size_w, size_h = self.resize_sizes[0], self.resize_sizes[1]
        adjust_width = 0
        adjust_height = 0

        ratio_src = src_width / src_height
        ratio_dst = size_w / size_h

        if ratio_src > ratio_dst:
            # 原始的w更长，按住w，放缩h
            adjust_width = size_w
            adjust_height = math.floor((size_w / src_width) * src_height)
        else:
            # 原始的h更长，按住h，放缩w
            adjust_height = size_h
            adjust_width = math.floor((size_h / src_height) * src_width)

        resize_image = self.opencv_resize_op(
            image, [adjust_width, adjust_height], 0, 0, byted_vision.INTER_LINEAR)
        return resize_image


class RandomResizedCrop:
    def __init__(self, size: int) -> None:
        self.crop = CustomCropOp()
        self.resize = ResizeOp()
        self.size = size
        self.scale = [0.08, 1.0]
        self.ratio = [3. / 4., 4. / 3.]

    def _get_params(self,
                    image: matx.NDArray,
                    scale: Tuple[float, float],
                    ratio: Tuple[float, float]) -> Tuple[int, int, int, int]:
        shape: List[int] = image.shape()
        org_height: int = shape[0]
        org_width: int = shape[1]
        org_area = org_height * org_width

        for _ in range(10):
            new_area = org_area * random.uniform(*scale)
            log_ratio = [math.log(item_ratio) for item_ratio in ratio]
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            new_height = int(round(math.sqrt(new_area * aspect_ratio)))
            new_width = int(round(math.sqrt(new_area / aspect_ratio)))

            if 0 < new_width <= org_width and 0 < new_height <= org_height:
                i = random.randint(0, org_width - new_width + 1)
                j = random.randint(0, org_height - new_height + 1)
                # 保证i + new_width <= org_width \ j + new_height <= org_height
                if (i + new_width) > org_width:
                    new_width = org_width - i
                if (j + new_height) > org_height:
                    new_height = org_height - j
                return i, j, new_width, new_height

        # fallback to center crop
        in_ratio = float(org_width) / float(org_height)
        if in_ratio < min(ratio):
            new_width = org_width
            new_height = int(round(new_width / min(ratio)))
        elif in_ratio > max(ratio):
            new_height = org_height
            new_width = int(round(new_height * max(ratio)))
        else:  # whole image
            new_width = org_width
            new_height = org_height
        i = (org_width - new_width) // 2
        j = (org_height - new_height) // 2

        # 保证i + new_width <= org_width \ j + new_height <= org_height
        if (i + new_width) > org_width:
            new_width = org_width - i
        if (j + new_height) > org_height:
            new_height = org_height - j

        return i, j, new_width, new_height

    def __call__(self, image: matx.NDArray) -> matx.NDArray:
        i, j, w, h = self._get_params(image, self.scale, self.ratio)
        image = self.crop(image, i, j, w, h)
        image = self.resize(
            image, (self.size, self.size), 0, 0, byted_vision.INTER_LINEAR)
        return image


class Resize:
    def __init__(self, sizes: int) -> None:
        self.resize_sizes = sizes
        self.resize = ResizeOp()

    def __call__(self, image: matx.NDArray) -> matx.NDArray:
        # Convention (h, w)
        size_h, size_w = self.resize_sizes, self.resize_sizes

        shape: List[int] = image.shape()
        src_height: int = shape[0]
        src_width: int = shape[1]

        if src_width < src_height:
            size_h = int(size_w * src_height / src_width)
        else:
            size_w = int(size_h * src_width / src_height)

        if (src_width <= src_height and src_width == size_h) or (src_height <= src_width and src_height == size_h):
            return image

        image = self.resize(
            image, (size_w, size_h), 0, 0, byted_vision.INTER_LINEAR)
        return image


class CenterCrop:
    def __init__(self, size: int) -> None:
        self.size = size
        self.center_crop = CenterCropOp()

    def __call__(self, image: matx.NDArray) -> matx.NDArray:
        image = self.center_crop(image, [self.size, self.size])
        return image


class RandomHorizontalFlip:
    def __init__(self) -> None:
        self.p = 0.5
        self.flip = FlipOp()

    def __call__(self, image: matx.NDArray) -> matx.NDArray:
        if random.random() < self.p:
            return self.flip(image, 1)
        return image
