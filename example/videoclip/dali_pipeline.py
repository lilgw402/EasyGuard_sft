# -*- coding: utf-8 -*-
'''
dali preprocess pipeline
'''
import random
import os
import numpy as np
import torch
import multiprocessing.dummy as mp

try:
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
except ImportError:
    raise ImportError(
        "Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")


class ImagePipeline(Pipeline):
    def __init__(self,
                 batch_size,
                 device_id,
                 external_data,
                 min_size,
                 max_size,
                 is_training=False,
                 seed=42,
                 dali_cpu=False,
                 num_threads=2,
                 prefetch_queue_depth=2):
        """
        预处理的pipeline。最简单的几个事情：
        1. decode
          (如果是training的话，带random crop)
        2. resize
        3. crop & normalize
        """
        super(ImagePipeline, self).__init__(
            batch_size, num_threads, device_id, seed=seed, prefetch_queue_depth=prefetch_queue_depth)
        self.external_data = external_data
        self.data_iter = iter(self.external_data)
        self.input_raw_image = ops.ExternalSource(device="cpu")
        self.queue = mp.Queue()

        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'

        # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
        # without additional reallocations
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        if is_training:
            self.decode = ops.ImageDecoderRandomCrop(device=decoder_device, output_type=types.RGB,
                                                     device_memory_padding=device_memory_padding,
                                                     host_memory_padding=host_memory_padding,
                                                     random_aspect_ratio=[0.8, 1.25],
                                                     random_area=[0.8, 1.0],
                                                     num_attempts=100)
        else:
            self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)

        self.res = ops.Resize(device="gpu",
                              resize_longer=max_size,
                              interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop_w=min_size,
                                            crop_h=max_size,
                                            crop_pos_x=0.5,
                                            crop_pos_y=0.5,
                                            mean=[0.485 * 255, 0.456 *
                                                  255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                                            out_of_bounds_policy="pad")

    def define_graph(self):
        self.raw_image = self.input_raw_image()
        image = self.decode(self.raw_image)
        image = self.res(image)
        image = self.cmnp(image)
        return image

    def iter_setup(self):
        try:
            data_dict = self.data_iter.__next__()
            self.feed_input(self.raw_image, data_dict.pop('image'))
            self.queue.put(data_dict)
        except StopIteration:
            self.data_iter = iter(self.external_data)
            raise StopIteration

    @property
    def batch_size(self):
        # forward compatibility
        return self._max_batch_size
