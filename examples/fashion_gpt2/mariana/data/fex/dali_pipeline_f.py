# -*- coding: utf-8 -*-
'''
Created on Nov-24-20 16:04
dali_pipline.py
@author: liuzhen.nlp
Description:
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
    Pipeline = object


class TrainFrameDecoderQueuePipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, external_data,
                 min_size, max_size,
                 seed=100, dali_cpu=False, prefetch_queue_depth=2):
        if Pipeline is object:
            raise ImportError(
                "Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")
        super(TrainFrameDecoderQueuePipeline, self).__init__(
            batch_size, num_threads, device_id, seed=seed, prefetch_queue_depth=prefetch_queue_depth)
        self.external_data = external_data
        self.data_iter = iter(self.external_data)
        frame_num = 8
        self.input_raw_images = [ops.ExternalSource(device="cpu") for _ in range(frame_num)]
        self.queue = mp.Queue()

        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'

        # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
        # without additional reallocations
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        self.decode = ops.ImageDecoderRandomCrop(device=decoder_device, output_type=types.RGB,
                                                 device_memory_padding=device_memory_padding,
                                                 host_memory_padding=host_memory_padding,
                                                 random_aspect_ratio=[
                                                     0.8, 1.25],
                                                 random_area=[0.1, 1.0],
                                                 num_attempts=100)
        self.min_size = min_size
        self.max_size = max_size
        self.res = ops.Resize(device=dali_device,
                              resize_shorter=self.min_size,
                              interp_type=types.INTERP_TRIANGULAR)

        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop_w=self.min_size,
                                            crop_h=self.max_size,
                                            crop_pos_x=0.5,
                                            crop_pos_y=0.5,
                                            mean=[0.485 * 255, 0.456 *
                                                  255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                                            out_of_bounds_policy="pad")
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        rng = self.coin()
        self.raw_images = [i() for i in self.input_raw_images]
        images = self.decode(self.raw_images)
        images = self.res(images)
        images = self.cmnp(images, mirror=rng)
        images = fn.stack(*images)
        return images

    def iter_setup(self):
        try:
            data_dict = self.data_iter.next()
            for i, inp in enumerate(self.raw_images):
                self.feed_input(inp, data_dict.pop('image%s' % i))
            self.queue.put(data_dict)

        except StopIteration:
            self.data_iter = iter(self.external_data)
            raise StopIteration

    @property
    def batch_size(self):
        # forward compatibility
        return self._max_batch_size

class ValFrameDecoderQueuePipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, external_data, min_size, max_size, seed=100, dali_cpu=False,
                 ):
        super(ValFrameDecoderQueuePipeline, self).__init__(
            batch_size, num_threads, device_id, seed=seed)
        self.external_data = external_data
        self.data_iter = iter(self.external_data)
        frame_num = 8
        self.input_raw_images = [ops.ExternalSource(device="cpu") for _ in range(frame_num)]
        self.queue = mp.Queue()

        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'

        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu",
                              resize_shorter=min_size,
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
        self.raw_images = [i() for i in self.input_raw_images]
        images = self.decode(self.raw_images)
        images = self.res(images)
        images = self.cmnp(images)
        images = fn.stack(*images)
        return images

    def iter_setup(self):
        try:
            data_dict = self.data_iter.next()
            for i, inp in enumerate(self.raw_images):
                self.feed_input(inp, data_dict.pop('image%s' % i))
            self.queue.put(data_dict)

        except StopIteration:
            self.data_iter = iter(self.external_data)
            raise StopIteration

    @property
    def batch_size(self):
        # forward compatibility
        return self._max_batch_size
