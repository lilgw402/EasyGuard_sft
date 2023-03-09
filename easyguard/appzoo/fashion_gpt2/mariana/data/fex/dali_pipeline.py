# -*- coding: utf-8 -*-
'''
Created on Nov-24-20 16:04
dali_pipline.py
@author: liuzhen.nlp
Description:
'''
import numpy as np
import random
import multiprocessing.dummy as mp
try:
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    Pipeline = object
    # raise ImportError(
    #     "Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")


class TrainImageDecoderPipeline(Pipeline):
    def __init__(
            self,
            batch_size,
            num_threads,
            device_id,
            external_data,
            crop,
            seed=100,
            dali_cpu=False,
            prefetch_queue_depth=2,
            need_text=False,
            need_temb=False,
            is_flip=None,
            rotate_angle=None,
            random_area_min=None,
            need_multi=False):
        if Pipeline is object:
            raise ImportError(
                "Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")
        super(
            TrainImageDecoderPipeline,
            self).__init__(
            batch_size,
            num_threads,
            device_id,
            seed=seed,
            prefetch_queue_depth=prefetch_queue_depth)
        self.external_data = external_data
        self.data_iter = iter(self.external_data)
        self.input_raw_images = ops.ExternalSource(device="cpu")
        self.queue = mp.Queue()

        is_flip = True if is_flip is None else is_flip
        rotate_angle = 0 if rotate_angle is None else rotate_angle
        random_area_min = 0.1 if random_area_min is None else random_area_min
        print(f'is_flip={is_flip}')
        print(f'rotate_angle={rotate_angle}')
        print(f'random_area_min={random_area_min}')
        self.is_flip = is_flip

        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'

        # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
        # without additional reallocations
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0

        self.decode = ops.decoders.Image(device=decoder_device,
                                         output_type=types.RGB,
                                         device_memory_padding=device_memory_padding,
                                         host_memory_padding=host_memory_padding)
        self.rotate = ops.Rotate(device=dali_device)
        self.rotate_angle = ops.random.Uniform(
            range=(-rotate_angle, rotate_angle)) if rotate_angle != 0 else None
        self.random_crop = ops.RandomResizedCrop(device=dali_device,
                                                 size=(crop, crop),
                                                 interp_type=types.INTERP_TRIANGULAR,
                                                 random_aspect_ratio=[0.8, 1.25],
                                                 random_area=[random_area_min, 1.0],
                                                 num_attempts=100)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

        self.coin = ops.random.CoinFlip(probability=0.5)

    @property
    def batch_size(self):
        # forward compatibility
        return self._max_batch_size

    def define_graph(self):

        rng = self.coin() if self.is_flip else 0
        self.raw_images = self.input_raw_images()
        images = self.decode(self.raw_images)

        if self.rotate_angle is not None:
            images = self.rotate(images, angle=self.rotate_angle())
        images = self.random_crop(images)
        images = self.cmnp(images, mirror=rng)
        return images

    def iter_setup(self):
        try:
            data_dict = next(self.data_iter)
            images = data_dict.pop('image')
            padding_size = self._max_batch_size - len(images)
            for _ in range(padding_size):
                images.append(random.choice(images))
            self.feed_input(self.raw_images, images)
            self.queue.put(data_dict)

        except StopIteration:
            self.data_iter = iter(self.external_data)
            raise StopIteration


class ValImageDecoderPipeline(Pipeline):
    def __init__(
            self,
            batch_size,
            num_threads,
            device_id,
            external_data,
            crop,
            size,
            seed=100,
            dali_cpu=False,
            need_text=False,
            need_temb=False):
        super(ValImageDecoderPipeline, self).__init__(
            batch_size, num_threads, device_id, seed=seed)
        self.external_data = external_data
        self.data_iter = iter(self.external_data)
        self.input_raw_images = ops.ExternalSource(device="cpu")
        self.queue = mp.Queue()

        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'

        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu",
                              resize_shorter=size,
                              interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            mean=[0.485 * 255, 0.456 *
                                                  255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    @property
    def batch_size(self):
        # forward compatibility
        return self._max_batch_size

    def define_graph(self):
        self.raw_images = self.input_raw_images()
        images = self.decode(self.raw_images)
        images = self.res(images)
        images = self.cmnp(images)
        return images

    def iter_setup(self):
        try:
            data_dict = next(self.data_iter)
            images = data_dict.pop("image")
            padding_size = self.batch_size - len(images)
            for _ in range(padding_size):
                images.append(random.choice(images))
            self.feed_input(self.raw_images, images)
            self.queue.put(data_dict)

        except StopIteration:
            self.data_iter = iter(self.external_data)
            raise StopIteration
