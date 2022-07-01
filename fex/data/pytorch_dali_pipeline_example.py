# -*- coding: utf-8 -*-
'''
Created on Feb-04-21 16:23
pytorch_dali_pipeline_example.py
@author: liuzhen.nlp
Description: 通过PytochDaliIter创建dali_pipline example
'''
import os
import torch
import multiprocessing.dummy as mp

try:
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    print("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

from fex.config import cfg
from fex.data import PytorchDaliIter
from tasks.image_classify.model import NET_MAP
from tasks.image_classify.data import DATASET_MAP


class TrainImageDecoderPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, external_data, crop, seed=100, dali_cpu=False, prefetch_queue_depth=2):
        super(TrainImageDecoderPipeline, self).__init__(
            batch_size, num_threads, device_id, seed=seed, prefetch_queue_depth=prefetch_queue_depth)
        self.external_data = external_data
        self.data_iter = iter(self.external_data)
        self.input_raw_images = ops.ExternalSource(device="cpu")
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
        self.res = ops.Resize(device=dali_device,
                              resize_x=crop,
                              resize_y=crop,
                              interp_type=types.INTERP_TRIANGULAR)

        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            mean=[0.485 * 255, 0.456 *
                                                  255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        rng = self.coin()
        self.raw_images = self.input_raw_images()

        images = self.decode(self.raw_images)
        images = self.res(images)
        images = self.cmnp(images, mirror=rng)

        return images

    def iter_setup(self):
        try:
            data_dict = next(self.data_iter)
            images = data_dict.pop("image")
            self.queue.put(data_dict)

            self.feed_input(self.raw_images, images)

        except StopIteration:
            self.data_iter = iter(self.external_data)
            raise StopIteration


def creat_json_dali_loader(config):
    train_dataset = DATASET_MAP[config.DATASET.DATASET](
        config,
        cfg.DATASET.TRAIN_PATH,
        rank=int(os.environ.get('RANK') or 0),
        world_size=int(os.environ.get('WORLD_SIZE') or 1),
        shuffle=False,
        repeat=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=8,
                                               num_workers=1,
                                               pin_memory=True,
                                               drop_last=True,
                                               collate_fn=train_dataset.collect_fn)

    train_pipeline = TrainImageDecoderPipeline(batch_size=8,
                                               num_threads=4,
                                               device_id=int(
                                                   os.environ.get('LOCAL_RANK') or 0),
                                               external_data=train_loader,
                                               crop=224,
                                               prefetch_queue_depth=4)

    train_dali_iter = PytorchDaliIter(dali_pipeline=train_pipeline,
                                      output_map=config.DATASET.MODEL_INPUT,
                                      auto_reset=True, last_batch_padded=True, fill_last_batch=False)

    return train_dali_iter


if __name__ == "__main__":

    cfg.update_cfg("./config/imagenet/resnet_emb_json_new_config.yaml")
    train_loader = creat_json_dali_loader(config=cfg)
    data_iter = iter(train_loader)
    print('-----> dali labels: ', data_iter.__next__())
