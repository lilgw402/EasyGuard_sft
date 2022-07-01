# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import cv2

from fex.config import cfg
from fex.data import PytorchDaliIter
from example.videoclip.dataset import VideoDataset
from example.videoclip.dataset_matx import VideoDatasetMatx, worker_init_fn
from example.videoclip.dali_pipeline import ImagePipeline
from example.videoclip.matx_vision_pipeline import VisionGPUVideoFramesPipe, VisionFrameLoader

from absl import app
from absl import flags

FLAGS = flags.FLAGS


def def_flags():
    flags.DEFINE_string('config_path',
                        './example/videoclip/config.yaml',
                        'config_path')
    flags.DEFINE_bool('mode', True, 'Train Pipeline')


def create_dali_loader(config, flag):
    if not flag:
        print('------------> Validation pipeline <------------')
        val_dataset = VideoDataset(config,
                                   cfg.dataset.val_path,
                                   rank=int(os.environ.get('RANK') or 0),
                                   world_size=int(
                                       os.environ.get('WORLD_SIZE') or 1),
                                   shuffle=False,
                                   repeat=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.TRAINER.VAL_BATCH_SIZE,
                                                 num_workers=0,
                                                 pin_memory=True,
                                                 drop_last=False,
                                                 collate_fn=val_dataset.collect_fn)

        val_pipeline = ImagePipeline(batch_size=config.TRAINER.VAL_BATCH_SIZE * 8,
                                     num_threads=2, device_id=int(os.environ.get('LOCAL_RANK') or 0),
                                     external_data=val_loader,
                                     min_size=244,
                                     max_size=244)

        val_dali_iter = PytorchDaliIter(dali_pipeline=val_pipeline,
                                        output_map=["image"],
                                        auto_reset=True,
                                        last_batch_padded=True,
                                        fill_last_batch=False)
        return val_dali_iter
    else:
        print('------------> Train pipeline <------------')
        train_dataset = VideoDataset(config,
                                     cfg.dataset.train_path,
                                     rank=int(os.environ.get('RANK') or 0),
                                     world_size=int(
                                         os.environ.get('WORLD_SIZE') or 1),
                                     shuffle=False,
                                     repeat=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.TRAINER.TRAIN_BATCH_SIZE,
                                                   num_workers=0,
                                                   pin_memory=True,
                                                   drop_last=True,
                                                   collate_fn=train_dataset.collect_fn)

        train_pipeline = ImagePipeline(batch_size=config.TRAINER.TRAIN_BATCH_SIZE * 8,
                                       num_threads=2, device_id=int(os.environ.get('LOCAL_RANK') or 0),
                                       external_data=train_loader,
                                       min_size=244,
                                       max_size=244,
                                       prefetch_queue_depth=2
                                       )
        train_dali_iter = PytorchDaliIter(dali_pipeline=train_pipeline,
                                          output_map=["image"],
                                          auto_reset=True,
                                          last_batch_padded=True,
                                          fill_last_batch=False)

        return train_dali_iter


def create_matx_vision_loader(config, flag):
    if not flag:
        print('------------> Validation pipeline <------------')
        val_dataset = VideoDatasetMatx(config,
                                       cfg.dataset.val_path,
                                       rank=int(os.environ.get('RANK') or 0),
                                       world_size=int(
                                           os.environ.get('WORLD_SIZE') or 1),
                                       shuffle=False,
                                       repeat=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.TRAINER.VAL_BATCH_SIZE,
                                                 num_workers=1,
                                                 pin_memory=True,
                                                 drop_last=False,
                                                 collate_fn=val_dataset.collect_fn,
                                                 worker_init_fn=worker_init_fn)

        val_matx_vision_iter = VisionFrameLoader(data_iter=val_loader,
                                                 thread_num=6,
                                                 device_id=int(os.environ.get(
                                                     'LOCAL_RANK') or 0),
                                                 image_dsr_height=244,
                                                 image_dsr_width=244,
                                                 resize_longer=244,
                                                 normalize_mean=[
                                                     0.485 * 255, 0.456 * 255, 0.406 * 255],
                                                 normalize_std=[
                                                     0.229 * 255, 0.224 * 255, 0.225 * 255],
                                                 scale=[0.8, 1.0],
                                                 ratio=[0.8, 1.25],
                                                 mode="val",
                                                 output_map=["images"])
        return val_matx_vision_iter
    else:
        print('------------> Train pipeline <------------')
        train_dataset = VideoDatasetMatx(config,
                                         cfg.dataset.train_path,
                                         rank=int(os.environ.get('RANK') or 0),
                                         world_size=int(
                                             os.environ.get('WORLD_SIZE') or 1),
                                         shuffle=False,
                                         repeat=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.TRAINER.TRAIN_BATCH_SIZE,
                                                   num_workers=1,
                                                   pin_memory=True,
                                                   drop_last=True,
                                                   collate_fn=train_dataset.collect_fn,
                                                   worker_init_fn=worker_init_fn)

        train_matx_vision_iter = VisionFrameLoader(data_iter=train_loader,
                                                   thread_num=6,
                                                   device_id=int(os.environ.get(
                                                       'LOCAL_RANK') or 0),
                                                   image_dsr_height=244,
                                                   image_dsr_width=244,
                                                   resize_longer=244,
                                                   normalize_mean=[
                                                       0.485 * 255, 0.456 * 255, 0.406 * 255],
                                                   normalize_std=[
                                                       0.229 * 255, 0.224 * 255, 0.225 * 255],
                                                   scale=[0.8, 1.0],
                                                   ratio=[0.8, 1.25],
                                                   mode="train",
                                                   output_map=["images"])
        return train_matx_vision_iter


def run_matx_vision_pipeline(_):
    image_dirs = "./matx_vision_proprocess_images"
    if not os.path.exists(image_dirs):
        os.makedirs(image_dirs)

    cfg.update_cfg(FLAGS.config_path)
    val_matx_vision_iter = create_matx_vision_loader(cfg, FLAGS.mode)

    it = iter(val_matx_vision_iter)
    batch_res = next(it)
    images = batch_res["images"]

    mean = torch.tensor(
        np.array([0.485 * 255, 0.456 * 255, 0.406 * 255])[:, None, None]).cuda()
    std = torch.tensor(
        np.array([0.229 * 255, 0.224 * 255, 0.225 * 255])[:, None, None]).cuda()
    output_denormalize = (images * std + mean).int().cpu()
    print('-------> matx_vision image shape: ', output_denormalize.shape)
    output_denormalize = output_denormalize.permute([0, 2, 3, 1]).numpy()

    for i in range(output_denormalize.shape[0]):
        cv2.imwrite(
            "./{}/matx_vision_index_{}.jpeg".format(image_dirs, i), output_denormalize[i])


def run_dali_pipeline(_):
    image_dirs = "./dali_proprocess_images"
    if not os.path.exists(image_dirs):
        os.makedirs(image_dirs)

    cfg.update_cfg(FLAGS.config_path)
    val_dali_iter = create_dali_loader(cfg, FLAGS.mode)

    it = iter(val_dali_iter)
    batch_res = next(it)
    images = batch_res["image"]

    mean = torch.tensor(
        np.array([0.485 * 255, 0.456 * 255, 0.406 * 255])[:, None, None]).cuda()
    std = torch.tensor(
        np.array([0.229 * 255, 0.224 * 255, 0.225 * 255])[:, None, None]).cuda()
    output_denormalize = (images * std + mean).int().cpu()
    print('-------> dali image shape: ', output_denormalize.shape)
    output_denormalize = output_denormalize.permute([0, 2, 3, 1]).numpy()

    for i in range(output_denormalize.shape[0]):
        cv2.imwrite(
            "./{}/dali_index_{}.jpeg".format(image_dirs, i), output_denormalize[i])


if __name__ == "__main__":
    def_flags()
    # app.run(run_dali_pipeline)
    app.run(run_matx_vision_pipeline)
