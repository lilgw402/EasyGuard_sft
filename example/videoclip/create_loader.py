"""
build video dataloader in a function
"""

import os
import torch

from fex.data import PytorchDaliIter

from example.videoclip.dataset import VideoDataset
from example.videoclip.dali_pipeline import ImagePipeline
from example.videoclip.dataset_matx import VideoDatasetMatx, worker_init_fn
from example.videoclip.matx_vision_pipeline import VisionFrameLoader


def creat_dali_loader(config):
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    train_dataset = VideoDataset(
        config,
        config.dataset.train_path,
        rank=int(os.environ.get('RANK') or 0),
        world_size=int(os.environ.get('WORLD_SIZE') or 1),
        shuffle=True,
        repeat=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAINER.TRAIN_BATCH_SIZE,
        num_workers=config.train.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=train_dataset.collect_fn)

    # TODO: 这里一些预处理的参数写死了，后面做成可配置的
    frame_num = 8
    image_w = 224
    image_h = 224
    train_pipeline = ImagePipeline(batch_size=config.TRAINER.TRAIN_BATCH_SIZE * frame_num,
                                   num_threads=2, device_id=int(os.environ.get('LOCAL_RANK') or 0),
                                   external_data=train_loader,
                                   min_size=image_w,
                                   max_size=image_h,
                                   prefetch_queue_depth=2,
                                   is_training=True
                                   )
    train_dali_iter = PytorchDaliIter(dali_pipeline=train_pipeline,
                                      output_map=["images"],
                                      auto_reset=True,
                                      last_batch_padded=True,
                                      fill_last_batch=False)

    val_dataset = VideoDataset(
        config,
        config.dataset.val_path,
        rank=int(os.environ.get('RANK') or 0),
        world_size=int(os.environ.get('WORLD_SIZE') or 1),
        shuffle=False,
        repeat=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.TRAINER.VAL_BATCH_SIZE,
                                             num_workers=config.val.num_workers,
                                             pin_memory=True,
                                             drop_last=False,
                                             collate_fn=val_dataset.collect_fn)

    val_pipeline = ImagePipeline(batch_size=config.TRAINER.VAL_BATCH_SIZE * frame_num,
                                 num_threads=2, device_id=int(os.environ.get('LOCAL_RANK') or 0),
                                 external_data=val_loader,
                                 min_size=224,
                                 max_size=224,
                                 is_training=False
                                 )
    val_dali_iter = PytorchDaliIter(dali_pipeline=val_pipeline,
                                    output_map=["images"],
                                    auto_reset=True,
                                    last_batch_padded=True,
                                    fill_last_batch=False)

    return train_dali_iter, val_dali_iter


def creat_matx_loader(config):
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    train_dataset = VideoDatasetMatx(config,
                                     config.dataset.train_path,
                                     rank=int(os.environ.get('RANK') or 0),
                                     world_size=int(
                                         os.environ.get('WORLD_SIZE') or 1),
                                     shuffle=True,
                                     repeat=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.TRAINER.TRAIN_BATCH_SIZE,
                                               num_workers=config.train.num_workers,
                                               pin_memory=True,
                                               drop_last=True,
                                               collate_fn=train_dataset.collect_fn,
                                               worker_init_fn=worker_init_fn)

    # TODO: 这里一些预处理的参数写死了，后面做成可配置的
    image_w = 224
    image_h = 224

    train_matx_vision_iter = VisionFrameLoader(data_iter=train_loader,
                                               thread_num=6,
                                               device_id=int(os.environ.get(
                                                   'LOCAL_RANK') or 0),
                                               image_dsr_height=image_h,
                                               image_dsr_width=image_w,
                                               resize_longer=image_h,
                                               normalize_mean=[
                                                   0.485 * 255, 0.456 * 255, 0.406 * 255],
                                               normalize_std=[
                                                   0.229 * 255, 0.224 * 255, 0.225 * 255],
                                               scale=[0.8, 1.0],
                                               ratio=[0.8, 1.25],
                                               mode="train",
                                               output_map=["images"])

    val_dataset = VideoDatasetMatx(config,
                                   config.dataset.val_path,
                                   rank=int(os.environ.get('RANK') or 0),
                                   world_size=int(
                                       os.environ.get('WORLD_SIZE') or 1),
                                   shuffle=False,
                                   repeat=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.TRAINER.VAL_BATCH_SIZE,
                                             num_workers=config.val.num_workers,
                                             pin_memory=True,
                                             drop_last=False,
                                             collate_fn=val_dataset.collect_fn,
                                             worker_init_fn=worker_init_fn)
    val_matx_vision_iter = VisionFrameLoader(data_iter=val_loader,
                                             thread_num=6,
                                             device_id=int(os.environ.get(
                                                 'LOCAL_RANK') or 0),
                                             image_dsr_height=image_h,
                                             image_dsr_width=image_w,
                                             resize_longer=image_h,
                                             normalize_mean=[
                                                 0.485 * 255, 0.456 * 255, 0.406 * 255],
                                             normalize_std=[
                                                 0.229 * 255, 0.224 * 255, 0.225 * 255],
                                             scale=[0.8, 1.0],
                                             ratio=[0.8, 1.25],
                                             mode="val",
                                             output_map=["images"])
    return train_matx_vision_iter, val_matx_vision_iter
