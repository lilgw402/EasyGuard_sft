# -*- coding: utf-8 -*-

import os
from absl import flags
from absl import app
import torch

from fex.config import cfg
from fex.engine import Trainer
from fex.utils.track import init_tracking
from example.image_classify.imagenet_json_dataset import ImageNetJsonDataset, get_transform
from example.image_classify.imagenet_json_dataset_matx import ImageNetJsonMatxVisionDataset
from example.image_classify.vision_pipline import VisionImageLoader
from example.image_classify.model import ResNetClassifier

FLAGS = flags.FLAGS


def def_flags():
    flags.DEFINE_string(
        "config_path", 'example/image_classify/resnet_json.yaml', "config_path")
    flags.DEFINE_string("output_path", "", "final ouput path")
    flags.DEFINE_string("dataset_type", "json",
                        "dataset type, can be [matx, json]")


def creat_default_loader(config):
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)

    train_transform = get_transform(mode="train")
    val_transform = get_transform(mode="val")

    train_dataset = ImageNetJsonDataset(config.DATASET.TRAIN_PATH,
                                        transform=train_transform,
                                        rank=int(os.environ.get('RANK') or 0),
                                        world_size=int(
                                            os.environ.get('WORLD_SIZE') or 1),
                                        shuffle=True,
                                        repeat=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.TRAINER.TRAIN_BATCH_SIZE,
                                               num_workers=config.TRAIN.NUM_WORKERS,
                                               pin_memory=True,
                                               drop_last=True,
                                               collate_fn=train_dataset.collect_fn)

    val_dataset = ImageNetJsonDataset(config.DATASET.VAL_PATH,
                                      transform=val_transform,
                                      rank=int(os.environ.get('RANK') or 0),
                                      world_size=int(
                                          os.environ.get('WORLD_SIZE') or 1),
                                      shuffle=False,
                                      repeat=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config.TRAINER.VAL_BATCH_SIZE,
                                             num_workers=config.VAL.NUM_WORKERS,
                                             pin_memory=True,
                                             drop_last=False,
                                             collate_fn=val_dataset.collect_fn)
    return train_loader, val_loader


def creat_matx_loader(config):
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)

    train_dataset = ImageNetJsonMatxVisionDataset(config.DATASET.TRAIN_PATH,
                                                  rank=int(
                                                      os.environ.get('RANK') or 0),
                                                  world_size=int(
                                                      os.environ.get('WORLD_SIZE') or 1),
                                                  shuffle=True,
                                                  repeat=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.TRAINER.TRAIN_BATCH_SIZE,
                                               num_workers=config.TRAIN.NUM_WORKERS,
                                               pin_memory=True,
                                               drop_last=True,
                                               collate_fn=train_dataset.collect_fn)
    train_matx_vision_iter = VisionImageLoader(data_iter=train_loader,
                                               thread_num=6,
                                               device_id=int(os.environ.get(
                                                   'LOCAL_RANK') or 0),
                                               image_dsr_height=224,
                                               image_dsr_width=224,
                                               resize_shorter=256,
                                               normalize_mean=[
                                                   0.485 * 255, 0.456 * 255, 0.406 * 255],
                                               normalize_std=[
                                                   0.229 * 255, 0.224 * 255, 0.225 * 255],
                                               scale=[0.1, 1.0],
                                               ratio=[0.8, 1.25],
                                               mode="train",
                                               output_map=["image"])

    val_dataset = ImageNetJsonMatxVisionDataset(config.DATASET.VAL_PATH,
                                                rank=int(
                                                    os.environ.get('RANK') or 0),
                                                world_size=int(
                                                    os.environ.get('WORLD_SIZE') or 1),
                                                shuffle=False,
                                                repeat=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=config.TRAINER.VAL_BATCH_SIZE,
                                             num_workers=config.VAL.NUM_WORKERS,
                                             pin_memory=True,
                                             drop_last=False,
                                             collate_fn=val_dataset.collect_fn)
    val_matx_vision_iter = VisionImageLoader(data_iter=val_loader,
                                             thread_num=6,
                                             device_id=int(os.environ.get(
                                                 'LOCAL_RANK') or 0),
                                             image_dsr_height=224,
                                             image_dsr_width=224,
                                             resize_shorter=256,
                                             normalize_mean=[
                                                 0.485 * 255, 0.456 * 255, 0.406 * 255],
                                             normalize_std=[
                                                 0.229 * 255, 0.224 * 255, 0.225 * 255],
                                             scale=[0.1, 1.0],
                                             ratio=[0.8, 1.25],
                                             mode="val",
                                             output_map=["image"])

    return train_matx_vision_iter, val_matx_vision_iter


def train(_):
    # 1. define model
    cfg.update_cfg(FLAGS.config_path)
    model = ResNetClassifier(config=cfg)
    init_tracking(project='nlp.fex.image_classify', cfg=cfg)

    # 2. train && validation data
    if FLAGS.dataset_type == 'json':
        train_loader, val_loader = creat_default_loader(cfg)
    elif FLAGS.dataset_type == 'matx':
        train_loader, val_loader = creat_matx_loader(cfg)
    else:
        train_loader, val_loader = creat_default_loader(cfg)

    # 3. train model
    trainer = Trainer(
        cfg=cfg, output_path=FLAGS.output_path)

    trainer.fit(model=model,
                train_dataloader=train_loader,
                val_dataloader=val_loader)


if __name__ == "__main__":
    def_flags()
    app.run(train)
