# -*- coding: utf-8 -*-
"""
train a image captioning model
"""

import os
from absl import flags
from absl import app

import torch

from fex.config import cfg
from fex.engine import Trainer
from fex.utils.track import init_tracking

from example.vision2text.igpt import IGPTNet
from example.vision2text.dataset import ImageCaptionDataset, get_transform

FLAGS = flags.FLAGS


def def_flags():
    flags.DEFINE_string("config_path", './example/vision2text/config.yaml', "config_path")
    flags.DEFINE_string("output_path", "v2t_model_output", "final ouput path")


def train(_):
    # 1. define model
    cfg.update_cfg(FLAGS.config_path)
    print(cfg)
    model = IGPTNet(config=cfg)
    # if cfg.NETWORK.PARTIAL_PRETRAIN:
    #     load_from_pretrain(model, cfg.NETWORK.PARTIAL_PRETRAIN, cfg.NETWORK.PARTIAL_PRETRAIN_PREFIX_CHANGES)
    init_tracking(project='nlp.fex.vision2text', cfg=cfg)

    # 2. train && validation data
    train_loader, val_loader = creat_loader(cfg)

    # 3. train model
    trainer = Trainer(cfg=cfg, output_path=FLAGS.output_path)

    trainer.fit(model=model,
                train_dataloader=train_loader,
                val_dataloader=val_loader)


def creat_loader(config):
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    train_transform = get_transform(mode="train")
    val_transform = get_transform(mode="val")

    train_dataset = ImageCaptionDataset(
        config,
        cfg.DATASET.TRAIN_PATH,
        transform=train_transform,
        rank=int(os.environ.get('RANK') or 0),
        world_size=int(os.environ.get('WORLD_SIZE') or 1),
        shuffle=True,
        repeat=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.TRAINER.TRAIN_BATCH_SIZE,
                                               num_workers=config.get('train.num_workers', 4),
                                               pin_memory=True,
                                               drop_last=True,
                                               collate_fn=train_dataset.collect_fn)

    val_dataset = ImageCaptionDataset(
        config,
        config.DATASET.VAL_PATH,
        transform=val_transform,
        rank=int(os.environ.get('RANK') or 0),
        world_size=int(os.environ.get('WORLD_SIZE') or 1),
        shuffle=False,
        repeat=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.TRAINER.VAL_BATCH_SIZE,
                                             num_workers=config.VAL.NUM_WORKERS,
                                             pin_memory=True,
                                             drop_last=False,
                                             collate_fn=val_dataset.collect_fn)

    return train_loader, val_loader


if __name__ == "__main__":
    def_flags()
    app.run(train)
