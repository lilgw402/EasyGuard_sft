# -*- coding: utf-8 -*-
'''
train a clip
'''
import os
import collections
from absl import flags
from absl import app
import torch

from fex import _logger as log
from fex.config import cfg
from fex.engine import Trainer
from fex.data import KVSampler, worker_init_fn, KVDataset
from fex.utils.hdfs_io import hmget, hglob
from fex.utils.load import load_from_pretrain
from fex.utils.track import init_tracking

from example.clip.model import CLIPNet
from example.clip.create_loader import create_torchvision_loader, create_dali_loader, create_bytedvision_loader

FLAGS = flags.FLAGS


def def_flags():
    flags.DEFINE_string("config_path", './example/clip/config.yaml', "config_path")
    flags.DEFINE_string("output_path", "", "final ouput path")
    flags.DEFINE_string("dataset_type", "torchvision", "preprocess mode")
    flags.DEFINE_string("update", "", "update config by string, like `TRAINER.TRAIN_BATCH_SIZE=500,TRAINER.END_EPOCH=32` ")


def train(_):
    # 1. define model
    cfg.update_cfg(FLAGS.config_path, FLAGS.update)
    print(cfg)
    model = CLIPNet(config=cfg)
    if cfg.network.partial_pretrain:
        load_from_pretrain(model, cfg.network.partial_pretrain, cfg.network.partial_pretrain_prefix_changes)
    init_tracking(project='nlp.fex.clip', cfg=cfg)

    # 2. train && validation data
    if FLAGS.dataset_type == 'torchvision':
        train_loader, val_loader = create_torchvision_loader(cfg)
    elif FLAGS.dataset_type == 'dali':
        train_loader, val_loader = create_dali_loader(cfg)
    elif FLAGS.dataset_type == 'bytedvision':
        train_loader, val_loader = create_bytedvision_loader(cfg)

    # 3. train model
    trainer = Trainer(cfg=cfg, output_path=FLAGS.output_path, resume=True)
    trainer.fit(model=model,
                train_dataloader=train_loader,
                val_dataloader=val_loader)


if __name__ == "__main__":
    def_flags()
    app.run(train)
