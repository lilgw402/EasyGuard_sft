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

from example.videoclip.model import VideoCLIPNet
from example.videoclip.create_loader import creat_dali_loader, creat_matx_loader

FLAGS = flags.FLAGS


def def_flags():
    flags.DEFINE_string(
        "config_path", './example/videoclip/config.yaml', "config_path")
    flags.DEFINE_string("output_path", "", "final ouput path")
    flags.DEFINE_string("mode", "matx", "loader mode")


def train(_):
    # 1. define model
    cfg.update_cfg(FLAGS.config_path)
    model = VideoCLIPNet(config=cfg)
    if cfg.network.partial_pretrain:
        load_from_pretrain(model, cfg.network.partial_pretrain,
                           cfg.network.partial_pretrain_prefix_changes)
    init_tracking(project='nlp.fex.videoclip', cfg=cfg)

    # 2. train && validation data
    if FLAGS.mode == "dali":
        train_loader, val_loader = creat_dali_loader(cfg)
    elif FLAGS.mode == "matx":
        train_loader, val_loader = creat_matx_loader(cfg)
    else:
        raise ValueError("Unkown loader mode: {}".format(FLAGS.mode))

    # 3. train model
    trainer = Trainer(cfg=cfg, output_path=FLAGS.output_path)
    trainer.fit(model=model,
                train_dataloader=train_loader,
                val_dataloader=val_loader)


if __name__ == "__main__":
    def_flags()
    app.run(train)
