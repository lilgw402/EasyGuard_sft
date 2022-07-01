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

from example.classify.torchvision_dataset import creat_torchvision_loader
from example.classify.model import MMClassifier

FLAGS = flags.FLAGS


def def_flags():
    flags.DEFINE_string("config_path", './example/classify/config.yaml', "config_path")
    flags.DEFINE_string("output_path", "", "final ouput path")


def train(_):
    # 1. define model
    cfg.update_cfg(FLAGS.config_path)
    model = MMClassifier(config=cfg)
    if cfg.NETWORK.PARTIAL_PRETRAIN:
        load_from_pretrain(model, cfg.NETWORK.PARTIAL_PRETRAIN, cfg.NETWORK.PARTIAL_PRETRAIN_PREFIX_CHANGES)
    init_tracking(project='nlp.fex.classify', cfg=cfg)

    # 2. train && validation data
    train_loader, val_loader = creat_torchvision_loader(cfg)

    # 3. train model
    trainer = Trainer(cfg=cfg, output_path=FLAGS.output_path)

    trainer.fit(model=model,
                train_dataloader=train_loader,
                val_dataloader=val_loader)


if __name__ == "__main__":
    def_flags()
    app.run(train)
