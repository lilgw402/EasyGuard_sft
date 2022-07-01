# -*- coding: utf-8 -*-

import os
from absl import flags
from absl import app
import torch
import logging

from fex.config import cfg
from fex.engine import Trainer
from fex.data.benchmark.imagenet import get_imagenet_dataloader
from fex.utils.distributed import rank_zero_info
from fex.utils.track import init_tracking

from example.imagenet_benchmark.model import ResNetClassifier
from example.imagenet_benchmark.benchmark import run_benchmark

FLAGS = flags.FLAGS


def def_flags():
    flags.DEFINE_string(
        "config_path", 'example/imagenet_benchmark/resnet_json.yaml', "config_path")
    flags.DEFINE_string("output_path", "", "final ouput path")
    flags.DEFINE_bool('do_train', True, 'training')
    flags.DEFINE_bool('do_eval', True, 'evaluate after training')
    flags.DEFINE_string("update", "", "update config by string, like `TRAINER.TRAIN_BATCH_SIZE=500,TRAINER.END_EPOCH=32` ")


def train(_):
    try:
        # 1. define model
        cfg.update_cfg(FLAGS.config_path, FLAGS.update)
        model = ResNetClassifier(config=cfg)

        # 2. training
        if FLAGS.do_train:
            init_tracking(project='nlp.fex.imagenet_benchmark', cfg=cfg)

            rank_zero_info(' === start training ! === ')
            # 2.1 train && validation data
            val_loader = get_imagenet_dataloader(
                data_path=cfg.data.val.path,
                batch_size=512,
                source=cfg.data.val.get('source', 'hdfs'),
                preprocess_type=cfg.data.val.get('preprocess_type', 'bytedvision'),
                mode='val'
            )
            train_loader = get_imagenet_dataloader(
                data_path=cfg.data.train.path,
                batch_size=cfg.TRAINER.TRAIN_BATCH_SIZE,
                source=cfg.data.train.get('source', 'hdfs'),
                preprocess_type=cfg.data.train.get('preprocess_type', 'bytedvision'),
                mode='train'
            )
            # 2.2 train model
            trainer = Trainer(
                cfg=cfg, output_path=FLAGS.output_path, resume=True)

            trainer.fit(model=model,
                        train_dataloader=train_loader,
                        val_dataloader=val_loader)

        # 3. evaluation
        rank = int(os.environ.get('RANK') or 0)
        if FLAGS.do_eval and rank == 0:
            rank_zero_info(' === start evaluation ! === ')
            run_benchmark(model)

    except:
        logging.exception('TRAIN ERROR')
        os.abort()
        #os.killpg(0, signal.SIGKILL)


if __name__ == "__main__":
    def_flags()
    app.run(train)
