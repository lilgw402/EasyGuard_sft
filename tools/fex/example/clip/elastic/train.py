# -*- coding: utf-8 -*-
'''
train a clip
'''
import logging
import os
from absl import flags
from absl import app
import math

from fex.config import cfg
from fex.utils.checkpointer import Checkpointer
from fex.utils.load import load_from_pretrain
from fex.utils.track import init_tracking

from example.clip.model import CLIPNet
from example.clip.elastic.create_loader import create_torchvision_loader, create_dali_loader, create_bytedvision_loader
from example.clip.elastic.elastic_trainer import ElasticTrainer

FLAGS = flags.FLAGS


def def_flags():
    flags.DEFINE_string("config_path", './example/clip/config.yaml', "config_path")
    flags.DEFINE_string("output_path", "", "final ouput path")
    flags.DEFINE_string("dataset_type", "torchvision", "preprocess mode")
    flags.DEFINE_string("bs_mode", "fix_local", "choose from {fix_local, fix_global, auto}")


def get_batch_size_fixed_global_bs(world_size):
    global_bs = 640 * 32
    return math.ceil(global_bs / world_size)


def get_batch_size(world_size):
    bs_map = {
        8: 768,
        16: 672,
        32: 640,
        64: 400,
        128: 256,
        256: 150,
        512: 80,   # not tested
    }
    if world_size <= 8:
        return bs_map[8]

    for upper_bound in sorted(bs_map.keys()):
        if world_size <= upper_bound:
            break
    else:
        raise RuntimeError(f'world size {world_size} is too large!')

    x2 = upper_bound
    x1 = upper_bound // 2
    bs = bs_map[x1] * (world_size / x1) ** math.log2(bs_map[x2] / bs_map[x1])
    return round(bs)


def train(_):
    try:
        # 1. define model
        cfg.update_cfg(FLAGS.config_path)
        if cfg.get('train.nce_world_size', -1) == -1:
            # if nce allgather is enabled, select an optimial batch size
            world_size = int(os.getenv('WORLD_SIZE') or 1)
            if FLAGS.bs_mode == 'fix_global':
                batch_size = get_batch_size_fixed_global_bs(world_size)
            elif FLAGS.bs_mode == 'fix_local':
                batch_size = cfg.TRAINER.TRAIN_BATCH_SIZE
            elif FLAGS.bs_mode == 'auto':
                batch_size = get_batch_size(world_size)
            else:
                raise RuntimeError('Unknown batch size mode')
            print(f'Setting batch size to {batch_size} for world size {world_size}')
            cfg.TRAINER.TRAIN_BATCH_SIZE = batch_size
        model = CLIPNet(config=cfg)
        if cfg.network.partial_pretrain:
            load_from_pretrain(model, cfg.network.partial_pretrain, cfg.network.partial_pretrain_prefix_changes)
        init_tracking(project='nlp.fex.clip', cfg=cfg)

        # 2. train && validation data
        latest_checkpoint = Checkpointer.find_latest_checkpoint_from_dir(FLAGS.output_path)
        if latest_checkpoint is not None:
            _, training_state_path = latest_checkpoint
        else:
            training_state_path = None
        if FLAGS.dataset_type == 'torchvision':
            train_loader, val_loader = create_torchvision_loader(cfg, training_state_path)
        elif FLAGS.dataset_type == 'dali':
            train_loader, val_loader = create_dali_loader(cfg, training_state_path)
        elif FLAGS.dataset_type == 'byted_vision':
            train_loader, val_loader = create_bytedvision_loader(cfg, training_state_path)

        # 3. train model
        trainer = ElasticTrainer(cfg=cfg, output_path=FLAGS.output_path, resume=True)
        trainer.fit(model=model,
                    train_dataloader=train_loader,
                    val_dataloader=val_loader)
    except:
        logging.exception('TRAIN ERROR')
        os.abort()


if __name__ == "__main__":
    def_flags()
    app.run(train)
