# -*- coding:utf-8 -*-
# Email:    jiangxubin@bytedance.com
# Created:  2023-03-01 12:52:19
# Modified: 2023-03-01 12:52:19
import os
import re
import sys
import argparse
from addict import Dict
from cruise import CruiseTrainer, CruiseCLI
from easyguard.utils.arguments import print_cfg
from utils.config import config
from utils.registry import get_model_module,get_data_module
from utils.util import load_conf, load_from_yaml,load_from_tcc,load_from_bbc,check_config,update_config,init_seeds
from utils.file_util import hmkdir, check_hdfs_exist

def prepare_folder(config):
    # print(config.trainer)
    os.makedirs(config.trainer.default_root_dir, exist_ok=True)
    if not check_hdfs_exist(config.trainer.default_hdfs_dir):
            hmkdir(config.trainer.default_hdfs_dir)

def prepare_gandalf_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="local yaml conf")
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--val", action="store_true")
    parser.add_argument("--trace", action="store_true")
    # origin_args = copy.deepcopy(sys.argv[1:])
    args, config_override = parser.parse_known_args()
    # Remove custom args for the compatability with cruise args
    for idx,arg in enumerate(sys.argv):
        if arg in ['--fit','--val','--trace']:
            sys.argv.pop(idx)
    if args.config:
        config = Dict(load_from_yaml(args.config))
    # update_config(config,config_override)
    config.update({'fit':args.fit,'val':args.val, 'trace':args.trace})
    # raise KeyError('One of train/val/trace mode must be specified')
    return Dict(config)

def prepare_trainer_components(config):
    model_module = get_model_module(config['model']['type'])
    data_module = get_data_module(config['data']['type'])
    return model_module, data_module

def cruise_cli_main():
    config = prepare_gandalf_args()
    # prepare_folder(config)
    model_module, data_module = prepare_trainer_components(config)
    cli = CruiseCLI(model_module,data_module,trainer_class=CruiseTrainer)
    cfg, trainer, model, datamodule = cli.parse_args()
    print_cfg(cfg)
    if config['fit']:
        trainer.fit(model, datamodule=datamodule)
    if config['val']:
        trainer.validate(model, datamodule=datamodule)
        # trainer.validate(model, datamodule.val_dataloader())
    if config['trace']:
        trainer.trace(model, datamodule=datamodule)

def main():
    from dataset import EcomLiveGandalfParquetAutoDisCruiseDataModule
    from models import EcomLiveGandalfAutoDisNNAsrCruiseModel
    cli = CruiseCLI(EcomLiveGandalfAutoDisNNAsrCruiseModel,EcomLiveGandalfParquetAutoDisCruiseDataModule,trainer_class=CruiseTrainer)
    cfg, trainer, model, datamodule = cli.parse_args()
    print_cfg(cfg)
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    # main()
    cruise_cli_main()
