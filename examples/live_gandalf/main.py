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
    # Extra comand line option
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--val", action="store_true")
    parser.add_argument("--trace", action="store_true")
    args, config_override = parser.parse_known_args()
    cruise_args = [arg for arg in sys.argv[1:] if len(re.findall('fit|val|trace',arg))==0]
    sys.argv = [sys.argv[0]]+cruise_args
    if args.config:
        config = Dict(load_from_yaml(args.config))
    else:
        raise FileNotFoundError('config file option must be specified')
    config.update({'fit':args.fit,'val':args.val, 'trace':args.trace})
    return Dict(config)

def prepare_trainer_components(config):
    model_module = get_model_module(config['model']['type'])
    data_module = get_data_module(config['data']['type'])
    return model_module, data_module

def main():
    config = prepare_gandalf_args()
    model_module, data_module = prepare_trainer_components(config)
    cli = CruiseCLI(model_module,data_module,trainer_class=CruiseTrainer)
    cfg, trainer, model, datamodule = cli.parse_args()
    print_cfg(cfg)
    if config['fit']:
        trainer.fit(model, datamodule=datamodule)
    if config['val']:
        trainer.validate(model, datamodule=datamodule)
    if config['trace']:
        trainer.trace(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
