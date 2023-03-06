# -*- coding:utf-8 -*-
# Email:    jiangxubin@bytedance.com
# Created:  2023-03-01 12:52:19
# Modified: 2023-03-01 12:52:19
import os
import re
import warnings
import argparse
import logging
from addict import Dict
from cruise import CruiseTrainer, CruiseCLI
from cruise.utilities.rank_zero import rank_zero_info
from cruise.trainer.logger import TensorBoardLogger
from cruise.trainer.logger.tracking import TrackingLogger
from easyguard.utils.arguments import print_cfg
from utils.registry import get_model,get_data_module
from utils.config import config
from utils.driver import reset_logger, get_logger, init_env,init_device, DIST_CONTEXT
from utils.util import load_conf, load_from_yaml,load_from_tcc,load_from_bbc,check_config,update_config,init_seeds
from utils.file_util import hmkdir, check_hdfs_exist
from pytorch_lightning.cli import LightningArgumentParser,LightningCLI
from pytorch_lightning.demos.boring_classes import DemoModel,BoringDataModule


def pl_cli_main():
    cli = LightningCLI(DemoModel,BoringDataModule,save_config_overwrite=True,save_config_filename='/mlx_devbox/users/jiangxubin/repo/121/EasyGuard/examples/live_gandalf/pl.yaml')

def cruise_cli_main():
    from dataset import GandalfCruiseDataModule
    from models import EcomLiveGandalfAutoDisNNAsrCruiseModel
    cli = CruiseCLI(EcomLiveGandalfAutoDisNNAsrCruiseModel,GandalfCruiseDataModule,trainer_class=CruiseTrainer)
    cfg, trainer, model, datamodule = cli.parse_args()
    print_cfg(cfg)
    trainer.fit(model, datamodule=datamodule)
    # similarly can manually call val, test, predict
    # trainer.validate(model, datamodule.val_dataloader())

if __name__ == "__main__":
    # pl_cli_main()
    cruise_cli_main()
    import torch
    from easyguard.core import AutoModel,AutoTokenizer
    asr_model_name = "deberta_base_6l"
    asr_encoder = AutoModel.from_pretrained(asr_model_name)
    from dataset.transforms.text_transforms.DebertaTokenizer import DebertaTokenizer
    # # asr_tokenizer = AutoTokenizer.from_pretrained(asr_model_name, padding='max_length', truncation=True,return_tensors="pt", max_length=20)
    asr_tokenizer = DebertaTokenizer('./models/weights/fashion_deberta_asr/deberta_3l/vocab.txt',max_len=20)
    inputs = asr_tokenizer('哈哈aaaa思思思思思思都是的师傅啦啦好卡啦')
    print(inputs,type(inputs),inputs['input_ids'].shape)
    print(torch.unsqueeze(inputs['token_type_ids'],0).shape)
    output = asr_encoder(torch.unsqueeze(inputs['input_ids'],0),torch.unsqueeze(inputs['attention_mask'],0),torch.unsqueeze(inputs['token_type_ids'],0))
    print(type(output),output)
    
