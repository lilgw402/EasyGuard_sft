# -*- coding: utf-8 -*-

import sys
import os

import torch
import torch.nn as nn

try:
    import cruise
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from cruise import CruiseTrainer, CruiseModule, CruiseCLI
from easyguard.appzoo.deberta_classification.data import DebertaDataModule
from easyguard.appzoo.deberta_classification.model import DebertaModel

cli = CruiseCLI(DebertaModel,
                trainer_class=CruiseTrainer,
                datamodule_class=DebertaDataModule,
                trainer_defaults={
                    'max_epochs': 3,
                    'val_check_interval': [2000, 1.0],
                    'summarize_model_depth': 2,
                    'checkpoint_monitor': 'loss',
                    'checkpoint_mode': 'min',
                    'default_hdfs_dir': 'hdfs://haruna/user/tianke/train_model',
                    'default_root_dir': '/opt/tiger/tianke',
                    'resume_ckpt_path': None
                })
cfg, trainer, model, datamodule = cli.parse_args()
# 训练模型
# trainer.fit(model, datamodule)
# 预测得分
datamodule.local_rank_zero_prepare()
datamodule.setup('predict')
model.partial_load_from_checkpoints('/opt/tiger/tianke/checkpoints/best_checkpoint/epoch=2-step=140625_success.ckpt', rename_params={})
# trainer.predict(model, datamodule.predict_dataloader(), sync_predictions=False)
# trace模型
trainer.trace(model, datamodule.predict_dataloader(), mode='jit')