# -*- coding: utf-8 -*-
import os, sys

try:
    import easyguard
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from easyguard.appzoo.multimodal_modeling.model import FashionProduct
from easyguard.appzoo.multimodal_modeling.data import FPDataModule
from easyguard.utils.arguments import print_cfg

from cruise import CruiseTrainer, CruiseCLI

cli = CruiseCLI(FashionProduct,
                trainer_class=CruiseTrainer,
                datamodule_class=FPDataModule,
                trainer_defaults={
                    'log_every_n_steps': 100,
                    'precision': 'fp16',
                    'max_epochs': 2,
                    'enable_versions': True,
                    'val_check_interval': 500,  # val after 1 epoch
                    'limit_val_batches': 100,
                    'gradient_clip_val': 2.0,
                    'sync_batchnorm': True,
                    'find_unused_parameters': True,
                    'summarize_model_depth': 2,
                    'checkpoint_monitor': 'loss',
                    'checkpoint_mode': 'min',
                    'default_hdfs_dir': None
                })
cfg, trainer, model, datamodule = cli.parse_args()
print_cfg(cfg)
trainer.fit(model, datamodule)
