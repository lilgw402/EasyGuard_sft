# -*- coding: utf-8 -*-

from easyguard.appzoo.language_modeling.model import LanguageModel
from easyguard.appzoo.language_modeling.data import LMDataModule

from cruise import CruiseTrainer, CruiseCLI

cli = CruiseCLI(LanguageModel,
                trainer_class=CruiseTrainer,
                datamodule_class=LMDataModule,
                trainer_defaults={
                    'log_every_n_steps': 50,
                    'precision': 'fp16',
                    'max_epochs': 1,
                    'enable_versions': True,
                    'val_check_interval': 1.0,  # val after 1 epoch
                    'limit_val_batches': 100,
                    'gradient_clip_val': 2.0,
                    'sync_batchnorm': True,
                    'find_unused_parameters': True,
                    'summarize_model_depth': 2,
                    'checkpoint_monitor': 'loss',
                    'checkpoint_mode': 'min',
                    'default_hdfs_dir': 'hdfs://harunasg/home/byte_magellan_govern/users/jiangjunjun.happy/xlmr14' # use your own path to save model
                }
                )
cfg, trainer, model, datamodule = cli.parse_args()
trainer.fit(model, datamodule)
