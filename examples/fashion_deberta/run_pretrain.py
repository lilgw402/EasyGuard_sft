"""An customizable fashion_deberta example"""
import sys
import os

try:
    import easyguard
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from cruise import CruiseTrainer, CruiseCLI

from easyguard.appzoo.fashion_deberta.model import FashionDebertaModel
from easyguard.appzoo.fashion_deberta.data import FashionDataModule
from easyguard.utils.arguments import print_cfg

"""
fashion-deberta: 基于search-deberta，进行mlm + cls + cl 预训练

warning: 如果计算auc score，记得验证集里需要包含所有label，比如一共20个类别，那么验证集需要20个类别中每个类别都需要有数据

"""

cli = CruiseCLI(FashionDebertaModel,
                trainer_class=CruiseTrainer,
                datamodule_class=FashionDataModule,
                trainer_defaults={
                    'max_epochs': 6,
                    'val_check_interval': [10000, 1.0],
                    'summarize_model_depth': 2,
                    'checkpoint_monitor': 'val_loss',
                    'checkpoint_mode': 'min',
                    'precision': 'fp16',
                    'enable_versions': False,
                    'resume_ckpt_path': 'auto',
                    'default_root_dir': '/mnt/bd/yangzheming/cruise/cruise_logs',
                    'default_hdfs_dir': 'hdfs://haruna/home/byte_ecom_govern/user/yangzheming/chinese/common_model/trails/ccr_v3_1_live_0.3b_mlm_cl/model_outputs'
                    })
cfg, trainer, model, datamodule = cli.parse_args()
print_cfg(cfg)
trainer.fit(model, datamodule)