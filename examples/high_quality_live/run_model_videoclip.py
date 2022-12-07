# -*- coding: utf-8 -*-
import os, sys

try:
    import easyguard
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from easyguard.appzoo.high_quality_live.model_videoclip import HighQualityLiveVideoCLIP
from easyguard.appzoo.high_quality_live.data import HighQualityLiveDataModule

from cruise import CruiseTrainer, CruiseCLI

import pdb

cli = CruiseCLI(HighQualityLiveVideoCLIP,
                trainer_class=CruiseTrainer,
                datamodule_class=HighQualityLiveDataModule,
                trainer_defaults={'summarize_model_depth': 3,})
# pdb.set_trace()
cfg, trainer, model, datamodule = cli.parse_args()

trainer.fit(model, datamodule)
