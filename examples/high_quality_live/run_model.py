# -*- coding: utf-8 -*-
import os, sys

try:
    import easyguard
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from easyguard.appzoo.authentic_modeling.model import AuthenticMM
from easyguard.appzoo.authentic_modeling.data import AuthenticDataModule

from cruise import CruiseTrainer, CruiseCLI

import pdb

cli = CruiseCLI(AuthenticMM,
                trainer_class=CruiseTrainer,
                datamodule_class=AuthenticDataModule,
                trainer_defaults={'summarize_model_depth': 3,})
# pdb.set_trace()
cfg, trainer, model, datamodule = cli.parse_args()

trainer.fit(model, datamodule)
