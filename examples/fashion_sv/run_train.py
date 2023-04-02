# -*- coding: utf-8 -*-
import os
import sys

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
try:
    import easyguard
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import random

import numpy as np
import torch
from cruise import CruiseCLI, CruiseTrainer
from examples.fashion_sv.data import SVDataModule
from examples.fashion_sv.sv_model import FashionSV

rand_seed = 42


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    try:
        torch.backends.cudnn.deterministic = True
    except:
        ...


# random seed
setup_seed(rand_seed)
cli = CruiseCLI(
    FashionSV,
    trainer_class=CruiseTrainer,
    datamodule_class=SVDataModule,
    trainer_defaults={
        "summarize_model_depth": 2,
    },
)
cfg, trainer, model, datamodule = cli.parse_args()

trainer.fit(model, datamodule)
