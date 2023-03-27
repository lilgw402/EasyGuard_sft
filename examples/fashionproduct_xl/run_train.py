# -*- coding: utf-8 -*-
import os
import sys

print(f'00000000000')
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
try:
    import easyguard
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import random

import numpy as np
import torch
from cruise import CruiseCLI, CruiseTrainer
from examples.fashionproduct_xl.data import FacDataModule
from examples.fashionproduct_xl.model import FrameAlbertClassify

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
    FrameAlbertClassify,
    trainer_class=CruiseTrainer,
    datamodule_class=FacDataModule,
    trainer_defaults={
        "summarize_model_depth": 2,
    },
)
cfg, trainer, model, datamodule = cli.parse_args()

trainer.fit(model, datamodule)
