# -*- coding: utf-8 -*-
print(f'==================================0')
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    import easyguard
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import random
print(f'==================================01')
import numpy as np
import torch
from cruise import CruiseCLI, CruiseTrainer
print(f'==================================02')
from examples.fashionproduct_xl.data import FacDataModule
print(f'==================================03')
from examples.fashionproduct_xl.model import FrameAlbertClassify
print(f'==================================04')
rand_seed = 42

print(f'==================================1')
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
print(f'==================================2')
cli = CruiseCLI(
    FrameAlbertClassify,
    trainer_class=CruiseTrainer,
    datamodule_class=FacDataModule,
    trainer_defaults={
        "summarize_model_depth": 2,
    },
)
print(f'==================================3')
cfg, trainer, model, datamodule = cli.parse_args()
print(f'==================================4')
trainer.fit(model, datamodule)
