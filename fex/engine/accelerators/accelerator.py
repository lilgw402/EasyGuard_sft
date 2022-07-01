# -*- coding: utf-8 -*-
'''
Created on Feb-19-21 16:36
accelerator.py
Description: accelerators的基类，便于后续其他加速方案的接入。
'''

from logging import Logger

import torch
from torch.optim import Optimizer

from fex.config import CfgNode
from fex.core.net import Net


class Accelerator:
    """
    Accelerator是所有accelerators的基类，新添加的accelerator需要继承该类。
    """

    def __init__(self, cfg: CfgNode, logger: Logger) -> None:
        self.cfg = cfg
        self.logger = logger
        self.accelerator_clip_grad_norm = float(self.cfg.ACCELERATOR.CLIP_GRAD_NORM)

    def set_up(self, model: Net):
        raise NotImplementedError("Set Up method not implement in Accelerator, please check! ")

    def broadcast(self):
        raise NotImplementedError("Broadcast method not implement in Accelerator, please check! ")

    def backward_step(self, loss: torch.Tensor):
        loss.backward()

    def optimizer_step(self, optimizer: Optimizer, model: Net) -> float:
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                    self.accelerator_clip_grad_norm)
        return float(total_norm)

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

    def get_metrics(self):
        return {}
