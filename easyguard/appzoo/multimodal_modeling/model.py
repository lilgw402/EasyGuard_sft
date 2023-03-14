# -*- coding: utf-8 -*-

import json
import math
import os
from collections import OrderedDict
from types import SimpleNamespace
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import _LRScheduler

from ptx.model import Model

try:
    import cruise
except ImportError:
    print(
        "[ERROR] cruise is not installed! Please refer this doc: https://bytedance.feishu.cn/wiki/wikcnGP7yzZAuKpPfL6jRJKl2ag"
    )

from easyguard.modelzoo.models.falbert.albert import ALBert
from easyguard.modelzoo.models.falbert.swin import SwinTransformer

from cruise import CruiseModule
from cruise.utilities.cloud_io import load
from cruise.utilities.hdfs_io import hexists, hopen

from ...core import AutoModel
from ...modelzoo.modeling_utils import ModelBase
from ...utils.losses import LearnableNTXentLoss, LearnablePCLLoss, SCELoss


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    optimizer (Optimizer): Wrapped optimizer.
    first_cycle_steps (int): First cycle step size.
    cycle_mult(float): Cycle steps magnification. Default: -1.
    max_lr(float): First cycle's max learning rate. Default: 0.1.
    min_lr(float): Min learning rate. Default: 0.001.
    warmup_steps(int): Linear warmup step size. Default: 0.
    gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
    last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(
            optimizer, last_epoch
        )

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps
                + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                + (self.max_lr - base_lr)
                * (
                    1
                    + math.cos(
                        math.pi
                        * (self.step_in_cycle - self.warmup_steps)
                        / (self.cur_cycle_steps - self.warmup_steps)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = (
                    int(
                        (self.cur_cycle_steps - self.warmup_steps)
                        * self.cycle_mult
                    )
                    + self.warmup_steps
                )
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(
                        math.log(
                            (
                                epoch
                                / self.first_cycle_steps
                                * (self.cycle_mult - 1)
                                + 1
                            ),
                            self.cycle_mult,
                        )
                    )
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps
                        * (self.cycle_mult**n - 1)
                        / (self.cycle_mult - 1)
                    )
                    self.cur_cycle_steps = (
                        self.first_cycle_steps * self.cycle_mult ** (n)
                    )
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class FashionBertv2(CruiseModule):
    def __init__(
        self,
        learning_rate: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.02,
    ):
        super(FashionBertv2, self).__init__()
        self.save_hparams()

    def setup(self, stage) -> None:
        self.model = AutoModel.from_pretrained("fashionbert-base")

        """
        Initialize loss
        """
        self.calc_nce_loss_vt = LearnableNTXentLoss(
            init_tau=self.model.config_fusion.init_tau,
            clamp=self.model.config_fusion.tau_clamp,
        )  # 底层图-文CLIP损失
        self.calc_pcl_loss_ff = LearnablePCLLoss(
            init_tau=self.model.config_fusion.init_tau,
            clamp=self.model.config_fusion.tau_clamp,
            num_labels=self.model.config_fusion.category_logits_level1 + 1,
        )  # 上层融合-融合CLIP损失，使用一级标签
        self.category_pred_loss = SCELoss(
            alpha=1.0,
            beta=0.5,
            num_classes=self.model.config_fusion.category_logits_level2 + 1,
        )  # 融合表征的预测损失

        self.fuse_category = nn.Sequential(
            nn.Linear(
                self.config_fusion.hidden_size,
                self.config_fusion.hidden_size * 2,
            ),
            nn.GELU(),
            nn.Dropout(self.config_fusion.embd_pdrop),
            nn.Linear(
                self.config_fusion.hidden_size * 2,
                self.config_fusion.category_logits_level2 + 1,
            ),
        )

    def cal_pt_loss(self, **kwargs):
        for key in ["t_rep", "v_rep", "fuse_rep", "label_l1"]:
            if key in kwargs:
                kwargs[key] = self.all_gather(kwargs[key].contiguous())
                kwargs[key] = kwargs[key].flatten(0, 1)

        vt_loss = self.calc_nce_loss_vt.forward(t_emb=kwargs["t_rep"], v_emb=kwargs["v_rep"])

        assert "label_l1" in kwargs
        ff_loss = self.calc_pcl_loss_ff.forward(f_emb=kwargs["fuse_rep"], label=kwargs["label_l1"])

        assert "label" in kwargs and "fuse_cat" in kwargs
        cat_loss = self.category_pred_loss.forward(pred=kwargs["fuse_cat"], labels=kwargs["label"])

        loss = (vt_loss + ff_loss + cat_loss) / 3

        return {
            "loss": loss,
            "vt_loss": vt_loss,
            "ff_loss": ff_loss,
            "cat_loss": cat_loss,
        }

    def training_step(self, batch, idx):
        token_ids, image, label, label_l1 = (
            batch["token_ids"],
            batch["image"],
            batch["label"],
            batch["label_l1"],
        )
        rep_dict = self.model(token_ids, image)
        rep_dict.update(batch)
        rep_dict['fuse_cat'] = self.fuse_category(rep_dict['fuse_emb'][:, 0])  # [B, num_categories]

        loss_dict = self.cal_pt_loss(**rep_dict)

        return loss_dict

    def validation_step(self, batch, idx):
        return self.training_step(batch, idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=self.hparams.betas,
            eps=self.hparams.eps,
            weight_decay=self.hparams.weight_decay,
        )

        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.trainer.total_steps,
            cycle_mult=1.0,
            max_lr=self.hparams.learning_rate,
            min_lr=0,
            warmup_steps=2000,
        )

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
