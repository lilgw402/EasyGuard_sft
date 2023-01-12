# -*- coding: utf-8 -*-

import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.scheduler import Scheduler
from timm.scheduler.step_lr import StepLRScheduler
from torch import optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import required


def build_optimizer(config, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, "no_weight_decay"):
        skip = model.no_weight_decay()
    if hasattr(model, "no_weight_decay_keywords"):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(model, skip, skip_keywords)

    opt_lower = config.optimizer.lower()
    optimizer = None
    if opt_lower == "sgd":
        optimizer = optim.SGD(
            parameters,
            momentum=config.momentum,
            nesterov=True,
            lr=config.base_lr,
            weight_decay=config.weight_decay,
        )
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(
            parameters,
            eps=config.optimizer_eps,
            betas=config.optimizer_betas,
            lr=config.base_lr,
            weight_decay=config.weight_decay,
        )

    return optimizer


def build_scheduler(config, optimizer, epochs, n_iter_per_epoch):
    num_steps = int(epochs * n_iter_per_epoch)
    warmup_steps = int(config.warmup_ratio * num_steps)
    decay_steps = int(config.lr_scheduler_decay_ratio * num_steps)

    lr_scheduler = None
    if config.lr_scheduler == "cosine":
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            # t_mul=1.,
            lr_min=config.min_lr,
            warmup_lr_init=config.warmup_lr,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    elif config.lr_scheduler == "linear":
        lr_scheduler = LinearLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min_rate=0.01,
            warmup_lr_init=config.warmup_lr,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    elif config.lr_scheduler == "step":
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=config.lr_scheduler_decay_rate,
            warmup_lr_init=config.warmup_lr,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )

    return lr_scheduler


class LinearLRScheduler(Scheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        t_initial: int,
        lr_min_rate: float,
        warmup_t=0,
        warmup_lr_init=0.0,
        t_in_epochs=True,
        noise_range_t=None,
        noise_pct=0.67,
        noise_std=1.0,
        noise_seed=42,
        initialize=True,
    ) -> None:
        super().__init__(
            optimizer,
            param_group_field="lr",
            noise_range_t=noise_range_t,
            noise_pct=noise_pct,
            noise_std=noise_std,
            noise_seed=noise_seed,
            initialize=initialize,
        )

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [
                (v - warmup_lr_init) / self.warmup_t for v in self.base_values
            ]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            total_t = self.t_initial - self.warmup_t
            lrs = [
                v - ((v - v * self.lr_min_rate) * (t / total_t))
                for v in self.base_values
            ]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (
            len(param.shape) == 1
            or name.endswith(".bias")
            or (name in skip_list)
            or check_keywords_in_name(name, skip_keywords)
        ):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{"params": has_decay}, {"params": no_decay, "weight_decay": 0.0}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
