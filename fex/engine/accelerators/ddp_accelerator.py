# -*- coding: utf-8 -*-
'''
Created on Nov-18-20 15:21
ddp_accelerator.py
@author: liuzhen.nlp
Description:
'''

import os
import random
from typing import Tuple, Union, Optional, Any
import numpy as np

import torch
import torch.distributed as distributed
from torch.optim import Optimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR

from fex.core.net import Net
from fex.engine.accelerators import Accelerator
from fex.utils.distributed import SubGroup


class DDPAccelerator(Accelerator):
    """
    DDPAccelerator 使用torch.nn.parallel.DistributedDataParallel进行分布式加速训练
    """

    def __init__(self, cfg, logger):
        super().__init__(cfg, logger)
        self.accelerator_rng_seed = self.cfg.TRAINER.RNG_SEED
        self.accelerator_syncbn = self.cfg.ACCELERATOR.SYNCBN
        self.nce_world_size = self.cfg.get('train.nce_world_size', -1)

    def set_up(self, model: Net, optimizer: Optimizer, lr_scheduler: LambdaLR,
               local_rank: int, world_size: int, rank: int) -> Tuple[DDP, Optimizer, LambdaLR]:
        """
        初始化 DDPAccelerator
        """
        torch.backends.cudnn.benchmark = False
        random.seed(self.accelerator_rng_seed)
        np.random.seed(self.accelerator_rng_seed)
        torch.random.manual_seed(self.accelerator_rng_seed)
        torch.cuda.manual_seed_all(self.accelerator_rng_seed)
        master_address = os.environ.get('MASTER_ADDR', "127.0.0.1")
        master_port = int(os.environ.get('MASTER_PORT', 34171))

        torch.cuda.set_device(local_rank)

        if self.accelerator_syncbn:
            model = self.configure_sync_batchnorm(model)

        model = model.cuda()
        if not torch.distributed.is_initialized():
            distributed.init_process_group(
                backend='nccl',
                init_method='tcp://{}:{}'.format(master_address, master_port),
                world_size=world_size,
                rank=rank,
                group_name='mtorch')
            self.logger.info(
                f'DDPAccelerator distributed, size: {world_size}, rank: {rank}, local rank: {local_rank}')
            if self.nce_world_size > 1 and self.nce_world_size < world_size:
                self.logger.info(f'[{rank}/{world_size}] Creating local process groups')
                SubGroup.init_sub_groups(rank, world_size, sub_world_size=self.nce_world_size, backend='nccl')

        self.broadcast(model)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

        return model, optimizer, lr_scheduler

    def broadcast(self, model: Net, src=0) -> None:
        """
        将model的参数做broadcast
        """
        for v in model.state_dict().values():
            distributed.broadcast(v, src)

    def configure_sync_batchnorm(self, model: Net) -> Net:
        """
        将model中的``torch.nn.modules.batchnorm._BatchNorm`` 转为 :class:`torch.nn.SyncBatchNorm`.
        """
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group=None)
        return model

    def backward_step(self, loss: torch.Tensor, optimizer: Optimizer):
        """
        backward step
        """
        loss.backward()

    def optimizer_step(self, optimizer: Optimizer, model: Net) -> float:
        """
        Gradient clipping
        """
        total_norm = 0
        if self.accelerator_clip_grad_norm > 0:
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.accelerator_clip_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        return float(total_norm)
