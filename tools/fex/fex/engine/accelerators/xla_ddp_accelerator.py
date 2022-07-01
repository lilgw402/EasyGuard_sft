# -*- coding: utf-8 -*-
'''
Created on May-6-21 10:30
xla_ddp_accelerator.py
@author: chengji.yao
Description:
'''

import os
import random
from typing import Tuple, Union, Optional, Any
import numpy as np

import torch
import torch.distributed as distributed
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from fex.core.net import Net
from fex.engine.accelerators import Accelerator
from fex.utils.distributed import SubGroup


class XlaDDPAccelerator(Accelerator):
    def __init__(self, cfg, logger):
        super().__init__(cfg, logger)
        self.accelerator_rng_seed = self.cfg.TRAINER.RNG_SEED
        try:
            import torch_xla.core.xla_model as xm
            import torch_xla.core.xla_env_vars as xenv
            import torch_xla.distributed.xla_multiprocessing as xmp
            from torch_xla.amp import GradScaler
        except ImportError:
            print('no torch_xla! Please install it if necessary.')
        self.scaler = GradScaler()
        self.nce_world_size = self.cfg.get('train.nce_world_size', -1)

    def set_up(self, model: Net, optimizer: Optimizer, lr_scheduler: LambdaLR, local_rank: int,
               world_size: int, rank: int):
        torch.backends.cudnn.benchmark = False
        random.seed(self.accelerator_rng_seed)
        np.random.seed(self.accelerator_rng_seed)
        torch.random.manual_seed(self.accelerator_rng_seed)
        torch.cuda.manual_seed_all(self.accelerator_rng_seed)
        master_address = os.environ.get('MASTER_ADDR', "127.0.0.1")
        master_port = int(os.environ.get('MASTER_PORT', 34171))

        distributed.init_process_group(
            backend='gloo',
            init_method='tcp://{}:{}'.format(master_address, master_port),
            world_size=world_size,
            rank=rank,
            group_name='mtorch')
        if self.nce_world_size > 1 and self.nce_world_size < world_size:
            self.logger.info(f'[{rank}/{world_size}] Creating local process groups')
            SubGroup.init_sub_groups(rank, world_size, sub_world_size=self.nce_world_size, backend='gloo')
        rank = distributed.get_rank()
        os.environ[xenv.LOCAL_ORDINAL] = str(local_rank)
        os.environ[xenv.ORDINAL] = str(rank)
        os.environ[xenv.WORLD_SIZE] = str(world_size)
        os.environ[xenv.MP_DEVICE] = 'GPU:{}'.format(rank)
        os.environ[xenv.LOCAL_WORKER] = '{}:{}'.format("localservice", rank)
        os.environ["PYTORCH_JIT"] = "0"
        xmp._setup_replication()
        xla_device = xm.xla_device()
        print('Use device {}, its real device is {}'.format(xla_device, xm.xla_real_devices([xla_device])[0]))
        model.to(xla_device)
        self.broadcast(model)

        optimizer, lr_scheduler = model.configure_optimizers()

        return model, optimizer, lr_scheduler

    def broadcast(self, model: Net, src=0) -> None:
        """
        将model的参数做broadcast
        """
        tensors = list(model.state_dict().values())
        xm.all_reduce('sum', tensors, scale=1.0 / int(os.environ[xenv.WORLD_SIZE]))
        xm.mark_step()

    def scaler_step(self, optimizer: Optimizer):
        self.scaler.step(optimizer)
        self.scaler.update()
        return

    def backward_step(self, loss: torch.Tensor, optimizer: Optimizer):
        """
        backward step
        """
        if self.cfg.XLA.AMP:
            self.scaler.scale(loss).backward()
        else:
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
