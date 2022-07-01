"""
使用ptx 的 amp  和 torch ddp 来做加速。
这里有两个好处：
1. ptx 的 amp 可以支持 O2，这样可以去掉 apex 的依赖，同时
2. ptx 的 amp 可以支持 module 粒度的决定 fp32 或 fp16，可以更动态的自主配置。

https://bytedance.feishu.cn/docs/doccnqGeVjrPkrBiBK9xSrW4vAQ#TntBCq

"""

from typing import Tuple, Union, Optional, Any

import torch
from torch.optim import Optimizer
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP

from fex.engine.accelerators import DDPAccelerator
from fex.core.net import Net
from fex import _logger as log

try:
    from ptx.train.fp16 import LossScaler, scale_loss as fp16_scale_loss, initialize as fp16_initialize, configure_fp16_optimizer, master_params
    from ptx.amp import amp_convert_network
except Exception as e:
    log.warning('ptx is not found, PtxAMPDDPAccelerator is not usable. You can install ptx from https://code.byted.org/nlp/ptx')


class PtxAMPDDPAccelerator(DDPAccelerator):
    def __init__(self, cfg, logger):
        """
        TODO: 这个版本对 batchnorm 并没有做处理，原始apex 是会做一些处理的，但处理过于复杂，这个版本没有支持。
        所以这个训练有batchnorm 的模型可能会有点问题。慎用，不保证无坑。
        """
        super().__init__(cfg, logger)

    def set_up(self, model: Net, optimizer: Optimizer, lr_scheduler: LambdaLR,
               local_rank: int, world_size: int, rank: int) -> Tuple[DDP, Optimizer, LambdaLR]:
        """
        初始化 DDPAccelerator，
        先做一堆 ddp 的处理，然后再做 ptx amp的处理
        """
        model = amp_convert_network(model, torch.float16, hack_forward=True)
        model, optimizer, lr_scheduler = super().set_up(model, optimizer, lr_scheduler, local_rank, world_size, rank)

        fp16_initialize()
        optimizer = configure_fp16_optimizer(optimizer)
        self.scaler = LossScaler('dynamic')

        # 对model 做一个 precast，手动将里面的模块转换成fp 16的计算，这样的好处是可以支持 O2，而且是动态的支持 O2。
        return model, optimizer, lr_scheduler

    def backward_step(self, loss: torch.Tensor, optimizer: Optimizer):
        """
        backward step
        """
        with fp16_scale_loss(loss, optimizer, self.scaler) as scaled_loss:
            scaled_loss.backward()

    def optimizer_step(self, optimizer: Optimizer, model: Net) -> float:
        """
        optimizer step,
        1. Gradient clipping (if has)
        """
        total_norm = 0
        if self.accelerator_clip_grad_norm > 0:
            params = master_params(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(params, self.accelerator_clip_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        return float(total_norm)

    def get_metrics(self):
        return {'loss_scale_0': self.scaler.loss_scale()}
