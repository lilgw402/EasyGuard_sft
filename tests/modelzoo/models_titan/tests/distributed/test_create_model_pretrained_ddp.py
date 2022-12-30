import os
import torch
import datetime
from typing import Union, Optional, Any, Tuple
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import titan
from titan import TOSHelper

tos_helper = TOSHelper(bucket='titan-modelzoo-public', access_key='BW7H90QZ6H7YR0U92WWM')


def distributed_available() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


class _DistEnv:
    def __init__(self) -> None:
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.num_nodes = int(self.world_size // self.local_world_size)
        self.node_rank = int(self.rank // self.local_world_size)

    def barrier(self) -> None:
        if self.world_size < 1:
            return
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        if torch.distributed.get_backend() == "nccl":
            torch.distributed.barrier(device_ids=[self.local_rank])
        else:
            torch.distributed.barrier()

    def init_process_group(self) -> None:
        if self.world_size > 1 and not torch.distributed.is_initialized():
            timeout_seconds = int(os.environ.get("CRS_NCCL_TIMEOUT_SECOND", 1800))
            torch.distributed.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=timeout_seconds))

DIST_ENV = _DistEnv()


def test_load_pretrained_ddp():
    DIST_ENV.init_process_group()
    if not distributed_available():
        print('DDP not available')
        return
    net = titan.create_model('resnet50', pretrained=True, tos_helper=tos_helper)
    net.to(torch.device(f'cuda:{DIST_ENV.local_rank}'))
    model = DDP(net, device_ids=[DIST_ENV.local_rank], find_unused_parameters=False)
    with torch.enable_grad():
        for _ in range(5):
            x = torch.rand(2, 3, 224, 224)
            y = model(x)
            loss = y.mean() * 0
            loss.backward()


if __name__ == '__main__':
    test_load_pretrained_ddp()
