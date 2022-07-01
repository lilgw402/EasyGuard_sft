import os
from typing import Dict, List, Union
import warnings

import torch
from torch import distributed as dist
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from fex import _logger as logger
from fex.config import CfgNode
from fex.core.net import Net
from fex.data import PytorchDaliIter
from fex.engine.trainer.common_trainer import Trainer

try:
    import tracking as tk
    TRACKING_FOUND = True
except ImportError:
    warnings.warn('Tracking not found. Please install by: pip install byted-tracking')
    TRACKING_FOUND = False


class ElasticTrainer(Trainer):
    def __init__(self, cfg: CfgNode, output_path: str = "./train_fex", resume: bool = False):
        super().__init__(cfg, output_path, resume)
        self.data_offsets = {}

    def train_start(self, model: Net, train_dataloader: DataLoader, val_dataloader: DataLoader = None) -> Union[Net, DataLoader]:
        """
        训练开始前置工作
        """
        # TODO: hack
        self.train_dataset = None
        if isinstance(train_dataloader, DataLoader):
            self.train_dataset = train_dataloader.dataset
        elif isinstance(train_dataloader, PytorchDaliIter):
            self.train_dataset = train_dataloader.dali_pipeline.external_data.dataset

        if self.train_dataset is not None:
            self.data_offsets = getattr(self.train_dataset, 'data_offsets', None) or {}
        else:
            logger.warning('Unknown train loader type. Will not resume data offsets!')

        return super().train_start(model, train_dataloader, val_dataloader)

    def train_epoch_end(self, model, *args, **kwargs):
        # reset data offsets
        if self.train_dataset is not None:
            self.train_dataset.data_offsets = None
        self.data_offsets = {}
        super().train_epoch_end(model, *args, **kwargs)

    def train_step_start(self, batch: Dict[str, Union[torch.Tensor, List]],
                         epoch: int, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        做一个train step 的前置工作
        如：
        将数据移到cuda，
        """
        batch_data_offsets = batch.pop('data_offsets', {})
        self.data_offsets.update(batch_data_offsets)
        return super().train_step_start(batch, epoch, batch_idx)

    def train_step_optimize(self, batch_idx: int, model: Net) -> None:
        """
        做一个train step的优化过程
        如：
        将更多的事情放在 self.accelerator.optimizer_step 里做
        更新参数等
        """
        if self.writer and batch_idx % self.log_frequent == 0:
            for n, v in model.named_parameters():
                if v.grad is not None:
                    # TODO: add to tracking add_scalar
                    self.writer.add_scalar(tag=f'WeightGradient/{n}',
                                           scalar_value=torch.abs(v.grad).mean(),
                                           global_step=self.global_step)

        if self.global_step % self.accelerator_gradient_accumulate_steps == 0:
            total_norm = self.accelerator.optimizer_step(self.optimizer, model)
            if total_norm:
                if self.writer and batch_idx % self.log_frequent == 0:
                    self.add_scalar(tag='Trainer/Grad-Norm',
                                    scalar_value=total_norm,
                                    global_step=self.global_step)
        if self.lr_scheduler is not None and not isinstance(
                self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_scheduler.step(self.actual_step)

    def save_checkpointer(self, batch_idx: int, model: Net, best_so_for: bool = False,
                          must: bool = False) -> None:
        """
        TODO: 后续补充
        """
        self._save_checkpointer(batch_idx=batch_idx,
                                model=model.module if hasattr(model, 'module') else model,
                                best_so_for=best_so_for,
                                must=must)

    @torch.no_grad()
    def writer_metrics(self, batch_idx: int, optimizer: Optimizer, outputs: Dict[str, torch.Tensor],
                       metrics_dict: Dict[str, Union[float, int]], **kwargs) -> None:
        """
        TODO: 后续补充
        """
        super().writer_metrics(batch_idx, optimizer, outputs, metrics_dict)

        if self.writer is None:
            return

        if batch_idx % self.log_frequent == 0:
            # log world size & batch size for elastic training
            self.add_scalar(tag='Elastic/World-Size',
                            scalar_value=self.world_size,
                            global_step=self.global_step)
            self.add_scalar(tag='Elastic/Batch-Size',
                            scalar_value=self.train_batch_size,
                            global_step=self.global_step)
            self.add_scalar(tag='Elastic/Global-Batch-Size',
                            scalar_value=self.train_batch_size * self.world_size,
                            global_step=self.global_step)

    def _save_checkpointer(self,
                           batch_idx: int,
                           model: Net,
                           best_so_for: bool,
                           must: bool = False):
        """
        保存训练状态
        """
        if (must or batch_idx % self.ckpt_frequent == 0) and self.checkpointer:
            model_state = model.state_dict()
            # collect data offsets from all workers
            world_size = int(os.getenv('WORLD_SIZE') or 1)
            rank = int(os.getenv('RANK') or 0)
            if world_size > 1:
                all_data_offsets = [None for _ in range(world_size)]
                # actually just gather is enough, but NCCL does not support gather
                dist.all_gather_object(
                    all_data_offsets,
                    self.data_offsets
                )
                data_offsets = {}
                for m in all_data_offsets:
                    data_offsets.update(m)
            else:
                data_offsets = self.data_offsets

            if rank == 0:
                training_states = {
                    'optimizer': self.optimizer.state_dict(),
                    'lr_scheduler': self.lr_scheduler.state_dict(),
                    # 'speed_monitor': self.speed_monitor.state_dict(),
                    'global_step': self.global_step,
                    'current_step': self.current_step,
                    'current_epoch': self.current_epoch,
                    'trained_samples_in_epoch': self.trained_samples_in_epoch,
                    'accelerator': self.accelerator.state_dict(),
                    'data_offsets': data_offsets
                }
                self.checkpointer.save_checkpoint(model_state=model_state,
                                                  epoch=self.global_step,
                                                  training_states=training_states,
                                                  is_best_so_far=best_so_for)
