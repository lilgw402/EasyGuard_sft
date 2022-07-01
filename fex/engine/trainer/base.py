"""
trainer base，trainer 的基类，实现了整个训练流程（loop）。
不同的trainer 需要继承基类，来实现具体训练各个步骤的逻辑。
"""

from typing import Dict, List, Union


import os

import torch
from torch.utils.data import DataLoader

from fex.utils.hdfs_io import hglob
from fex.utils.distributed import rank_zero_warn, rank_zero_info
from fex.utils.stop_watch import StopWatch
from fex.config import CfgNode
from fex.core.net import Net


class TrainerBase:
    """
    Trainer 的基类，实现了训练（fit）流程。
    规定了Trainer 的接口。可以继承这个类，实现不同功能的Trainer。
    这个类无法直接使用，只是定义了流程，需要继承他，实现每个流程所需做的事情。

    一般情况下，是不需要用户做Trainer 实现的，除非有特别定制化的需求。

    如果需要实现一个Trainer，
    需要实现的函数列表：
    trainer 相关的：
    1. trainer_init
    2. train_start
    3. train_epoch_start
    4. train_epoch_end (optional)
    一个 train step 内做的事情：
    5. train_step_start
    6. train_step_forward
    7. train_step_backward
    8. train_step_optimize
    9. run_validation
    10. train_step_end

    """

    def __init__(self, cfg: CfgNode):
        """
        Trainer初始化需要两个参数
        Args:
            cfg (CfgNode): 模型训练的配置文件.
            resume: 是否resume
        """
        self.cfg = cfg
        self.interrupted = False
        self.stop_watch = StopWatch()
        self.global_step = 0

        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))

        self.train_begin_epoch = self.cfg.TRAINER.BEGIN_EPOCH  # 如果resume的话，这个会被重定义
        self.train_end_epoch = self.cfg.TRAINER.END_EPOCH
        self.train_dataset_size = self.cfg.TRAINER.TRAIN_DATASET_SIZE
        self.train_batch_size = self.cfg.TRAINER.TRAIN_BATCH_SIZE
        self.total_epoch = self.train_end_epoch - self.train_begin_epoch
        self.train_begin_batch = 0  # will override it when resuming from ckpt
        self.step_per_epoch = self.train_dataset_size // (self.train_batch_size * self.world_size)
        self.total_step = self.step_per_epoch * self.total_epoch
        self.train_dataloader = None

        self.extra_metric_name_list = self.cfg.get('TRAINER.EXTRA_METRICS', [])

    def fit(self, model: Net,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader = None) -> None:
        """
        Trainer的训练入口，需要传入定义好的model和对应的tain dataloader 和 validation dataloader.

        Args:
            model (Net): 自定义的model .
            train_dataloader (Optional[DataLoader], optional): train数据集的dataloader.
            val_dataloader (Optional[DataLoader], optional): validation数据集的dataloader.
        """

        # init env
        self.trainer_init()

        # setup fit ...
        model, train_dataloader, val_dataloader = self.train_start(model, train_dataloader, val_dataloader)
        self.train_dataloader = train_dataloader
        try:
            # run all epochs
            for epoch in range(self.train_begin_epoch, self.train_end_epoch):
                # train epoch start
                self.train_epoch_start(model, epoch, train_dataloader)
                # train epoch
                self.train_epoch_run(model, epoch, train_dataloader, val_dataloader)
                # train epoch end
                self.train_epoch_end(model, epoch, val_dataloader)
            # train end
            self.train_end(model)

        except KeyboardInterrupt:
            rank_zero_warn('Detected KeyboardInterrupt, attempting graceful shutdown...')
            if not self.interrupted:
                self.interrupted = True
                self.train_end(model)
        self.train_dataloader = None

    def train_epoch_run(self,
                        model: Net,
                        epoch: int,
                        train_dataloader: DataLoader,
                        val_dataloader: DataLoader = None) -> None:
        """
        训练一个epoch的具体执行，是训练loop的核心函数
        具体操作：
        1. get data: next train_dataloader，取一个batch的数据
        2. train step start:
        3. forward:
        4. backward:
        5. optimize:
        6. validation (check if needed)
        7. train step end:
        backward，更新参数
        """
        self.stop_watch.start()
        train_dataloader_iter = iter(train_dataloader)
        for batch_idx in range(self.train_begin_batch, self.step_per_epoch):
            # 1. get data
            try:
                batch = next(train_dataloader_iter)
            except StopIteration:
                rank_zero_info('reset loader .. ')
                train_dataloader_iter = iter(train_dataloader)
                batch = next(train_dataloader_iter)
            metrics_dic: Dict[str, Union[float, int]] = {}
            data_iter_elapsed = self.stop_watch.elapsed()

            # 2. train step start
            self.stop_watch.start()
            batch = self.train_step_start(batch, epoch, batch_idx)
            train_step_start_elapsed = self.stop_watch.elapsed()

            # 3. forward
            self.stop_watch.start()
            output_dict = self.train_step_forward(batch, model)
            forward_elapsed = self.stop_watch.elapsed()

            # 4. backward
            self.stop_watch.start()
            self.train_step_backward(output_dict["loss"])
            backward_step_elapsed = self.stop_watch.elapsed()

            # 5. optimize, update params
            self.stop_watch.start()
            self.train_step_optimize(batch_idx, model)
            optimizer_step_elapsed = self.stop_watch.elapsed()

            # 6. run validation
            val_metrics_dic = self.run_validation(model, val_dataloader)

            # 7. train step end
            output_dict["loss"] = output_dict["loss"].item()
            metrics_dic["data_iter_elapsed"] = data_iter_elapsed
            metrics_dic["train_step_start_elapsed"] = train_step_start_elapsed
            metrics_dic["forward_elapsed"] = forward_elapsed
            metrics_dic["backward_step_elapsed"] = backward_step_elapsed
            metrics_dic["optimizer_step_elapsed"] = optimizer_step_elapsed
            metrics_dic["loss"] = output_dict["loss"]
            for metric_name in self.extra_metric_name_list[:]:
                metric_value = output_dict.get(metric_name)
                if not isinstance(metric_value, torch.Tensor):
                    self.extra_metric_name_list.remove(metric_name)
                    print('Metric value[%s] type error, only torch.Tensor can be passed!' % metric_name)
                    continue

                metrics_dic[metric_name] = metric_value.detach().mean().item()
            self.train_step_end(batch_idx, model, output_dict, metrics_dic, val_metrics_dic)

    def trainer_init(self):
        """
        trainer 的一些初始化工作。
        如：
        1. 检查output 目录是否存在，如不存在则创建
        2. 初始化checkpointer、log writer
        3. 检查必要配置是否存在
        """
        raise NotImplementedError

    def train_start(self, model: Net, train_dataloader: DataLoader, val_dataloader: DataLoader) -> None:
        """
        开始训练前的工作。
        如：
        1. 初始化optimizer， lr_scheduler
        2. 打印model 的summary
        """
        raise NotImplementedError

    def train_epoch_start(self, model: Net, epoch: int, train_loader: DataLoader) -> None:
        """
        训练时，每个epoch开始前的工作
        如：
        1. 如果使用了sampler，每个epoch前需要 `sampler.set_epoch(epoch)`
        2. model.train() 以及 optimizer.zero_grad()
        """
        raise NotImplementedError

    def train_epoch_end(self, model: Net, epoch: int, val_dataloader: DataLoader):
        """
        训练一个epoch 结束做的工作
        如：保存checkpoint
        """
        raise NotImplementedError

    def train_end(self, model: Net):
        """
        训练完整结束后的收尾工作：
        如：
        1. 保存最后的ckpt
        2. writer close
        3. 清空cache
        """
        raise NotImplementedError

    def train_step_start(self, batch: Dict[str, Union[torch.Tensor, List]],
                         epoch: int, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        做一个train step 的前置工作
        如：
        将数据移到cuda，
        """
        raise NotImplementedError

    def train_step_forward(self, batch: Dict[str, torch.Tensor], model: Net) -> Dict[str, torch.Tensor]:
        """
        做一个train step 的前向过程
        如：
        调用模型的train_step，
        以及梯度累积的一些事情
        """
        raise NotImplementedError

    def train_step_backward(self, loss: torch.Tensor):
        """
        做一个train step 的后向过程
        如：
        调用accelerator 的 backward_step
        """
        raise NotImplementedError

    def train_step_optimize(self, batch_idx: int, model: Net) -> None:
        """
        做一个train step的优化过程
        如：
        更新参数等
        """
        raise NotImplementedError

    def train_step_end(self, batch_idx: int, model: Net, outputs_dict: Dict[str, torch.Tensor], metrics_dict: Dict[str,
                       Union[float, int]], val_metrics_dic: Dict[str, Union[float, int, torch.Tensor]]) -> None:
        """
        做一个train step的结束的后置操作
        如：
        写metric
        保存checkpoint
        """
        raise NotImplementedError

    def run_validation(self, model: Net, val_loader: DataLoader = None) -> Dict[str, torch.Tensor]:
        """
        跑 validation 的过程
        如：
        遍历val_loader，计算结果，计算metric
        """
        raise NotImplementedError
