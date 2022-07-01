"""
基础的trainer
"""

import os
from collections import namedtuple
from typing import Dict, List, Union, Tuple
from logging import Logger
import string
import random
import warnings

import numpy
import torch
from torch import distributed as dist
from torch.utils.data import DataLoader
from torch import optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

try:
    from torch.cuda.amp import autocast
except Exception as e:
    print(e)

from fex import _logger as logger
from fex.config import CfgNode
from fex.core.net import Net
from fex.data import PytorchDaliIter
from fex.utils.distributed import rank_zero_only, local_rank_zero_only, log, rank_zero_info, log_file, rank_zero_warn
from fex.utils.hdfs_io import hcopy, hexists, hmkdir, hglob
from fex.utils.torch_io import load as torch_io_load
from fex.utils.speedmonitor import Speedmonitor
from fex.utils.checkpointer import Checkpointer
from fex.utils.sync_tensor import sync_tensor
from fex.utils.load import load_from_pretrain, device_mapping
from fex.utils.remote_tensorboard_writer import SummaryWriter, RemoteSummaryWriter
from fex.utils.stop_watch import StopWatch
from fex.engine.trainer.base import TrainerBase
from fex.engine.accelerators import ACCELERATOR_MAP

try:
    import tracking as tk
except ImportError:
    warnings.warn('Tracking not found. Please install by: pip install byted-tracking')


class Trainer(TrainerBase):
    """
    基本款的 Trainer
    实现了基础的训练过程，绝大部分情况可以用他
    """

    def __init__(self, cfg: CfgNode, output_path: str = "./train_fex", resume: bool = False):
        super().__init__(cfg)
        self.cfg = cfg
        self.output_path = output_path
        self.optimizer = None
        self.lr_scheduler = None
        self.val_dataset_size = self.cfg.TRAINER.VAL_DATASET_SIZE
        self.val_batch_size = self.cfg.TRAINER.VAL_BATCH_SIZE
        self.log_frequent = self.cfg.TRAINER.LOG_FREQUENT
        self.val_frequent = self.cfg.TRAINER.VAL_FREQUENT
        self.ckpt_frequent = self.cfg.TRAINER.CHECKPOINT_FREQUENT
        self.model_to_channels_last = self.cfg.TRAINER.CHANNELS_LAST
        self.val_step = self.cfg.TRAINER.VAL_STEPS if self.cfg.TRAINER.VAL_STEPS > 1 else self.val_dataset_size // (self.val_batch_size * self.world_size)
        self.val_metric_name = self.cfg.get('TRAINER.VAL_METRIC_NAME', '')

        self.accelerator_gradient_accumulate_steps = int(self.cfg.ACCELERATOR.GRAD_ACCUMULATE_STEPS)

        self.checkpointer = None
        self.writer = None
        self._current_step = 0
        self._current_epoch = 0
        self.trained_samples_in_epoch = 0
        self.actual_step = 0
        self.best_val_metric = -1e9
        self.accelerator = ACCELERATOR_MAP[self.cfg.ACCELERATOR.ACCELERATOR](cfg, log)
        self.resume = resume
        self.random_string = ''.join(random.sample(string.ascii_letters + string.digits, 16))
        self.use_torch_autocast = self.cfg.ACCELERATOR.ACCELERATOR in ['TorchAMPDDP', 'PtxAMPDDP']  # TODO: 有点hack，先这么着

        self.base_name_list = ['data_iter', 'train_step_start', 'forward', 'backward_step', 'optimizer_step', 'loss']
        self.base_metric_set = {'data_iter_elapsed', 'train_step_start_elapsed', 'forward_elapsed', 'backward_step_elapsed', 'optimizer_step_elapsed', 'loss'}  # TODO: 有点hack
        self.speed_monitor = Speedmonitor(
            batch_size=self.train_batch_size,
            logger=logger,
            frequent=self.log_frequent,
            batches_per_epoch=self.step_per_epoch,
            total_epochs=self.total_epoch,
            gradient_accumulate_steps=self.accelerator_gradient_accumulate_steps)

    def trainer_init(self) -> None:
        """
        trainer 初始化
        """
        rank_zero_info("Trainer init ...")
        self._check_ouput()
        self._init_checkpointer_once()
        self._init_writer()
        self.cfg.check_core_attr()

    def train_start(self, model: Net, train_dataloader: DataLoader, val_dataloader: DataLoader = None) -> Union[Net, DataLoader]:
        """
        训练开始前置工作
        """
        torch.cuda.empty_cache()
        model.set_train_base(self)
        if self.model_to_channels_last:
            model = model.to(memory_format=torch.channels_last)  # model to channel_last
        self.optimizer, self.lr_scheduler = model.configure_optimizers()

        training_state = None
        if self.resume:
            training_state = self.restore_checkpoint(model)

        model, self.optimizer, self.lr_scheduler = self.accelerator.set_up(
            model, self.optimizer, self.lr_scheduler, self.local_rank, self.world_size, self.rank)

        if isinstance(self.accelerator, ACCELERATOR_MAP["ApexDDP"]):
            self.reinit_scheduler_properties(self.optimizer, self.lr_scheduler)

        if training_state is not None:
            self.optimizer.load_state_dict(training_state['optimizer'])
            if self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(training_state['lr_scheduler'])
            self.accelerator.load_state_dict(training_state['accelerator'])

        self.summary_parameters(model, logger)
        self._save_to_output_path()
        return model, train_dataloader, val_dataloader

    def train_epoch_start(self, model: Net, epoch: int, train_dataloader: DataLoader) -> None:
        """
        训练一个epoch开始前的工作
        """
        self.current_epoch = epoch
        if hasattr(train_dataloader, 'sampler') and hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)
        rank_zero_info("PROGRESS: %.2f%%, Epoch %d/%d, %d steps per epoch" % (
            100.0 * epoch / self.train_end_epoch, epoch, self.train_end_epoch, self.step_per_epoch))
        model.train()
        self.optimizer.zero_grad()

    def train_epoch_end(self, model, epoch, val_dataloader):
        """
        训练一个epoch 结束做的工作
        如：保存checkpoint
        """
        self.save_checkpointer(self.global_step, model, must=True)
        self.train_begin_batch = 0
        self.trained_samples_in_epoch = 0

    def train_step_start(self, batch: Dict[str, Union[torch.Tensor, List]],
                         epoch: int, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        做一个train step 的前置工作
        如：
        将数据移到cuda，
        """
        cuda_batch = self._move_to_cuda(batch)
        self.actual_step = self.step_per_epoch * epoch + batch_idx
        self.current_step = batch_idx
        return cuda_batch

    def train_step_forward(self, batch: Dict[str, torch.Tensor], model: Net) -> Dict[str, torch.Tensor]:
        """
        做一个train step 的前向过程
        调用模型的train_step，以及梯度累积的一些事情
        """
        with autocast(enabled=self.use_torch_autocast):  # 默认是 false，只有用官方的amp时才会为true
            output_dict = model(**batch)  # model.module.train_step
            loss = output_dict.get("loss")

            # fetch the next batch
            try:
                from dlx import ScheduledDataLoader
                loader = ScheduledDataLoader.lookup(self.train_dataloader)
                if loader is not None:
                    loader.fetch()
            except ImportError as e:
                warnings.warn("Unable to import dlx. Some dataloading acceleration is disabled.")

            if loss is None:
                raise ValueError(
                    "Loss not in model training step outputs, please check!")

            if torch.isnan(loss):
                logger.error("nan loss encountered, kill the process")
                raise KeyboardInterrupt("Nan loss encountered, kill the process")

            if self.accelerator_gradient_accumulate_steps > 1:
                loss = loss / self.accelerator_gradient_accumulate_steps

            output_dict["loss"] = loss
            return output_dict

    def train_step_backward(self, loss: torch.Tensor):
        """
        做一个train step 的后向过程
        调用accelerator 的 backward_step
        """
        self.accelerator.backward_step(loss, self.optimizer)

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
            self.lr_scheduler.step()

    def train_step_end(self, batch_idx: int, model: Net, outputs_dict: Dict[str, torch.Tensor], metrics_dict: Dict[str,
                       Union[float, int]], val_metrics_dic: Dict[str, Union[float, int, torch.Tensor]]) -> None:
        """
        做两件事情：1. print metrics 2. save checkpoint
        增加val_metrics_dic: 可以用来选择save best.th
        """
        self.trained_samples_in_epoch += self.train_batch_size * self.world_size
        # print metrics
        self.writer_metrics(batch_idx, self.optimizer, outputs_dict, metrics_dict)
        # select best ckpt.
        best_so_far = False
        must = False
        if val_metrics_dic and self.val_metric_name and self.val_metric_name in val_metrics_dic:
            cur_val_metric = val_metrics_dic[self.val_metric_name]
            if cur_val_metric > self.best_val_metric:
                if isinstance(self.best_val_metric, torch.Tensor):
                    cur_val_metric = cur_val_metric.mean().item()
                self.best_val_metric = cur_val_metric
                best_so_far = True
                must = True
                rank_zero_info('Best {} so far: {}'.format(self.val_metric_name, self.best_val_metric))
        self.save_checkpointer(self.global_step, model, must=must, best_so_for=best_so_far)
        self.global_step += 1

    def run_validation(self, model: Net, val_loader: DataLoader = None) -> Dict[str, torch.Tensor]:
        """
        跑 validation 的过程
        如：
        遍历val_loader，计算结果，计算metric
        """
        if self.val_frequent <= 0 or self.global_step % self.val_frequent != 0:
            return
        model.eval()
        outputs: List[Dict[str, torch.Tensor]] = []
        rank_zero_info("Going to valdation on step {}".format(self.global_step))
        val_loader_iter = iter(val_loader)
        for index in range(self.val_step):
            try:
                batch = next(val_loader_iter)
            except StopIteration:
                print('finish loading val loader on index %s .. ' % index)
                break
            cuda_batch = self._move_to_cuda(batch)
            # 调用Net的validation_step
            process_res = model(**cuda_batch)
            outputs.append(process_res)

        # 对于val的结果，由Net的validation_epoch_end来处理
        process_outputs = model.module.validation_epoch_end(outputs)

        # 判断 process_outputs中的内容是否为空
        if process_outputs:
            for key, value in process_outputs.items():
                process_outputs[key] = sync_tensor(value, reduce_op="avg", device_rank=self.local_rank)
            rank_zero_info("Validation result on step {} : {} ".format(self.global_step, process_outputs))
        else:
            rank_zero_warn("Validation result is None !")
        model.train()
        return process_outputs

    def train_end(self, model: Net):
        """
        训练结束的工作
        """
        self.save_checkpointer(self.global_step, model, must=True)
        rank_zero_only(hcopy)(log_file, self.output_path)
        if self.writer:
            self.writer.close()
        torch.cuda.empty_cache()
        rank_zero_info("Train success ...")

    @rank_zero_only
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
        if self.writer is None:
            logger.warning("Writer is None in TrainBase, skip write metrics!")
            return

        must = kwargs.get("must", False)
        if batch_idx % self.log_frequent == 0 or must:
            for group_i, param_group in enumerate(optimizer.param_groups):
                # self.writer.add_scalar(tag='LR/Init-LR/Group_{}'.format(group_i), # 这个log not necessary
                #                        scalar_value=param_group['initial_lr'],
                #                        global_step=self.global_step)
                self.add_scalar(tag='Trainer/LR-Group-{}'.format(group_i),
                                scalar_value=param_group['lr'],
                                global_step=self.global_step)
                # TODO: 增加参数的avg log
            accel_metrics = self.accelerator.get_metrics()
            for accel_key, accel_val in accel_metrics.items():
                self.add_scalar(tag=f'Trainer/Accel-{accel_key}',
                                scalar_value=accel_val,
                                global_step=self.global_step)
            self.add_scalar(tag='Trainer/Loss',
                            scalar_value=float(metrics_dict["loss"] * self.accelerator_gradient_accumulate_steps),
                            global_step=self.global_step)
        self.speed_monitor_tolog(metrics_dict)

    @property
    def current_step(self):
        return self._current_step

    @current_step.setter
    def current_step(self, batch_idx: int) -> None:
        """
        TODO: 后续补充
        """
        if not isinstance(batch_idx, int):
            raise ValueError("batch index must be an integer!")
        if batch_idx < 0 or batch_idx > self.step_per_epoch:
            logger.error(
                "batch index must between 0 ~ {}, please check!".format(self.step_per_epoch))

        self._current_step = batch_idx

    @property
    def current_epoch(self):
        return self._current_epoch

    @current_epoch.setter
    def current_epoch(self, cur_epoch: int) -> None:
        """
        TODO: 后续补充
        """
        if not isinstance(cur_epoch, int):
            raise ValueError("current epoch must be an integer!")
        if cur_epoch < self.train_begin_epoch or cur_epoch > self.train_end_epoch:
            logger.error(
                "current epoch must between {} ~ {}, please check!".format(
                    self.train_begin_epoch,
                    self.train_end_epoch))

        self._current_epoch = cur_epoch

    @rank_zero_only
    def summary_parameters(self, model: Net, logger: Logger = None):
        """
        Summary Parameters of Model
        :param model: torch.nn.module_name
        :param logger: logger
        :return: None
        """

        self._print_and_log('>> Trainable Parameters:', logger)
        trainable_paramters = [(str(n), str(v.dtype), str(tuple(v.shape)), str(v.numel()))
                               for n, v in model.named_parameters() if v.requires_grad]
        max_lens = [max([len(item) + 4 for item in col])
                    for col in zip(*trainable_paramters)]
        raw_format = '|' + \
            '|'.join(['{{:{}s}}'.format(max_len)
                      for max_len in max_lens]) + '|'
        raw_split = '-' * (sum(max_lens) + len(max_lens) + 1)
        self._print_and_log(raw_split, logger)
        self._print_and_log(raw_format.format(
            'Name', 'Dtype', 'Shape', '#Params'), logger)
        self._print_and_log(raw_split, logger)

        for name, dtype, shape, number in trainable_paramters:
            self._print_and_log(raw_format.format(
                name, dtype, shape, number), logger)
            self._print_and_log(raw_split, logger)

        num_trainable_params = sum(
            [v.numel() for v in model.parameters() if v.requires_grad])
        total_params = sum([v.numel() for v in model.parameters()])
        non_trainable_params = total_params - num_trainable_params
        self._print_and_log('>> {:25s}\t{:.2f}\tM'.format(
            '# TrainableParams:', num_trainable_params / (1.0 * 10 ** 6)), logger)
        self._print_and_log('>> {:25s}\t{:.2f}\tM'.format(
            '# NonTrainableParams:', non_trainable_params / (1.0 * 10 ** 6)), logger)
        self._print_and_log('>> {:25s}\t{:.2f}\tM'.format(
            '# TotalParams:', total_params / (1.0 * 10 ** 6)), logger)

    @local_rank_zero_only
    def speed_monitor_tolog(self, batch_end_params: Dict[str, Union[float, int]]) -> None:
        """
        TODO: 后续补充
        """
        extra_metric_dic = {}
        for k, v in batch_end_params.items():
            if k not in self.base_metric_set:
                extra_metric_dic[k] = v

        BatchEndParam = namedtuple('BatchEndParams', self.base_name_list)

        batch_end_params = BatchEndParam(
            data_iter=batch_end_params["data_iter_elapsed"],
            train_step_start=batch_end_params["train_step_start_elapsed"],
            forward=batch_end_params["forward_elapsed"],
            backward_step=batch_end_params["backward_step_elapsed"],
            optimizer_step=batch_end_params["optimizer_step_elapsed"],
            loss=batch_end_params["loss"])
        self.speed_monitor(self.current_epoch,
                           self.current_step,
                           batch_end_params,
                           extra_metric_dic)

    def checkpoint_check(self, ckpt_path: str) -> str:
        tested_ckpt_path: str = None
        temp_ckpt_paths: List[str] = []

        if ckpt_path == "last":
            temp_ckpt_paths = hglob(os.path.join(self.output_path, "model*.th"), sort_by_time=True)
        elif ckpt_path == "best":
            temp_ckpt_paths = hglob(os.path.join(self.output_path, "best.th"))
        else:
            temp_ckpt_paths = hglob(ckpt_path)

        if len(temp_ckpt_paths) >= 1:
            tested_ckpt_path = temp_ckpt_paths[-1]

        return tested_ckpt_path

    def _print_and_log(self, string: str, logger: Logger = None):
        """
        TODO: 后续补充
        """
        if logger is None:
            logger.info(string)
        else:
            logger.info(string)

    def _init_checkpointer_once(self) -> None:
        """
        TODO: 后续补充
        """
        if not self.checkpointer:
            self.checkpointer = Checkpointer(self.output_path)

    def _init_writer(self) -> None:
        """
        TODO: 后续补充
        """
        if self.writer:
            return
        metrics_dir = os.path.join(
            self.output_path, 'rank{}'.format(self.rank))
        if not hexists(metrics_dir):
            hmkdir(metrics_dir)

        if metrics_dir.startswith("hdfs://"):
            temp_metrics_dir = os.path.join(
                '/tmp', f'{self.random_string}_temp_train_log', f'rank{self.rank}')
            if not os.path.exists(temp_metrics_dir):
                os.makedirs(temp_metrics_dir)
            self.writer = RemoteSummaryWriter(
                hdfs_path=metrics_dir, remote_flush_secs=120, log_dir=temp_metrics_dir)
        else:
            self.writer = SummaryWriter(log_dir=metrics_dir)

    def _move_to_cuda(self, batch_dict: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        TODO: 后续补充
        """
        if isinstance(batch_dict, list):
            batch_dict = batch_dict[0]
        for k in batch_dict:
            if isinstance(batch_dict[k], torch.Tensor):
                if not batch_dict[k].is_cuda:
                    batch_dict[k] = batch_dict[k].cuda(non_blocking=True)
        return batch_dict

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

            training_states = {
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else {},
                # 'speed_monitor': self.speed_monitor.state_dict(),
                'global_step': self.global_step,
                'current_step': self.current_step,
                'current_epoch': self.current_epoch,
                'trained_samples_in_epoch': self.trained_samples_in_epoch,
                'accelerator': self.accelerator.state_dict(),
            }
            self.checkpointer.save_checkpoint(model_state=model_state,
                                              epoch=self.global_step,
                                              training_states=training_states,
                                              is_best_so_far=best_so_for)

    def restore_checkpoint(self, model: Net):
        """
        Restores a model from a serialization_dir to the last saved checkpoint.
        This includes a training state (typically consisting of an epoch count and optimizer state),
        which is serialized separately from  model parameters. This function should only be used to
        continue training - if you wish to load a model for inference/load parts of a model into a new
        computation graph, you should use the native Pytorch functions:
        `` model.load_state_dict(torch.load("/path/to/model/weights.th"))``

        If ``self._serialization_dir`` does not exist or does not contain any checkpointed weights,
        this function will do nothing and return empty dicts.

        Returns
        -------
        states: Tuple[Dict[str, Any], Dict[str, Any]]
            The model state and the training state.
        """
        latest_checkpoint = self.checkpointer.find_latest_checkpoint()

        if latest_checkpoint is None:
            # No checkpoint to restore, start at 0
            log.error('Cannot find checkpoint, resume failed')
            return None

        model_path, training_state_path = latest_checkpoint

        # Load the parameters onto CPU, then transfer to GPU.
        # This avoids potential OOM on GPU for large models that
        # load parameters onto GPU then make a new GPU copy into the parameter
        # buffer. The GPU transfer happens implicitly in load_state_dict.
        log.info(f'[Resuming]: ckpt from {model_path} ... ')
        load_from_pretrain(model.module if hasattr(model, 'module') else model, model_path, [])

        log.info(f'[Resuming]: training status from {training_state_path} ... ')
        training_state = torch_io_load(training_state_path, map_location=device_mapping(-1))
        resume_epoch = training_state['current_epoch']  # + 1 #TODO: 这里这个+1 的逻辑，稍微有点trick，默认了上个epoch是结束了，现在resume是从下一个epoch开始。这个逻辑在加载中间保存ckpt的时候会有问题，会跳过上一个epoch的后半部分训练了
        self.global_step = training_state['global_step'] + 1  # step must + 1, otherwise will re-run this step

        # resume train begin epoch
        self.train_begin_epoch = resume_epoch
        # resume train begin batch
        self.trained_samples_in_epoch = training_state['trained_samples_in_epoch']
        self.train_begin_batch = self.trained_samples_in_epoch // (self.train_batch_size * self.world_size)
        log.info(f'[Resuming]: starting from epoch: {resume_epoch}/{self.train_end_epoch}, '
                 f'batch: {self.train_begin_batch}/{self.step_per_epoch}, global_step: {self.global_step}')

        self.optimizer.load_state_dict(training_state['optimizer'])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(training_state['lr_scheduler'])
        # self.speed_monitor.load_state_dict(training_state['speed_monitor'])
        return training_state

    def _check_ouput(self):
        """
        检查用户提供的output_path是否存在: 如果不存在, 会自动创建.
        support: 支持path为本地目录和hdfs远程目录
        """
        if not hexists(self.output_path):
            try:
                hmkdir(self.output_path)
            except Exception as e:
                print("Hmkdir has exception: {}".format(str(e)))

    @rank_zero_only
    def _save_to_output_path(self):
        """
        将训练的config和model_file保存到output_path中
        """
        hcopy(self.cfg.MOUDEL_FILE, self.output_path)
        config_file_name = os.path.split(self.cfg.CONFIG_PATH)[-1]
        tmp_local_config_file_path = os.path.join('/tmp', f'{self.random_string}_{config_file_name}')
        self.cfg.dump(config_name=tmp_local_config_file_path)
        hcopy(tmp_local_config_file_path, os.path.join(self.output_path, config_file_name))

    def reinit_scheduler_properties(self, optimizer: Optimizer, scheduler: _LRScheduler) -> None:
        """
        在ApexDDP情况下, 需要重新初始化 scheduler, 避免出现lr_scheduler warning.
        issue: https://github.com/pytorch/pytorch/issues/27595
        issue: https://github.com/PyTorchLightning/pytorch-lightning/issues/841
        """
        # check that we dont mix users optimizers and schedulers
        if scheduler.optimizer == optimizer:
            # Find the mro belonging to the base lr scheduler class
            for i, mro in enumerate(scheduler.__class__.__mro__):
                if mro == optim.lr_scheduler._LRScheduler:
                    idx = i
            scheduler.__class__.__mro__[idx].__init__(scheduler, optimizer)

    def add_scalar(self, tag, scalar_value, global_step):
        self.writer.add_scalar(tag=tag,
                               scalar_value=scalar_value,
                               global_step=global_step)
        if int(os.getenv('RANK', 0)) == 0:
            tk.log({tag: scalar_value}, step=global_step)

    def add_histogram(self, tag, values, global_step):
        self.writer.add_histogram(tag=tag, values=values, global_step=global_step)
        if int(os.getenv('RANK', 0)) == 0:
            tk.log({tag: tk.plot.histogram(values)})
