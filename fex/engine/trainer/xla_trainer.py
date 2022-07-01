"""
xla trainer
"""
from typing import Dict, List, Union, Tuple

import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from fex import _logger as logger
from fex.utils.distributed import rank_zero_info, rank_zero_warn
from fex.config import CfgNode
from fex.core.net import Net
from fex.engine.trainer.common_trainer import Trainer
from fex.utils.sync_tensor import sync_tensor


class XLATrainer(Trainer):
    """
    xla trainer
    需要重载以下函数：
    1. train_start: 需要将loader 都转为 XLALoader
    2. train_step_start: 不做move_to_cuda 这一步
    3. train_step_forward: 不判断 nan loss，xla 会崩
    4. train_step_optimize: xla 的更新步骤稍有不同
    5. run_evaluation: 有一些 device_id，move_to_cuda 的不同
    """

    def __init__(self, cfg: CfgNode, output_path: str = "./train_fex"):
        super().__init__(cfg, output_path)

        try:
            import torch_xla
            import torch_xla.core.xla_model as xm
            from torch_xla.amp import autocast
        except ImportError:
            print('no torch_xla! Please install it if necessary.')

    def train_start(self, model: Net, train_dataloader: DataLoader, val_dataloader: DataLoader = None) -> Union[Net, DataLoader]:
        """
        训练开始前置工作，
        需要将dataloader 转成 xla 的loader
        """
        model.set_train_base(self)
        if self.model_to_channels_last:
            model = model.to(memory_format=torch.channels_last)  # model to channel_last
        self.optimizer, self.lr_scheduler = model.configure_optimizers()
        model, self.optimizer, self.lr_scheduler = self.accelerator.set_up(
            model, self.optimizer, self.lr_scheduler, self.local_rank, self.world_size, self.rank)
        self.summary_parameters(model, logger)
        self._save_to_output_path()
        return model, XLALoaderWrapper(train_dataloader), XLALoaderWrapper(val_dataloader)

    def train_step_start(self, batch: Dict[str, Union[torch.Tensor, List]],
                         epoch: int, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        做一个train step 的前置工作
        如：
        将数据移到cuda，
        """
        self.global_step = self.step_per_epoch * epoch + batch_idx
        self.current_step = batch_idx
        return batch

    def train_step_forward(self, batch: Dict[str, torch.Tensor], model: Net) -> Dict[str, torch.Tensor]:
        """
        做一个train step 的前向过程
        调用模型的train_step，以及梯度累积的一些事情
        """
        with autocast(self.cfg.XLA.AMP):
            output_dict = model(**batch)  # model.module.train_step
            loss = output_dict.get("loss")
            if self.accelerator_gradient_accumulate_steps > 1:
                loss = loss / self.accelerator_gradient_accumulate_steps

            output_dict["loss"] = loss
            return output_dict

    def train_step_optimize(self, batch_idx: int, model: Net) -> None:
        """
        做一个train step的优化过程
        如：
        更新参数等
        """
        if self.world_size > 1:
            gradients = xm._fetch_gradients(self.optimizer)
            xm.all_reduce('sum', gradients, scale=1.0 / self.world_size)
        if self.cfg.XLA.AMP:
            self.accelerator.scaler_step(self.optimizer)
        else:
            self.optimizer.step()

        self.optimizer.zero_grad()
        if self.lr_scheduler is not None and not isinstance(
                self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_scheduler.step()

    def run_evaluation(self, model: Net, val_loader: DataLoader) -> Dict[str, torch.Tensor]:
        """
        TODO: 后续补充
        """
        # TODO: val_steps没有设置，默认为所有的step
        model.eval()
        outputs: List[Dict[str, torch.Tensor]] = []
        rank_zero_info("Going to valdation on step {}".format(self.global_step))
        val_loader_iter = iter(val_loader)
        device_rank = 0
        for index in range(self.val_step):
            try:
                batch = next(val_loader_iter)
            except StopIteration:
                print('finish loading val loader on index %s .. ' % index)
                break
            # 调用Net的validation_step
            process_res = model(**batch)
            outputs.append(process_res)

        # 对于val的结果，由Net的validation_epoch_end来处理
        process_outputs = model.validation_epoch_end(outputs)

        # 判断 process_outputs中的内容是否为空
        if process_outputs:
            for key, value in process_outputs.items():
                process_outputs[key] = sync_tensor(value, reduce_op="avg", device_rank=device_rank)
            rank_zero_info("Validation result on step {} : {} ".format(self.global_step, process_outputs))
        else:
            rank_zero_warn("Validation result is None !")
        model.train()
        return process_outputs

    def _save_checkpointer(self, batch_idx: int, model: Net, optimizer: Optimizer, best_so_for: bool,
                           must: bool = False):
        """
        xla device 上的模型，需要转换一下才能存储
        """
        if (must or batch_idx % self.ckpt_frequent == 0) and self.checkpointer:
            model_state = model.state_dict()
            training_states = optimizer.state_dict()

            def convert_fn(tensors):
                return [tensor.cpu() for tensor in tensors]

            def select_fn(v):
                return isinstance(v, torch.Tensor) and v.device.type == 'xla'

            model_state = xm.ToXlaTensorArena(convert_fn, select_fn).transform(model_state)
            training_states = xm.ToXlaTensorArena(convert_fn, select_fn).transform(training_states)

            self.checkpointer.save_checkpoint(model_state=model_state,
                                              epoch=self.global_step,
                                              training_states=training_states,
                                              is_best_so_far=best_so_for)


class XLALoaderWrapper:
    def __init__(self, loader):
        self._loader = loader
        self._device = str(xm.xla_device())

    def __next__(self):
        data = next(self._loader)
        new_data = {}
        if isinstance(data, list):
            data = data[0]
        for k in data:
            if isinstance(data[k], torch.Tensor):
                try:
                    new_data[k] = torch_xla._XLAC._xla_tensor_from_cuda(data[k], self._device)
                except Exception as e:
                    print(e)
            else:
                new_data[k] = data[k]
        return new_data

    def __len__(self):
        return len(self._loader)

    def __iter__(self):
        return self
