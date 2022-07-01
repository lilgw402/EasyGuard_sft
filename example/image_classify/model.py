#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Description: resnet分类器
'''
from typing import Dict, Tuple, List
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from fex.optim import optimization
from fex.optim.lr_scheduler import WarmupMultiStepLR

from fex.core.net import Net  # TODO: import 有点丑，看看如何去掉core.net
from fex.nn.backbone.resnet import resnet50


class ResNetClassifier(Net):
    """ resnet 图像分类器 """

    def __init__(self, config, model=None, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        # pretrained arguments is DEPRECATED, since we will re init weight 2 line later, please
        self.resnet = resnet50() if model is None else model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.init_weights()
        self.model_to_channels_last = config.TRAINER.CHANNELS_LAST

    def init_weights(self):
        def init_weight_module(module):
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                torch.nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.bias.data.zero_()
                # strategy in this paper https://arxiv.org/pdf/1812.01187.pdf
                module.weight.data.fill_(1.)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.apply(init_weight_module)

    def forward(self, **batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        这样写主要是为了在各种accelerators下, forward可以做完training_step和validation_step
        """
        if self.training:
            return self.training_step(**batch)
        else:
            return self.validation_step(**batch)

    def _forward(self, image: torch.Tensor):
        logits = self.resnet(image)
        if isinstance(logits, dict):
            logits = logits['cls_score']  # [bsz, 1000]
        scores = torch.nn.functional.softmax(logits, dim=1)
        return {"logits": logits, "scores": scores}

    def training_step(self, **batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        主要是forward model的输出和计算loss
        """
        outputs_dict: Dict[str, torch.Tensor] = {}
        image = batch["image"]
        if self.model_to_channels_last:
            image = self.data_to_channels_last(image)
        target = batch["label"]

        outputs = self._forward(image)
        loss = self.criterion(outputs["logits"], target)

        # if self.do_with_frequence(config_frequence=self.log_frequent):
        #     process_outputs = self.cal_acc(output=outputs["scores"], target=target, topk=(1, 5))
        #     # 将training中top1_acc和top5_acc到tensorboard
        #     self.log_dict(process_outputs)

        outputs_dict["loss"] = loss
        return outputs_dict

    @torch.no_grad()
    def validation_step(self, **batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        validation阶段预测model的输出
        """
        total_outputs: Dict[str, torch.Tensor] = {}
        image = batch["image"]
        if self.model_to_channels_last:
            image = self.data_to_channels_last(image)
        outputs = self._forward(image)
        total_outputs.update(outputs)

        if "label" in batch:
            target = batch.get("label")
            res = self.cal_acc(
                output=outputs["logits"], target=target, topk=(1, 5))
            total_outputs.update(res)
        return total_outputs

    @torch.no_grad()
    def validation_epoch_end(self, total_outputs: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        对validation的所有step的结果进行处理，最终输出为一个Dict[str, torch.Tensor]
        """
        res_out: Dict[str, torch.Tensor] = {}
        top1_acc = []
        top5_acc = []
        for item in total_outputs:
            top1_acc.append(item["top1_acc"].mean().item())
            top5_acc.append(item["top5_acc"].mean().item())

        top1_acc = torch.tensor(top1_acc).mean()
        top5_acc = torch.tensor(top5_acc).mean()
        res_out["top1_acc"] = top1_acc
        res_out["top5_acc"] = top5_acc
        # 将validation中top1_acc和top5_acc到tensorboard
        self.log_dict(res_out)
        return res_out

    def configure_optimizers(self) -> Tuple[Optimizer, LambdaLR]:
        """
        Model定制optimizer和lr_scheduler
        """

        no_decay = ['bias', 'bn']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': self.config.TRAIN.WD},
            {'params': [p for n, p in self.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optm = torch.optim.SGD(optimizer_grouped_parameters, self.config.TRAIN.LR,
                               momentum=self.config.TRAIN.MOMENTUM,
                               weight_decay=self.config.TRAIN.WD)

        if self.config.TRAIN.LR_SCHEDULE == 'step':
            lr_iters = [int(epoch * self.step_per_epoch)
                        for epoch in self.config.TRAIN.LR_STEP]
            lr_scheduler = WarmupMultiStepLR(
                optimizer=optm,
                milestones=lr_iters,
                gamma=0.1,
                warmup_factor=self.config.TRAIN.WARMUP_FACTOR,
                warmup_iters=5 * self.step_per_epoch,
                warmup_method="linear",
            )
        elif self.config.TRAIN.LR_SCHEDULE == 'linear':
            lr_scheduler = optimization.get_linear_schedule_with_warmup(
                optimizer=optm,
                num_warmup_steps=5 * self.step_per_epoch,
                num_training_steps=self.total_step)
        elif self.config.TRAIN.LR_SCHEDULE == 'cosine':
            lr_scheduler = optimization.get_cosine_schedule_with_warmup(
                optimizer=optm,
                num_warmup_steps=5 * self.step_per_epoch,
                num_training_steps=self.total_step)

        return optm, lr_scheduler

    @torch.no_grad()
    def cal_acc(self, output: torch.Tensor, target: torch.Tensor, topk: Tuple[int] = (1,)) -> Dict[str, torch.Tensor]:
        """
        Computes the accuracy over the k top predictions for the specified values of k
        """
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = {}
        for k in topk:
            correct_k = correct[:k].contiguous(
            ).view(-1).float().sum(0, keepdim=True)
            res['top%s_acc' % k] = correct_k.mul_(100.0 / batch_size)
        return res

    @torch.no_grad()
    def trace(self, frames: torch.Tensor) -> torch.Tensor:
        cls_scroe = self.resnet(frames)['cls_score']
        return cls_scroe
