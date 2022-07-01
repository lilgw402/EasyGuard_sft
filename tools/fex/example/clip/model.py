#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLIP 的训练
"""
from typing import Dict, Tuple, List

import torch
import torch.nn.functional as F
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import LambdaLR

# unified optimizers from data/optimizer
try:
    from byted_optimizer.torch import AdaBeliefW
except Exception as e:
    print(f'Error: {e}; you can install byted_optimizer with: `pip3 install https://d.scm.byted.org/api/v2/download/data.aml.optimizer_1.0.0.2.tar.gz --no-cache-dir -i https://bytedpypi.byted.org/simple/`')

from fex.core import Net
from fex.model import CLIP
from fex.optim import optimization
from fex.optim.optimization import AdamW
from fex.optim.lr_scheduler import WarmupMultiStepLR


class CLIPNet(CLIP):
    """
    继承了 clip
    """

    def __init__(self, config, *args, **kwargs):

        # 模型结构相关
        visual_type = config.get('network.visual_type', 'VitB32')
        visual_config = config.get('network.visual_config', {})
        project_mode = config.get('network.project_mode', 'default')
        text_type = config.get('network.text_type', 'ALBert')
        # 训练相关配置
        gpuwise_nce = True
        nce_world_size = config.get('train.nce_world_size', None)
        if nce_world_size is None:
            gpuwise_nce = config.get('train.gpuwise_nce', True)
            nce_world_size = -1 if gpuwise_nce else 1
        init_tau = config.get('train.init_tau', 0.07)
        tau_clamp = config.get('train.tau_clamp', 4.6051)
        super().__init__(config=config,
                         visual_type=visual_type,
                         visual_config=visual_config,
                         text_type=text_type,
                         project_mode=project_mode,
                         gpuwise_nce=gpuwise_nce,
                         init_tau=init_tau,
                         tau_clamp=tau_clamp,
                         nce_world_size=nce_world_size,
                         *args, **kwargs)

        self.init_weights()
        self.freeze_params(self.config.get('train.freeze_prefix', []))

    def init_weights(self):
        def init_weight_module(module):
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                torch.nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, (torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.apply(init_weight_module)

    def freeze_params(self, freeze_prefix):
        for name, param in self.named_parameters():
            for prefix in freeze_prefix:
                if name.startswith(prefix):
                    param.requires_grad = False

    def forward(self, **batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        这样写主要是为了在各种accelerators下, forward可以做完training_step和validation_step
        """
        if self.training:
            return self.training_step(**batch)
        else:
            return self.validation_step(**batch)

    def training_step(self,
                      image: torch.Tensor,
                      input_ids: torch.Tensor,
                      input_mask: torch.Tensor,
                      input_segment_ids: torch.Tensor = None):
        result = super().forward(image=image,
                                 input_ids=input_ids,
                                 input_mask=input_mask,
                                 input_segment_ids=input_segment_ids
                                 )
        self.log('nce_temperature', result.pop('nce_temperature'))
        return result

    @torch.no_grad()
    def validation_step(self,
                        image: torch.Tensor,
                        input_ids: torch.Tensor,
                        input_mask: torch.Tensor,
                        input_segment_ids: torch.Tensor = None,
                        **kwargs):
        """
        因为要label embedding的acc要将1000个分类的embedding计算出来，稍微有点麻烦
        作为近似的，我们计算同batch下的recall指标，更好的表示nce拟合的效果
        """

        # TODO: hack
        if image.shape[0] < 10:
            return {'v2t@10': torch.tensor([0.]),
                    't2v@10': torch.tensor([0.])}

        t_emb = self.encode_text(input_ids=input_ids, input_segment_ids=input_segment_ids, input_mask=input_mask)
        v_emb = self.encode_image(image=image)
        t_emb = F.normalize(t_emb, dim=1)
        v_emb = F.normalize(v_emb, dim=1)
        inter_mat = torch.mm(t_emb, v_emb.t())
        _, rank_vt = inter_mat.topk(10, dim=0)
        _, rank_tv = inter_mat.topk(10, dim=1)  # [bsz, 10]
        rank_vt = rank_vt.transpose(0, 1)  # [bsz, 10]

        total = torch.tensor(v_emb.shape[0])
        vt_correct = torch.tensor([0])
        tv_correct = torch.tensor([0])
        for i, gt in enumerate(rank_vt):
            if i in gt:
                vt_correct += 1
        for i, gt in enumerate(rank_tv):
            if i in gt:
                tv_correct += 1

        # self.log('V->T@10', vt_correct / total)
        # self.log('T->V@10', tv_correct / total)

        return {'v2t@10': vt_correct / total,
                't2v@10': tv_correct / total}

    def configure_optimizers(self) -> Tuple[Optimizer, LambdaLR]:
        """
        Model定制optimizer和lr_scheduler
        """
        no_decay = ['bias', 'bn', 'norm']
        no_dacay_params_dict = {'params': [], 'weight_decay': 0.0}
        normal_params_dict = {'params': [], 'weight_decay': self.config.train.wd}

        for n, p in self.named_parameters():
            if any(nd in n for nd in no_decay):
                no_dacay_params_dict['params'].append(p)
            else:
                normal_params_dict['params'].append(p)
        optimizer_grouped_parameters = [no_dacay_params_dict, normal_params_dict]

        if self.config.train.optim == 'SGD':
            optm = torch.optim.SGD(optimizer_grouped_parameters, self.config.train.lr,
                                   momentum=self.config.train.momentum,
                                   weight_decay=self.config.train.wd)
        elif self.config.train.optim == 'AdamW':
            optm = AdamW(optimizer_grouped_parameters,
                         lr=self.config.train.lr,
                         betas=(0.9, 0.999),
                         eps=1e-6,
                         weight_decay=self.config.train.wd,
                         correct_bias=False
                         )
        elif self.config.train.optim == 'AdaBeliefW':
            # adabelief+lamb+wd, from data/optimizer
            optm = AdaBeliefW(optimizer_grouped_parameters,
                              lr=self.config.train.lr,
                              betas=(0.9, 0.999),
                              eps=1e-6,
                              weight_decay=self.config.train.wd,
                              correct_bias=False,
                              lamb=self.config.train.lamb
                              )
        elif self.config.train.optim == 'Adam':
            optm = Adam(optimizer_grouped_parameters,
                        lr=self.config.train.lr,
                        betas=(0.9, 0.999),
                        eps=1e-6,
                        weight_decay=self.config.train.wd,
                        )

        warmup_steps = int(self.config.get('train.warmup_factor', 0.1) * self.total_step)
        if self.config.train.lr_schedule == 'step':
            lr_iters = [int(epoch * self.step_per_epoch) for epoch in self.config.train.lr_step]
            lr_scheduler = WarmupMultiStepLR(
                optimizer=optm,
                milestones=lr_iters,
                gamma=0.1,
                warmup_factor=self.config.train.warmup_factor,
                warmup_iters=warmup_steps,
                warmup_method="linear",
            )
        elif self.config.train.lr_schedule == 'linear':
            lr_scheduler = optimization.get_linear_schedule_with_warmup(
                optimizer=optm,
                num_warmup_steps=warmup_steps,
                num_training_steps=self.total_step)
        elif self.config.train.lr_schedule == 'cosine':
            lr_scheduler = optimization.get_cosine_schedule_with_warmup(
                optimizer=optm,
                num_warmup_steps=warmup_steps,
                num_training_steps=self.total_step,
                num_cycles=self.config.get('train.lr_num_cycles', 0.5))
        elif self.config.train.lr_schedule == 'onecycle':
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optm,
                max_lr=self.config.train_lr,
                total_steps=self.total_step
            )

        return optm, lr_scheduler
