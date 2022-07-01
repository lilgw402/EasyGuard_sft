#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
CLIP Model
'''

from typing import Dict, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from fex.config import CfgNode
from fex.nn import ALBert
from fex.nn.visual_tokenizer import create_visual_tokenizer

from fex.utils.distributed import AllGather, SubGroup
from fex.nn.loss import LearnableNTXentLoss
allgather = AllGather.apply

from fex.core import Net

try:
    from ptx.model.deberta.model import DebertaBare, DebertaEncoder, FastDebertaEncoder
    from ptx.model.deberta.disentangled_attention import DAConfig
except Exception as e:
    log.warning('ptx is not installed!')


class CLIP(Net):
    """
    双塔模型
    一侧是 Text Encoder: 一般是一个6层 albert
    一侧是 Visual Encoder: 一般是 Resnet 或者 Vit 等

    """

    def __init__(self,
                 config: CfgNode = None,
                 visual_type: str = 'VitB32',
                 visual_config: dict = {
                     'output_dim': 512,
                     'vit_dropout': 0.1,
                     'vit_emb_dropout': 0.0,
                     'patch_length': 49
                 },
                 text_type: str = 'ALBert',
                 project_mode: str = 'default',
                 gpuwise_nce: bool = False,
                 init_tau: float = 0.07,
                 tau_clamp: float = 4.6051,
                 nce_world_size: int = None,
                 **kwargs):
        super().__init__(config, **kwargs)

        if nce_world_size is None:
            # falling back to gpuwise_nce
            nce_world_size = -1 if gpuwise_nce else 1

        self.nce_world_size = nce_world_size  # GPU 同步 batch 负例
        self.visual_type = visual_type

        if text_type == 'ALBert':
            self.text = ALBert(config.BERT)  # TODO: 这里写死了用albert，后面做成可配置的
        elif text_type == 'Deberta':
            self.text = DebertaBare(**config.BERT)
        self.visual = create_visual_tokenizer(visual_type, **(visual_config))
        self.t_projector, self.v_projector = self.init_projector(project_mode=project_mode, visual_size=visual_config.get('output_dim'))
        self.calc_nce_loss = LearnableNTXentLoss(init_tau=init_tau, clamp=tau_clamp)

    def forward(self,
                image: torch.Tensor,
                input_ids: torch.Tensor,
                input_mask: torch.Tensor,
                input_segment_ids: torch.Tensor = None,
                *args, **kwargs):
        # 1. encode
        v_emb = self.encode_image(image)
        t_emb = self.encode_text(input_ids, input_mask, input_segment_ids)

        # 2. nce loss
        if self.nce_world_size <= 0 or self.nce_world_size >= self.trainer.world_size:
            # global nce
            t_emb = allgather(t_emb, self.trainer.rank, self.trainer.world_size)
            v_emb = allgather(v_emb, self.trainer.rank, self.trainer.world_size)
        elif self.nce_world_size > 1:
            # sub group nce
            sub_group = SubGroup.group
            assert sub_group is not None, 'sub group is not initialized'
            t_emb = allgather(t_emb, sub_group.rank(), sub_group.size(), sub_group)
            v_emb = allgather(v_emb, sub_group.rank(), sub_group.size(), sub_group)
        nce_loss = self.calc_nce_loss(v_emb=v_emb, t_emb=t_emb)
        return {'loss': nce_loss, 'nce_temperature': self.calc_nce_loss.tau}

    def encode_image(self, image: torch.Tensor):
        v_out = self.visual(image)
        if self.visual_type == 'RN50':  # resnet 的逻辑不太兼容，后面统一一下
            v_out = torch.mean(v_out['body5'], dim=[2, 3])  # [bsz, 2048]
        v_emb = self.v_projector(v_out)
        return v_emb

    def encode_text(self,
                    input_ids: torch.Tensor,
                    input_mask: torch.Tensor,
                    input_segment_ids: torch.Tensor = None
                    ):
        if input_segment_ids is None:
            input_segment_ids = torch.zeros_like(input_ids, device=input_ids.device)
        # t_out = self.text(input_ids=input_ids,
        #                   input_segment_ids=input_segment_ids,
        #                   input_mask=input_mask)['pooled_output']
        # t_emb = self.t_projector(t_out)

        t_out = self.text(input_ids,
                          input_segment_ids,
                          input_mask,
                          output_pooled=True)['pooled_output']
        t_emb = self.t_projector(t_out)

        return t_emb

    def init_projector(self,
                       project_mode: str = 'default',
                       hidden_size: int = 768,
                       visual_size: int = 512
                       ):
        if project_mode == 'default':
            t_projector = torch.nn.Linear(hidden_size, 128)
            v_projector = torch.nn.Linear(visual_size, 128, bias=False)
        elif project_mode == '1024':
            v_projector = torch.nn.Sequential(
                torch.nn.Linear(visual_size, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 1024)
            )
            t_projector = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 1024)
            )
        elif project_mode == '1024_v2':  # deprecated
            t_projector = torch.nn.Linear(hidden_size, 1024)
            v_projector = torch.nn.Sequential(
                torch.nn.Linear(visual_size, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 1024, bias=False)
            )
        else:
            raise ValueError(f'project mode [{project_mode}] is not valid')
        return t_projector, v_projector
