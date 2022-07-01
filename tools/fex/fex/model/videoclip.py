#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Video CLIP Model
'''

import torch

from fex.nn.backbone.falbert import FrameALBert
from fex.model.clip import CLIP
from fex.config import CfgNode

from fex.core import Net
from fex.utils.distributed import AllGather
from fex.nn.loss import LearnableNTXentLoss
allgather = AllGather.apply


class VideoCLIP(Net):
    def __init__(self,
                 config: CfgNode = None,
                 project_mode: str = 'default',
                 gpuwise_nce: bool = False,
                 init_tau: float = 0.07,
                 tau_clamp: float = 4.6051,
                 **kwargs):
        super().__init__(config, **kwargs)

        self.gpuwise_nce = gpuwise_nce  # GPU 同步 batch 负例
        self.falbert = FrameALBert(config)
        self.t_projector, self.v_projector = self.init_projector(project_mode=project_mode)
        self.calc_nce_loss = LearnableNTXentLoss(init_tau=init_tau, clamp=tau_clamp)

    def forward(self,
                images: torch.Tensor,
                images_mask: torch.Tensor,
                input_ids: torch.Tensor,
                input_mask: torch.Tensor,
                input_segment_ids: torch.Tensor = None,
                title_input_ids: torch.Tensor = None,
                title_input_mask: torch.Tensor = None,
                title_input_segment_ids: torch.Tensor = None,
                *args, **kwargs):

        # query
        t_emb = self.encode_text(input_ids=input_ids, input_segment_ids=input_segment_ids, input_mask=input_mask)['pooled_output']
        t_emb = self.t_projector(t_emb)

        # doc
        if title_input_ids is not None:  # 如果有 title_input_ids，就按多模态的方式来编码
            v_emb = self.encode_multimodal(
                input_ids=title_input_ids,
                input_segment_ids=title_input_segment_ids,
                input_mask=title_input_mask,
                images=images,
                images_mask=images_mask)['pooled_output']
        else:  # 否则是指对图片编码
            v_emb = self.encode_image(images=images, images_mask=images_mask)['pooled_output']
        v_emb = self.v_projector(v_emb)

        # loss
        if self.gpuwise_nce:
            t_emb = allgather(t_emb, self.trainer.rank, self.trainer.world_size)
            v_emb = allgather(v_emb, self.trainer.rank, self.trainer.world_size)
        nce_loss = self.calc_nce_loss(v_emb=v_emb, t_emb=t_emb)
        return {'loss': nce_loss, 'nce_temperature': self.calc_nce_loss.tau}

    def encode_image(self,
                     images: torch.Tensor,
                     images_mask: torch.Tensor = None,
                     visual_embeds: torch.tensor = None
                     ):
        if len(images.shape) == 4:
            images = images.unsqueeze(1)
        if images_mask is None:
            if visual_embeds is None:
                images_mask = torch.ones(images.shape[0:2], device=images.device, dtype=torch.long)
            else:
                images_mask = torch.ones(visual_embeds.shape[0:2], device=visual_embeds.device, dtype=torch.long)
        v_out = self.falbert(frames=images, frames_mask=images_mask, visual_embeds=visual_embeds, mode='v')
        return v_out

    def encode_text(self,
                    input_ids: torch.Tensor,
                    input_mask: torch.Tensor,
                    input_segment_ids: torch.Tensor = None
                    ):
        if input_segment_ids is None:
            input_segment_ids = torch.zeros_like(input_ids, device=input_ids.device)
        t_out = self.falbert(input_ids=input_ids, input_segment_ids=input_segment_ids, input_mask=input_mask, mode='t')
        return t_out

    def encode_multimodal(self, input_ids, input_segment_ids, input_mask, images=None, images_mask=None, visual_embeds=None, *args, **kwargs):
        if images_mask is None:
            if visual_embeds is None:
                images_mask = torch.ones(images.shape[0:2], device=images.device, dtype=torch.long)
            else:
                images_mask = torch.ones(visual_embeds.shape[0:2], device=visual_embeds.device, dtype=torch.long)
        mmout = self.falbert(input_ids=input_ids, input_segment_ids=input_segment_ids, input_mask=input_mask,
                             frames=images, frames_mask=images_mask, visual_embeds=visual_embeds, mode='tv')
        return mmout

    def init_projector(self,
                       project_mode: str = 'default',
                       hidden_size: int = 768,
                       ):
        if project_mode == 'default':
            projector = torch.nn.Linear(hidden_size, 128)
            v_projector = projector
            t_projector = projector
        else:
            raise ValueError(f'project mode [{project_mode}] is not valid')
        return t_projector, v_projector
