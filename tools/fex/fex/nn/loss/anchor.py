#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: Huang Wenguan (huangwenguan@bytedance.com)
@Date: 2020-05-18 22:44:12
LastEditTime: 2020-11-17 21:26:36
LastEditors: Huang Wenguan
@Description: loss
'''

import torch
import torch.nn.functional as F


class MILNCELoss(torch.nn.Module):
    """copy from https://github.com/antoine77340/MIL-NCE_HowTo100M/blob/master/loss.py """

    def __init__(self, temperature=1.):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor_embd, candidate_embd):
        """
        anchor_embd: [bsz, dim]
        candidate_embd: [bsz * candidate, dim]
        """
        x = torch.matmul(anchor_embd, candidate_embd.t())
        x = x.view(anchor_embd.shape[0], anchor_embd.shape[0], -1) # [bsz, bsz, candidate]
        nominator = x * torch.eye(x.shape[0], device=x.device)[:, :, None]
        nominator = nominator.sum(dim=1) / self.temperature # [bsz, candidate] 每个视频和每个pos文本的分，作为分子
        nominator = torch.logsumexp(nominator, dim=1) # 分子
        denominator = torch.cat((x, x.permute(1, 0, 2)), dim=1).view(x.shape[0], -1) / self.temperature # [bsz, bsz * candidate]
        denominator = torch.logsumexp(denominator, dim=1)
        return torch.mean(denominator - nominator)


class MaxMarginLoss(torch.nn.Module):
    """ max margin loss """

    def __init__(self, margin=0.1):
        super().__init__()
        self.margin = margin

    def forward(self, text_embd, video_embd):
        sim_matrix = torch.matmul(F.normalize(text_embd), F.normalize(video_embd).t())
        diag = sim_matrix.diagonal()
        # TODO: 有一点常数的问题，对角线上会总是有margin的loss，但可能问题不大？没有回传东西
        loss = F.relu(self.margin + sim_matrix - diag.view(-1, 1)) + \
            F.relu(self.margin + sim_matrix - diag.view(1, -1))
        return loss.mean(), diag  # buggy


class NTXentLoss(torch.nn.Module):
    """
    https://arxiv.org/pdf/2002.05709.pdf 
    https://github.com/google-research/simclr/blob/2435f07519badc8ff13bbf8c17cd2c01c54c63e2/tf2/objective.py#L35
    https://github.com/sthalles/SimCLR/blob/e8a690ae4f4359528cfba6f270a9226e3733b7fa/loss/nt_xent.py#L5
    """
    def __init__(self, temperature=1.):
        super().__init__()
        self.temperature = temperature
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction="sum")

    def forward(self, text_embd, visual_embd):
        """ TODO: batch wise 
        text_embd: [bsz, dim]
        visual_embd: [bsz, dim]
        """
        
        representations = torch.cat([text_embd, visual_embd], dim=0) # [bsz, bsz*2]
        similarity_matrix = self.dot(representations, representations)
        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)
    
    def dot(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v
