#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: Huang Wenguan (huangwenguan@bytedance.com)
Date: 2020-08-22 15:01:07
LastEditTime: 2020-08-22 23:48:59
LastEditors: Huang Wenguan
Description: mostly modified from https://github.com/karpathy/minGPT
'''

from typing import Dict, Tuple, List

import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from fex.core.net import Net
from fex.nn import resnet18, resnet50
from fex import _logger as logger
from fex.optim.optimization import AdamW
from fex.optim import optimization


class IGPTNet(Net):

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        config = config.NETWORK
        self.igpt = IGPT(config)



    def forward(self, **batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        这样写主要是为了在各种accelerators下, forward可以做完training_step和validation_step
        """
        if self.training:
            return self.training_step(**batch)
        else:
            return self.validation_step(**batch)


    def training_step(self, image, x, y=None):
        """
        image: [bsz, c, h, w]
        x: [bsz, seq_len - 1]
        y: [bsz, seq_len]
        """
        logits, loss = self.igpt(image=image, idx=x, targets=y)
        self.log('loss', loss)
        return {'loss': loss, 'logits': logits}

    @torch.no_grad()
    def validation_step(self, image, x=None, y=None):
        """
        image: [bsz, c, h, w]
        x: [bsz, seq_len - 1]
        y: [bsz, seq_len]
        """
        logits, loss = self.igpt(image=image, idx=x, targets=y)
        self.log('loss', loss)
        return {'loss': loss, 'logits': logits}

    def configure_optimizers(self) -> Tuple[Optimizer, LambdaLR]:
        """
        Model定制optimizer和lr_scheduler
        """
        no_decay = ['bias', 'bn', 'norm']
        no_dacay_params_dict = {'params': [], 'weight_decay': 0.0}
        normal_params_dict = {'params': [], 'weight_decay': self.config.TRAIN.WD}
        
        for n, p in self.named_parameters():
            if any(nd in n for nd in no_decay):
                no_dacay_params_dict['params'].append(p)
            else:
                normal_params_dict['params'].append(p)
        optimizer_grouped_parameters = [no_dacay_params_dict, normal_params_dict]

        # TODO: 再看看这些参数
        optm = AdamW(optimizer_grouped_parameters,
                    lr=self.config.TRAIN.LR,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=self.config.TRAIN.WD,
                    correct_bias=False
                    )

        # TODO: 第一个epoch做warmup
        lr_scheduler = optimization.get_linear_schedule_with_warmup(
            optimizer=optm,
            num_warmup_steps=1 * self.step_per_epoch,
            num_training_steps=self.total_step)

        return optm, lr_scheduler


class IGPT(nn.Module):
    """  image captioning gpt """

    def __init__(self, config):
        super().__init__()

        self.config = config

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.seq_length, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.num_hidden_layers)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.seq_length = config.seq_length
        self.apply(self._init_weights)

        # TODO: 比较hack，先这样吧
        self.pretrained_model_path = config.pretrained_model_path
        if config.visual_type == 'RN18':
            self.resnet = resnet18(pretrained=self.pretrained_model_path,
                                expose_stages=[5])
        elif config.visual_type == 'RN50':
            self.resnet = resnet50(pretrained=self.pretrained_model_path,
                                expose_stages=[5])
        self.resnet.frozen_parameters(frozen_stages=[1,2,3,4,5], frozen_bn=True)
        self.fm_proj = nn.Linear(config.visual_hidden, config.n_embd)

        self.PAD = 2 # 忽略 pad位置的loss

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_block_size(self):
        return self.seq_length

    def forward(self, image, idx=None, targets=None):
        feature_map = self.resnet(image)['body5']
        visual_token = torch.mean(feature_map, dim=[-1, -2])
        visual_token = self.fm_proj(visual_token).unsqueeze(1)

        if idx is not None:
            b, t = idx.size()
            assert t <= self.seq_length, "Cannot forward, model block size is exhausted."
            
            # forward the GPT model
            token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
            position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
            x = token_embeddings + position_embeddings
            x = torch.cat([visual_token, x], dim=1)
        else:
            x = visual_token
        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=self.PAD)

        return logits, loss


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(math.pi / 2) * (x + 0.044715 * x ** 3)))


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.proj_up = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.proj_down = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.resid_pdrop)

        # self.mlp = nn.Sequential(
        #     nn.Linear(config.n_embd, 4 * config.n_embd),
        #     #nn.GELU(),
        #     self.activ,
        #     nn.Linear(4 * config.n_embd, config.n_embd),
        #     nn.Dropout(config.resid_pdrop),
        # )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        m = self.ln2(x)
        m = self.proj_up(m)
        m = torch.nn.functional.gelu(m)
        #m = gelu(m)
        m = self.proj_down(m)
        m = self.dropout(m)
        x = x + m
        #x = x + self.mlp(self.ln2(x)) # TODO: origin impl, but buggy
        return x


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    I believe I could have just used torch.nn.MultiheadAttention but their documentation
    is all but absent and code ugly so I don't trust it, rolling my own here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.seq_length, config.seq_length))
                                     .view(1, 1, config.seq_length, config.seq_length))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, -1e4) # todo: just use float('-inf') instead?
        
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

