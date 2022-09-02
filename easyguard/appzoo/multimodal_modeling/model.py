# -*- coding: utf-8 -*-
import os.path

import yaml
from types import SimpleNamespace
from typing import Union, List, Tuple

import math
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler

try:
    import cruise
except ImportError:
    print("[ERROR] cruise is not installed! Please refer this doc: https://bytedance.feishu.cn/wiki/wikcnGP7yzZAuKpPfL6jRJKl2ag")

from cruise import CruiseModule
from cruise.utilities.cloud_io import load
from cruise.utilities.hdfs_io import hexists, hopen

from .albert import ALBert
from .swin import SwinTransformer
from ...utils.losses import LearnableNTXentLoss, LearnablePCLLoss, SCELoss

class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001,
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1
        ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class FashionBertv2(CruiseModule):
    def __init__(self, config_text: str = "hdfs://haruna/home/byte_ecom_govern/user/liuyuhang/pretrain/fashion_bert_v2/config_text.yaml",
                 config_visual: str = "hdfs://haruna/home/byte_ecom_govern/user/liuyuhang/pretrain/fashion_bert_v2/config_visual.yaml",
                 config_fusion: str = "hdfs://haruna/home/byte_ecom_govern/user/liuyuhang/pretrain/fashion_bert_v2/config_fusion.yaml",
                 learning_rate: float = 1e-4,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.02,
                 load_pretrained: str = "hdfs://haruna/home/byte_search_nlp_lq/multimodal/modelhub/swinb224_ds_20220119/model1.th"):
        super(FashionBertv2, self).__init__()
        self.save_hparams()

    def setup(self, stage) -> None:
        """
        Load yaml file as config class
        """
        assert hexists(self.hparams.config_text) and hexists(self.hparams.config_visual) and hexists(self.hparams.config_fusion)
        with hopen(self.hparams.config_text) as fp:
            self.config_text = SimpleNamespace(**yaml.load(fp, yaml.Loader))
        with hopen(self.hparams.config_visual) as fp:
            self.config_visual = SimpleNamespace(**yaml.load(fp, yaml.Loader))
        with hopen(self.hparams.config_fusion) as fp:
            self.config_fusion = SimpleNamespace(**yaml.load(fp, yaml.Loader))

        """
        Initialize modules
        """
        self.text = ALBert(self.config_text)
        self.visual = SwinTransformer(
            img_size=self.config_visual.img_size,
            num_classes=self.config_visual.output_dim,
            embed_dim=self.config_visual.embed_dim,
            depths=self.config_visual.depths,
            num_heads=self.config_visual.num_heads
        )
        self.visual_feat = nn.Linear(1024, self.config_visual.output_dim)
        self.visual_pos = nn.Embedding(256, self.config_visual.output_dim)
        self.visual_fuse = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=self.config_visual.output_dim,
                                                     nhead=8,
                                                     batch_first=True,
                                                     activation="gelu"),
            num_layers=1
        )
        self.visual_pooler = nn.Sequential(
            nn.Linear(self.config_visual.output_dim, self.config_visual.output_dim),
            nn.GELU()
        )

        self.t_projector = nn.Linear(self.config_text.hidden_size,
                                     self.config_fusion.hidden_size)
        self.v_projector = nn.Linear(self.config_visual.output_dim,
                                     self.config_fusion.hidden_size)

        self.t_projector_fuse = nn.Linear(self.config_text.hidden_size,
                                          self.config_fusion.hidden_size)
        self.v_projector_fuse = nn.Linear(self.config_visual.output_dim,
                                          self.config_fusion.hidden_size)
        self.segment_embedding = nn.Embedding(2, self.config_fusion.hidden_size)
        self.ln_text = nn.LayerNorm(self.config_fusion.hidden_size)
        self.ln_visual = nn.LayerNorm(self.config_fusion.hidden_size)
        self.ln_cls = nn.LayerNorm(self.config_fusion.hidden_size)
        self.fuse_cls = nn.Linear(self.config_fusion.hidden_size * 2, self.config_fusion.hidden_size)

        self.fuse_dropout = nn.Dropout(0.2)
        self.fuse = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.config_fusion.hidden_size,
                nhead=8,
                batch_first=True,
                activation="gelu"
            ),
            num_layers=self.config_fusion.num_layers
        )
        self.fuse_pooler = nn.Linear(self.config_fusion.hidden_size, self.config_fusion.hidden_size)

        """
        Initialize output layer
        """
        self.fuse_category = nn.Sequential(
            nn.Linear(self.config_fusion.hidden_size, self.config_fusion.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(self.config_fusion.embd_pdrop),
            nn.Linear(self.config_fusion.hidden_size * 2, self.config_fusion.category_logits_level2 + 1)
        )

        """
        Initialize loss
        """
        self.calc_nce_loss_vt = LearnableNTXentLoss(
            init_tau=self.config_fuse.init_tau,
            clamp=self.config_fuse.tau_clamp
        )  # 底层图-文CLIP损失
        self.calc_pcl_loss_ff = LearnablePCLLoss(
            init_tau=self.config_fuse.init_tau,
            clamp=self.config_fuse.tau_clamp,
            num_labels=self.config_fuse.category_logits_level1 + 1
        )  # 上层融合-融合CLIP损失，使用一级标签
        self.category_pred_loss = SCELoss(
            alpha=1.0,
            beta=0.5,
            num_classes=self.config_fusion.category_logits_level2 + 1
        )  # 融合表征的预测损失

        """
        Initialize some fixed parameters.
        """
        self.PAD = 2
        self.MASK = 1
        self.SEP = 3
        # 文本的position ids
        self.text_position_ids = nn.Parameter(
            data=torch.arange(0, 512, dtype=torch.int64),
            requires_grad=False
        )  # [512, ]
        self.image_position_ids = nn.Parameter(
            data=torch.arange(0, 512, dtype=torch.int64),
            requires_grad=False
        )  # [512, ]
        self.text_segment_ids = nn.Parameter(
            data=torch.zeros(size=(512,), dtype=torch.int64),
            requires_grad=False
        )  # [512, ]
        self.image_segment_ids = nn.Parameter(
            data=torch.ones(size=(512,), dtype=torch.int64),
            requires_grad=False
        )  # [512, ]
        self.image_masks = nn.Parameter(
            data=torch.ones(size=(512,), dtype=torch.int64),
            requires_grad=False
        )  # [512, ]
        self.clsf_masks = nn.Parameter(
            data=torch.ones(size=(1,), dtype=torch.int64),
            requires_grad=False
        )

        self.initialize_weights()

    def initialize_weights(self):
        if hexists(self.hparams.load_pretrained):
            state_dict = load(self.hparams.load_pretrained, map_location="cpu")
            self.load_state_dict(state_dict, strict=False)

    def forward_text(self, token_ids: torch.Tensor):
        text_masks = (token_ids != self.PAD).long()
        text_segment_ids = (token_ids == self.SEP).long()
        text_segment_ids = torch.cumsum(text_segment_ids, dim=-1) - text_segment_ids
        text_segment_ids = torch.clamp(text_segment_ids, min=0, max=1)
        batch_size, text_length = token_ids.shape
        position_ids = self.text_position_ids[:text_length].unsqueeze(0).expand((batch_size, -1))  # [B, M]

        t_out = self.text(input_ids=token_ids,
                          input_segment_ids=text_segment_ids,
                          input_mask=text_masks,
                          position_ids=position_ids)
        t_emb, t_rep = t_out['encoded_layers'][-1], \
                       t_out['pooled_output']
        t_rep = self.t_projector(t_rep)

        return t_rep, t_emb  # [B, 512], [B, 256, 768]

    def forward_image(self, image: torch.Tensor):
        img_out = self.visual(image, return_dict=True)
        v_emb, v_rep = self.visual_feat(img_out["feature_map"]), img_out["pooled_out"]

        v_cat = torch.cat([v_rep.unsqueeze(1), v_emb], dim=1)  # [B, 1 + N, d_v]
        batch_size, image_length, _ = v_cat.shape
        position_ids = self.image_position_ids[:image_length].unsqueeze(0).expand((batch_size, -1))
        v_cat = v_cat + self.visual_pos(position_ids)
        v_cat = self.visual_fuse(v_cat)  # [B, 1 + N, d_v]
        v_rep = self.visual_pooler(v_cat[:, 0])  # [B, d_v]
        v_rep = self.v_projector(v_rep)

        return v_rep, v_cat  # [B, 512], [B, 1 + 49, 512]

    def forward_fuse(self, t_emb, t_rep,
                     text_masks, v_emb, v_rep):
        batch_size, text_length, _ = t_emb.shape
        text_segment_ids = self.text_segment_ids[:text_length].unsqueeze(0).expand((batch_size, -1))
        t_emb = self.ln_text(self.t_projector_fuse(t_emb) +
                             self.segment_embedding(text_segment_ids))  # [B, M, d_f]

        batch_size, image_length, _ = v_emb.shape
        image_segment_ids = self.image_segment_ids[:image_length].unsqueeze(0).expand((batch_size, -1))
        v_emb = self.ln_visual(self.v_projector_fuse(v_emb) +
                               self.segment_embedding(image_segment_ids))  # [B, 1 + N, d_f]
        cls_emb = self.ln_cls(self.fuse_cls(torch.cat([t_rep, v_rep], dim=-1))).unsqueeze(1)  # [B, 1, d_f]
        fuse_emb = self.fuse_dropout(torch.cat([cls_emb, t_emb, v_emb], dim=1))  # [B, 1 + M + 1 + N, d_f]

        image_masks = self.image_masks[:image_length].unsqueeze(0).expand((batch_size, -1))
        cls_masks = self.clsf_masks.unsqueeze(0).expand((batch_size, -1))
        fuse_mask = torch.cat([cls_masks, text_masks, image_masks], dim=1) == 0  # [B, 1 + M + 1 + N], True indicates mask
        fuse_emb = self.fuse(fuse_emb, src_key_padding_mask=fuse_mask)  # [B, 1 + M + 1 + N, d_f]

        fuse_rep = self.fuse_pooler(fuse_emb[:, 0])  # [B, d_f]
        fuse_cat = self.fuse_category(fuse_emb[:, 0])  # [B, num_categories]

        return fuse_rep, fuse_cat

    def forward(self,
                token_ids: torch.Tensor,
                image: torch.Tensor,
                label: torch.Tensor = None,
                **kwargs):
        text_masks = (token_ids != self.PAD).long()
        t_rep, t_emb = self.forward_text(token_ids)
        v_rep, v_emb = self.forward_image(image)
        fuse_rep, fuse_cat = self.forward_fuse(
            t_emb, t_rep, text_masks, v_emb, v_rep
        )

        return {
            "t_rep": t_rep,
            "v_rep": v_rep,
            "fuse_rep": fuse_rep,
            "fuse_cat": fuse_cat
        }

    def cal_pt_loss(self, **kwargs):
        for key in ["t_rep", "v_rep", "fuse_rep", "label_l1"]:
            if key in kwargs:
                kwargs[key] = self.all_gather(kwargs[key].contiguous())

        vt_loss = self.calc_nce_loss_vt(kwargs["t_rep"], kwargs["v_rep"])

        assert "label_l1" in kwargs
        ff_loss = self.calc_pcl_loss_ff(kwargs["fuse_rep"], kwargs["label_l1"])

        assert "label" in kwargs and "fuse_cat" in kwargs
        cat_loss = self.category_pred_loss(kwargs["fuse_cat"], kwargs["label"])

        loss = (vt_loss + ff_loss + cat_loss) / 3

        return {
            "loss": loss,
            "vt_loss": vt_loss,
            "ff_loss": ff_loss,
            "cat_loss": cat_loss
        }

    def training_step(self, batch, idx):
        token_ids, image, label, label_l1 = batch['token_ids'], batch['image'], batch['label'], batch["label_l1"]
        rep_dict = self.forward(token_ids, image)
        rep_dict.update(batch)

        loss_dict = self.cal_pt_loss(**rep_dict)
        loss_dict["logits"] = rep_dict["fuse_cat"]
        loss_dict["label"] = label

        return loss_dict

    def validation_step(self, batch, idx):
        return self.training_step(batch, idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=self.hparams.betas,
            eps=self.hparams.eps,
            weight_decay=self.hparams.weight_decay
        )

        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.trainer.total_steps,
            cycle_mult=1.0,
            max_lr=self.hparams.learning_rate,
            min_lr=0,
            warmup_steps=2000
        )

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
