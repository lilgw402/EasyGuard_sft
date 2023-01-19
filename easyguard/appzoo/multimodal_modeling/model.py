# -*- coding: utf-8 -*-

import os
import math
import json
from collections import OrderedDict
from types import SimpleNamespace
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import _LRScheduler
from ptx.model import Model

try:
    import cruise
except ImportError:
    print(
        "[ERROR] cruise is not installed! Please refer this doc: https://bytedance.feishu.cn/wiki/wikcnGP7yzZAuKpPfL6jRJKl2ag"
    )

from cruise import CruiseModule
from cruise.utilities.cloud_io import load
from cruise.utilities.hdfs_io import hexists, hopen

from easyguard.modelzoo.models.albert import ALBert
from easyguard.modelzoo.models.swin import SwinTransformer

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

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1,
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

        super(CosineAnnealingWarmupRestarts, self).__init__(
            optimizer, last_epoch
        )

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps
                + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                + (self.max_lr - base_lr)
                * (
                    1
                    + math.cos(
                        math.pi
                        * (self.step_in_cycle - self.warmup_steps)
                        / (self.cur_cycle_steps - self.warmup_steps)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = (
                    int(
                        (self.cur_cycle_steps - self.warmup_steps)
                        * self.cycle_mult
                    )
                    + self.warmup_steps
                )
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(
                        math.log(
                            (
                                epoch
                                / self.first_cycle_steps
                                * (self.cycle_mult - 1)
                                + 1
                            ),
                            self.cycle_mult,
                        )
                    )
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps
                        * (self.cycle_mult**n - 1)
                        / (self.cycle_mult - 1)
                    )
                    self.cur_cycle_steps = (
                        self.first_cycle_steps * self.cycle_mult ** (n)
                    )
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class FashionBertv2(CruiseModule):
    def __init__(
        self,
        config_text: str = "hdfs://haruna/home/byte_ecom_govern/user/liuyuhang/pretrain/fashion_bert_v2/config_text.yaml",
        config_visual: str = "hdfs://haruna/home/byte_ecom_govern/user/liuyuhang/pretrain/fashion_bert_v2/config_visual.yaml",
        config_fusion: str = "hdfs://haruna/home/byte_ecom_govern/user/liuyuhang/pretrain/fashion_bert_v2/config_fusion.yaml",
        learning_rate: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.02,
        load_pretrained: str = "hdfs://haruna/home/byte_search_nlp_lq/multimodal/modelhub/swinb224_ds_20220119/model1.th",
    ):
        super(FashionBertv2, self).__init__()
        self.save_hparams()

    def setup(self, stage) -> None:
        """
        Load yaml file as config class
        """
        assert (
            hexists(self.hparams.config_text)
            and hexists(self.hparams.config_visual)
            and hexists(self.hparams.config_fusion)
        )
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
            num_heads=self.config_visual.num_heads,
        )
        self.visual_feat = nn.Linear(1024, self.config_visual.output_dim)
        self.visual_pos = nn.Embedding(256, self.config_visual.output_dim)
        self.visual_fuse = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.config_visual.output_dim,
                nhead=8,
                batch_first=True,
                activation="gelu",
            ),
            num_layers=1,
        )
        self.visual_pooler = nn.Sequential(
            nn.Linear(
                self.config_visual.output_dim, self.config_visual.output_dim
            ),
            nn.GELU(),
        )

        self.t_projector = nn.Linear(
            self.config_text.hidden_size, self.config_fusion.hidden_size
        )
        self.v_projector = nn.Linear(
            self.config_visual.output_dim, self.config_fusion.hidden_size
        )

        self.t_projector_fuse = nn.Linear(
            self.config_text.hidden_size, self.config_fusion.hidden_size
        )
        self.v_projector_fuse = nn.Linear(
            self.config_visual.output_dim, self.config_fusion.hidden_size
        )
        self.segment_embedding = nn.Embedding(2, self.config_fusion.hidden_size)
        self.ln_text = nn.LayerNorm(self.config_fusion.hidden_size)
        self.ln_visual = nn.LayerNorm(self.config_fusion.hidden_size)
        self.ln_cls = nn.LayerNorm(self.config_fusion.hidden_size)
        self.fuse_cls = nn.Linear(
            self.config_fusion.hidden_size * 2, self.config_fusion.hidden_size
        )

        self.fuse_dropout = nn.Dropout(0.2)
        self.fuse = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.config_fusion.hidden_size,
                nhead=8,
                batch_first=True,
                activation="gelu",
            ),
            num_layers=self.config_fusion.num_layers,
        )
        self.fuse_pooler = nn.Linear(
            self.config_fusion.hidden_size, self.config_fusion.hidden_size
        )

        """
        Initialize output layer
        """
        self.fuse_category = nn.Sequential(
            nn.Linear(
                self.config_fusion.hidden_size,
                self.config_fusion.hidden_size * 2,
            ),
            nn.GELU(),
            nn.Dropout(self.config_fusion.embd_pdrop),
            nn.Linear(
                self.config_fusion.hidden_size * 2,
                self.config_fusion.category_logits_level2 + 1,
            ),
        )

        """
        Initialize loss
        """
        self.calc_nce_loss_vt = LearnableNTXentLoss(
            init_tau=self.config_fusion.init_tau,
            clamp=self.config_fusion.tau_clamp,
        )  # 底层图-文CLIP损失
        self.calc_pcl_loss_ff = LearnablePCLLoss(
            init_tau=self.config_fusion.init_tau,
            clamp=self.config_fusion.tau_clamp,
            num_labels=self.config_fusion.category_logits_level1 + 1,
        )  # 上层融合-融合CLIP损失，使用一级标签
        self.category_pred_loss = SCELoss(
            alpha=1.0,
            beta=0.5,
            num_classes=self.config_fusion.category_logits_level2 + 1,
        )  # 融合表征的预测损失

        """
        Initialize some fixed parameters.
        """
        self.PAD = 2
        self.MASK = 1
        self.SEP = 3
        # 文本的position ids
        self.text_position_ids = nn.Parameter(
            data=torch.arange(0, 512, dtype=torch.int64), requires_grad=False
        )  # [512, ]
        self.image_position_ids = nn.Parameter(
            data=torch.arange(0, 512, dtype=torch.int64), requires_grad=False
        )  # [512, ]
        self.text_segment_ids = nn.Parameter(
            data=torch.zeros(size=(512,), dtype=torch.int64),
            requires_grad=False,
        )  # [512, ]
        self.image_segment_ids = nn.Parameter(
            data=torch.ones(size=(512,), dtype=torch.int64), requires_grad=False
        )  # [512, ]
        self.image_masks = nn.Parameter(
            data=torch.ones(size=(512,), dtype=torch.int64), requires_grad=False
        )  # [512, ]
        self.clsf_masks = nn.Parameter(
            data=torch.ones(size=(1,), dtype=torch.int64), requires_grad=False
        )

        self.initialize_weights()

    def initialize_weights(self):
        if hexists(self.hparams.load_pretrained):
            state_dict_ori = self.state_dict()
            state_dict = load(self.hparams.load_pretrained, map_location="cpu")
            state_dict_new = OrderedDict()
            for key, value in state_dict.items():
                if (
                    key in state_dict_ori
                    and state_dict_ori[key].shape == state_dict[key].shape
                ):
                    state_dict_new[key] = value
            self.load_state_dict(state_dict_new, strict=False)

    def forward_text(self, token_ids: torch.Tensor):
        text_masks = (token_ids != self.PAD).long()
        text_segment_ids = (token_ids == self.SEP).long()
        text_segment_ids = (
            torch.cumsum(text_segment_ids, dim=-1) - text_segment_ids
        )
        text_segment_ids = torch.clamp(text_segment_ids, min=0, max=1)
        batch_size, text_length = token_ids.shape
        position_ids = (
            self.text_position_ids[:text_length]
            .unsqueeze(0)
            .expand((batch_size, -1))
        )  # [B, M]

        t_out = self.text(
            input_ids=token_ids,
            input_segment_ids=text_segment_ids,
            input_mask=text_masks,
            position_ids=position_ids,
        )
        t_emb, t_rep = t_out["encoded_layers"][-1], t_out["pooled_output"]
        t_rep = self.t_projector(t_rep)

        return t_rep, t_emb  # [B, 512], [B, 256, 768]

    def forward_image(self, image: torch.Tensor):
        img_out = self.visual(image, return_dict=True)
        v_emb, v_rep = (
            self.visual_feat(img_out["feature_map"]),
            img_out["pooled_out"],
        )

        v_cat = torch.cat([v_rep.unsqueeze(1), v_emb], dim=1)  # [B, 1 + N, d_v]
        batch_size, image_length, _ = v_cat.shape
        position_ids = (
            self.image_position_ids[:image_length]
            .unsqueeze(0)
            .expand((batch_size, -1))
        )
        v_cat = v_cat + self.visual_pos(position_ids)
        v_cat = self.visual_fuse(v_cat)  # [B, 1 + N, d_v]
        v_rep = self.visual_pooler(v_cat[:, 0])  # [B, d_v]
        v_rep = self.v_projector(v_rep)

        return v_rep, v_cat  # [B, 512], [B, 1 + 49, 512]

    def forward_fuse(self, t_emb, t_rep, text_masks, v_emb, v_rep):
        batch_size, text_length, _ = t_emb.shape
        text_segment_ids = (
            self.text_segment_ids[:text_length]
            .unsqueeze(0)
            .expand((batch_size, -1))
        )
        t_emb = self.ln_text(
            self.t_projector_fuse(t_emb)
            + self.segment_embedding(text_segment_ids)
        )  # [B, M, d_f]

        batch_size, image_length, _ = v_emb.shape
        image_segment_ids = (
            self.image_segment_ids[:image_length]
            .unsqueeze(0)
            .expand((batch_size, -1))
        )
        v_emb = self.ln_visual(
            self.v_projector_fuse(v_emb)
            + self.segment_embedding(image_segment_ids)
        )  # [B, 1 + N, d_f]
        cls_emb = self.ln_cls(
            self.fuse_cls(torch.cat([t_rep, v_rep], dim=-1))
        ).unsqueeze(
            1
        )  # [B, 1, d_f]
        fuse_emb = self.fuse_dropout(
            torch.cat([cls_emb, t_emb, v_emb], dim=1)
        )  # [B, 1 + M + 1 + N, d_f]

        image_masks = (
            self.image_masks[:image_length]
            .unsqueeze(0)
            .expand((batch_size, -1))
        )
        cls_masks = self.clsf_masks.unsqueeze(0).expand((batch_size, -1))
        fuse_mask = (
            torch.cat([cls_masks, text_masks, image_masks], dim=1) == 0
        )  # [B, 1 + M + 1 + N], True indicates mask
        fuse_emb = self.fuse(
            fuse_emb, src_key_padding_mask=fuse_mask
        )  # [B, 1 + M + 1 + N, d_f]

        fuse_rep = self.fuse_pooler(fuse_emb[:, 0])  # [B, d_f]
        fuse_cat = self.fuse_category(fuse_emb[:, 0])  # [B, num_categories]

        return fuse_rep, fuse_cat

    def forward(
        self,
        token_ids: torch.Tensor,
        image: torch.Tensor,
        label: torch.Tensor = None,
        **kwargs,
    ):
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
            "fuse_cat": fuse_cat,
        }

    def cal_pt_loss(self, **kwargs):
        for key in ["t_rep", "v_rep", "fuse_rep", "label_l1"]:
            if key in kwargs:
                kwargs[key] = self.all_gather(kwargs[key].contiguous())
                kwargs[key] = kwargs[key].flatten(0, 1)

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
            "cat_loss": cat_loss,
        }

    def training_step(self, batch, idx):
        token_ids, image, label, label_l1 = (
            batch["token_ids"],
            batch["image"],
            batch["label"],
            batch["label_l1"],
        )
        rep_dict = self.forward(token_ids, image)
        rep_dict.update(batch)

        loss_dict = self.cal_pt_loss(**rep_dict)

        return loss_dict

    def validation_step(self, batch, idx):
        return self.training_step(batch, idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=self.hparams.betas,
            eps=self.hparams.eps,
            weight_decay=self.hparams.weight_decay,
        )

        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.trainer.total_steps,
            cycle_mult=1.0,
            max_lr=self.hparams.learning_rate,
            min_lr=0,
            warmup_steps=2000,
        )

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


class AttentionPooler(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooler, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, x):
        alpha = F.softmax(self.mlp(x), dim=-2)
        out = (alpha * x).sum(-2)

        return out

class AttMaxPooling(nn.Module):
    """
    使用Attention Aggregation + Max Pooling来聚合多个特征，得到一个融合特征；
    使用FC层映射到新的空间，用于对比学习；
    """
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super(AttMaxPooling, self).__init__()

        self.att_pool = AttentionPooler(input_dim)
        self.max_pool = nn.AdaptiveMaxPool1d(output_size=1)

        self.fc = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        """
        :param x: [B, N, d_h]
        """
        x_att = self.att_pool(x)  # [B, d_h]
        x_max = self.max_pool(x.transpose(1, 2)).squeeze()  # [B, d_h]
        x_out = self.fc(torch.cat([x_att, x_max], dim=-1))

        return x_out

class FashionProduct(CruiseModule):
    def __init__(
        self,
        config_visual: str = "hdfs://haruna/home/byte_ecom_govern/user/liuyuhang/pretrain/fashion_product/config_visual.yaml",
        config_fusion: str = "hdfs://haruna/home/byte_ecom_govern/user/liuyuhang/pretrain/fashion_product/config_fusion.yaml",
        learning_rate: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.02,
        load_pretrained: str = "hdfs://haruna/home/byte_ecom_govern/user/liuyuhang/pretrain/fashion_bert_v2/model_state_epoch_83332.th",
        pretrained_model_dir: str = "hdfs://haruna/home/byte_ecom_govern/user/yangzheming/asr_model/zh_deberta_base_l6_emd_20210720",
        local_pretrained_model_dir_prefix: str = "/opt/tiger/liuyuhang/ckpt/",
    ):
        super(FashionProduct, self).__init__()
        self.save_hparams()
        
        """
        Used for loading Deberta model from ptx.
        """
        suffix = self.hparams.pretrained_model_dir.strip("/").split("/")[-1]
        self.local_pretrained_model_dir = (
            f"{self.hparams.local_pretrained_model_dir_prefix}/{suffix}"
        )
    
    def local_rank_zero_prepare(self) -> None:
        # download the tokenizer once per node
        if not os.path.exists(self.local_pretrained_model_dir):
            os.makedirs(
                self.hparams.local_pretrained_model_dir_prefix, exist_ok=True
            )
            os.system(
                f"hdfs dfs -copyToLocal {self.hparams.pretrained_model_dir} {self.local_pretrained_model_dir}"
            )
    
    def setup(self, stage) -> None:
        """
        Load yaml file as config class
        """
        assert (
            hexists(self.hparams.config_visual)
            and hexists(self.hparams.config_fusion)
        )
        with hopen(self.hparams.config_visual) as fp:
            self.config_visual = SimpleNamespace(**yaml.load(fp, yaml.Loader))
        with hopen(self.hparams.config_fusion) as fp:
            self.config_fusion = SimpleNamespace(**yaml.load(fp, yaml.Loader))
        
        """
        Text Encoder
        """
        self.deberta = Model.from_option('file:%s|strict=false' % (self.local_pretrained_model_dir))

        """
        Visual Encoder
        """
        self.visual = SwinTransformer(
            img_size=self.config_visual.img_size,
            num_classes=self.config_visual.output_dim,
            embed_dim=self.config_visual.embed_dim,
            depths=self.config_visual.depths,
            num_heads=self.config_visual.num_heads,
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
            nn.Linear(self.config_visual.output_dim, 
                      self.config_visual.output_dim),
            nn.GELU()
        )

        """
        Calculate CLIP loss
        """
        self.t_projector = nn.Linear(768,
                                     self.config_fusion.hidden_size)
        self.v_projector = nn.Linear(self.config_visual.output_dim,
                                     self.config_fusion.hidden_size)
        
        """
        Fuse multiple text/vision fields features
        """
        self.t_projector_fuse = nn.Linear(768,
                                          self.config_fusion.hidden_size)
        self.v_projector_fuse = nn.Linear(self.config_visual.output_dim,
                                          self.config_fusion.hidden_size)
        self.fuse_segment_embedding = nn.Embedding(64, self.config_fusion.hidden_size)
        self.ln_text = nn.LayerNorm(self.config_fusion.hidden_size)
        self.ln_visual = nn.LayerNorm(self.config_fusion.hidden_size)
        self.ln_cls = nn.LayerNorm(self.config_fusion.hidden_size)
        self.fuse_cls = nn.Parameter(
            data=torch.randn((1, self.config_fusion.hidden_size)),
            requires_grad=True
        )

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
        self.fuse_max_pooler = nn.AdaptiveMaxPool1d(output_size=1)

        """
        Pretraining mode, prepare loss functions
        """
        if True:
            """
            (1) 1 modality-level clip loss
            """
            init_tau = self.config_fusion.init_tau
            tau_clamp = self.config_fusion.tau_clamp
            self.modality_clip = LearnableNTXentLoss(init_tau, tau_clamp)

            """
            (2) 1 category-level loss
            """
            self.category_logits = nn.Sequential(
                nn.Linear(self.config_fusion.hidden_size, self.config_fusion.hidden_size * 2),
                nn.GELU(),
                nn.Dropout(self.config_fusion.embd_pdrop),
                nn.Linear(self.config_fusion.hidden_size * 2, self.config_fusion.category_logits_level2 + 1)
            )  # 预测二级类目
            self.category_sce = SCELoss(
                alpha=1.0,
                beta=0.5,
                num_classes=self.config_fusion.category_logits_level2 + 1
            )  # 二级类目预测损失

            """
            (3) 1 property prediction loss. Use following properties as targets.
            """
            ner_tasks = self.config_fusion.ner_tasks
            ner_task_dict = self.config_fusion.ner_task_dict
            self.ner_task_dict = None
            self.ner_tasks = ner_tasks
            self.ner_heads = None
            self.ner_kl = nn.KLDivLoss(reduction="batchmean")
            if len(ner_tasks) > 0:
                assert hexists(ner_task_dict), "ner task dict {} does not exist!".format(ner_task_dict)
                with hopen(ner_task_dict, "r") as fp:
                    self.ner_task_dict = json.load(fp)
                for task in ner_tasks:
                    assert task in self.ner_task_dict, "task {} is not supported in ner tasks.".format(task)
                self.ner_heads = nn.ModuleList([
                    AttMaxPooling(self.config_fusion.hidden_size, len(self.ner_task_dict[task]["label2idx"]) + 1) for task in ner_tasks])

        """
        Some fixed parameters, easy for tracing.
        """
        self.PAD = 2
        self.MASK = 1
        self.SEP = 3

        self.text_segment_ids = nn.Parameter(
            data=torch.zeros((512, ), dtype=torch.int64),
            requires_grad=False
        )  # [512, ]
        self.text_position_ids = nn.Parameter(
            data=torch.arange(0, 512, dtype=torch.int64),
            requires_grad=False
        )  # [512, ]self.config_fusion.hidden_size
        self.image_position_ids = nn.Parameter(
            data=torch.arange(0, 512, dtype=torch.int64),
            requires_grad=False
        )  # [512, ]
        self.initialize_weights()
    
    def initialize_weights(self):
        if hexists(self.hparams.load_pretrained):
            print("loading weights from {}".format(self.hparams.load_pretrained))
            state_dict_ori = self.state_dict()
            state_dict = load(self.hparams.load_pretrained, map_location="cpu")
            state_dict_new = OrderedDict()
            for key, value in state_dict.items():
                if (
                    key in state_dict_ori
                    and state_dict_ori[key].shape == state_dict[key].shape
                ):
                    state_dict_new[key] = value
            self.load_state_dict(state_dict_new, strict=False)

    def encode_text(self, input_ids: torch.Tensor):
        batch_size, num_text, text_length = input_ids.shape
        input_ids = input_ids.view((batch_size * num_text, -1))  # [B * num_text, max_len]
        input_segment_ids = self.text_segment_ids[:text_length].unsqueeze(0).expand((batch_size * num_text, -1))  # [B, max_len]
        input_masks = (input_ids != self.PAD).long()  # [B, max_len]
        input_position_ids = self.text_position_ids[:text_length].unsqueeze(0).expand((batch_size * num_text, -1))  # [B, max_len]

        t_out = self.deberta(input_ids=input_ids,
                             segment_ids=input_segment_ids,
                             attention_mask=input_masks,
                             output_pooled=True)
        t_emb, t_rep = t_out['sequence_output'], \
                       t_out['pooled_output']

        t_emb = t_emb.view((batch_size, num_text, text_length, -1))  # [B, num_text, max_len, 768]
        t_rep = t_rep.view((batch_size, num_text, -1))  # [B, num_text, 768]

        return t_emb, t_rep

    def encode_image(self, image: torch.Tensor):
        batch_size, num_image, _, __, ___ = image.shape
        image = image.flatten(0, 1)  # [B * num_image, 3, 224, 224]

        img_out = self.visual(image, return_dict=True)
        v_emb, v_rep = self.visual_feat(img_out["feature_map"]), img_out["pooled_out"]
        v_cat = torch.cat([v_rep.unsqueeze(1), v_emb], dim=1)

        batch_size_all, image_length, _ = v_cat.shape
        position_ids = self.image_position_ids[:image_length].unsqueeze(0)
        v_cat = v_cat + self.visual_pos(position_ids)

        v_cat = self.visual_fuse(v_cat).view((batch_size, num_image, image_length, -1))
        v_rep = self.visual_pooler(v_cat[:, :, 0])

        return v_cat, v_rep

    def allgather(self, x):
        return self.all_gather(x.contiguous()).flatten(0, 1)

    def forward_pretrain(self,
                         main_images,
                         desc_images,
                         sku_images,
                         main_ocrs,
                         desc_ocrs,
                         sku_ocrs,
                         product_name,
                         other_text,
                         **kwargs
                         ):
        """
        :param main_images: [B, num_main_images, 3, 244, 244]
        :param desc_images: [B, num_desc_images, 3, 244, 244]
        :param sku_images: [B, num_sku_images, 3, 244, 244]
        :param main_ocrs: [B, num_main_ocrs, max_main_len]
        :param desc_ocrs: [B, num_desc_ocrs, max_desc_len]
        :param sku_ocrs: [B, num_sku_ocrs, max_sku_len]
        :param product_name: [B, max_product_len]
        :param other_text: [B, max_other_len]
        :param fuse_mask: [B, num_segments], 1 indicates valid segment, 0 indicates mask
        """

        """
        1. Encode inputs.
        """
        _, main_text_rep = self.encode_text(main_ocrs)
        _, desc_text_rep = self.encode_text(desc_ocrs)
        _, sku_text_rep = self.encode_text(sku_ocrs)
        _, product_text_rep = self.encode_text(product_name.unsqueeze(1))
        _, other_text_rep = self.encode_text(other_text.unsqueeze(1))

        _, main_image_rep = self.encode_image(main_images)
        _, desc_image_rep = self.encode_image(desc_images)
        _, sku_image_rep = self.encode_image(sku_images)

        """
        2. Modality-level clip loss
        """
        batch_size = main_text_rep.shape[0]
        fuse_cls = self.fuse_cls.unsqueeze(0).expand((batch_size, -1, -1))
        fuse_inputs_image = torch.cat([
            self.ln_cls(fuse_cls + self.fuse_segment_embedding.weight[0]),
            self.ln_visual(self.v_projector_fuse(main_image_rep) + self.fuse_segment_embedding.weight[6]),
            self.ln_visual(self.v_projector_fuse(desc_image_rep) + self.fuse_segment_embedding.weight[7]),
            self.ln_visual(self.v_projector_fuse(sku_image_rep) + self.fuse_segment_embedding.weight[8])
        ], dim=1)
        fuse_inputs_text = torch.cat([
            self.ln_cls(fuse_cls + self.fuse_segment_embedding.weight[0]),
            self.ln_text(self.t_projector_fuse(main_text_rep) + self.fuse_segment_embedding.weight[1]),
            self.ln_text(self.t_projector_fuse(desc_text_rep) + self.fuse_segment_embedding.weight[2]),
            self.ln_text(self.t_projector_fuse(sku_text_rep) + self.fuse_segment_embedding.weight[3]),
            self.ln_text(self.t_projector_fuse(product_text_rep) + self.fuse_segment_embedding.weight[4]),
            self.ln_text(self.t_projector_fuse(other_text_rep) + self.fuse_segment_embedding.weight[5])
        ], dim=1)
        fuse_image = self.fuse(fuse_inputs_image)[:, 0]  # [B, d_h]
        fuse_text = self.fuse(fuse_inputs_text)[:, 0]  # [B, d_h]
        fuse_image, fuse_text = self.allgather(fuse_image), self.allgather(fuse_text)
        loss_clip_modality = self.modality_clip(fuse_image, fuse_text)

        """
        3. Category-level loss.
        """
        fuse_inputs = torch.cat([
            fuse_inputs_text, fuse_inputs_image[:, 1:]
        ], dim=1)  # [B, 39, 512]
        fuse_inputs = self.fuse_dropout(fuse_inputs)
        fuse_emb = self.fuse(fuse_inputs)  # [B, 1 + 38, d_f]

        logits_cat = self.category_logits(fuse_emb[:, 0])  # [B, num_categories + 1]
        loss_sce = self.category_sce(logits_cat, kwargs["label"])

        """
        4. Property prediction loss.
        """
        # KLDiv Loss for ner
        loss_ner = 0.
        if len(self.ner_tasks) > 0:
            for i in range(len(self.ner_tasks)):
                key = "ner_{}".format(i)
                logits = self.ner_heads[i](fuse_emb)
                label = kwargs[key]
                loss_ner = loss_ner + self.ner_kl(F.log_softmax(logits, dim=-1), label)
            loss_ner = loss_ner / len(self.ner_tasks)

        """
        5. Gather all losses.
        """
        loss = loss_clip_modality + loss_sce + loss_ner

        return {
            "loss": loss,
            "loss_clip_modality": loss_clip_modality,
            "loss_sce": loss_sce,
            "loss_ner": loss_ner,
            "logits": logits_cat,
            "label": kwargs["label"]
        }

    def training_step(self, batch, idx):
        return self.forward_pretrain(**batch)

    @torch.no_grad()
    def validation_step(self, batch, idx):
        return self.forward_pretrain(**batch)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=self.hparams.betas,
            eps=self.hparams.eps,
            weight_decay=self.hparams.weight_decay,
        )

        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.trainer.total_steps,
            cycle_mult=1.0,
            max_lr=self.hparams.learning_rate,
            min_lr=0,
            warmup_steps=2000,
        )

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
