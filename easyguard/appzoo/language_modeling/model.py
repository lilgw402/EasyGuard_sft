# -*- coding: utf-8 -*-

import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoTokenizer

try:
    import cruise
except ImportError:
    print(
        "[ERROR] cruise is not installed! Please refer this doc: https://bytedance.feishu.cn/wiki/wikcnGP7yzZAuKpPfL6jRJKl2ag"
    )

from cruise import CruiseModule
from cruise.utilities.distributed import DIST_ENV

from ...core.optimizers import build_optimizer, build_scheduler
from ...modelzoo.modeling_utils import load_pretrained
from ...modelzoo.models.mdeberta_v2 import DebertaV2ForMaskedLM
from ...modelzoo.models.fashionxlm import DebertaV2ForMaskedLMMoE, MLP
from ...modelzoo.models.fashionxlm import XLMRobertaForMaskedLM
from ...utils.losses import cross_entropy, LearnableNTXentLoss


class LanguageModel(CruiseModule):
    def __init__(
        self,
        pretrained_model_name_or_path: str = None,
        cl_enable: bool = False,
        cl_temp: float = 0.05,
        cl_weight: float = 1.0,
        ntx_enable: bool = False,
        classification_task_enable: bool = False,
        classification_task_head: int = 1422,
        hidden_size: int = 768,
        load_pretrain: Optional[
            str
        ] = "hdfs://harunava/home/byte_magellan_va/user/jiangjunjun.happy/xlmr16/model_state_epoch_400000.th",
        all_gather_limit: int = -1,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.05,
        base_lr: float = 5e-4,
        warmup_lr: float = 5e-7,
        min_lr: float = 5e-6,
        lr_scheduler: str = "cosine",
        lr_scheduler_decay_ratio: float = 0.8,
        lr_scheduler_decay_rate: float = 0.1,
        optimizer: str = "adamw",
        optimizer_eps: float = 1e-8,
        optimizer_betas: Tuple[float, ...] = (0.9, 0.999),
        momentum: float = 0.9,
    ):
        super().__init__()
        self.save_hparams()
        self.moe = False
        if pretrained_model_name_or_path == "fashionxlm-mdeberta-v3-base":
            self.backbone = DebertaV2ForMaskedLM.from_pretrained('microsoft/mdeberta-v3-base')
        elif pretrained_model_name_or_path == "fashionxlm-mdeberta-moe-adv":
            self.moe = True
            self.backbone = DebertaV2ForMaskedLMMoE.from_pretrained('microsoft/mdeberta-v3-base')
        elif pretrained_model_name_or_path == "xlm-roberta-base":
            self.backbone = XLMRobertaForMaskedLM.from_pretrained('xlm-roberta-base')
        else:
            self.backbone = AutoModelForMaskedLM.from_pretrained(
                pretrained_model_name_or_path
            )

        # self.model = self.backbone
        # self.init_weights()
        # self.freeze_params(self.config.TRAIN.freeze_prefix)

        # use contrast learning loss
        self.cl_enable = cl_enable
        if self.cl_enable:
            self.cl_temp = cl_temp  # tempure of softmax
            self.cl_weight = cl_weight  # weighted loss

            self.ntx_enable = ntx_enable
            self.ntx_loss_layer = LearnableNTXentLoss()
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # use classification learning loss
        self.classification_task_enable = classification_task_enable
        if self.classification_task_enable:
            self.classification_task_head = classification_task_head
            cla_input_size = hidden_size * 2 if self.moe else hidden_size
            self.classifier = torch.nn.Linear(cla_input_size, self.classification_task_head)

        if self.moe:
            self.moe_hard_heads = nn.ModuleDict(
                    {k: MLP(hidden_size, hidden_size, hidden_size) for k in
                     ['GB', 'ID', 'TH', 'MY', 'VN', 'PH']})
            self.moe_share_head = MLP(hidden_size, hidden_size, hidden_size)
            self.kl_loss = nn.KLDivLoss(reduction="batchmean")
            self.adv_head = nn.Linear(hidden_size, 6)

        # setup nce allgather group if has limit
        nce_limit = self.hparams.all_gather_limit
        if nce_limit < 0:
            # no limit
            self.nce_group = None
        elif nce_limit == 0:
            # no all_gather
            self.nce_group = False
        else:
            raise NotImplementedError(
                "Using sub-groups in NCCL is not well implemented."
            )
            group_rank_id = self.trainer.global_rank // nce_limit
            group_ranks = [
                group_rank_id * nce_limit + i for i in range(nce_limit)
            ]
            self.nce_group = torch.distributed.new_group(
                ranks=group_ranks, backend="nccl"
            )
            self.print(
                "Create non-global allgather group from ranks:",
                group_ranks,
                "group size:",
                self.nce_group.size(),
            )

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

    def rank_zero_prepare(self):
        # load partial pretrain
        if self.hparams.load_pretrain:
            load_pretrained(self.hparams.load_pretrain, self)

    def cl_loss(self, cls_status):
        batch_size = cls_status.shape[0]
        z1, z2 = (
            cls_status[0 : batch_size // 2, :],
            cls_status[batch_size // 2 :, :],
        )

        # all gather to increase effective batch size
        if self.nce_group is not False:
            # [bsz, n] -> [group, bsz, n]
            group_z1 = self.all_gather(
                z1, group=self.nce_group, sync_grads="rank"
            )
            group_z2 = self.all_gather(
                z2, group=self.nce_group, sync_grads="rank"
            )
            # [group, bsz, n] -> [group * bsz, n]
            z1 = group_z1.view((-1, cls_status.shape[-1]))
            z2 = group_z2.view((-1, cls_status.shape[-1]))

        if self.ntx_enable:
            loss = self.ntx_loss_layer(z1, z2)
        else:
            # cosine similarity as logits
            self.logit_scale.data.clamp_(-np.log(100), np.log(100))
            logit_scale = self.logit_scale.exp()
            self.log("logit_scale", logit_scale)
            logits_per_z1 = logit_scale * z1 @ z2.t()
            logits_per_z2 = logit_scale * z2 @ z1.t()

            bsz = logits_per_z1.shape[0]
            labels = torch.arange(bsz, device=logits_per_z1.device)  # bsz

            loss_v = self.cl_loss_layer(logits_per_z1, labels)
            loss_t = self.cl_loss_layer(logits_per_z2, labels)
            loss = (loss_v + loss_t) / 2

        return loss

    def forward(self, input_ids, token_type_ids, attention_mask, labels, classification_label=None, language=None):
        """
        input_ids: [bsz, seq_len]
        input_segment_ids: [bsz, seq_len]
        input_mask: [bsz, seq_len]
        label: [bsz]
        classification_label: [bsz]
        """
        output_dict = {}

        # mlm loss
        mmout = self.backbone(
            input_ids,
            attention_mask,
            token_type_ids,
            labels=labels,
            output_hidden_states=True,
        )
        loss1 = mmout.loss
        self.log("mlm_loss", loss1)
        output_dict["loss"] = loss1

        # mlm acc
        mlm_acc = self.mlm_acc(mmout.logits, labels)
        output_dict['mlm_acc'] = mlm_acc

        # cl loss
        if self.cl_enable:
            hidden_states = mmout.hidden_states[
                -1
            ]  # batch * sen_len * emd_size
            cls_status = hidden_states[:, 0, :]
            moe_acc = 0.0
            aux_loss = 0.0
            entroy, sep_loss = 0, 0
            if self.moe:
                cls_status_ = torch.stack([self.moe_hard_heads[k](xi) for xi, k in zip(cls_status, language)],
                                              dim=0)  # language proj
                cls_status_share = self.moe_share_head(cls_status)
                entroy, sep_loss, acc_input = self.hard_moe_loss(cls_status, cls_status_share, language)
                moe_acc = self.cla_acc(acc_input[0], acc_input[1])
                cls_status_ = torch.cat((cls_status_, cls_status_share), dim=1)
            else:
                cls_status_ = cls_status
            loss2 = self.cl_loss(cls_status_)
            self.log('cl_loss', loss2)
            cl_acc = self.cl_acc(cls_status_)
            loss = loss1 + self.cl_weight * loss2 + self.moe_aux_weight * aux_loss + entroy + self.moe_share_weight * sep_loss
            output_dict['mlm_loss'] = loss1
            output_dict['aux_loss'] = aux_loss
            output_dict['cl_loss'] = loss2
            output_dict['entroy'] = entroy
            output_dict['sep_loss'] = sep_loss
            output_dict['loss'] = loss
            output_dict.update(cl_acc)
            output_dict['moe_acc'] = moe_acc

        # classification task
        if self.classification_task_enable:
            hidden_states = mmout.hidden_states[
                -1
            ]  # batch * sen_len * emd_size
            cls_status = hidden_states[:, 0, :]  # batch * emd_size
            moe_acc = 0.0
            aux_loss = 0.0
            entroy, sep_loss = 0, 0
            if self.moe:
                cls_status_ = torch.stack([self.moe_hard_heads[k](xi) for xi, k in zip(cls_status, language)],
                                          dim=0)  # language proj
                cls_status_share = self.moe_share_head(cls_status)
                entroy, sep_loss, acc_input = self.hard_moe_loss(cls_status, cls_status_share, language)
                moe_acc = self.cla_acc(acc_input[0], acc_input[1])
                cls_status_ = torch.cat((cls_status_, cls_status_share), dim=1)
            else:
                cls_status_ = cls_status

            logits = self.classifier(cls_status_)  # batch * label_size
            loss3 = cross_entropy(logits, classification_label)
            cla_acc = self.cla_acc(logits, classification_label)

            loss = loss1 + loss3 + aux_loss + entroy + self.moe_share_weight * sep_loss
            output_dict['mlm_loss'] = loss1
            output_dict['cla_loss'] = loss3
            output_dict['aux_loss'] = aux_loss
            output_dict['entroy'] = entroy
            output_dict['sep_loss'] = sep_loss
            output_dict['loss'] = loss
            output_dict['cla_acc'] = cla_acc
            output_dict['moe_acc'] = moe_acc

        return output_dict

    def training_step(self, batch, idx):
        return self(**batch)

    def validation_step(self, batch, idx):
        return self(**batch)

    def validation_epoch_end(self, outputs) -> None:
        gathered_results = DIST_ENV.all_gather_object(outputs)
        all_results = []
        for item in gathered_results:
            all_results.extend(item)
        acc_all = [out["mlm_acc"] for out in all_results]
        total_acc = sum(acc_all) / len(acc_all)
        self.log("total_mlm_acc", total_acc, console=True)
        print("total_mlm_acc", total_acc)

        if self.classification_task_enable:
            # cls acc
            cla_acc_all = [out["cla_acc"] for out in all_results]
            total_cla_acc = sum(cla_acc_all) / len(cla_acc_all)
            self.log("total_cla_acc", total_cla_acc, console=True)
            print("total_cla_acc", total_cla_acc)

        if self.cl_enable:
            trans2origin_all = [out["trans2origin@1"] for out in all_results]
            trans2origin_acc = sum(trans2origin_all) / len(trans2origin_all)
            self.log("total_trans2origin_acc", trans2origin_acc, console=True)
            print("total_trans2origin_acc", trans2origin_acc)

            origin2trans_all = [out["origin2trans@1"] for out in all_results]
            origin2trans_acc = sum(origin2trans_all) / len(origin2trans_all)
            self.log("total_origin2trans_acc", origin2trans_acc, console=True)
            print("total_origin2trans_acc", origin2trans_acc)


    def configure_optimizers(self):
        optimizer = build_optimizer(self.hparams, self)
        lr_scheduler = build_scheduler(
            self.hparams,
            optimizer,
            self.trainer.max_epochs,
            self.trainer.steps_per_epoch
            // self.trainer._accumulate_grad_batches,
        )
        return [optimizer], [lr_scheduler]

    def lr_scheduler_step(self, schedulers, **kwargs):
        # timm lr scheduler is called every step
        for scheduler in schedulers:
            scheduler.step_update(
                self.trainer.global_step
                // self.trainer._accumulate_grad_batches
            )

    def on_fit_start(self) -> None:
        self.rank_zero_print('===========My custom fit start function is called!============')

    @torch.no_grad()
    def mlm_acc(self, mlm_logits, mlm_labels, PAD_IDX=-100) -> torch.Tensor:
        """
        :param mlm_logits:  [batch_size, src_len, src_vocab_size]
        :param mlm_labels:  [batch_size, src_len]
        :param cl_logits:  [batch_size, batch_size]
        :param PAD_IDX:
        :return:
        """
        mlm_pred = mlm_logits.argmax(axis=2).reshape(-1)
        mlm_true = mlm_labels.reshape(-1)

        mlm_acc = mlm_pred.eq(mlm_true)  # 计算预测值与正确值比较的情况
        mask = torch.logical_not(mlm_true.eq(PAD_IDX))  # 找到真实标签中，mask位置的信息。 mask位置为TRUE
        mlm_acc = mlm_acc.logical_and(mask)  # 去掉acc中非mask的部分
        mlm_correct = mlm_acc.sum().item()
        mlm_total = mask.sum().item()
        mlm_acc = float(mlm_correct) / mlm_total
        return torch.tensor(mlm_acc)

    @torch.no_grad()
    def cla_acc(self, cla_logits=None, cla_label=None) -> torch.Tensor:
        """
        :param cla_logits:  [batch_size, label_num]
        :param cla_label:   [batch_size]
        :return:
        """
        cla_correct = (cla_logits.argmax(1) == cla_label).float().sum()
        cla_total = len(cla_label)
        cla_acc = float(cla_correct) / cla_total
        return torch.tensor(cla_acc)
