# -*- coding: utf-8 -*-
"""
FrameAlbert Classification
"""
import math
import os.path
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

try:
    import cruise
except ImportError:
    print(
        "[ERROR] cruise is not installed! Please refer this doc: https://bytedance.feishu.cn/wiki/wikcnGP7yzZAuKpPfL6jRJKl2ag"
    )

from cruise import CruiseModule
from cruise.utilities.cloud_io import load
from cruise.utilities.distributed import DIST_ENV
from cruise.utilities.hdfs_io import hexists, hopen
from sklearn.metrics import roc_auc_score

from easyguard.appzoo.authentic_modeling.utils import (
    CosineAnnealingWarmupRestarts,
    accuracy,
)
from easyguard.modelzoo.models.falbert import FrameALBert

from ...utils.losses import (
    LearnableNTXentLoss,
    LearnablePCLLoss,
    SCELoss,
    cross_entropy,
)
from .optimization import *
from .optimization import AdamW


# def p_fix_r(output, labels, fix_r):
#     output_sort = output[(-output).argsort()]
#     labels_sort = labels[(-output).argsort()]
#     num_pos = np.sum(labels == 1)
#     recall_sort = np.cumsum(labels_sort) / float(num_pos)
#     index = np.abs(recall_sort - fix_r).argmin()
#     thr = output_sort[index]
#     precision = np.sum(((output >= thr) == labels) * labels) / np.sum(
#         output >= thr
#     )
#     return precision, recall_sort[index], thr


class FrameAlbertClassify(CruiseModule):
    def __init__(
            self,
            config_backbone,
            config_fusion,
            config_optim,
            load_pretrained: str = None,
            low_lr_prefix: list = [],
            prefix_changes: list = [],
    ):
        super(FrameAlbertClassify, self).__init__()
        self.save_hparams()

    def setup(self, stage) -> None:
        """
        Load yaml file as config class
        """
        with open(self.hparams.config_backbone) as fp:
            self.config_backbone = SimpleNamespace(**yaml.load(fp, yaml.Loader))
        with open(self.hparams.config_fusion) as fp:
            self.config_fusion = SimpleNamespace(**yaml.load(fp, yaml.Loader))
        with open(self.hparams.config_optim) as fp:
            self.config_optim = SimpleNamespace(**yaml.load(fp, yaml.Loader))
        """
        Initialize modules
        """
        self.falbert = FrameALBert(self.config_backbone)
        """
        Initialize output layer
        """
        # fuse_emb_size = 768
        # self.classifier = torch.nn.Linear(
        #     fuse_emb_size, self.config_fusion.class_num
        # )
        feat_emb_size = self.config_fusion.feat_emb_size
        # self.classifier_concat = torch.nn.Linear(feat_emb_size, self.config_fusion.class_num)
        self.multi_heads = torch.nn.Linear(feat_emb_size * 2, self.config_fusion.head_num * self.config_fusion.class_num)
        # self.softmax = nn.Softmax(dim=1)
        # self.criterion = torch.nn.CrossEntropyLoss()
        self.ce = torch.nn.CrossEntropyLoss()
        """
        Initialize some fixed parameters.
        """
        self.init_weights()
        if self.hparams.load_pretrained:
            prefix_changes = [prefix_change.split('->') for prefix_change in self.hparams.prefix_changes]
            rename_params = {pretrain_prefix: new_prefix for pretrain_prefix, new_prefix in prefix_changes}
            self.partial_load_from_checkpoints(
                self.hparams.load_pretrained,
                map_location='cpu',
                rename_params=rename_params
            )
        self.freeze_params(self.config_backbone.freeze_prefix)

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

    def criterion(self, logits, label, use_gather=False):
        if use_gather:
            # gather_logits = allgather(logits.contiguous(), self.trainer.rank, self.trainer.world_size)
            # gather_label = allgather(label.contiguous(), self.trainer.rank, self.trainer.world_size)
            # loss = self.ce(gather_logits, gather_label)
            loss = self.ce(logits, label)
            return loss
        else:
            loss = self.ce(logits, label)
            return loss

    def maxpooling_with_mask(self, hidden_state, attention_mask):
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).half()
        mask_expanded = 1e4 * (mask_expanded - 1)
        hidden_masked = hidden_state + mask_expanded  # sum instead of multiple
        max_pooling = torch.max(hidden_masked, dim=1)[0]

        return max_pooling

    def multi_heads_with_mask(self, hidden_state, head_mask):
        x = self.multi_heads(hidden_state).reshape(-1, self.head_num, self.class_num)
        head_mask = head_mask.unsqueeze(-1).expand(x.size()).half()
        head_mask = 1e4 * (head_mask - 1)
        x = head_mask + x
        x = torch.max(x, dim=1)[0]
        return x

    def forward_step(
            self,
            input_ids,
            input_segment_ids,
            input_mask,
            frames=None,
            frames_mask=None,
            visual_embeds=None,
            head_mask=None,
    ):
        mmout = self.encode_multimodal(
            input_ids=input_ids,
            input_segment_ids=input_segment_ids,
            input_mask=input_mask,
            images=frames,
            images_mask=frames_mask,
            visual_embeds=visual_embeds,
        )
        cls_emb = mmout['pooled_output']
        last_hidden_state = mmout['encoded_layers'][-1]
        attention_mask = torch.cat(
            [input_mask, torch.ones([frames_mask.shape[0], 1], device=frames_mask.device), frames_mask], dim=1)
        max_pooling = self.maxpooling_with_mask(hidden_state=last_hidden_state, attention_mask=attention_mask)

        concat_feat = torch.cat([cls_emb, max_pooling], dim=1)

        logits = self.multi_heads_with_mask(concat_feat, head_mask)

        return {"feat": concat_feat, "logits": logits}

    def encode_multimodal(
            self,
            input_ids,
            input_segment_ids,
            input_mask,
            images=None,
            images_mask=None,
            visual_embeds=None,
            *args,
            **kwargs,
    ):
        if images_mask is None:
            if visual_embeds is None:
                images_mask = torch.ones(
                    images.shape[0:2], device=images.device, dtype=torch.long
                )
            else:
                images_mask = torch.ones(
                    visual_embeds.shape[0:2],
                    device=visual_embeds.device,
                    dtype=torch.long,
                )
        mmout = self.falbert(
            input_ids=input_ids,
            input_segment_ids=input_segment_ids,
            input_mask=input_mask,
            frames=images,
            frames_mask=images_mask,
            visual_embeds=visual_embeds,
            mode="tv",
        )
        return mmout

    def training_step(self, batch, idx):
        token_ids, segment_ids, attn_mask, image, image_mask, head_mask = (
            batch["input_ids"],
            batch["input_segment_ids"],
            batch["input_mask"],
            batch["frames"],
            batch["frames_mask"],
            batch["head_mask"]
        )
        rep_dict = self.forward_step(
            input_ids=token_ids,
            input_segment_ids=segment_ids,
            input_mask=attn_mask,
            frames=image,
            frames_mask=image_mask,
            head_mask=head_mask,
        )
        rep_dict.update({"label": batch["label"]})
        loss = self.criterion(rep_dict["logits"], rep_dict["label"], use_gather=False)
        res = {
            "loss": loss,
            "train_lr": self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        }
        acc_dict = self.cal_acc(rep_dict["logits"], label=rep_dict["label"], topk=(1, 5))
        for k, v in acc_dict.items():
            res.update({f'train_{k}': v})

        return res

    def validation_step(self, batch, idx):
        token_ids, segment_ids, attn_mask, image, image_mask, head_mask = (
            batch["input_ids"],
            batch["input_segment_ids"],
            batch["input_mask"],
            batch["frames"],
            batch["frames_mask"],
            batch["head_mask"]
        )
        rep_dict = self.forward_step(
            input_ids=token_ids,
            input_segment_ids=segment_ids,
            input_mask=attn_mask,
            frames=image,
            frames_mask=image_mask,
            head_mask=head_mask,
        )
        rep_dict.update({"label": batch["label"]})
        loss = self.criterion(rep_dict["logits"], rep_dict["label"], use_gather=False)
        res = {"val_loss": loss}

        acc_dict = self.cal_acc(rep_dict["logits"], label=rep_dict["label"], topk=(1, 5))
        for k, v in acc_dict.items():
            res.update({f'val_{k}': v})

        return res

    def validation_epoch_end(self, outputs) -> None:
        gathered_results = DIST_ENV.all_gather_object(outputs)
        # all_results = []
        # for item in gathered_results:
        #     all_results.extend(item)
        # acc_all = [out["acc"] for out in all_results]
        # total_acc = sum(acc_all) / len(acc_all)
        # self.log("total_val_acc", total_acc, console=True)
        res_out = {}
        all_results = []
        for item in gathered_results:
            all_results.extend(item)
        val_loss_all = [out["val_loss"] for out in all_results]
        top1_acc_all = [out["val_top1_acc"] for out in all_results]
        top5_acc_all = [out["val_top5_acc"] for out in all_results]
 
        val_loss = sum(val_loss_all) / len(val_loss_all)
        top1_acc = sum(top1_acc_all) / len(top1_acc_all)
        top5_acc = sum(top5_acc_all) / len(top5_acc_all)

        res_out["val_loss"] = val_loss
        res_out["val_top1_acc"] = top1_acc
        res_out["val_top5_acc"] = top5_acc

        self.log_dict(res_out, console=True)
        self.log("val_loss", val_loss, console=True)
        self.log("val_top1_acc", top1_acc, console=True)
        self.log("val_top5_acc", top1_acc, console=True)

    def trace_before_step(self, batch):
        token_ids, segment_ids, attn_mask, image, image_mask, head_mask = (
            batch["input_ids"],
            batch["input_segment_ids"],
            batch["input_mask"],
            batch["frames"],
            batch["frames_mask"],
            batch["head_mask"]
        )
        return token_ids, segment_ids, attn_mask, image, image_mask, head_mask

    def trace_step(self, token_ids, segment_ids, attn_mask, image, image_mask, head_mask):
        rep_dict = self.forward_step(
            input_ids=token_ids,
            input_segment_ids=segment_ids,
            input_mask=attn_mask,
            frames=image,
            frames_mask=image_mask,
            head_mask=head_mask,
        )

        logits = rep_dict["logits"]

        return logits

    def trace_after_step(self, result):
        pass

    def configure_optimizers(self):
        no_decay = ["bias", "bn", "norm"]
        no_dacay_params_dict = {"params": [], "weight_decay": 0.0}
        low_lr_params_dict = {
            "params": [],
            "weight_decay": self.config_optim.weight_decay,
            "lr": self.config_optim.learning_rate * 0.1,
        }
        normal_params_dict = {
            "params": [],
            "weight_decay": self.config_optim.weight_decay,
        }

        low_lr_keys = []
        for n, p in self.named_parameters():
            low_lr = False
            for low_lr_prefix in self.hparams.low_lr_prefix:
                if n.startswith(low_lr_prefix):
                    low_lr = True
                    low_lr_params_dict['params'].append(p)
                    low_lr_keys.append(n)
                    break
            if low_lr:
                continue

            if any(nd in n for nd in no_decay):
                no_dacay_params_dict["params"].append(p)
            # elif n.startswith("albert"):
            #     low_lr_params_dict["params"].append(p)
            else:
                normal_params_dict["params"].append(p)

        if low_lr_keys:
            print(f'low_lr_keys: {low_lr_keys}')

        optimizer_grouped_parameters = [
            no_dacay_params_dict,
            low_lr_params_dict,
            normal_params_dict,
        ]

        if self.config_optim.optim == "SGD":
            optm = torch.optim.SGD(
                optimizer_grouped_parameters,
                self.config_optim.learning_rate,
                momentum=self.config_optim.momentum,
                weight_decay=self.config_optim.weight_decay,
            )
        elif self.config_optim.optim == "AdamW":
            optm = AdamW(
                optimizer_grouped_parameters,
                lr=self.config_optim.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-6,
                weight_decay=self.config_optim.weight_decay,
                correct_bias=False,
            )

        if self.config_optim.lr_schedule == "linear":
            print(f'warmup: {self.config_optim.warmup_steps_factor * self.trainer.steps_per_epoch}')
            print(f'total step: {self.trainer.total_steps}')
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=optm,
                num_warmup_steps=self.config_optim.warmup_steps_factor
                                 * self.trainer.steps_per_epoch,
                num_training_steps=self.trainer.total_steps,
            )
        elif self.config_optim.lr_schedule == "cosine":
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=optm,
                num_warmup_steps=self.config_optim.warmup_steps_factor * self.trainer.steps_per_epoch,
                num_training_steps=self.trainer.total_steps,
            )
        elif self.config_optim.lr_schedule == "onecycle":
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optm,
                max_lr=self.config_optim.learning_rate,
                total_steps=self.trainer.total_steps,
            )

        return {"optimizer": optm, "lr_scheduler": lr_scheduler}

    def lr_scheduler_step(self, schedulers, **kwargs,) -> None:
        """
        默认是per epoch的lr schedule, 改成per step的
        """
        for scheduler in schedulers:
            scheduler.step()

    @torch.no_grad()
    def cal_acc(self, output: torch.Tensor, label: torch.Tensor, topk: Tuple[int] = (1,)):
        """
        Computes the accuracy over the k top predictions for the specified values of k
        """
        maxk = max(topk)
        batch_size = label.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(label.view(1, -1).expand_as(pred))

        res = {}
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res[f"top{k}_acc"] = correct_k.mul_(100.0 / batch_size)
        return res
