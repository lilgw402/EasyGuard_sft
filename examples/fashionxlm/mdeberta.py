# -*- coding: utf-8 -*-
""" XLMR模型 """
import os
import sys
from typing import Union, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate

try:
    import cruise
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from cruise import CruiseTrainer, CruiseModule, CruiseCLI
from cruise.data_module import CruiseDataModule
from cruise.utilities.hdfs_io import hlist_files
from cruise.data_module.cruise_loader import DistributedCruiseDataLoader

from transformers import AutoTokenizer, AutoModelForMaskedLM

from model_utils import load_pretrained
from optimizer import build_optimizer
from lr_scheduler import build_scheduler
from modeling_deberta_v2 import DebertaV2ForMaskedLM


class TextProcessor:
    def __init__(self, x_key, y_key, region_key, pre_tokenize, mlm_probability, cl_enable, cla_task_enable, category_key,
                 max_len, tokenizer):
        self._x_key = x_key
        self._y_key = y_key
        self._region_key = region_key
        self._pre_tokenize = pre_tokenize
        self._mlm_probability = mlm_probability
        self._cl_enable = cl_enable
        self._cla_task_enable = cla_task_enable
        self._category_key = category_key
        self._max_len = max_len
        self._tokenizer = tokenizer

    def transform(self, data_dict: dict):
        # == 文本 ==
        if not self._pre_tokenize:  # do not pre tokenize
            text = data_dict.get(self._x_key, " ")
            text_token = self._tokenizer(text, padding='max_length', max_length=self._max_len, truncation=True)
            # text_token['token_type_ids'] = [0] * self._max_len    # for "xlm-roberta-base"
        else:
            text_token = data_dict[self._x_key]
            text_token[0] = self._tokenizer.cls_token
            text_token[-1] = self._tokenizer.sep_token
            text_token_ids = self._tokenizer.convert_tokens_to_ids(text_token)
            text_token['input_ids'] = text_token_ids
            text_token['input_mask'] = [1] * len(text_token_ids[:self._max_len]) + [0] * (self._max_len - len(text_token_ids))
            text_token['token_type_ids'] = [0] * self._max_len

        language = data_dict[self._region_key]

        input_dict = {'language': language,
                      'input_ids': torch.Tensor(text_token['input_ids']).long(),
                      'input_mask': torch.Tensor(text_token['attention_mask']).long(),
                      'input_segment_ids': torch.Tensor(text_token['token_type_ids']).long(),
                      'labels': torch.Tensor(text_token['input_ids']).long()}

        # == 标签 ==
        if self._cla_task_enable:
            label = int(data_dict[self._category_key])
            input_dict['classification_label'] = torch.tensor(label)

        return input_dict

    def batch_transform(self, batch_data):
        # batch_data: List[Dict[modal, modal_value]]
        out_batch = {}
        if self._cla_task_enable:
            keys = ('input_ids', 'input_mask', 'input_segment_ids', 'labels', 'classification_label')
        else:
            keys = ('input_ids', 'input_mask', 'input_segment_ids', 'labels')

        for k in keys:
            out_batch[k] = default_collate([data[k] for data in batch_data])
            if self._cl_enable:
                out_batch[k] = torch.cat((out_batch[k], out_batch[k]), 0)

        out_batch['input_ids'], out_batch['labels'] = self.torch_mask_tokens(out_batch['input_ids'])

        return out_batch

    def torch_mask_tokens(self, inputs, special_tokens_mask=None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self._mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self._tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
                                   ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self._tokenizer.convert_tokens_to_ids(self._tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self._tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class LMDataModule(CruiseDataModule):
    def __init__(self,
                 train_batch_size: int = 16,
                 val_batch_size: int = 16,
                 paths: Union[str, List[str]] = 'hdfs://harunava/home/byte_magellan_va/user/jiangjunjun.happy/tiktok_text_category',
                 data_size: int = 200000000,
                 val_step: int = 500,
                 num_workers: int = 1,
                 tokenizer: str = 'microsoft/mdeberta-v3-base',
                 x_key: str = 'text',
                 y_key: str = 'label',
                 region_key: str = 'region',
                 pre_tokenize: bool = False,
                 mlm_probability: float = 0.15,
                 cl_enable: bool = False,
                 cla_task_enable: bool = False,
                 category_key: str = 'category',
                 max_len: int = 256,
                 ):
        super().__init__()
        self.save_hparams()

    def local_rank_zero_prepare(self) -> None:
        # download the tokenizer once per node
        AutoTokenizer.from_pretrained(self.hparams.tokenizer)

    def setup(self, stage) -> None:
        paths = self.hparams.paths
        if isinstance(paths, str):
            paths = [paths]
        # split train/val
        files = hlist_files(paths)
        if not files:
            raise RuntimeError(f"No valid files can be found matching `paths`: {paths}")
        files = sorted(files)
        # use the last file as validation
        self.train_files = files
        self.val_files = files[0:16]

        self.x_key = self.hparams.x_key
        self.y_key = self.hparams.y_key
        self.region_key = self.hparams.region_key
        self.pre_tokenize = self.hparams.pre_tokenize
        self.mlm_probability = self.hparams.mlm_probability
        self.cl_enable = self.hparams.cl_enable
        self.cla_task_enable = self.hparams.cla_task_enable
        self.max_len = self.hparams.max_len

        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer)

    def train_dataloader(self):

        return DistributedCruiseDataLoader(
            data_sources=[self.train_files],
            keys_or_columns=[None],
            batch_sizes=[self.hparams.train_batch_size],
            num_workers=self.hparams.num_workers,
            num_readers=[1],
            decode_fn_list=[[]],
            processor=TextProcessor(
                self.x_key, self.y_key, self.region_key, self.pre_tokenize, self.mlm_probability, self.cl_enable, self.cla_task_enable,
                self.hparams.category_key, self.max_len, self.tokenizer
            ),
            predefined_steps=self.hparams.data_size // self.hparams.train_batch_size // self.trainer.world_size,
            source_types=['jsonl'],
            shuffle=True,
        )

    def val_dataloader(self):

        return DistributedCruiseDataLoader(
            data_sources=[self.val_files],
            keys_or_columns=[None],
            batch_sizes=[self.hparams.val_batch_size],
            num_workers=self.hparams.num_workers,
            num_readers=[1],
            decode_fn_list=[[]],
            processor=TextProcessor(self.x_key, self.y_key, self.region_key, self.pre_tokenize, self.mlm_probability, self.cl_enable, self.cla_task_enable,
                self.hparams.category_key, self.max_len, self.tokenizer
                ),
            predefined_steps=self.hparams.val_step,
            source_types=['jsonl'],
            shuffle=False,
        )


class LearnableNTXentLoss(torch.nn.Module):
    def __init__(self, init_tau=0.07, clamp=4.6051):
        super().__init__()
        self.tau = torch.nn.Parameter(torch.tensor([np.log(1.0 / init_tau)], dtype=torch.float32))
        self.calc_ce = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.clamp = clamp  # 4.6051 等价于CLAMP 100, 初始值是2.6593，

    def forward(self, v_emb=None, t_emb=None, logits=None):
        """
        v_emb: batch 对比loss的一边
        t_emb: batch 对比loss的另一边
        logits: 需要计算对比loss的矩阵，default: None
        """
        self.tau.data = torch.clamp(self.tau.data, 0, self.clamp)
        if logits is None:
            bsz = v_emb.shape[0]
            v_emb = F.normalize(v_emb, dim=1)
            t_emb = F.normalize(t_emb, dim=1)
            logits = torch.mm(v_emb, t_emb.t()) * self.tau.exp()  # [bsz, bsz]
        else:
            bsz = logits.shape[0]
            logits = logits * self.tau.exp()
        labels = torch.arange(bsz, device=logits.device)  # bsz

        loss_v = self.calc_ce(logits, labels)
        loss_t = self.calc_ce(logits.t(), labels)
        loss = (loss_v + loss_t) / 2
        return loss


class Mdeberta(CruiseModule):
    def __init__(self,
                 name: str = "microsoft/mdeberta-v3-base",
                 cl_enable: bool = False,
                 cl_temp: float = 0.05,
                 cl_weight: float = 1.0,
                 ntx_enable: bool = False,
                 classification_task_enable: bool = False,
                 classification_task_head: int = 1422,
                 hidden_size: int = 768,
                 load_pretrain: Optional[str] = "hdfs://harunava/home/byte_magellan_va/user/jiangjunjun.happy/xlmr16/model_state_epoch_400000.th",
                 all_gather_limit: int = -1,

                 warmup_ratio: float = 0.1,
                 weight_decay: float = 0.05,
                 base_lr: float = 5e-4,
                 warmup_lr: float = 5e-7,
                 min_lr: float = 5e-6,
                 lr_scheduler: str = 'cosine',
                 lr_scheduler_decay_ratio: float = 0.8,
                 lr_scheduler_decay_rate: float = 0.1,
                 optimizer: str = 'adamw',
                 optimizer_eps: float = 1e-8,
                 optimizer_betas: Tuple[float, ...] = (0.9, 0.999),
                 momentum: float = 0.9,
                 ):
        super().__init__()
        self.save_hparams()

        # self.model = AutoModelForMaskedLM.from_pretrained(name)
        self.model = DebertaV2ForMaskedLM.from_pretrained(name)

        # self.init_weights()
        # self.freeze_params(self.config.TRAIN.freeze_prefix)

        self.cl_enable = cl_enable  # use contrast learning loss
        self.cl_temp = cl_temp  # tempure of softmax
        self.cl_weight = cl_weight  # weighted loss
        self.cl_loss_layer = torch.nn.CrossEntropyLoss()

        self.ntx_enable = ntx_enable
        self.ntx_loss_layer = LearnableNTXentLoss()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.classification_task_enable = classification_task_enable
        self.classification_task_head = classification_task_head
        if self.classification_task_enable:
            self.classifier = torch.nn.Linear(hidden_size, self.classification_task_head)
            self.cla_loss_layer = torch.nn.CrossEntropyLoss()

        # setup nce allgather group if has limit
        nce_limit = self.hparams.all_gather_limit
        if nce_limit < 0:
            # no limit
            self.nce_group = None
        elif nce_limit == 0:
            # no all_gather
            self.nce_group = False
        else:
            raise NotImplementedError("Using sub-groups in NCCL is not well implemented.")
            group_rank_id = self.trainer.global_rank // nce_limit
            group_ranks = [group_rank_id * nce_limit + i for i in range(nce_limit)]
            self.nce_group = torch.distributed.new_group(ranks=group_ranks, backend='nccl')
            self.print('Create non-global allgather group from ranks:', group_ranks, 'group size:', self.nce_group.size())

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
        z1, z2 = cls_status[0:batch_size // 2, :], cls_status[batch_size // 2:, :]

        # all gather to increase effective batch size
        if self.nce_group is not False:
            # [bsz, n] -> [group, bsz, n]
            group_z1 = self.all_gather(z1, group=self.nce_group, sync_grads='rank')
            group_z2 = self.all_gather(z2, group=self.nce_group, sync_grads='rank')
            # [group, bsz, n] -> [group * bsz, n]
            z1 = group_z1.view((-1, cls_status.shape[-1]))
            z2 = group_z2.view((-1, cls_status.shape[-1]))

        if self.ntx_enable:
            loss = self.ntx_loss_layer(z1, z2)
        else:
            # cosine similarity as logits
            self.logit_scale.data.clamp_(-np.log(100), np.log(100))
            logit_scale = self.logit_scale.exp()
            self.log('logit_scale', logit_scale)
            logits_per_z1 = logit_scale * z1 @ z2.t()
            logits_per_z2 = logit_scale * z2 @ z1.t()

            bsz = logits_per_z1.shape[0]
            labels = torch.arange(bsz, device=logits_per_z1.device)  # bsz

            loss_v = self.cl_loss_layer(logits_per_z1, labels)
            loss_t = self.cl_loss_layer(logits_per_z2, labels)
            loss = (loss_v + loss_t) / 2

        return loss

    def forward(self, input_ids, input_segment_ids, input_mask, labels, classification_label=None):
        """
        input_ids: [bsz, seq_len]
        input_segment_ids: [bsz, seq_len]
        input_mask: [bsz, seq_len]
        label: [bsz]
        classification_label: [bsz]
        """
        output_dict = {}

        # mlm loss
        mmout = self.model(input_ids, input_mask, input_segment_ids, labels=labels, output_hidden_states=True)
        loss1 = mmout.loss
        self.log('mlm_loss', loss1)
        output_dict['loss'] = loss1

        # cl loss
        if self.cl_enable:
            hidden_states = mmout.hidden_states[-1]  # batch * sen_len * emd_size
            cls_status = hidden_states[:, 0, :]
            loss2 = self.cl_loss(cls_status)
            self.log('cl_loss', loss2)
            loss = loss1 + self.cl_weight * loss2

            output_dict['mlm_loss'] = loss1
            output_dict['cl_loss'] = loss2
            output_dict['loss'] = loss

        # classification task
        if self.classification_task_enable:
            hidden_states = mmout.hidden_states[-1]  # batch * sen_len * emd_size
            cls_status = hidden_states[:, 0, :]  # batch * emd_size
            logits = self.classifier(cls_status)  # batch * label_size
            loss3 = self.cla_loss_layer(logits, classification_label)
            loss = loss1 + loss3

            output_dict['mlm_loss'] = loss1
            output_dict['cla_loss'] = loss3
            output_dict['loss'] = loss

        return output_dict

    def training_step(self, batch, idx):
        return self(**batch)

    def validation_step(self, batch, idx):
        return self(**batch)

    def configure_optimizers(self):
        optimizer = build_optimizer(self.hparams, self)
        lr_scheduler = build_scheduler(
            self.hparams, optimizer,
            self.trainer.max_epochs,
            self.trainer.steps_per_epoch // self.trainer._accumulate_grad_batches)
        return [optimizer], [lr_scheduler]

    def lr_scheduler_step(self, schedulers, **kwargs):
        # timm lr scheduler is called every step
        for scheduler in schedulers:
            scheduler.step_update(self.trainer.global_step // self.trainer._accumulate_grad_batches)

    def on_fit_start(self) -> None:
        self.rank_zero_print('===========My custom fit start function is called!============')


if __name__ == '__main__':
    cli = CruiseCLI(Mdeberta,
                    trainer_class=CruiseTrainer,
                    datamodule_class=LMDataModule,
                    trainer_defaults={
                        'log_every_n_steps': 50,
                        'precision': 'fp16',
                        'max_epochs': 1,
                        'enable_versions': True,
                        'val_check_interval': 1.0,  # val after 1 epoch
                        'limit_val_batches': 100,
                        'gradient_clip_val': 2.0,
                        'sync_batchnorm': True,
                        'find_unused_parameters': True,
                        'summarize_model_depth': 2,
                        'checkpoint_monitor': 'loss',
                        'checkpoint_mode': 'min',
                        'default_hdfs_dir': 'hdfs://harunasg/home/byte_magellan_govern/users/jiangjunjun.happy/xlmr14' # use your own path to save model
                        }
                    )
    cfg, trainer, model, datamodule = cli.parse_args()
    trainer.fit(model, datamodule)
