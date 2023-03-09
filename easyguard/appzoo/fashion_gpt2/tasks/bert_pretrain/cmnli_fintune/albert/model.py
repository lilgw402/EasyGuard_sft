from typing import List, Optional, Dict
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from cruise import CruiseModule, CruiseCLI, CruiseConfig
from mariana.data.bert_pretrain.datamodule.cmnli import NLIFinetuneDatamodule
from mariana.utils.exp_helper import ExpHelper
from mariana.models.ptx_bert import BertModel
from mariana.optim.optimizer import AdamW  # diff to pytorch native AdamW?
from mariana.optim.lr_scheduler import get_linear_schedule_with_warmup  # use compose?


class PoolingClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, num_labels: int, hidden_size: int = 768, hidden_dropout_prob: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


bert_default_config = {
    "vocab_size": 145608,
    "embedding_dim": 256,
    "dim": 768,
    "dim_ff": 3072,
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-8,
    "layernorm_type": "v0",
    "layernorm_fp16": True,
    "head_layernorm_type": "v0",
    "act": "gelu",
    "max_len": 512,
    "n_heads": 12,
    "n_layers": 6,
    "n_segments": 2,
    "p_drop_attn": 0.1,
    "p_drop_hidden": 0.1,
    "padding_index": 2,
    "ignore_index": 0,
    "calc_mlm_accuracy": False,
    "extra_transformer_config": {
      "layernorm_fp16": True,
      "dropout_in_ffn": False,
      "omit_other_attn_output": True,
      "mha_acts_unite_d01": True,
      "fuse_qkv_projs": True,
      "use_ft_softmax": False,
      "use_ft_layernorm": False,
      "use_ft_linear_in_attn": False,
      "use_ft_transpose_in_attn": False,
      "use_ft_mm_in_attn": False,
      "use_ft_linear_in_attn_out": False,
      "use_ft_linear_in_ffn": False,
      "use_ft_ffn_linear_fusion": False,
      "use_ffn_output_dropout": False,
      "use_ft_attn_out_proj_dropout_fusion": False
    },
    "omit_other_output": True,
    "use_ft_linear_amap": False,
    "use_ft_layernorm_amap": False
  }


class NLIFinetuneModel(CruiseModule):
    def __init__(self,
                 bert: CruiseConfig = bert_default_config,
                 pretrained_path: str = 'hdfs://haruna/home/byte_arnold_lq/data/reckon/mlxlab_aml_test/tasks/3110044/trials/10111340/output/checkpoints/epoch=1-step=612792.ckpt',
                 pretrained_rename_prefix: Dict[str, str] = {'albert': 'pretrained'},
                 num_labels: int = 3,
                 hidden_size: int = -1,
                 lr: float = 3e-5,
                 wd: float = 0.01,
                 warmup_step_rate: float = 0.1,
                 ):
        super().__init__()
        self.save_hparams()
        self.pretrained = BertModel(**self.hparams.bert)
        if hasattr(self.pretrained, 'bare_mode'):
            self.pretrained.bare_mode = True
        # TODO(Zhi): allow NLI model to be customized
        self.dropout = torch.nn.Dropout(0.1)
        self.num_labels = num_labels
        if hidden_size < 0:
            hidden_size = bert.dim
        assert isinstance(hidden_size, int) and hidden_size > 0, "Invalid model dim/hidden_size"
        self.classifier = PoolingClassificationHead(num_labels=num_labels, hidden_size=hidden_size, hidden_dropout_prob=0.1)
        self.classifier.apply(self._init_weights)

        if self.num_labels == 1:
            self.loss = torch.nn.MSELoss()
        else:
            self.loss = torch.nn.CrossEntropyLoss()

    def rank_zero_prepare(self):
        if 'mp_rank' in self.hparams.pretrained_path:
            # zero2 checkpoints has key 'module'
            from cruise.utilities.cloud_io import load as crs_load
            state_dict = crs_load(self.hparams.pretrained_path, map_location='cpu')['module']
            state_dict = {k[7:]: v for k, v in state_dict.items()}
            self.partial_load_from_checkpoints(state_dict, rename_params=self.hparams.pretrained_rename_prefix)
        else:
            self.partial_load_from_checkpoints(self.hparams.pretrained_path, map_location='cpu', rename_params=self.hparams.pretrained_rename_prefix)

    def forward(self,
                input_ids: torch.Tensor,
                input_mask: torch.Tensor,
                segment_ids: torch.Tensor,
                position_ids: torch.Tensor):
        bert_output = self.pretrained(
                input_ids=input_ids,
                position_ids=position_ids,
                segment_ids=segment_ids,
                attention_mask=input_mask)
        sequence_output = bert_output['sequence_output']
        enc_out = sequence_output[:, 0, :]
        enc_out = self.dropout(enc_out)
        logits = self.classifier(enc_out)
        if 'loss' in bert_output:
            return logits, bert_output['loss']
        return logits, 0

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        input_mask = batch['input_mask']
        segment_ids = batch['segment_ids']
        position_ids = batch['position_ids']
        label = batch['labels'].long()
        logits, prev_loss = self.forward(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            position_ids=position_ids,
        )
        if self.num_labels == 1:
            probs = F.sigmoid(logits.squeeze(-1))
        else:
            probs = F.softmax(logits, dim=-1)

        output_dict = {}
        if self.num_labels == 1:
            loss = self.loss(probs, label)
        else:
            loss = self.loss(logits, label.long().view(-1))
            acc = ((label == torch.argmax(probs, dim=-1).long()).sum().float() / label.numel()).item()
            output_dict['acc'] = acc
        output_dict['loss'] = loss + prev_loss
        return output_dict

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        input_mask = batch['input_mask']
        segment_ids = batch['segment_ids']
        position_ids = batch['position_ids']
        label = batch['labels'].long()
        mask = label > -1
        logits, prev_loss = self.forward(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            position_ids=position_ids,
        )
        preds = torch.argmax(logits, dim=-1).long()
        # TODO: current acc is not super accurate because there are some duplicated samples
        # to ensure zero3 batch shape, we can check if it's repeated by '_num_valid_samples' attribute
        acc = ((label[mask] == preds[mask]).sum().float() / label[mask].numel()).item()
        return {'val_acc': acc}

    def configure_optimizers(self):
        no_decay = ['bias', 'bn', 'norm', 'ln', 'attn_', 'Norm']
        no_dacay_params_dict = {'params': [], 'weight_decay': 0.0}
        normal_params_dict = {'params': [], 'weight_decay': self.hparams.wd}

        for n, p in self.named_parameters():
            if any(nd in n for nd in no_decay):
                no_dacay_params_dict['params'].append(p)
            else:
                normal_params_dict['params'].append(p)
        optimizer_grouped_parameters = [
            no_dacay_params_dict,
            normal_params_dict]

        optm = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.lr,
            betas=(0.9, 0.999),
            eps=1e-6,
            weight_decay=self.hparams.wd,
            correct_bias=False  # TODO(Zhi): align with master branch ADAMW
            )

        warmup_steps = self.hparams.warmup_step_rate * self.trainer.total_steps
        lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=optm,
                num_warmup_steps=warmup_steps,
                num_training_steps=self.trainer.total_steps)

        return [optm], [lr_scheduler]

    def lr_scheduler_step(
        self,
        schedulers,
        **kwargs,
    ) -> None:
        r"""
        默认是per epoch的lr schedule, 改成per step的
        """
        # if self.trainer.global_step == 0:
        #     # skip first step
        #     return
        for scheduler in schedulers:
            scheduler.step()

    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


if __name__ == '__main__':
    helper = ExpHelper(__file__)
    cli = CruiseCLI(
        NLIFinetuneModel,
        datamodule_class=NLIFinetuneDatamodule,
        trainer_defaults={
            'logger': 'console',
            'precision': 16,
            'enable_versions': False,
            'find_unused_parameters': True,
            'max_epochs': 2,
            "default_hdfs_dir": helper.hdfs_prefix,
            "project_name": helper.project_name,
            'val_check_interval': 1.0,
            'summarize_model_depth': 2,
            'gradient_clip_val': 1.0,
            'checkpoint_monitor': 'step',
            'checkpoint_mode': 'max',
            'enable_checkpoint': False})

    cfg, trainer, model, datamodule = cli.parse_args()
    trainer.fit(model, datamodule)
