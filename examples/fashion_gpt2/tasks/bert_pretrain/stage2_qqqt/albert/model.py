from typing import List, Optional, Dict
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from cruise import CruiseModule, CruiseCLI, CruiseConfig
from mariana.data.bert_pretrain.datamodule.qqqt import QqqtFinetuneDatamodule
from mariana.utils.exp_helper import ExpHelper
from mariana.models.ptx_bert import BertModel
from mariana.optim.optimizer import AdamW  # diff to pytorch native AdamW?
from mariana.optim.lr_scheduler import get_linear_schedule_with_warmup  # use compose?
from mariana.metrics.ndcg import evaluate_ndcg


class MarginOutputLayer(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, output_dropout: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.output_dropout = torch.nn.Dropout(p=output_dropout)
        self.eps = eps

    def forward(self, cls_output, seq_output, input_mask, segment_ids, margin=0.3):
        input_len = torch.sum(input_mask, dim=1)
        max_seq_len = input_mask.shape[1]
        shift_mask = (torch.arange(0, max_seq_len, 1, device=input_mask.device) < torch.unsqueeze(input_len, dim=-1)).to(torch.int32)
        second_sep_mask = input_mask - shift_mask
        text_b_mask = segment_ids - second_sep_mask
        text_b_len = torch.sum(text_b_mask, axis=1, keepdim=True)
        text_b_output = seq_output * text_b_mask.unsqueeze(dim=2).to(torch.float32)
        text_b_avg = torch.sum(text_b_output, dim=1) / torch.max(text_b_len, dim=1, keepdim=True).values.to(torch.float32)

        output_layer = torch.cat([cls_output, text_b_avg], dim=1)
        output_layer = self.output_dropout(output_layer)
        if self.training:
            split_size = output_layer.shape[0] // 3
            query_output, pos_output, neg_output = torch.split(output_layer, split_size, dim=0)
        else:
            split_size = output_layer.shape[0] // 2
            query_output, pos_output = torch.split(output_layer, split_size, dim=0)
            neg_output = None

        query_output = torch.nn.functional.normalize(query_output, p=2, dim=1, eps=self.eps)
        pos_output = torch.nn.functional.normalize(pos_output, p=2, dim=1, eps=self.eps)
        if neg_output is not None:
            neg_output = torch.nn.functional.normalize(neg_output, p=2, dim=1, eps=self.eps)
        pos_sim = torch.sum(query_output * pos_output, dim=1)
        pos_prob = (pos_sim + 1) * 0.5

        if self.training:
            neg_sim = torch.sum(query_output * neg_output, dim=1)
            per_example_loss = torch.maximum(torch.zeros_like(pos_sim, device=pos_sim.device), margin + neg_sim - pos_sim)
            loss = per_example_loss.mean()
        else:
            loss, per_example_loss = 0.0, 0.0
        return loss, per_example_loss, pos_sim, pos_prob


bert_default_config = {
    "vocab_size": 145608,
    "embedding_dim": 256,
    "dim": 768,
    "dim_ff": 3072,
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-8,
    "layernorm_type": "v0",
    "layernorm_fp16": False,
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
      "layernorm_fp16": False,
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


class QqqtFinetuneModel(CruiseModule):
    def __init__(self,
                 bert: CruiseConfig = bert_default_config,
                 pretrained_path: str = 'hdfs://haruna/home/byte_arnold_lq/data/reckon/mlxlab_aml_test/tasks/3110044/trials/10111340/output/checkpoints/epoch=1-step=612792.ckpt',
                 pretrained_rename_prefix: Dict[str, str] = {'albert': 'pretrained'},
                 lr: float = 6e-6,
                 wd: float = 0.01,
                 margin: float = 0.3,
                 ):
        super().__init__()
        self.save_hparams()
        self.pretrained = BertModel(**self.hparams.bert)
        if hasattr(self.pretrained, 'bare_mode'):
            self.pretrained.bare_mode = True
        self.margin_out = MarginOutputLayer(output_dropout=0.1)

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
        pooled_output = bert_output['pooled_output']
        if 'loss' in bert_output:
            return sequence_output, pooled_output, bert_output['loss']
        return sequence_output, pooled_output, 0

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        input_mask = batch['input_mask']
        segment_ids = batch['segment_ids']
        pos_input_ids = batch['pos_input_ids']
        pos_segment_ids = batch['pos_segment_ids']
        pos_input_mask = batch['pos_input_mask']
        neg_input_ids = batch['neg_input_ids']
        neg_segment_ids = batch['neg_segment_ids']
        neg_input_mask = batch['neg_input_mask']
        position_ids = batch['position_ids']
        # during training, concat input, pos input and neg input along the first dim
        all_input_mask = torch.cat([input_mask, pos_input_mask, neg_input_mask], dim=0)
        all_segment_ids = torch.cat([segment_ids, pos_segment_ids, neg_segment_ids], dim=0)
        sequence_output, pooled_output, prev_loss = self.forward(
            input_ids=torch.cat([input_ids, pos_input_ids, neg_input_ids], dim=0),
            input_mask=all_input_mask,
            segment_ids=all_segment_ids,
            position_ids=torch.cat([position_ids, position_ids, position_ids], dim=0),
        )

        # margin loss
        loss, _, pos_sim, pos_prob = self.margin_out(
                pooled_output, sequence_output, all_input_mask, all_segment_ids, margin=self.hparams.margin)

        output_dict = {}
        output_dict['loss'] = loss + prev_loss
        return output_dict

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        input_mask = batch['input_mask']
        segment_ids = batch['segment_ids']
        pos_input_ids = batch['pos_input_ids']
        pos_segment_ids = batch['pos_segment_ids']
        pos_input_mask = batch['pos_input_mask']
        position_ids = batch['position_ids']
        _num_valid_samples = batch.get('_num_valid_samples', input_ids.shape[0])
        # during training, concat input, pos input along the first dim
        all_input_mask = torch.cat([input_mask, pos_input_mask], dim=0)
        all_segment_ids = torch.cat([segment_ids, pos_segment_ids], dim=0)
        sequence_output, pooled_output, prev_loss = self.forward(
            input_ids=torch.cat([input_ids, pos_input_ids], dim=0),
            input_mask=all_input_mask,
            segment_ids=all_segment_ids,
            position_ids=torch.cat([position_ids, position_ids], dim=0),
        )

        # margin loss
        _, _, _, pos_prob = self.margin_out(
                pooled_output, sequence_output, all_input_mask, all_segment_ids)
        pos_prob = pos_prob[:_num_valid_samples]
        return {'pos_prob': pos_prob}

    def validation_epoch_end(self, outputs) -> None:
        assert self.trainer._datamodule is not None
        labels = self.trainer._datamodule.labels
        # collect predictions from all ranks
        rank_probs = torch.cat([out['pos_prob'] for out in outputs], dim=0)
        all_rank_probs = self.all_gather(rank_probs, sync_grads=False).reshape(-1).cpu().numpy()
        self.rank_zero_info(f'All probs collected: {all_rank_probs.shape}')
        predicts = []
        predict = []
        label_idx = 0
        for prob in all_rank_probs:
            predict.append(prob)
            if len(predict) == len(labels[label_idx]):
                predicts.append(predict)
                predict = []
                label_idx += 1
        self.rank_zero_info(f"Num predictions: {len(predicts)}")
        result = evaluate_ndcg(labels, predicts)
        self.log_dict(result, console=True)
        return result

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

        return optm


if __name__ == '__main__':
    helper = ExpHelper(__file__)
    cli = CruiseCLI(
        QqqtFinetuneModel,
        datamodule_class=QqqtFinetuneDatamodule,
        trainer_defaults={
            'logger': 'console',
            'precision': 16,
            'enable_versions': False,
            'find_unused_parameters': True,
            'max_epochs': 1,
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
