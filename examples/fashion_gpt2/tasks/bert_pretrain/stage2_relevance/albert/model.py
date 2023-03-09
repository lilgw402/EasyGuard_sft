from typing import List, Optional, Dict
import os
import torch
import torch.nn.functional as F

from cruise import CruiseModule, CruiseCLI, CruiseConfig
from mariana.data.bert_pretrain.datamodule.relevance import RelevanceDatamodule
# from mariana.data.bert_pretrain.benchmark import SUPPORTED_BENCHMARKS
SUPPORTED_BENCHMARKS = {}  # no online benchmark for stage 2, checkout qqqt for offline evaluation
from mariana.utils.exp_helper import ExpHelper
from mariana.models.ptx_bert import BertModel, init_weights
from mariana.optim import mariana_optimizer_kwargs_defaults


# Config adapter for ALBert
albert_default_config = {
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
    "calc_mlm_accuracy": True,
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
    "omit_other_output": False,
    "use_ft_linear_amap": False,
    "use_ft_layernorm_amap": False
  }


class AlbertPretrainModel(CruiseModule):
    """Albert pretrain"""
    def __init__(self,
                 bert: CruiseConfig = albert_default_config,
                 freeze_prefix: Optional[List[str]] = None,
                 partial_pretrain: Optional[str] = None,
                 partial_pretrain_rename: Optional[Dict[str, str]] = None,
                 pooled_output_dropout: float = 0.1,
                 fixed_margin: Optional[float] = None,
                 mlm_loss_scale: float = 1.0,
                 margin_loss_scale: float = 5.0,
                 ):
        super().__init__()
        self.save_hparams()  # save to self.hparams

        # 文本
        self.albert = BertModel(**self.hparams.bert)
        self.pooled_dropout = torch.nn.Dropout(pooled_output_dropout)
        self.pooled_hidden = torch.nn.Linear(self.hparams.bert.dim, self.hparams.bert.dim // 2)
        self.pooled_hidden_act = torch.nn.GELU()
        self.pooled_score = torch.nn.Linear(self.hparams.bert.dim // 2, 1)
        self.pooled_hidden.apply(init_weights)
        self.pooled_score.apply(init_weights)
        self.freeze_params(self.hparams.freeze_prefix or [])

    def setup(self):
        # In DDP rank 0 load pretrain weights is enough
        if self.trainer.global_rank == 0 and self.hparams.partial_pretrain:
            rename_params = self.hparams.partial_pretrain_rename or {}
            self.partial_load_from_checkpoints(
                self.hparams.partial_pretrain,
                rename_params=rename_params, verbose=True)

    def freeze_params(self, freeze_prefix):
        for name, param in self.named_parameters():
            for prefix in freeze_prefix:
                if name.startswith(prefix):
                    self.rank_zero_print('freeze_params:', name)
                    param.requires_grad = False

    def forward(
        self,
        input_ids,
        input_mask,
        position_ids,
        segment_ids,
        labels=None,
        masked_lm_weights=None,
        masked_lm_positions=None,
        masked_lm_ids=None,
        sentence_label=None,
    ):
        return self.albert.forward(
            input_ids=input_ids, position_ids=position_ids, segment_ids=segment_ids,
            masked_tokens=labels, sentence_label=sentence_label,
            masked_lm_positions=masked_lm_positions, masked_lm_ids=masked_lm_ids,
            masked_lm_weights=masked_lm_weights,
        )

    def _preprocess_batch(self, batch):
        """Concat qp and qn inputs along the batch dimension"""
        out_batch = {}
        for name in ('input_ids', 'segment_ids', 'input_mask', 'masked_lm_positions',
                     'masked_lm_ids', 'masked_lm_weights', 'labels'):
            out_batch[name] = torch.cat([batch['qp_' + name], batch['qn_' + name]], dim=0)
        out_batch['position_ids'] = torch.cat([batch['position_ids'], batch['position_ids']], dim=0)
        return out_batch

    def training_step(self, batch, batch_idx):
        # log lr
        scheduler = self.trainer.lr_scheduler_configs[0].scheduler
        if hasattr(scheduler, 'get_lr'):
            self.log('lr', scheduler.get_lr()[0], console=True)
        else:
            self.log('lr', scheduler.get_last_lr()[0], console=True)
        self.deberta._omit_other_output = False
        self.albert.bare_mode = False
        loss_dict = self.forward(**self._preprocess_batch(batch))
        if self.albert.local_metrics:
            self.log_dict(self.albert.local_metrics, console=True)
        # pn loss
        pooled_output = loss_dict['pooled_output']
        pooled_output = self.pooled_dropout(pooled_output)
        pooled_output = self.pooled_hidden_act(self.pooled_hidden(pooled_output))
        pn_scores = torch.sigmoid(self.pooled_score(pooled_output))
        split_size = pn_scores.shape[0] // 2
        qp_scores, qn_scores = torch.split(pn_scores, split_size, dim=0)
        self.log('qp_score', qp_scores.mean().item(), console=True)
        self.log('qn_score', qn_scores.mean().item(), console=True)
        if isinstance(self.hparams.fixed_margin, float):
            margin = self.hparams.fixed_margin
            self.rank_zero_warn(f'Using global fixed margin value: {margin}.')
        else:
            margin = batch['margin']
        per_example_loss = F.relu(margin + qn_scores - qp_scores)  # torch.maximum(0, xxx)
        pn_loss = per_example_loss.mean()
        self.log('margin_loss', pn_loss, console=True)
        total_loss = loss_dict['loss'] * self.hparams.mlm_loss_scale + pn_loss * self.hparams.margin_loss_scale
        return {'loss': total_loss}

    def validation_step(self, batch, batch_idx):
        self.albert._omit_other_output = True
        self.albert.bare_mode = True
        benchmark = batch.pop('benchmark', None)
        if not self.trainer.default_hdfs_dir:
            self.rank_zero_warn("Benchmark requires HDFS dir which is disabled in trainer, skip.")
            return {}
        batch['output_path'] = os.path.join(self.trainer.default_hdfs_dir, 'benchmark_output')
        if benchmark not in SUPPORTED_BENCHMARKS:
            self.rank_zero_warn(f"Unrecognized benchmark {benchmark} received, will skip it")
            result = {}
        else:
            self.rank_zero_info(f"Running {batch_idx}-th benchmark: {benchmark} with kwargs: {batch}")
            batch['model'] = self
            batch['hidden_size'] = self.albert.encoder.dim
            bench = SUPPORTED_BENCHMARKS[benchmark](**batch)
            result = bench.run()  # can be empty if not rank0, but it's okay
        return result

    def configure_optimizers(self, optimizer_kwargs):
        """
        Model定制optimizer和lr_scheduler
        """
        no_decay = ['bias', 'bn', 'norm', 'ln']
        no_dacay_params_dict = {'params': [], 'weight_decay': 0.0}
        normal_params_dict = {'params': [], 'weight_decay': optimizer_kwargs["optimizer"]["params"]["weight_decay"]}

        for n, p in self.named_parameters():
            if any(nd in n for nd in no_decay):
                no_dacay_params_dict['params'].append(p)
            else:
                normal_params_dict['params'].append(p)
        optimizer_grouped_parameters = [
            no_dacay_params_dict,
            normal_params_dict]

        optimizers = super()._configure_optimizers(optimizer_grouped_parameters, optimizer_kwargs)
        lr_schedulers = super()._configure_schedulers(optimizers, optimizer_kwargs)
        return optimizers, lr_schedulers

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


if __name__ == '__main__':
    helper = ExpHelper(__file__)
    from cruise.trainer.callback import ModelCheckpoint
    ckpter = ModelCheckpoint(monitor='step',
                             save_last=False,
                             save_top_k=-1,
                             every_n_train_steps=20000,
                             every_n_epochs=1,
                             save_on_train_epoch_end=True,
                             enable_trace=False)
    cli = CruiseCLI(
        AlbertPretrainModel,
        datamodule_class=RelevanceDatamodule,
        trainer_defaults={
            'precision': 16,
            'enable_versions': False,
            'find_unused_parameters': True,
            'max_epochs': 1,
            "default_hdfs_dir": helper.hdfs_prefix,
            "project_name": helper.project_name,
            'val_check_interval': -1,
            'summarize_model_depth': 2,
            'gradient_clip_val': 1.0,
            'checkpoint_monitor': 'step',
            'checkpoint_mode': 'max',
            'callbacks': [ckpter],
            'optimizer_kwargs': mariana_optimizer_kwargs_defaults,
        }
    )

    cli.add_argument('--val-only', default=False, action='store_true', dest='val_only')

    cfg, trainer, model, datamodule = cli.parse_args()
    if cfg.val_only:
        trainer.validate(model, datamodule=datamodule)
    else:
        trainer.fit(model, datamodule)
