from typing import List, Optional, Dict
import os
import torch
import torch.nn.functional as F

from cruise import CruiseModule, CruiseCLI, CruiseConfig
from mariana.data.bert_pretrain.datamodule.mlm import MLMPretrainDatamodule
from mariana.data.bert_pretrain.benchmark import SUPPORTED_BENCHMARKS
from mariana.utils.exp_helper import ExpHelper
from mariana.models.ptx_bert import BertModel
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


class AlbertPretrainModel(CruiseModule):
    """Albert pretrain"""
    def __init__(self,
                 bert: CruiseConfig = albert_default_config,
                 freeze_prefix: Optional[List[str]] = None,
                 partial_pretrain: Optional[List[str]] = None,
                 partial_pretrain_rename: Optional[Dict[str, str]] = None,
                 ):
        super().__init__()
        self.save_hparams()  # save to self.hparams

        # 文本
        self.albert = BertModel(**self.hparams.bert)
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
                    print('freeze_params:', name)
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

    def training_step(self, batch, batch_idx):
        # log lr
        scheduler = self.trainer.lr_scheduler_configs[0].scheduler
        if hasattr(scheduler, 'get_lr'):
            self.log('lr', scheduler.get_lr()[0], console=True)
        else:
            self.log('lr', scheduler.get_last_lr()[0], console=True)
        self.albert._omit_other_output = True
        self.albert.bare_mode = False
        loss_dict = self.forward(**batch)
        return loss_dict

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
        datamodule_class=MLMPretrainDatamodule,
        trainer_defaults={
            'precision': 16,
            'enable_versions': False,
            'find_unused_parameters': False,
            'max_epochs': 2,
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
