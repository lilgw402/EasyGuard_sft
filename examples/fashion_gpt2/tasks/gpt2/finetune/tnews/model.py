from typing import List, Optional, Dict, Union
import math
import os
import torch
import torch.nn.functional as F
from torch import nn

from cruise import CruiseModule, CruiseCLI, CruiseConfig
from cruise.utilities.distributed import DIST_ENV
from mariana.utils.exp_helper import ExpHelper
from mariana.models.gpt2 import Conv1D
from mariana.data.gpt.datamodule.classification import ClassificationGPTDatamodule
from mariana.models.gpt2_finetune import GPT2LMModelwClassificationHead,GPT2LMModelwClassificationHeadRMPAD
from mariana.models.gpt2 import get_subsequent_mask
from mariana.utils.generate import play_console, play_file
from mariana.utils.rh2 import dump_metrics_to_hdfs, set_metric_prefix
from mariana.optim import mariana_optimizer_kwargs_defaults

network_config = {
    "hidden_size": 2048,
    "n_embed": 512,  # vocab embedding
    "n_inner": 8192,
    "n_head": 16,
    "n_layer": 24,
    "vocab_size": 145664,
    "max_position_embeddings": 2048,
    "layer_norm_epsilon": 1.0e-5,
    "activation_function": "gelu_new",
    "resid_pdrop": 0.1,
    "embd_pdrop": 0.1,
    "attn_pdrop": 0.1,
    "scale_attn_weights": True,  # TODO:
    "scale_attn_by_inverse_layer_idx": False,  # TODO:
    "reorder_and_upcast_attn": False,  # TODO:
    "initializer_range": 0.02,
    "gradient_checkpointing": False,
    "initializer_range": 0.02,
    "gradient_checkpointing": False,
    "gradient_checkpointing_ln": False,
    "gradient_checkpointing_mlp": False,
    "gradient_checkpointing_start_layers": 0,
    "tie_weight": True,
    "pad_idx": -100,
    "num_labels": 17, #task_config.get("num_labels", 17),  # Depends on the classification label size, 17 for TNEWS
    "lm_loss_weight": 0.0,
    "classification_head_bias": False,
    "use_ft_flash_attn": False,
    "use_ft_linear": False,
    "use_ft_layernorm": False,
    "use_rmpad_lmloss": False,
    "use_rmpad_lnmlp": False,
    "use_rmpad_attn": False,
    "use_rmpad": False,
    "pad_output": False,
  }

# model config for Alice 1.3b
# --model.network.pad_idx=2
# --model.max_position_embeddings=2048
'''
network_config_alice_1b3 = CruiseConfig({
    "hidden_size": 2048,
    "n_embed": 512,  # vocab embedding
    "n_inner": 8192,
    "n_head": 16,
    "n_layer": 24,
    "vocab_size": 145664,
    "max_position_embeddings": 2048,
    "layer_norm_epsilon": 1.0e-5,
    "activation_function": "gelu_new",
    "resid_pdrop": 0.1,
    "embd_pdrop": 0.1,
    "attn_pdrop": 0.1,
    "scale_attn_weights": True,  # TODO:
    "scale_attn_by_inverse_layer_idx": False,  # TODO:
    "reorder_and_upcast_attn": False,  # TODO:
    "initializer_range": 0.02,
    "gradient_checkpointing": False,
    "tie_weight": True,
    "pad_idx": 2,
    "num_labels": 17,#task_config.get("num_labels", 17),  # Depends on the classification label size 17 for tnnews
    "lm_loss_weight": 0.0,
  })
'''

class GPT2ClassificationModel(CruiseModule):
    """Deberta pretrain"""
    def __init__(self,
                 network: CruiseConfig = network_config,
                 freeze_prefix: Optional[List[str]] = None,
                 partial_pretrain: Optional[str] = None,
                 partial_pretrain_rename: Optional[Dict[str, str]] = None,
                 ):
        super().__init__()
        self.save_hparams()  # save to self.hparams

        # 文本
        self.gpt = GPT2LMModelwClassificationHeadRMPAD(self.hparams)
        self.init_weights()
        self.freeze_params(self.hparams.freeze_prefix or [])
        self.average_token_rate = 0
        self.use_rmpad = network.get('use_rmpad', False)
        self.pad_output = network.get('pad_output', False)

        self.best_val_performance_value = 0.0
        self.best_val_performance_batch_idx = -1


    def setup(self):
        # In DDP rank 0 load pretrain weights is enough
        if self.trainer.global_rank == 0 and self.hparams.partial_pretrain:
            rename_params = self.hparams.partial_pretrain_rename or {}
            if 'mp_rank' in self.hparams.partial_pretrain:
                # zero2 checkpoints has key 'module'
                from cruise.utilities.cloud_io import load as crs_load
                state_dict = crs_load(self.hparams.partial_pretrain, map_location='cpu')['module']
                state_dict = {k[7:]: v for k, v in state_dict.items()}
                self.partial_load_from_checkpoints(state_dict, rename_params=rename_params)
            else:
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
        attention_mask,
        labels=None,
        classification_labels = None,
        actual_seq_length = None, 
    ):
        attention_mask = get_subsequent_mask(attention_mask)
        # print("original attention mask: {} with size: {}".format(attention_mask, attention_mask.size()))

        model_out = self.gpt(input_ids=input_ids, attention_mask=attention_mask, 
                            labels=labels, classification_labels = classification_labels, 
                            actual_seq_length=actual_seq_length,
                             use_rmpad=self.use_rmpad, pad_output=self.pad_output)
        return model_out

    def training_step(self, batch, batch_idx):
        # log lr
        # (TODO) Need to confirm if we still need lr scheduler
        scheduler = self.trainer.lr_scheduler_configs[0].scheduler
        if hasattr(scheduler, 'get_lr'):
            self.log('lr', scheduler.get_lr()[0], console=True)
        else:
            self.log('lr', scheduler.get_last_lr()[0], console=True)
        # in hf model, labels will be shifted by 1, so here labels = input_ids
        batch['labels'] = batch['input_ids']
        # print("batch information: {}, with batch_idx: {}".format(batch, batch_idx))
        model_out = self.forward(**batch)
        loss = model_out['loss']

        classification_logits = model_out['classification_logits']
        preds = torch.argmax(classification_logits, dim=-1).long()

        acc_nums = (torch.squeeze(batch['classification_labels']) == preds).sum().float()

        batch_size = batch['classification_labels'].numel() * 1.0
        
        acc = acc_nums / batch_size
        
        output = {
            "loss": loss,
            'train_loss': loss, 
            'train_lm_loss': model_out['lm_loss'], 
            'train_classification_loss': model_out['classification_loss'],
            "train_acc": acc,
        }
        if batch_idx % 500 == 0:
            print("Training:  gt_labels: {}, predicted labels: {}, classification_logits: {}, train_acc: {}, accurate_nums: {}, batch_size: {}, batch_idx: {}".format(
                            torch.squeeze(batch['classification_labels']), preds, classification_logits, acc, acc_nums, batch_size, batch_idx))
        return output


    # def training_epoch_end(self, outputs) -> None:
    #     # check https://code.byted.org/data/mariana/blob/HEAD/tasks/bert_pretrain/stage2_qqqt/albert/model.py#L199 on how to calculate NDGC
        
    #     try:
    #         mean_acc = torch.mean(torch.as_tensor([out['train_acc'] for out in outputs]))
    #         self.rank_zero_info(f"train_mean_acc: {mean_acc}")
    #         self.log('train_acc_global_mean_per_epoch', mean_acc, console=True)
    #     except:
    #         print("Exception in processing train_mean_acc because train_acc is: {}, set mean_acc = 0.0".format(out["train_acc"]))
    #         mean_acc = 0.0

    #     return mean_acc


    def validation_step(self, batch, batch_idx):
        # in hf model, labels will be shifted by 1, so here labels = input_ids
        batch['labels'] = batch['input_ids']
        model_out = self.forward(**batch)
        loss = model_out['loss']
        # lm_loss = model_out['lm_loss']
        # classification_loss = model_out['classification_loss']
        classification_logits = model_out['classification_logits']

        preds = torch.argmax(classification_logits, dim=-1).long()
                
        acc_nums = (torch.squeeze(batch['classification_labels']) == preds).sum().float()
        batch_size = batch['classification_labels'].numel() * 1.0
        acc = acc_nums / batch_size

        output = {
            'val_loss': loss, 
            'val_lm_loss': model_out['lm_loss'], 
            'val_classification_loss': model_out['classification_loss'], 
            "val_acc": acc,
            'num_sample': batch_size, 
            'num_correct': acc_nums,
        }
        
        # self.log_dict(output, console=True)

        if batch_idx % 500 == 0:
            print("Test gt_labels: {}, predicted labels: {}, classification_logits: {}, val_acc: {}, accurate_nums: {}, batch_size: {}, batch_idx: {}".format(
                            torch.squeeze(batch['classification_labels']), preds, classification_logits, acc, acc_nums, batch_size, batch_idx))
        
        # if acc >= self.best_val_performance_value:
        #     self.best_val_performance_value = acc
        #     self.best_val_performance_batch_idx = batch_idx
        # print("PRINT: best_val_performance_batch: {}, best_val_performance_value (acc): {}".format(self.best_val_performance_batch_idx, self.best_val_performance_value))
        # self.rank_zero_info(f"RankZERO: best_val_performance_batch (from_rank_zero_info): {self.best_val_performance_batch_idx}")
        # self.rank_zero_info(f"RankZERO: best_val_performance_value (acc) (from_rank_zero_info): {self.best_val_performance_value}")
        return output

    def validation_epoch_end(self, outputs) -> None:

        # check https://code.byted.org/data/mariana/blob/HEAD/tasks/bert_pretrain/stage2_qqqt/albert/model.py#L199 on how to calculate NDGC
        
        rank_total_sample = torch.as_tensor([out['num_sample'] for out in outputs])
        all_rank_total_sample_list = self.all_gather_object(rank_total_sample)
        all_rank_total_sample = [sum(num) for num in all_rank_total_sample_list]


        rank_total_correct = torch.as_tensor([out['num_correct'] for out in outputs])
        all_rank_total_correct_list = self.all_gather_object(rank_total_correct)
        all_rank_total_correct = [sum(num) for num in all_rank_total_correct_list]

      
        acc = sum(all_rank_total_correct) * 1.0 / sum(all_rank_total_sample)
        self.rank_zero_info(f"All Rank Acc: val_acc_per_epoch of current step: {acc}")
        self.rank_zero_info(f"RankZERO Acc: all_rank_total_sample: {sum(all_rank_total_sample)}, rank_total_correct: {sum(all_rank_total_correct)}, val_acc_per_epoch of current step: {acc}")

        self.log_dict({
            'all_rank_acc': acc,
            'all_rank_num_samples': sum(all_rank_total_sample),
            'all_rank_num_corrects': sum(all_rank_total_correct)
        }, console=True)

        if acc >= self.best_val_performance_value:
            self.best_val_performance_value = acc

            dump_metrics_to_hdfs({
                'best_acc': self.best_val_performance_value
            })

        self.rank_zero_info(f"ALL Rank Acc: current best_val_performance_value (acc): {self.best_val_performance_value}")

        # deprecated 
        # mean_acc = torch.mean(torch.as_tensor([out['val_acc'] for out in outputs]))
        # self.rank_zero_info(f"RankZERO: val_acc_global_mean_per_epoch of current step (torch.mean): {mean_acc}")
        # self.log('val_acc_global_mean_per_epoch', mean_acc, console=True)
        # if mean_acc.item() >= self.best_val_performance_value:
        #     self.best_val_performance_value = mean_acc.item()
        # self.rank_zero_info(f"RankZERO: current best_val_performance_value (acc): {self.best_val_performance_value}")
        # return mean_acc
        return acc


    @torch.no_grad()
    def decode(self, input_ids: torch.Tensor, input_mask: torch.Tensor, *args, **kwargs):
        """For generation task"""
        model_out = self.gpt(input_ids=input_ids, attention_mask=input_mask)
        return model_out

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

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.hparams.network.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.hparams.network.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if "c_proj.weight" in name or 'attn_ow' in name: # deepspeed transformer kernel 是 attn_ow
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(mean=0.0, std=(self.hparams.network.initializer_range / math.sqrt(2 * self.hparams.network.n_layer)))


if __name__ == '__main__':
    helper = ExpHelper(__file__)
    from cruise.trainer.callback import ModelCheckpoint
    ckpter = ModelCheckpoint(monitor='step',
                             save_last=False,
                             save_top_k=0,
                             every_n_train_steps=0,
                             every_n_epochs=0,
                             save_on_train_epoch_end=False,
                             enable_trace=False)
    cli = CruiseCLI(
        GPT2ClassificationModel,
        datamodule_class=ClassificationGPTDatamodule,
        trainer_defaults={
            'precision': 16,
            'enable_versions': False,
            'log_every_n_steps': 20,
            'find_unused_parameters': False,
            'max_epochs': 10,
            "default_hdfs_dir": helper.hdfs_prefix,
            "project_name": helper.project_name,
            'val_check_interval': 1,
            'summarize_model_depth': 2,
            'gradient_clip_val': 1.0,
            'checkpoint_monitor': 'step',
            'checkpoint_mode': 'max',
            'callbacks': [ckpter],
            'optimizer_kwargs': mariana_optimizer_kwargs_defaults, 
            })
    cli.add_argument('--val-only', default=False, action='store_true', dest='val_only')
    cli.add_argument('--play', default=False, action='store_true', dest='play')
    cli.add_argument('--play-file', default='', type=str, help='generate by samples loaded from file')
    cli.add_argument('--play-file-limit', default=-1, type=int, help="If >0, limit how many lines to generate.")
    cli.add_argument('--generate-trial-num', default=5, type=int, help="generation trial num, default is 5")
    cli.add_argument('--generate-steps', default=256, type=int, help='decode sequence length/steps')
    cli.add_argument('--generate-temp', default=0.7, type=float, help='Smaller tempreature logits become more steep')
    cli.add_argument('--generate-do-sample', default=True, type=bool, help='multinomial sample if True')
    cli.add_argument('--generate-topk', default=5, type=int, help='sample top-k')
    cli.add_argument('--generate-topp', default=None, type=float, help='sample at least top-p probability')
    cli.add_argument('--generate-n-eos', default=1, type=int, help='Stop until n-eos tokens')


    cfg, trainer, model, datamodule = cli.parse_args()

    set_metric_prefix(cfg.data.task_name + '/')

    if datamodule.hparams.bsz_warmup:
        warmup_rate = trainer._config['trainer']['optimizer_kwargs']['scheduler']['params']['warmup_step_rate']
        max_epochs = trainer._config['trainer']['max_epochs']
        datamodule.hparams.warmup_step_rate = warmup_rate * max_epochs
      
    if cfg.val_only:
        trainer.validate(model, datamodule=datamodule)
    elif cfg.play_file or cfg.play:
        assert DIST_ENV.world_size == 1, "Play mode only support single card"
        datamodule.rank_zero_prepare()
        datamodule.local_rank_zero_prepare()
        datamodule.setup()
        tokenizer = datamodule.tokenizer
        assert tokenizer is not None, "Invalid tokenizer from datamodule"
        model.rank_zero_prepare()
        model.local_rank_zero_prepare()
        model.setup()
        if cfg.play_file:
            print("\nFile play mode.")
            play_file(cfg.play_file, tokenizer, model.cuda(), cfg.generate_trial_num,
                      steps=cfg.generate_steps, temperature=cfg.generate_temp, do_sample=cfg.generate_do_sample,
                      top_k=cfg.generate_topk, top_p=cfg.generate_topp, until_n_eos=cfg.generate_n_eos,
                      limit_samples=cfg.play_file_limit)
        else:
            print("\nConsole play mode.")
            play_console(tokenizer, model.cuda(), cfg.generate_trial_num,
                         steps=cfg.generate_steps, temperature=cfg.generate_temp, do_sample=cfg.generate_do_sample,
                         top_k=cfg.generate_topk, top_p=cfg.generate_topp, until_n_eos=cfg.generate_n_eos)
    else:
        if datamodule.hparams.dyn_bsz:
            # tricky impl for DynBszBuffer
            dynbsz_buf_cb = DynBszBufferCallback(
                            trainer=trainer,
                            every_n_train_steps = int(os.environ.get('MARIANA_CUSTOM_SAVE_INTERVAL', 10000)),
                            resume_ckpt_path = model.hparams.partial_pretrain if model.hparams.partial_pretrain else None,
                            )
            trainer.callbacks = [dynbsz_buf_cb] + trainer.callbacks
        trainer.fit(model, datamodule)
