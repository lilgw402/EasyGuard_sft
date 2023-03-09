from typing import List, Optional, Dict
import os
import torch
import torch.nn.functional as F

from cruise import CruiseModule, CruiseCLI, CruiseConfig
from mariana.data.fex.datamodule.dy_cover import DyCoverDataModule
from mariana.data.fex.benchmark import SUPPORTED_BENCHMARKS
from mariana.utils.exp_helper import ExpHelper
from mariana.models.swin_transformer import SwinTransformer
from mariana.models.albert import ALBert
from mariana.loss.ntxent import LearnableNTXentLoss
from mariana.optim.optimizer import AdamW  # diff to pytorch native AdamW?
from mariana.optim.lr_scheduler import get_linear_schedule_with_warmup  # use compose?


# Config adapter for ALBert
albert_default_config = {
  'embedding_size': 256,
  'frozen_layers': -1,
  'hidden_dropout_prob': 0.1,
  'hidden_size': 768,
  'initializer_range': 0.02,
  'intermediate_size': 3072,
  'layernorm_eps': 1.0e-06,
  'max_position_embeddings': 512,
  'num_attention_heads': 12,
  'num_hidden_layers': 6,
  'project_embedding_first': False,
  'type_vocab_size': 16,
  'vocab_size': 145608,
  'with_pooler': True,
  'word_embedding_frozen': False,
}

class CLIPDyCoverSwinAlbertTower(CruiseModule):
    """Swin + Albert 双塔"""
    def __init__(self,
                 bert: CruiseConfig = albert_default_config,
                 large_proj: bool = False,
                 # Visual
                 visual_model: str = 'swin',
                 # Swin
                 swin_depths: List[int] = [2, 2, 18, 2],
                 swin_num_heads: List[int] = [6, 12, 24, 48],
                 swin_embed_dim: int = 192,
                 # NCE
                 tau: float = 0.07,
                 nce_clamp: float = 4.6051,
                 gpuwise_nce: bool = True,
                 freeze_prefix: Optional[List[str]] = None,
                 # Optimizer
                 bert_lr_decay_rate: float = 1.0,
                 emb_loss_weight: float = 4.0,
                 emb_loss_warmup_step: int = 5000,
                 with_cosine_loss: bool = False,
                 cos_loss_margin: float = 0.1,
                 with_nce_loss: bool = True,
                 need_norm: bool = True,
                 bert_embed: str = 'cls',
                 partial_pretrain: Optional[List[str]] = ['hdfs://haruna/home/byte_data_aml_research/user/ding.zhou/s1_large/swin_large_albert_6l_lr2_5e4_eps50_bs160_g128/model_state_epoch_1500000.th'],
                 partial_pretrain_rename: Optional[Dict[str, str]] = None,
                 ):
        super().__init__()
        self.save_hparams()  # save to self.hparams
        # 文本
        self.albert = ALBert(self.hparams.bert)
        if large_proj:
            self.projector = torch.nn.Sequential(
                torch.nn.Linear(self.hparams.bert.hidden_size, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 1024)
            )
        else:
            self.projector = torch.nn.Linear(self.hparams.bert.hidden_size, 128)

        # 视觉
        visual_output_dim = 512
        if visual_model.lower() == 'swin':
            self.resnet = SwinTransformer(num_classes=visual_output_dim,
                                          embed_dim=swin_embed_dim,
                                          depths=swin_depths,
                                          num_heads=swin_num_heads)
        else:
            raise ValueError(f"unknown visual model: {visual_model}")
        total_visual_params = sum(
            [v.numel() for v in self.resnet.parameters()])
        print(f'{visual_model} Params: {total_visual_params}')

        if large_proj:
            self.fc128 = torch.nn.Sequential(
                torch.nn.Linear(512, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 1024),
                torch.nn.ReLU(),
                torch.nn.Linear(1024, 1024)
            )
        else:
            self.fc128 = torch.nn.Linear(512, 128, bias=False)

        self.calc_ce = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.calc_cosine = torch.nn.CosineSimilarity(eps=1e-4)
        self.calc_nce_loss = LearnableNTXentLoss(
            init_tau=tau, clamp=nce_clamp)

        self.init_weights()
        self.freeze_params(freeze_prefix or [])

    def setup(self):
        # In DDP rank 0 load pretrain weights is enough
        if self.trainer.global_rank == 0 and self.hparams.partial_pretrain:
            rename_params = self.hparams.partial_pretrain_rename or {}
            self.partial_load_from_checkpoints(
                self.hparams.partial_pretrain,
                rename_params=rename_params, verbose=True)

    def init_weights(self):
        # https://github.com/google-research/vision_transformer/issues/34
        # https://github.com/google-research/vision_transformer/blob/4317e064a0a54b825b5b9ff634482954788b8d84/vit_jax/models.py#L129
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
                    print('freeze_params:', name)
                    param.requires_grad = False

    def forward(self, image, t_emb=None, input_ids=None, input_segment_ids=None, input_mask=None):
        # 1. visual_embed
        output = self.resnet(image)
        v_emb = self.fc128(output)

        # 2. text embedding
        if t_emb is None:
            tout = self.albert(
                input_ids=input_ids,
                input_segment_ids=input_segment_ids,
                input_mask=input_mask)
            if self.hparams.bert_embed == 'cls':
                cls_emb = tout['pooled_output']
                t_emb = self.projector(cls_emb)
            elif self.hparams.bert_embed == 'avg':
                tokens_emb = tout['encoded_layers'][-1]
                tokens_emb = tokens_emb * input_mask.unsqueeze(-1)
                tokens_emb = tokens_emb.sum(1)
                tokens_emb = tokens_emb / input_mask.sum(1, keepdim=True)
                t_emb = self.projector(tokens_emb)
            elif self.hparams.bert_embed == 'cls+avg':
                cls_emb = tout['pooled_output']
                tokens_emb = tout['encoded_layers'][-1]
                tokens_emb = tokens_emb * input_mask.unsqueeze(-1)
                tokens_emb = tokens_emb.sum(1)
                tokens_emb = tokens_emb / input_mask.sum(1, keepdim=True)
                t_emb = tokens_emb + cls_emb
                t_emb = self.projector(t_emb)
            else:
                raise ValueError(
                    'Not Support bert_embed: {}, only [`cls`, `avg`, `cls+avg`]'.format(self.hparams.bert_embed))
        return v_emb, t_emb

    def training_step(self, batch, batch_idx):
        image = batch['image']
        input_ids = batch['input_ids']
        input_segment_ids = batch['input_segment_ids']
        input_mask = batch['input_mask']
        # log lr
        scheduler = self.trainer.lr_scheduler_configs[0].scheduler
        if hasattr(scheduler, 'get_lr'):
            self.log('lr', scheduler.get_lr()[0], console=True)
        else:
            self.log('lr', scheduler.get_last_lr()[0], console=True)
        # 1&2
        v_emb, t_emb = self(image, input_ids=input_ids, input_segment_ids=input_segment_ids, input_mask=input_mask)

        # 3. cosine embedding loss
        emb_loss = 0.
        if self.hparams.with_cosine_loss:
            sim_score = self.calc_cosine(v_emb, t_emb)
            emb_loss = torch.clamp(1 - sim_score, min=self.hparams.cos_loss_margin, max=1)
            if self.hparams.emb_loss_warmup_step > 0:
                emb_warmup_weight = min(self.trainer.global_step / self.hparams.emb_loss_warmup_step, 1)
            else:
                emb_warmup_weight = 1.
            self.log('emb_loss', emb_loss.mean().float(), console=True)
            self.log('sims', sim_score.mean().float())
            emb_loss = emb_loss * self.hparams.emb_loss_weight * emb_warmup_weight
            emb_loss = emb_loss.mean()

        # 4. nce loss
        # 可能有点buggy，没有考虑同label的情况，后面fix看看
        nce_loss = 0.
        if self.hparams.with_nce_loss:
            self.log('before_nce_temperature', self.calc_nce_loss.tau.float())
            if self.hparams.need_norm:
                t_emb = F.normalize(t_emb, dim=1)
                v_emb = F.normalize(v_emb, dim=1)
            if self.hparams.gpuwise_nce:
                # [bsz, n] -> [group, bsz, n]
                t_emb = self.all_gather(t_emb, sync_grads='rank')
                v_emb = self.all_gather(v_emb, sync_grads='rank')
                # [group, bsz, n] -> [group * bsz, n]
                t_emb = t_emb.view((-1, t_emb.shape[-1]))
                v_emb = v_emb.view((-1, v_emb.shape[-1]))
            nce_loss = self.calc_nce_loss(v_emb=v_emb, t_emb=t_emb)
            self.log('nce_loss', nce_loss.float(), console=True)
            self.log('nce_temperature', self.calc_nce_loss.tau.float())

        # final. merge loss
        loss = emb_loss + nce_loss
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
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
            bench = SUPPORTED_BENCHMARKS[benchmark](**batch)
            result = bench.run()  # can be empty if not rank0, but it's okay
        return result

    @torch.no_grad()
    def encode(
            self,
            image=None,
            input_ids=None,
            input_segment_ids=None,
            input_mask=None,
            mode='t',
            *args,
            **kwargs):
        """Used for benchmark."""
        if mode == 't':
            tout = self.albert(
                input_ids=input_ids,
                input_segment_ids=input_segment_ids,
                input_mask=input_mask)
            if self.hparams.bert_embed == 'cls':
                cls_emb = tout['pooled_output']
                t_emb = self.projector(cls_emb)
            elif self.hparams.bert_embed == 'avg':
                tokens_emb = tout['encoded_layers'][-1]
                tokens_emb = tokens_emb * input_mask.unsqueeze(-1)
                tokens_emb = tokens_emb.sum(1)
                tokens_emb = tokens_emb / input_mask.sum(1, keepdim=True)
                t_emb = self.projector(tokens_emb)
            elif self.hparams.bert_embed == 'cls+avg':
                cls_emb = tout['pooled_output']
                tokens_emb = tout['encoded_layers'][-1]
                tokens_emb = tokens_emb * input_mask.unsqueeze(-1)
                tokens_emb = tokens_emb.sum(1)
                tokens_emb = tokens_emb / input_mask.sum(1, keepdim=True)
                t_emb = tokens_emb + cls_emb
                t_emb = self.projector(t_emb)
            else:
                raise ValueError(
                    'Not Support bert_embed: {}, only [`cls`, `avg`, `cls+avg`]'.format(self.hparams.bert_embed))
            return {'pooled_out': t_emb}
        elif mode == 'v':
            if len(image.shape) == 4:
                encoder_output = self.resnet(image)
                v_emb = self.fc128(encoder_output)
            elif len(image.shape) == 5:
                # frame的
                bsz, fnum, c, h, w = image.shape
                image = image.reshape([bsz * fnum, c, h, w])
                encoder_output = self.resnet(image)
                v_emb = self.fc128(encoder_output)
                v_emb = v_emb.reshape([bsz, fnum, -1])
            else:
                raise ValueError(f"Invalid image shape: {image.shape}")
            return {'pooled_out': v_emb, 'visual_emb': encoder_output}

    def configure_optimizers(self, optimizer_kwargs):
        """
        Model定制optimizer和lr_scheduler
        """
        no_decay = ['bias', 'bn', 'norm', 'ln']
        no_dacay_params_dict = {'params': [], 'weight_decay': 0.0}
        low_lr_params_dict = {
            'params': [],
            'weight_decay': optimizer_kwargs["optimizer"]["params"]["weight_decay"],
            'lr': optimizer_kwargs["optimizer"]["params"]["lr"] * self.hparams.bert_lr_decay_rate}
        normal_params_dict = {'params': [], 'weight_decay': optimizer_kwargs["optimizer"]["params"]["weight_decay"]}

        for n, p in self.named_parameters():
            if any(nd in n for nd in no_decay):
                no_dacay_params_dict['params'].append(p)
            elif n.startswith('albert'):
                low_lr_params_dict['params'].append(p)
            else:
                normal_params_dict['params'].append(p)
        optimizer_grouped_parameters = [
            no_dacay_params_dict,
            low_lr_params_dict,
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
        CLIPDyCoverSwinAlbertTower,
        datamodule_class=DyCoverDataModule,
        trainer_defaults={
            'precision': 16,
            'enable_versions': False,
            'find_unused_parameters': False,
            'max_epochs': 25,
            "default_hdfs_dir": helper.hdfs_prefix,
            "project_name": helper.project_name,
            'val_check_interval': [20000, 1.0],
            'summarize_model_depth': 2,
            'gradient_clip_val': 1.0,
            'checkpoint_monitor': 'step',
            'checkpoint_mode': 'max',
            'callbacks': [ckpter]})
    cli.add_argument('--val-only', default=False, action='store_true', dest='val_only')
    cli.add_argument('--zero3', default=False, action='store_true', dest='zero3')

    cfg, trainer, model, datamodule = cli.parse_args()
    if cfg.val_only:
        trainer.validate(model, datamodule=datamodule)
    else:
        trainer.fit(model, datamodule)
