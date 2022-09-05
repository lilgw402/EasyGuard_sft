"""An customizable clip example"""
from base64 import b64decode
import sys
import os
import io
from typing import Union, List, Tuple
import math

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data._utils.collate import default_collate
from torch.optim.lr_scheduler import _LRScheduler
from torchvision import transforms as T
from transformers import AutoTokenizer
from titan import create_model, TOSHelper

try:
    import cruise
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from cruise import CruiseTrainer, CruiseModule, CruiseCLI
from cruise.data_module import CruiseDataModule
from cruise.utilities.hdfs_io import hlist_files


class ImageTextProcessor:
    def __init__(self, image_keys, text_fields, im_transform, tokenizer, context_length):
        self._image_keys = image_keys
        self._text_fields = text_fields
        self._tokenizer = tokenizer
        self._context_length = context_length
        self._im_transform = im_transform

    def transform(self, data_dict: dict):
        # get image by key, order matters
        for im_key in self._image_keys:
            image = data_dict.get(im_key, None)
            if image is not None:
                break
        else:
            raise KeyError(f"Unable to find image item by keys: {self._image_keys}, available keys: {data_dict.keys()}")
        texts = []
        for text_key in self._text_fields:
            texts.append(data_dict.get(text_key, ""))
        text = " ".join(texts)
        text_token = self._tokenizer(text, padding='max_length', max_length=self._context_length, truncation=True)
        # decode image
        image_str = b64decode(image)
        image = Image.open(io.BytesIO(image_str)).convert("RGB")
        image = self._im_transform(image)
        return {'image': image,
                'input_ids': torch.Tensor(text_token['input_ids']).long(),
                'attention_mask': torch.Tensor(text_token['attention_mask'])}

    def batch_transform(self, batch_data):
        # batch_data: List[Dict[modal, modal_value]]
        out_batch = {}
        for k in ('image', 'input_ids', 'attention_mask'):
            out_batch[k] = default_collate([data[k] for data in batch_data])
        return out_batch


class CLIPDataModule(CruiseDataModule):
    def __init__(self,
                 train_batch_size: int = 128,
                 val_batch_size: int = 64,
                 paths: Union[str, List[str]] = 'hdfs://haruna/home/byte_search_nlp_lq/user/fex/data/laion_tuchong_merge_en_dedup/part-*',
                 data_size: int = 600000000,
                 val_step: int = 500,
                 image_keys: Union[str, List[str]] = 'b64_binary',
                 text_fields: Union[str, List[str]] = 'text',
                 num_workers: int = 16,
                 im_size: int = 224,
                 resize_ratio: float = 0.75,
                 tokenizer: str = 'bert-base-uncased',
                 context_length: int = 77,
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
        self.train_files = files[:-1]
        self.val_files = files[-1]

        image_keys = self.hparams.image_keys
        if isinstance(image_keys, str):
            image_keys = [image_keys]
        self.image_keys = image_keys
        text_fields = self.hparams.text_fields
        if isinstance(text_fields, str):
            text_fields = [text_fields]
        self.text_fields = text_fields
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer)

    def train_dataloader(self):
        from cruise.data_module.cruise_loader import DistributedCruiseDataLoader
        return DistributedCruiseDataLoader(
            data_sources=[self.train_files],
            keys_or_columns=[None],
            batch_sizes=[self.hparams.train_batch_size],
            num_workers=self.hparams.num_workers,
            num_readers=[1],
            decode_fn_list=[[]],
            processor=ImageTextProcessor(
                self.image_keys,
                self.text_fields,
                T.Compose([
                    T.RandomResizedCrop(self.hparams.im_size,
                                        scale=(self.hparams.resize_ratio, 1.),
                                        ratio=(1., 1.)),
                    T.ToTensor(),
                    T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
                ]),
                self.tokenizer,
                self.hparams.context_length
            ),
            predefined_steps=self.hparams.data_size // self.hparams.train_batch_size // self.trainer.world_size,
            source_types=['jsonl'],
            shuffle=True,
        )

    def val_dataloader(self):
        from cruise.data_module.cruise_loader import DistributedCruiseDataLoader
        return DistributedCruiseDataLoader(
            data_sources=[self.train_files],
            keys_or_columns=[None],
            batch_sizes=[self.hparams.train_batch_size],
            num_workers=self.hparams.num_workers,
            num_readers=[1],
            decode_fn_list=[[]],
            processor=ImageTextProcessor(
                self.image_keys,
                self.text_fields,
                T.Compose([
                    T.Resize(int(self.hparams.im_size / 0.875)),
                    T.CenterCrop(self.hparams.im_size),
                    T.ToTensor(),
                    T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
                ]),
                self.tokenizer,
                self.hparams.context_length
            ),
            predefined_steps=self.hparams.val_step,
            source_types=['jsonl'],
            shuffle=False,
        )


class CLIPModel(CruiseModule):
    def __init__(self,
                 visual_model: str = 'resnet50', # resnet50, resnet101, swin-base-det/moe
                 language_model: str = 'bert',
                 language_model_version: str = 'bert-base-uncased',
                 embed_dim: int = 1024,
                 context_length: int = 77,
                 all_gather_limit: int = -1,
                 learning_rate: float = 5e-4,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.002,
                 titan_backend: str = 'titan',
                 titan_bucket: str = 'titan-modelzoo-public',
                 titan_access_key: str = 'BW7H90QZ6H7YR0U92WWM'
                 ):
        super().__init__()
        self.save_hparams()
        self.tos_helper = TOSHelper(bucket=titan_bucket, access_key=titan_access_key)

    def setup(self, stage) -> None:
        import titan
        assert titan.__version__ >= '0.0.5', 'bytedtitan>=0.0.5 is required'
        self.visual = create_model(
            model_name=self.hparams.visual_model, backend=self.hparams.titan_backend, pretrained=True,
            pretrained_version=None, features_only=False, num_classes=0, tos_helper=self.tos_helper)
        self.transformer = create_model(
            model_name=self.hparams.language_model, backend=self.hparams.titan_backend, pretrained=True,
            pretrained_version=self.hparams.language_model_version, features_only=False,
            tos_helper=self.tos_helper)
        text_width = self.transformer.last_out_channels
        self.context_length = self.hparams.context_length
        self.ln_final = LayerNorm(text_width)
        self.visual_projection = nn.Parameter(torch.empty(self.visual.last_out_channels, self.hparams.embed_dim))
        self.text_projection = nn.Parameter(torch.empty(text_width, self.hparams.embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.cross_entropy = torch.nn.CrossEntropyLoss(ignore_index=-1)

        self.initialize_parameters()

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

    def initialize_parameters(self):
        # we have loaded visual and lingural backbones from pretrained models so skip init self.visual/self.transformer
        if self.visual_projection is not None:
            nn.init.normal_(self.visual_projection, std=self.visual.last_out_channels ** -0.5)
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.last_out_channels ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def training_step(self, batch, idx):
        image, input_ids, attention_mask = batch['image'], batch['input_ids'], batch['attention_mask']
        image_features = self.encode_image(image)
        text_features = self.encode_text(input_ids, attention_mask)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # all gather to increase effective batch size
        if self.nce_group is not False:
            # [bsz, n] -> [group, bsz, n]
            group_image_features = self.all_gather(image_features, group=self.nce_group, sync_grads='rank')
            group_text_features = self.all_gather(text_features, group=self.nce_group, sync_grads='rank')
            # [group, bsz, n] -> [group * bsz, n]
            image_features = group_image_features.view((-1, image_features.shape[-1]))
            text_features = group_text_features.view((-1, text_features.shape[-1]))

        # cosine similarity as logits
        self.logit_scale.data.clamp_(-np.log(100), np.log(100))
        logit_scale = self.logit_scale.exp()
        self.log('logit_scale', logit_scale)
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        bsz = logits_per_image.shape[0]
        labels = torch.arange(bsz, device=logits_per_image.device)  # bsz

        loss_v = self.cross_entropy(logits_per_image, labels)
        loss_t = self.cross_entropy(logits_per_text, labels)
        loss = (loss_v + loss_t) / 2

        with torch.no_grad():
            acc_i = (torch.argmax(logits_per_image, 1) == labels.long()).sum().float() / labels.numel()
            acc_t = (torch.argmax(logits_per_text, 0) == labels.long()).sum().float() / labels.numel()
        self.log('acc_i', acc_i, console=True)
        self.log('acc_t', acc_t, console=True)

        self.log('loss_v', loss_v, console=True)
        self.log('loss_t', loss_t, console=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        # should better use better metrics in validation, currently only return loss to indicate whether it's overfitting
        image, input_ids, attention_mask = batch['image'], batch['input_ids'], batch['attention_mask']
        image_features = self.encode_image(image)
        text_features = self.encode_text(input_ids, attention_mask)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        bsz = image_features.shape[0]
        labels = torch.arange(bsz, device=logits_per_image.device)  # bsz

        loss_v = self.cross_entropy(logits_per_image, labels)
        loss_t = self.cross_entropy(logits_per_text, labels)
        loss = (loss_v + loss_t) / 2

        acc_i = (torch.argmax(logits_per_image, 1) == labels.long()).sum().float() / labels.numel()
        acc_t = (torch.argmax(logits_per_text, 0) == labels.long()).sum().float() / labels.numel()
        self.log('val_acc_i', acc_i)
        self.log('val_acc_t', acc_t)
        return {'val_loss': loss, 'val_acc_i': acc_i, 'val_acc_t': acc_t}

    def encode_image(self, image):
        return self.visual(image.type(self.dtype)) @ self.visual_projection

    def encode_text(self, text, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones(text.shape, device=text.device).long()
        x = self.transformer(text, attention_mask)
        x = self.ln_final(x).type(self.dtype)
        x = x @ self.text_projection
        return x

    def forward(self, image, text, attention_mask=None):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text, attention_mask=attention_mask)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=self.hparams.betas,
            eps=self.hparams.eps,
            weight_decay=self.hparams.weight_decay
        )

        # Source: https://github.com/openai/CLIP/issues/107
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.trainer.total_steps,
            cycle_mult=1.0,
            max_lr=self.hparams.learning_rate,
            min_lr=0,
            warmup_steps=2000
        )

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def trace_before_step(self, batch):
        # tracer don't like dict of input
        x = [batch['image'], batch['input_ids'], batch['attention_mask']]
        if x[2] is None:
            return x[:2]
        return x

    def trace_step(self, image, text, mask=None):
        return self(image, text, mask=mask)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


# https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup/blob/master/cosine_annealing_warmup/scheduler.py
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
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001,
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1
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

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


if __name__ == '__main__':
    cli = CruiseCLI(CLIPModel,
                    trainer_class=CruiseTrainer,
                    datamodule_class=CLIPDataModule,
                    trainer_defaults={
                        'max_epochs': 5,
                        'val_check_interval': [10000, 1.0],
                        'summarize_model_depth': 2,
                        'checkpoint_monitor': 'val_loss',
                        'checkpoint_mode': 'min'})
    cfg, trainer, model, datamodule = cli.parse_args()
    trainer.fit(model, datamodule)
