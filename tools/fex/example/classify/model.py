""" 分类模型 """

from typing import Dict, List, Tuple

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from fex.optim import optimization
from fex.optim.lr_scheduler import WarmupMultiStepLR
from fex.optim.optimization import AdamW
from fex.core.net import Net
from fex.nn import FrameALBert


class MMClassifier(Net):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.class_num = self.config.CLASS_NUM

        self.backbone = FrameALBert(config)
        self.classifier = torch.nn.Linear(config.BERT.hidden_size, self.class_num)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.init_weights()
        self.freeze_params(self.config.TRAIN.freeze_prefix)

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

    def forward(self, **batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        这样写主要是为了在各种accelerators下, forward可以做完training_step和validation_step
        """
        if self.training:
            return self.training_step(**batch)
        else:
            return self.validation_step(**batch)

    def training_step(self, input_ids, input_segment_ids, input_mask,
                      label,
                      frames=None,
                      frames_mask=None,
                      visual_embeds=None):
        """
        input_ids: [bsz, seq_len]
        input_segment_ids: [bsz, seq_len]
        input_mask: [bsz, seq_len]
        label: [bsz],
        frames (optional): [bsz, frame_num, c, h, w]
        frames_mask (optional): [bsz, frame_num]
        visual_embeds (optional): [bsz, frame_num, visual_dim]

        frames 表示多帧的binary信息，visual_embeds 表示 多帧的 embedding信息，
        二者必须有一个是有的：
        如果传入frames，就会在训练过程中继续多帧的embedding，
        如果传入visual_embeds，就会直接使用visual_embeds（这种情况是提前从视觉服务取到多帧embedding了）

        frames_mask 是用来mask掉一些帧的。因为不是所有视频的帧数都一样。可以用这个字段干掉padding的帧。
        在dataset里会默认对不够帧数的视频pad一些全黑的帧，模型内计算时mask掉。
        如果不传frames_mask，默认所有帧都是valid的，都参与计算。
        """

        mmout = self.backbone(input_ids=input_ids,
                              input_segment_ids=input_segment_ids,
                              input_mask=input_mask,
                              frames=frames,
                              frames_mask=frames_mask,
                              visual_embeds=visual_embeds
                              )

        cls_emb = mmout['pooled_output']  # mmout 里还有 encoded_layers 等字段，可以支持一些fancy的玩法
        logits = self.classifier(cls_emb)  # [bsz, class_num]
        loss = self.criterion(logits, label)
        self.log('loss', loss)
        return {'loss': loss}

    @torch.no_grad()
    def validation_step(self, input_ids, input_segment_ids, input_mask,
                        label,
                        frames=None,
                        frames_mask=None,
                        visual_embeds=None):

        mmout = self.backbone(input_ids=input_ids,
                              input_segment_ids=input_segment_ids,
                              input_mask=input_mask,
                              frames=frames,
                              frames_mask=frames_mask,
                              visual_embeds=visual_embeds
                              )
        cls_emb = mmout['pooled_output']
        logits = self.classifier(cls_emb)
        res = self.cal_acc(logits, label=label, topk=(1, 5))
        return res

    @torch.no_grad()
    def validation_epoch_end(self, total_outputs: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        对validation的所有step的结果进行处理，最终输出为一个Dict[str, torch.Tensor]
        """
        res_out: Dict[str, torch.Tensor] = {}
        top1_acc = []
        top5_acc = []
        for item in total_outputs:
            top1_acc.append(item["top1_acc"].mean().item())
            top5_acc.append(item["top5_acc"].mean().item())

        top1_acc = torch.tensor(top1_acc).mean()
        top5_acc = torch.tensor(top5_acc).mean()
        res_out["top1_acc"] = top1_acc
        res_out["top5_acc"] = top5_acc
        # 将validation中top1_acc和top5_acc到tensorboard
        self.log_dict(res_out)
        return res_out

    def configure_optimizers(self) -> Tuple[Optimizer, LambdaLR]:
        """
        Model定制optimizer和lr_scheduler
        """
        no_decay = ['bias', 'bn', 'norm']
        no_dacay_params_dict = {'params': [], 'weight_decay': 0.0}
        low_lr_params_dict = {'params': [], 'weight_decay': self.config.TRAIN.WD, 'lr': self.config.TRAIN.LR * 0.1}
        normal_params_dict = {'params': [], 'weight_decay': self.config.TRAIN.WD}

        for n, p in self.named_parameters():
            if any(nd in n for nd in no_decay):
                no_dacay_params_dict['params'].append(p)
            elif n.startswith('albert'):
                low_lr_params_dict['params'].append(p)
            else:
                normal_params_dict['params'].append(p)
        optimizer_grouped_parameters = [no_dacay_params_dict, low_lr_params_dict, normal_params_dict]

        if self.config.TRAIN.OPTIM == 'SGD':
            optm = torch.optim.SGD(optimizer_grouped_parameters, self.config.TRAIN.LR,
                                   momentum=self.config.TRAIN.MOMENTUM,
                                   weight_decay=self.config.TRAIN.WD)
        elif self.config.TRAIN.OPTIM == 'AdamW':
            optm = AdamW(optimizer_grouped_parameters,
                         lr=self.config.TRAIN.LR,
                         betas=(0.9, 0.999),
                         eps=1e-6,
                         weight_decay=self.config.TRAIN.WD,
                         correct_bias=False
                         )

        if self.config.TRAIN.LR_SCHEDULE == 'step':
            lr_iters = [int(epoch * self.step_per_epoch) for epoch in self.config.TRAIN.LR_STEP]
            lr_scheduler = WarmupMultiStepLR(
                optimizer=optm,
                milestones=lr_iters,
                gamma=0.1,
                warmup_factor=self.config.TRAIN.WARMUP_FACTOR,
                warmup_iters=5 * self.step_per_epoch,
                warmup_method="linear",
            )
        elif self.config.TRAIN.LR_SCHEDULE == 'linear':
            lr_scheduler = optimization.get_linear_schedule_with_warmup(
                optimizer=optm,
                num_warmup_steps=5 * self.step_per_epoch,
                num_training_steps=self.total_step)
        elif self.config.TRAIN.LR_SCHEDULE == 'cosine':
            lr_scheduler = optimization.get_cosine_schedule_with_warmup(
                optimizer=optm,
                num_warmup_steps=5 * self.step_per_epoch,
                num_training_steps=self.total_step)
        elif self.config.TRAIN.LR_SCHEDULE == 'onecycle':
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optm,
                max_lr=self.config.TRAIN.LR,
                total_steps=self.total_step
            )

        return optm, lr_scheduler

    @torch.no_grad()
    def cal_acc(self, output: torch.Tensor, label: torch.Tensor, topk: Tuple[int] = (1,)) -> Dict[str, torch.Tensor]:
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
            res['top%s_acc' % k] = correct_k.mul_(100.0 / batch_size)
        return res
