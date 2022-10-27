import sys
import os
import math
import numpy as np

import torch
import torch.nn as nn
from ptx.model import Model

from sklearn import metrics
from transformers import get_polynomial_decay_schedule_with_warmup

try:
    import cruise
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from cruise import CruiseModule 
from cruise.utilities.distributed import DIST_ENV
from .deberta_config import DebertaConfig

from ...utils.losses import LearnableNTXentLoss, SCELoss

class FashionDebertaModel(CruiseModule):
    def __init__(self,
                 learning_rate: float = 2e-4,
                 weight_decay: float = 0.1,
                 warmup_ratio: float = 0.0,
                 pretrained_model_dir: str = 'hdfs://haruna/home/byte_ecom_govern/user/yangzheming/asr_model/zh_deberta_base_l6_emd_20210720',
                 local_pretrained_model_dir_prefix: str = '/opt/tiger/yangzheming/ckpt/',
                 cl_enable: bool = True,
                 cl_weight: float = 10.0,
                 ntx_enable: bool = True,
                 all_gather_limit: int = -1,
                 classification_task_enable: bool = False,
                 cls_weight: float = 1.0,
                 cls_class_num: int = 2,
                 sce_loss_enable: bool = False,     
                 cat_max_pool_enable: bool = False, # concat[cls, max_pooling] if true, for cls head
                 auc_score_enable: bool = False, # need to guarantee eval dataset contains all class label
                 ):
        super().__init__()
        self.save_hparams()
        suffix =self.hparams.pretrained_model_dir.strip("/").split("/")[-1]
        self.local_pretrained_model_dir = f"{self.hparams.local_pretrained_model_dir_prefix}/{suffix}"

    def local_rank_zero_prepare(self) -> None:
        # download the tokenizer once per node
        if not os.path.exists(self.local_pretrained_model_dir):
            os.makedirs(self.hparams.local_pretrained_model_dir_prefix, exist_ok=True)
            os.system(f"hdfs dfs -copyToLocal {self.hparams.pretrained_model_dir} {self.local_pretrained_model_dir}")      

    def setup(self, stage) -> None:
        self.deberta = Model.from_option('file:%s|strict=false' % (self.local_pretrained_model_dir))
        self.bert_config_path = os.path.join(self.local_pretrained_model_dir, "config.json")
        self.config = DebertaConfig.from_json_file(self.bert_config_path)

        self.mlm_prediction = DebertaLMPredictionHead(self.config)
        self.cls_prediction = Prediction(self.config.hidden_size * 2, self.hparams.cls_class_num) \
                                if self.hparams.cat_max_pool_enable else nn.Linear(self.config.hidden_size, self.hparams.cls_class_num)
        # self.initialize_parameters()
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.cls_loss_fct = SCELoss(alpha=1.0, beta=0.5, num_classes=self.hparams.cls_class_num) \
                                if self.hparams.sce_loss_enable else self.loss_fct

        # use contrast learning loss
        self.cl_enable = self.hparams.cl_enable
        if self.cl_enable:
            self.cl_weight = self.hparams.cl_weight  # weighted loss
            self.ntx_enable = self.hparams.ntx_enable
            if self.ntx_enable:
                self.ntx_loss_layer = LearnableNTXentLoss()
            else:
                self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

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
            # group_rank_id = self.trainer.global_rank // nce_limit
            # group_ranks = [group_rank_id * nce_limit + i for i in range(nce_limit)]
            # self.nce_group = torch.distributed.new_group(ranks=group_ranks, backend="nccl")
            # self.print("Create non-global allgather group from ranks:", group_ranks, "group size:", self.nce_group.size())

        self.train_return_loss_dict = dict()
        self.val_return_loss_dict = dict()
    
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

            loss_v = self.loss_fct(logits_per_z1, labels)
            loss_t = self.loss_fct(logits_per_z2, labels)
            loss = (loss_v + loss_t) / 2

        return loss

    def training_step(self, batch, idx):
        input_ids, input_masks, input_segment_ids, mlm_labels = batch['input_ids'], batch['input_masks'], batch['input_segment_ids'], batch['mlm_labels']
        output = self.deberta(input_ids=input_ids, segment_ids=input_segment_ids, attention_mask = input_masks, output_pooled=True)

        mlm_prediction_logits = self.mlm_prediction(output['sequence_output'])

        mlm_loss = self.loss_fct(mlm_prediction_logits.reshape(-1, self.config.vocab_size),
                                mlm_labels.reshape(-1))

        if self.hparams.classification_task_enable:
            classification_labels = batch['classification_labels']
            if self.hparams.cat_max_pool_enable:
               max_pooling_output = torch.max((output['sequence_output'] * input_masks.unsqueeze(-1)), axis=1).values # [bz, hidden]
               classification_pred_logits = self.cls_prediction(output['pooled_output'], max_pooling_output)   # [bz, cls_class_num]
            else:
                classification_pred_logits = self.cls_prediction(output['pooled_output'])
            # cls loss
            cls_loss = self.cls_loss_fct(classification_pred_logits.reshape(-1, self.hparams.cls_class_num),
                                classification_labels.reshape(-1))
            self.train_return_loss_dict["train_cls_loss"] = cls_loss
            self.train_return_loss_dict["loss"] = cls_loss * self.hparams.cls_weight + mlm_loss

        # TODO: support using cl and cls the same time
        elif self.hparams.cl_enable:
            cls_status = output['pooled_output']
            cl_loss = self.cl_loss(cls_status)
            self.train_return_loss_dict["train_cl_loss"] = cl_loss
            self.train_return_loss_dict["loss"] = cl_loss * self.hparams.cl_weight + mlm_loss

        # only mlm loss   
        else:
            self.train_return_loss_dict["loss"] = mlm_loss
        self.train_return_loss_dict["train_mlm_loss"] = mlm_loss

        return self.train_return_loss_dict

    def validation_step(self, batch, batch_idx):
        input_ids, input_masks, input_segment_ids, mlm_labels = batch['input_ids'], batch['input_masks'], batch['input_segment_ids'], batch['mlm_labels']
    
        output = self.deberta(input_ids=input_ids, segment_ids=input_segment_ids, attention_mask = input_masks, output_pooled=True)

        mlm_prediction_logits = self.mlm_prediction(output['sequence_output'])

        mlm_loss = self.loss_fct(mlm_prediction_logits.reshape(-1, self.config.vocab_size),
                                mlm_labels.reshape(-1))
        
        if self.hparams.classification_task_enable:
            classification_labels = batch['classification_labels']
            if self.hparams.cat_max_pool_enable:
               max_pooling_output = torch.max((output['sequence_output'] * input_masks.unsqueeze(-1)), axis=1).values # [bz, hidden]
               classification_pred_logits = self.cls_prediction(output['pooled_output'], max_pooling_output)   # [bz, cls_class_num]
            else:
                classification_pred_logits = self.cls_prediction(output['pooled_output'])
            # cls loss
            cls_loss = self.cls_loss_fct(classification_pred_logits.reshape(-1, self.hparams.cls_class_num),
                                classification_labels.reshape(-1))
            self.val_return_loss_dict["val_mlm_loss"] = mlm_loss
            self.val_return_loss_dict["val_cls_loss"] = cls_loss
            self.val_return_loss_dict["val_loss"] = cls_loss + mlm_loss
            # acc
            acc_result = accuracy(mlm_prediction_logits, mlm_labels, classification_pred_logits, classification_labels)
            mlm_acc, _, _, risk_acc, _, _ = acc_result
            self.val_return_loss_dict["mlm_acc"] = mlm_acc
            self.val_return_loss_dict["cls_acc"] = risk_acc
            # prepare for auc
            if self.hparams.auc_score_enable:
                ## multi class y_scores require (n_sample, n_class) shape
                if self.hparams.cls_class_num > 2:
                    pred_probs = nn.Softmax(-1)(classification_pred_logits)
                ## 2 class y_scores require (n_sample,) shape
                else:
                    pred_probs = nn.Softmax(-1)(classification_pred_logits)[:,-1]
                self.val_return_loss_dict["pred_probs"] = pred_probs.cpu().detach().numpy()
                self.val_return_loss_dict["classification_labels"] = classification_labels.cpu().detach().numpy()
        elif self.hparams.cl_enable:
           cls_status = output['pooled_output']
           cl_loss = self.cl_loss(cls_status)
           self.val_return_loss_dict["val_cl_loss"] = cl_loss
           self.val_return_loss_dict["val_mlm_loss"] = mlm_loss
           self.val_return_loss_dict["val_loss"] = cl_loss * self.hparams.cl_weight + mlm_loss
           acc_result = accuracy(mlm_prediction_logits, mlm_labels)
           mlm_acc, _, _ = acc_result
           self.val_return_loss_dict["mlm_acc"] = mlm_acc
        # only mlm loss   
        else:
            self.val_return_loss_dict["val_loss"] = mlm_loss
            self.val_return_loss_dict["val_mlm_loss"] = mlm_loss
            acc_result = accuracy(mlm_prediction_logits, mlm_labels)
            mlm_acc, _, _ = acc_result
            self.val_return_loss_dict["mlm_acc"] = mlm_acc
        self.val_return_loss_dict["cur_learning_rate"] = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        return self.val_return_loss_dict

    def validation_epoch_end(self, outputs) -> None:
        gathered_results = DIST_ENV.all_gather_object(outputs)
        all_results = []
        for item in gathered_results:
            all_results.extend(item)
        acc_all = [out["mlm_acc"] for out in all_results]
        total_acc = sum(acc_all) / len(acc_all)
        self.log("total_mlm_acc", total_acc, console=True)
        print("total_mlm_acc", total_acc)
        if self.hparams.classification_task_enable:
            # cls acc
            cls_acc_all = [out["cls_acc"] for out in all_results]
            total_cls_acc = sum(cls_acc_all) / len(cls_acc_all)
            self.log("total_cls_acc", total_cls_acc, console=True)
            print("total_cls_acc", total_cls_acc)
            # auc score
            if self.hparams.auc_score_enable:
                classification_labels_all = np.concatenate([out["classification_labels"] for out in all_results], axis=0) 
                pred_probs_all = np.concatenate([out["pred_probs"] for out in all_results], axis=0) 
                # multi class
                if self.hparams.cls_class_num > 2:
                    auc_score = metrics.roc_auc_score(classification_labels_all, pred_probs_all, multi_class = "ovr")
                else:
                    auc_score = metrics.roc_auc_score(classification_labels_all, pred_probs_all)
                self.log("total_auc_score", auc_score, console=True)
                print("total_auc_score", auc_score)

    def forward(self, input_ids, input_masks, input_segment_ids):
        output = self.deberta(input_ids=input_ids, segment_ids=input_segment_ids, attention_mask = input_masks, output_pooled=True)
        return output

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
                "initial_lr": self.hparams.learning_rate
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "initial_lr": self.hparams.learning_rate
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                        int(self.trainer.steps_per_epoch * self.hparams.warmup_ratio),
                                                        int(self.trainer.total_steps)
                                                        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def trace_before_step(self, batch):
        # tracer don't like dict of input
        x = [batch['input_ids'], batch['input_segment_ids'], batch['input_masks']]
        return x

    def trace_step(self, input_ids, input_segment_ids, input_masks):
        return self(input_ids, input_segment_ids, input_masks)

class DebertaLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = DebertaPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class DebertaPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = nn.GELU()

        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer_norm_eps = 1e-6 
        linear = nn.Linear(in_features, out_features)
        nn.init.normal_(linear.weight, std=math.sqrt(2. / in_features))
        nn.init.zeros_(linear.bias)
        self.model = nn.Sequential(
            linear,
            nn.GELU(),
            LayerNorm(out_features, eps=self.layer_norm_eps)
        )

    def forward(self, x):
        return self.model(x)

class Prediction(nn.Module):
    def __init__(self, inp_features, out_features):
        super().__init__()
        self.hidden_size = 768
        self.dense = nn.Sequential(
            Linear(inp_features, self.hidden_size),
            nn.Linear(self.hidden_size, out_features)
        )

    def forward(self, a, b):
        return self.dense(torch.cat([a, b], dim=-1))


def accuracy(mlm_logits, mlm_labels, risk_logits=None, risk_label=None, PAD_IDX=-100):
    """
    :param mlm_logits:  [src_len,batch_size,src_vocab_size]
    :param mlm_labels:  [src_len,batch_size]
    :param risk_logits:  [batch_size,2]
    :param risk_label:  [batch_size]
    :param PAD_IDX:
    :return:
    """
    mlm_pred = mlm_logits.argmax(axis=2).reshape(-1)

    mlm_true = mlm_labels.reshape(-1)

    mlm_acc = mlm_pred.eq(mlm_true)  # 计算预测值与正确值比较的情况
    mask = torch.logical_not(mlm_true.eq(PAD_IDX))  # 找到真实标签中，mask位置的信息。 mask位置为TRUE
    mlm_acc = mlm_acc.logical_and(mask)  # 去掉acc中非mask的部分
    mlm_correct = mlm_acc.sum().item()
    mlm_total = mask.sum().item()
    mlm_acc = float(mlm_correct) / mlm_total
    if risk_logits is not None and risk_label is not None:
        risk_correct = (risk_logits.argmax(1) == risk_label).float().sum()
        risk_total = len(risk_label)
        risk_acc = float(risk_correct) / risk_total
        return [mlm_acc, mlm_correct, mlm_total, risk_acc, risk_correct, risk_total]
    else:
        return [mlm_acc, mlm_correct, mlm_total]