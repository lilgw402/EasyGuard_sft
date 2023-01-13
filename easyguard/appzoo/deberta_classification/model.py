# -*- coding: utf-8 -*-

import csv
import os
import sys
from typing import Tuple

import torch
import torch.nn as nn
from ptx.model import Model
from ptx.train.optimizer import Optimizer

try:
    import cruise
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from cruise import CruiseCLI, CruiseModule, CruiseTrainer


class DebertaModel(CruiseModule):
    def __init__(
        self,
        max_seq_len: int = 512,
        learning_rate: float = 2e-5,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        pretrained_model_dir="hdfs://haruna/user/tianke/pretrain_models/ddp_subword_bs8x8_1696w",
        local_dir_prefix="/opt/tiger/tianke",
    ):
        super().__init__()
        self.save_hparams()
        suffix = self.hparams.pretrained_model_dir.split("/")[-1]
        self.local_pretrained_model_dir = (
            f"{self.hparams.local_dir_prefix}/{suffix}"
        )

        if not os.path.exists(self.hparams.local_dir_prefix):
            os.makedirs(self.hparams.local_dir_prefix, exist_ok=True)

        if not os.path.exists(self.local_pretrained_model_dir):
            os.system(
                f"hdfs dfs -copyToLocal {self.hparams.pretrained_model_dir} {self.hparams.local_dir_prefix}"
            )

        self.deberta = Model.from_option(
            "file:%s|strict=false" % (self.local_pretrained_model_dir)
        )
        self.dropout = torch.nn.Dropout(0.1)
        self.num_labels = 2
        self.classifier = torch.nn.Linear(
            in_features=self.deberta.encoder.dim, out_features=self.num_labels
        )

        self.loss = nn.CrossEntropyLoss()
        # self.freeze_parameters()

    def local_rank_zero_prepare(self) -> None:
        # download the tokenizer once per node
        # if not os.path.exists(self.local_pretrained_model_dir):
        #     os.makedirs(self.hparams.local_pretrained_model_dir_prefix, exist_ok=True)
        #     os.system(f"hdfs dfs -copyToLocal {self.hparams.pretrained_model_dir} {self.local_pretrained_model_dir}")
        self.csv_writer = csv.writer(
            open(self.hparams.local_dir_prefix + "/cruise_predict.csv", "w")
        )

    def forward(self, input_ids, attention_mask, segment_ids, labels=None):
        """
        input_ids: [bsz, seq_len]
        input_segment_ids: [bsz, seq_len]
        attention_mask: [bsz, seq_len]
        labels: [bsz]
        """
        pooled_output = self.deberta(
            input_ids,
            segment_ids=segment_ids,
            attention_mask=attention_mask,
            output_pooled=True,
        )["pooled_output"]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss = self.loss(logits.view(-1, self.num_labels), labels.view(-1))
            return (logits, loss)
        else:
            return logits

    def training_step(self, batch, idx):
        output_dict = {}
        input_ids, attention_mask, segment_ids, labels = (
            batch["input_ids"],
            batch["attention_mask"],
            batch["segment_ids"],
            batch["labels"],
        )
        logits, loss = self(input_ids, attention_mask, segment_ids, labels)
        output_dict["loss"] = loss

        return output_dict

    def validation_step(self, batch, idx):
        output_dict = {}
        input_ids, attention_mask, segment_ids, labels = (
            batch["input_ids"],
            batch["attention_mask"],
            batch["segment_ids"],
            batch["labels"],
        )
        logits, loss = self(input_ids, attention_mask, segment_ids, labels)
        pred_labels = torch.argmax(logits, dim=1)
        acc = torch.sum(pred_labels == labels).item() / len(pred_labels)

        output_dict["loss"] = loss
        output_dict["acc"] = acc

        return output_dict

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(
        #     filter(lambda p: p.requires_grad, model.parameters()),
        #     lr=self.hparams.learning_rate,
        #     betas=self.hparams.betas,
        #     eps=self.hparams.eps,
        #     weight_decay=self.hparams.weight_decay
        # )
        optimizer = Optimizer.from_option(
            list(self.named_parameters()),
            {"type": "adam", "lr": self.hparams.learning_rate},
        )

        return {"optimizer": optimizer}

    def freeze_parameters(self, num_layers_frozen=6):
        params_exclusive = []
        if num_layers_frozen:
            params_exclusive = set(params_exclusive)
            # for i in range(num_layers_frozen):
            params_exclusive.add("deberta")
            params_exclusive = list(params_exclusive)

        for name, param in self.named_parameters():
            if any(
                [name.startswith(pop_name) for pop_name in params_exclusive]
            ):
                param.requires_grad = False

    def predict_step(self, batch, idx):
        print(idx, batch["input_ids"].size()[0])
        output_dict = {}
        input_ids, attention_mask, segment_ids, labels = (
            batch["input_ids"],
            batch["attention_mask"],
            batch["segment_ids"],
            batch["labels"],
        )
        logits, loss = self(input_ids, attention_mask, segment_ids, labels)
        softmax = torch.nn.Softmax(dim=1)
        softmax_score = softmax(logits)
        scores = softmax_score[:, 1:].squeeze()

        pred_labels = torch.argmax(logits, dim=1)
        acc = torch.sum(pred_labels == labels).item() / len(pred_labels)

        for label, predict, score in zip(
            labels.cpu().numpy().tolist(),
            pred_labels.cpu().numpy().tolist(),
            scores.cpu().numpy().tolist(),
        ):
            self.csv_writer.writerow([label, predict, score])
            # print('label:{} predict:{} score:{}'.format(label, predict, score)

        # output_dict['score'] = scores.cpu().numpy().tolist()
        # output_dict['acc'] = acc

        return

    def trace_before_step(self, batch):
        # tracer don't like dict of input
        input_ids, attention_mask, segment_ids, labels = (
            batch["input_ids"],
            batch["attention_mask"],
            batch["segment_ids"],
            batch["labels"],
        )
        return (input_ids, attention_mask, segment_ids)

    def trace_step(self, input_ids, attention_mask, segment_ids):
        return self(input_ids, attention_mask, segment_ids)

    def on_fit_start(self) -> None:
        self.rank_zero_print(
            "===========My custom fit start function is called!============"
        )
