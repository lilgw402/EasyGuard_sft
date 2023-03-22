# -*- coding: utf-8 -*-

from typing import Any, List, Union

import torch
from easyguard import AutoTokenizer
from torch.utils.data._utils.collate import default_collate

try:
    import cruise
except ImportError:
    print(
        "[ERROR] cruise is not installed! Please refer this doc: https://bytedance.feishu.cn/wiki/wikcnGP7yzZAuKpPfL6jRJKl2ag"
    )

from dataclasses import dataclass

from cruise.data_module import CruiseDataModule
from cruise.data_module.cruise_loader import DistributedCruiseDataLoader
from cruise.utilities.hdfs_io import hlist_files


@dataclass
class TextPreProcessor:
    x_key: str
    y_key: str
    region_key: str
    pre_tokenize: Any  # mlm_probability, cl_enable, cla_task_enable, category_key,
    max_len: int
    tokenizer: Any

    def transform(self, data_dict: dict):
        # == 文本 ==
        if not self.pre_tokenize:  # do not pre tokenize
            text = data_dict.get(self.x_key, " ")
            text_token = self.tokenizer(
                text,
                padding="max_length",
                max_length=self.max_len,
                truncation=True,
            )
            if "token_type_ids" not in self.tokenizer.model_input_names:
                text_token["token_type_ids"] = [0] * self.max_len
        else:
            text_token = data_dict[self.x_key]
            text_token[0] = self.tokenizer.cls_token
            text_token[-1] = self.tokenizer.sep_token
            text_token_ids = self.tokenizer.convert_tokens_to_ids(text_token)
            text_token["input_ids"] = text_token_ids
            text_token["attention_mask"] = [1] * len(
                text_token_ids[: self.max_len]
            ) + [0] * (self.max_len - len(text_token_ids))
            text_token["token_type_ids"] = [0] * self.max_len

        language = data_dict[self.region_key]

        input_dict = {
            "language": language,
            "input_ids": torch.Tensor(text_token["input_ids"]).long(),
            "attention_mask": torch.Tensor(text_token["attention_mask"]).long(),
            "token_type_ids": torch.Tensor(text_token["token_type_ids"]).long(),
        }

        # == 标签 ==
        label = int(data_dict[self.y_key])
        input_dict["labels"] = torch.tensor(label)

        return input_dict

    def batch_transform(self, batch_data):
        # batch_data: List[Dict[modal, modal_value]]
        out_batch = {}
        keys = (
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "labels",
            "language",
        )

        for k in keys:
            out_batch[k] = default_collate([data[k] for data in batch_data])

        return out_batch


class SequenceClassificationData(CruiseDataModule):
    def __init__(
        self,
        train_batch_size: int = 16,
        val_batch_size: int = 16,
        train_file: str = "hdfs://harunava/home/byte_magellan_va/user/jiangjunjun.happy/xtreme_ecom/ansa_train.json",
        val_file: str = "hdfs://harunava/home/byte_magellan_va/user/jiangjunjun.happy/xtreme_ecom/leaderboard/ansa_test.json",
        data_size: int = 200000000,
        val_step: int = 500,
        num_workers: int = 1,
        tokenizer: str = "fashionxlm-base",
        x_key: str = "text",
        y_key: str = "label",
        region_key: str = "country",
        pre_tokenize: bool = False,
        #  mlm_probability: float = 0.15,
        #  cl_enable: bool = False,
        #  cla_task_enable: bool = False,
        max_len: int = 256,
    ):
        super().__init__()
        self.save_hparams()

    def local_rank_zero_prepare(self) -> None:
        # download the tokenizer once per node
        AutoTokenizer.from_pretrained(self.hparams.tokenizer)

    def setup(self, stage) -> None:
        self.train_files = [self.hparams.train_file]
        self.val_files = [self.hparams.train_file]

        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer)

    def train_dataloader(self):
        return DistributedCruiseDataLoader(
            data_sources=[self.train_files],
            keys_or_columns=[None],
            batch_sizes=[self.hparams.train_batch_size],
            num_workers=self.hparams.num_workers,
            num_readers=[1],
            decode_fn_list=[[]],
            processor=TextPreProcessor(
                self.hparams.x_key,
                self.hparams.y_key,
                self.hparams.region_key,
                self.hparams.pre_tokenize,
                # self.hparams.mlm_probability, self.hparams.cl_enable, self.hparams.cla_task_enable, self.hparams.category_key,
                self.hparams.max_len,
                self.tokenizer,
            ),
            predefined_steps=self.hparams.data_size
            // self.hparams.train_batch_size
            // self.trainer.world_size,
            source_types=["jsonl"],
            shuffle=True,
        )

    def val_dataloader(self):
        return DistributedCruiseDataLoader(
            data_sources=[self.val_files],
            keys_or_columns=[None],
            batch_sizes=[self.hparams.val_batch_size],
            num_workers=self.hparams.num_workers,
            num_readers=[1],
            decode_fn_list=[[]],
            processor=TextPreProcessor(
                self.hparams.x_key,
                self.hparams.y_key,
                self.hparams.region_key,
                self.hparams.pre_tokenize,
                # self.hparams.mlm_probability, self.hparams.cl_enable, self.hparams.cla_task_enable, self.hparams.category_key,
                self.hparams.max_len,
                self.tokenizer,
            ),
            predefined_steps=self.hparams.val_step,
            source_types=["jsonl"],
            shuffle=False,
        )
