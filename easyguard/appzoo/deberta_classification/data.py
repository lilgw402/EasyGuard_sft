# -*- coding: utf-8 -*-

import sys
import os
from typing import Tuple
import csv

import torch
import torch.nn as nn
from torch.utils.data._utils.collate import default_collate

from ptx.model import Model
from ptx.vocabulary import Vocabulary
from ptx.train.optimizer import Optimizer

from text_tokenizer import BpeTokenizer
from text_cutter import Cutter

try:
    import cruise
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from cruise import CruiseTrainer, CruiseModule, CruiseCLI
from cruise.data_module import CruiseDataModule
from cruise.data_module.cruise_loader import DistributedCruiseDataLoader
from cruise.utilities.hdfs_io import hlist_files


class TextProcessor:
    def __init__(self, text_label_field, text_field, tokenizer, vocab, cutter, max_sen_len):
        self._text_label_field = text_label_field
        self._text_field = text_field
        self._tokenizer = tokenizer
        self._vocab = vocab
        self._max_seq_len = max_sen_len
        self._cutter = cutter

    def transform(self, data_dict: dict):
        if not self._text_field in data_dict:
            raise KeyError(f"Unable to find text by keys: {self._text_field} available keys: {data_dict.keys()}")

        text = data_dict.get(self._text_field, "")
        tokens = []
        for word in self._cutter.cut(text):
            tokens.extend(self._tokenizer(word))
        if len(tokens) > self._max_seq_len - 2:
            tokens = tokens[:self._max_seq_len - 2]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        input_ids = self._vocab(tokens)
        attention_mask = [1] * len(input_ids)

        pad_id = self._vocab("[PAD]")
        padding_length = self._max_seq_len - len(input_ids)
        input_ids += [pad_id] * padding_length
        attention_mask += [0] * padding_length
        segment_ids += [0] * padding_length

        assert len(input_ids) == self._max_seq_len
        assert len(attention_mask) == self._max_seq_len
        assert len(segment_ids) == self._max_seq_len

        label = int(data_dict.get(self._text_label_field, ""))

        input_dict = {
            'input_ids': torch.Tensor(input_ids).long(),
            'attention_mask': torch.Tensor(attention_mask).long(),
            'segment_ids': torch.Tensor(segment_ids).long(),
            'labels': label
        }

        return input_dict

    def batch_transform(self, batch_data):
        # batch_data: List[Dict[modal, modal_value]]
        out_batch = {}
        keys = ('input_ids', 'attention_mask', 'segment_ids', 'labels')

        for k in keys:
            out_batch[k] = default_collate([data[k] for data in batch_data])

        return out_batch


class DebertaDataModule(CruiseDataModule):
    def __init__(self,
                 train_batch_size: int = 16,
                 val_batch_size: int = 32,
                 train_paths: str = 'hdfs://haruna/user/tianke/data/model_train_data/cruise_data/1_0/train',
                 val_paths: str = 'hdfs://haruna/user/tianke/data/model_train_data/cruise_data/1_0/predict',
                 data_size: int = 1200000,
                 val_data_size: int = 10000,
                 val_step: int = 100,
                 text_label_field: str = 'label',
                 text_field: str = 'text',
                 num_workers: int = 1,
                 pretrained_model_dir: str = 'hdfs://haruna/user/tianke/pretrain_models/ddp_subword_bs8x8_1696w',
                 local_dir_prefix='/opt/tiger/tianke',
                 cutter_resource_dir: str = "hdfs://haruna/user/tianke/tools/libcut_data_zh_20200827fix2",
                 max_sen_len: int = 512,
                 ):
        super().__init__()
        self.save_hparams()
        suffix_model = self.hparams.pretrained_model_dir.split("/")[-1]
        self.local_pretrained_model_dir = f"{self.hparams.local_dir_prefix}/{suffix_model}"

        suffix_cutter = self.hparams.cutter_resource_dir.split("/")[-1]
        self.local_cutter_dir = f"{self.hparams.local_dir_prefix}/{suffix_cutter}"

    def local_rank_zero_prepare(self) -> None:
        # download the tokenizer/pretrained_model once per node
        if not os.path.exists(self.hparams.local_dir_prefix):
            os.makedirs(self.hparams.local_dir_prefix, exist_ok=True)

        if not os.path.exists(self.local_pretrained_model_dir):
           os.system(f"hdfs dfs -copyToLocal {self.hparams.pretrained_model_dir} {self.hparams.local_dir_prefix}")

        # download cutter resource
        if not os.path.exists(self.local_cutter_dir):
            os.system(f"hdfs dfs -copyToLocal {self.hparams.cutter_resource_dir} {self.hparams.local_dir_prefix}")

    def setup(self, stage) -> None:
        self.train_files = hlist_files([self.hparams.train_paths])
        self.val_files = hlist_files([self.hparams.val_paths])

        self.tokenizer = BpeTokenizer(self.local_pretrained_model_dir + "/vocab.txt", wordpiece_type="bert",
                                      lower_case=False)
        self.vocab = Vocabulary(self.local_pretrained_model_dir + "/vocab.txt", oov_token='[UNK]')
        self.cutter = Cutter("CRF_LARGE", self.local_cutter_dir)

    def train_dataloader(self):
        return DistributedCruiseDataLoader(
            data_sources=[self.train_files],
            keys_or_columns=[None],
            batch_sizes=[self.hparams.train_batch_size],
            num_workers=self.hparams.num_workers,
            num_readers=[1],
            decode_fn_list=[[]],
            processor=TextProcessor(
                self.hparams.text_label_field,
                self.hparams.text_field,
                self.tokenizer,
                self.vocab,
                self.cutter,
                self.hparams.max_sen_len
            ),
            predefined_steps=self.hparams.data_size // self.hparams.val_batch_size // self.trainer.world_size,
            source_types=['jsonl'],
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
            processor=TextProcessor(
                self.hparams.text_label_field,
                self.hparams.text_field,
                self.tokenizer,
                self.vocab,
                self.cutter,
                self.hparams.max_sen_len
            ),
            predefined_steps=self.hparams.val_step,
            source_types=['jsonl'],
            shuffle=False,
        )

    def predict_dataloader(self):
        return DistributedCruiseDataLoader(
            data_sources=[self.val_files],
            keys_or_columns=[None],
            batch_sizes=[self.hparams.val_batch_size],
            num_workers=self.hparams.num_workers,
            num_readers=[1],
            decode_fn_list=[[]],
            processor=TextProcessor(
                self.hparams.text_label_field,
                self.hparams.text_field,
                self.tokenizer,
                self.vocab,
                self.cutter,
                self.hparams.max_sen_len
            ),
            predefined_steps=self.hparams.val_data_size // self.hparams.val_batch_size,
            source_types=['jsonl'],
            shuffle=False,
        )