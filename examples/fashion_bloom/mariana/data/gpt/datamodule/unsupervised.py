"""Unsupervised datamodule for GPT pretraining"""
import logging
import os
import tempfile
import torch
from typing import Union, List
from cruise import CruiseDataModule
from cruise.data_module.tools import create_dataloader_by_cfg
from torch.utils.data._utils.collate import default_collate
from cruise.data_module import DistributedCruiseDataLoader
from cruise.data_module.gpu_wrapper import GPUPrefetcher
from cruise.utilities.hdfs_io import hlist_files, hcopy
from cruise.utilities import DIST_ENV
from ..tokenization import CasterTokenizer


class RawTextProcessor:
    r"""
        Args:
            tokenizer: the name of the pretrained tokenizer, e.g., "bigscience/bloom"
            text_keys: keys that contains text as values in the input.
            max_seq_len: max length that the model accept, if data is not enough,
                         pad_token_id will be used.
            drop_last: if text length is not divisible by max_seq_len, set this
                       field to False will pad the remainder.
    """
    def __init__(self, tokenizer: str, text_keys: Union[str, List[str]], max_seq_len:int, drop_last:bool = False, **kwargs):
        if not isinstance(text_keys, list):
            text_keys = [text_keys]
        self.text_keys = text_keys
        if not isinstance(tokenizer, str):
            # from created tokenizer object
            self.tokenizer = tokenizer
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_seq_len = max_seq_len
        self.drop_last = drop_last
        # We will automatically convert token list to tensor
        kwargs.pop('return_tensors', None)
        self.kwargs = kwargs

    def transform(self, data_dict):
        text_dict = {}
        for key in self.text_keys:
            text_output = self.tokenizer(data_dict[key], **self.kwargs)
            for k, v in text_output.items():
                if k not in text_dict:
                    text_dict[k] = [v]
                else:
                    text_dict[k].append(v)
        # append EOS token到末尾, 中途不加token，这个处理方法待定
        for k, v in text_dict.items():
            if 'mask' in k:
                eos_token_id = 1  # EOS mask is still 1
            else:
                eos_token_id = self.tokenizer.eos_token_id
            v[-1] += [eos_token_id]
        return self.group_texts(text_dict)

    def batch_transform(self, batch_data):
        return default_collate(batch_data)

    def group_texts(self, examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        mod_rst = total_length % self.max_seq_len
        total_length = (total_length // self.max_seq_len) * self.max_seq_len
        if mod_rst:
            if total_length < self.max_seq_len:
                for key in concatenated_examples:
                    if 'mask' in key:
                        pad_token_id = 0
                    else:
                        pad_token_id = self.tokenizer.pad_token_id
                    concatenated_examples[key] = concatenated_examples[key] + [pad_token_id] * (self.max_seq_len-total_length)
                total_length = self.max_seq_len
            elif not self.drop_last:
                for key in concatenated_examples:
                    concatenated_examples[key] = concatenated_examples[key][:total_length] + concatenated_examples[key][-self.max_seq_len:]
                total_length = total_length + self.max_seq_len

        # Split by chunks of max_len.
        outputs = []
        for i in range(0, total_length, self.max_seq_len):
            result = {
                k: torch.as_tensor(t[i : i + self.max_seq_len])
                for k, t in concatenated_examples.items()
            }
            # result["labels"] = result["input_ids"].clone()
            outputs.append(result)
        return outputs


class UnsupGPTDatamodule(CruiseDataModule):
    """GPT pretrain dataset module.

    It supports reading from raw text dataset and process using pretrained tokenizers.

    Data v1: hdfs://haruna/home/byte_data_aml_research/user/xuwei.yi/dataset/normal_split
    Data v1+: hdfs://haruna/home/byte_data_aml_research/user/xuwei.yi/dataset/dataset_v1
    Data v2: hdfs://haruna/home/byte_data_aml_research/user/corpus/sampled/
    """
    def __init__(self,
                 train_path: str = 'hdfs://haruna/home/byte_data_aml_research/user/corpus/sampled/',
                 val_path: str = '',
                 train_size: int = 1101492224,  # actually 1101493088, make it divisible by 4096 so less chance to stuck in spliting parquet files
                 train_batch_size: int = 32,
                 train_num_workers: int = 4,
                 val_batch_size: int = 32,
                 val_num_workers: int = 1,
                 max_seq_len: int = 1024,
                 text_keys: List[str] = ['content_split'],
                 tokenizer: str = 'hdfs://haruna/home/byte_data_aml_research/user/zhangzhi.joshua/tokenizer/zh_0620_newcut_caster_145665_lowercase',
                 gpu_prefetch: bool = False,
                 ):
        super().__init__()
        self.save_hparams()
        self.tokenizer = None

    def local_rank_zero_prepare(self) -> None:
        if self.hparams.tokenizer.startswith('hdfs'):
            # try download it to local once per node and load it in setup
            tmp_dir = os.path.join(tempfile.gettempdir(), os.path.basename(self.hparams.tokenizer))
            hcopy(self.hparams.tokenizer, tmp_dir)
        else:
            logging.info(f"Prefetching HF tokenizers {self.hparams.tokenizer} on local rank zero...")
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained(self.hparams.tokenizer)

    def setup(self):
        if self.hparams.tokenizer.startswith('hdfs'):
            # try download it to local once per node and load it in setup
            tmp_dir = os.path.join(tempfile.gettempdir(), os.path.basename(self.hparams.tokenizer))
            self.tokenizer = CasterTokenizer.from_pretrained(tmp_dir, max_len=-1)
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer, max_len=-1)

    def train_dataloader(self):
        train_steps = -1
        if self.hparams.train_size > 0:
            train_steps = self.hparams.train_size // (self.hparams.train_batch_size * DIST_ENV.world_size)
            assert train_steps > 0, f"train_size={self.hparams.train_size} may be too small to split to batch_size * world_size"
        train_files = [x for x in hlist_files([self.hparams.train_path]) if x.endswith('.parquet')]
        self.rank_zero_info(f"Fetched {len(train_files)} training files.")
        loader = DistributedCruiseDataLoader(
            data_sources=[train_files],
            batch_sizes=[self.hparams.train_batch_size],
            num_workers=self.hparams.train_num_workers,
            predefined_steps=train_steps,
            source_types=['parquet'],
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            parquet_cache_on=True,
            keys_or_columns=None,
            num_readers=[1],
            decode_fn_list=None,
            processor=RawTextProcessor(
                tokenizer=self.tokenizer if self.tokenizer is not None else self.hparams.tokenizer,
                text_keys=self.hparams.text_keys,
                max_seq_len=self.hparams.max_seq_len,
                drop_last=False),
            transform_output_many=True,
        )
        if self.hparams.gpu_prefetch:
            loader = GPUPrefetcher(loader)
        return loader

    def val_dataloader(self):
        if not self.hparams.val_path:
            return iter([])
        val_steps = -1
        val_files = [x for x in hlist_files([self.hparams.val_path]) if x.endswith('.parquet')]
        self.rank_zero_info(f"Fetched {len(val_files)} val files.")
        loader = DistributedCruiseDataLoader(
            data_sources=[val_files],
            batch_sizes=[self.hparams.val_batch_size],
            num_workers=self.hparams.val_num_workers,
            predefined_steps=val_steps,
            source_types=['parquet'],
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            parquet_cache_on=True,
            keys_or_columns=None,
            num_readers=[1],
            decode_fn_list=None,
            processor=RawTextProcessor(
                tokenizer=self.tokenizer if self.tokenizer is not None else self.hparams.tokenizer,
                text_keys=self.hparams.text_keys,
                max_seq_len=self.hparams.max_seq_len,
                drop_last=False),
            transform_output_many=True,
        )
        if self.hparams.gpu_prefetch:
            loader = GPUPrefetcher(loader)
        return loader

