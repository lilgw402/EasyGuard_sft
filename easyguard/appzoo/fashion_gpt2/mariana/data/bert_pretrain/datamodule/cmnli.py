"""NLI finetune datamodule"""
from typing import List, Dict, Any

import torch
from cruise import CruiseDataModule
from cruise.data_module import DistributedCruiseDataLoader
from cruise.data_module.gpu_wrapper import GPUPrefetcher
from cruise.utilities.hdfs_io import hlist_files
from cruise.utilities import DIST_ENV
from ...fex.tokenization import BertTokenizer


class RawNLIProcessor:
    def __init__(self, vocab_file, batch_size, max_seq_len=128, padding_index: int = 2):
        self._pad_id = padding_index
        # bsz is required to ensure zero3 mode all batches have same shape
        self._bsz = batch_size
        self.tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=False)
        self.vocab = self.tokenizer.vocab
        self.label_mapping = {"neutral": 0, "contradiction": 1, "entailment": 2}
        self.cls_id = self.vocab['[CLS]']
        self.pad_id = self.vocab['[PAD]']
        self.sep_id = self.vocab['[SEP]']
        self.max_a_len = int(max_seq_len * 0.5)
        self.max_b_len = max_seq_len
        self.max_len = max_seq_len

    def transform(self, data_dict):
        label_id = self.label_mapping[data_dict["label"]]
        input_a = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(data_dict["sentence1"]))
        input_b = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(data_dict["sentence2"]))
        # concat bert pair
        input_a = input_a[:self.max_a_len - 2]
        input_b = input_b[:self.max_b_len - 1]
        input_a = [self.cls_id] + input_a + [self.sep_id]
        input_a_segment = [0] * len(input_a)
        input_b = input_b + [self.sep_id]
        input_b_segment = [1] * len(input_b)
        input_ab = input_a + input_b
        # rm last input id
        input_ab = input_ab[:self.max_len - 1]
        # pad with sep_id
        input_ab = input_ab + [self.sep_id]
        input_ab_segment = input_a_segment + input_b_segment
        input_ab_segment = input_ab_segment[:self.max_len]
        input_ab_mask = [1] * len(input_ab_segment)
        if len(input_ab_segment) < self.max_len:
            input_ab_segment += [0] * (self.max_len - len(input_ab_segment))
            input_ab_mask += [0] * (self.max_len - len(input_ab_mask))
        if len(input_ab) < self.max_len:
            input_ab += [self.sep_id] * (self.max_len - len(input_ab))
        assert len(input_ab_segment) == self.max_len, f"invalid segment_id length: {len(input_ab_segment)}"
        assert len(input_ab) == self.max_len, f"invalid input_id length: {len(input_ab)}"
        assert len(input_ab_mask) == self.max_len, f"invalid input_mask length: {len(input_ab_mask)}"

        input_ab_mask = torch.tensor(input_ab_mask).long()
        pos_ids = list(range(len(input_ab)))

        instance_data = {
            'input_ids': torch.tensor(input_ab).long(),
            'input_mask': torch.tensor(input_ab_mask).long(),
            'segment_ids': torch.tensor(input_ab_segment).long(),
            'position_ids': torch.tensor(pos_ids).long(),
            'labels': torch.tensor(int(label_id)).long(),
        }
        return instance_data

    def batch_transform(self, batch_data):
        # stack tensors in dicts
        result_data = {}
        if len(batch_data) < self._bsz:
            offset = self._bsz - len(batch_data)
            result_data['_num_valid_samples'] = len(batch_data)
            batch_data += [batch_data[0]] * offset
        else:
            result_data['_num_valid_samples'] = self._bsz

        for key in batch_data[0].keys():
            result_data[key] = torch.stack([dd[key] for dd in batch_data], dim=0)
        return result_data


class NLIProcessor:
    def __init__(self, batch_size, padding_index: int = 2):
        self._pad_id = padding_index
        # bsz is required to ensure zero3 mode all batches have same shape
        self._bsz = batch_size

    def transform(self, data_dict):
        input_ab = data_dict['input_ids']
        input_ab_mask = data_dict['input_mask']
        label_id = data_dict['label']
        input_ab_segment = data_dict['segment_ids']
        pos_ids = list(range(len(input_ab)))

        instance_data = {
            'input_ids': torch.tensor(input_ab).long(),
            'input_mask': torch.tensor(input_ab_mask).long(),
            'segment_ids': torch.tensor(input_ab_segment).long(),
            'position_ids': torch.tensor(pos_ids).long(),
            'labels': torch.tensor(int(label_id)).long(),
        }
        return instance_data

    def batch_transform(self, batch_data):
        # stack tensors in dicts
        result_data = {}
        if len(batch_data) < self._bsz:
            offset = self._bsz - len(batch_data)
            result_data['_num_valid_samples'] = len(batch_data)
            batch_data += [batch_data[0]] * offset
        else:
            result_data['_num_valid_samples'] = self._bsz

        for key in batch_data[0].keys():
            result_data[key] = torch.stack([dd[key] for dd in batch_data], dim=0)
        return result_data


class NLIFinetuneDatamodule(CruiseDataModule):
    """Demo LM pretrain dataset module.

    The trainset creates the a dataloader from the previously dumped chinese pretrain
    data on hdfs://haruna/home/byte_search_nlp_lq/user/zhaoshenjian.01/data/bert/zh_sdr_qa_site_web_20200518/*.tfrecord
    It was converted to parquet files to get rid of tensorflow dependency.

    TODO(Zhi): add validation.
    """
    def __init__(self,
                 train_path: str = 'hdfs://haruna/home/byte_data_aml_research/user/zhangzhi.joshua/data/cmnli/train_oldcut.parquet',
                 val_path: str = 'hdfs://haruna/home/byte_data_aml_research/user/zhangzhi.joshua/data/cmnli/dev_oldcut.parquet',
                 train_size: int = -1,
                 train_batch_size: int = 64,
                 train_num_workers: int = 4,
                 val_batch_size: int = 32,
                 val_num_workers: int = 2,
                 gpu_prefetch: bool = False,
                 padding_index: int = 2,
                 ):
        super().__init__()
        self.save_hparams()

    def train_dataloader(self):
        train_steps = -1
        if self.hparams.train_size > 0:
            train_steps = self.hparams.train_size // (self.hparams.train_batch_size * DIST_ENV.world_size)
            assert train_steps > 0, f"train_size={self.hparams.train_size} may be too small to split to batch_size * world_size"
        train_files = [x for x in hlist_files([self.hparams.train_path]) if x.endswith('parquet')]
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
            processor=NLIProcessor(
                batch_size=self.hparams.train_batch_size,
                padding_index=self.hparams.padding_index),
        )
        if self.hparams.gpu_prefetch:
            loader = GPUPrefetcher(loader)
        return loader

    def val_dataloader(self):
        val_steps = -1
        val_files = [x for x in hlist_files([self.hparams.val_path]) if x.endswith('parquet')]
        self.rank_zero_info(f"Fetched {len(val_files)} training files.")
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
            processor=NLIProcessor(
                batch_size=self.hparams.val_batch_size,
                padding_index=self.hparams.padding_index),
        )
        if self.hparams.gpu_prefetch:
            loader = GPUPrefetcher(loader)
        return loader
