"""Relevance data pretraining datamodule"""
import logging
from typing import List, Dict, Any

import torch
from cruise import CruiseDataModule
from cruise.data_module import DistributedCruiseDataLoader
from cruise.data_module.gpu_wrapper import GPUPrefetcher
from cruise.utilities.hdfs_io import hlist_files
from cruise.utilities import DIST_ENV
from ...benchmark_dataloader import DelegateBenchmarkLoader


class RelevanceProcessor:
    def __init__(self, batch_size):
        # bsz is required to ensure zero3 mode all batches have same shape
        self._bsz = batch_size

    def transform(self, data_dict):
        qp_input_ids = data_dict["qp_input_ids"]
        qp_input_mask = data_dict["qp_input_mask"]
        qp_segment_ids = data_dict["qp_segment_ids"]
        qp_masked_lm_positions = data_dict["qp_masked_lm_positions"]
        qp_masked_lm_ids = data_dict["qp_masked_lm_ids"]
        qp_masked_lm_weights = data_dict["qp_masked_lm_weights"]
        qn_input_ids = data_dict["qn_input_ids"]
        qn_input_mask = data_dict["qn_input_mask"]
        qn_segment_ids = data_dict["qn_segment_ids"]
        qn_masked_lm_positions = data_dict["qn_masked_lm_positions"]
        qn_masked_lm_ids = data_dict["qn_masked_lm_ids"]
        qn_masked_lm_weights = data_dict["qn_masked_lm_weights"]
        margin = data_dict['margin'][0]
        pos_ids = list(range(len(qp_input_ids)))

        qp_masked_labels = [0] * len(qp_input_ids)
        for i, (mask, mpos, mtid) in enumerate(zip(qp_masked_lm_weights, qp_masked_lm_positions, qp_masked_lm_ids)):
            if not mask:
                break
            qp_masked_labels[mpos] = mtid

        qn_masked_labels = [0] * len(qn_input_ids)
        for i, (mask, mpos, mtid) in enumerate(zip(qn_masked_lm_weights, qn_masked_lm_positions, qn_masked_lm_ids)):
            if not mask:
                break
            qn_masked_labels[mpos] = mtid
        instance_data = {
            'qp_input_ids': torch.tensor(qp_input_ids).long(),
            'qp_input_mask': torch.tensor(qp_input_mask),
            'qp_segment_ids': torch.tensor(qp_segment_ids).long(),
            'qp_masked_lm_positions': torch.tensor(qp_masked_lm_positions).long(),
            'qp_masked_lm_ids': torch.tensor(qp_masked_lm_ids).long(),
            'qp_masked_lm_weights': torch.tensor(qp_masked_lm_weights),
            'qp_labels': torch.tensor(qp_masked_labels).long(),
            'qn_input_ids': torch.tensor(qn_input_ids).long(),
            'qn_input_mask': torch.tensor(qn_input_mask),
            'qn_segment_ids': torch.tensor(qn_segment_ids).long(),
            'qn_masked_lm_positions': torch.tensor(qn_masked_lm_positions).long(),
            'qn_masked_lm_ids': torch.tensor(qn_masked_lm_ids).long(),
            'qn_masked_lm_weights': torch.tensor(qn_masked_lm_weights),
            'qn_labels': torch.tensor(qn_masked_labels).long(),
            'margin': torch.tensor(margin),
            'position_ids':  torch.tensor(pos_ids).long()
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


class RelevanceDatamodule(CruiseDataModule):
    """Relevance dataset module.

    The trainset creates the a dataloader from the previously relevance data
    data on hdfs://haruna/home/byte_search_nlp_cr/user/zhouxincheng/data/relevance/qt_gs_lower_hash_0827_20210318
    It was converted to parquet files to get rid of tensorflow dependency and the parquet file contains data that has been preprossed already.
    """
    def __init__(self,
                 train_path: str = 'hdfs://haruna/home/byte_data_aml_research/user/zhangzhi.joshua/data/bert_pretrain/qt_gs_lower_hash_0827_20210318',
                 train_size: int = -1,
                 train_batch_size: int = 128,
                 train_num_workers: int = 4,
                 gpu_prefetch: bool = False,
                 padding_index: int = 2,
                 val_benchmarks: List[str] = [],
                 val_kwargs: Dict[str, Any] = {},
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
            processor=RelevanceProcessor(
                batch_size=self.hparams.train_batch_size),
        )
        if self.hparams.gpu_prefetch:
            loader = GPUPrefetcher(loader)
        return loader

    def val_dataloader(self):
        return DelegateBenchmarkLoader(
            self.hparams.val_benchmarks,
            **self.hparams.val_kwargs)
