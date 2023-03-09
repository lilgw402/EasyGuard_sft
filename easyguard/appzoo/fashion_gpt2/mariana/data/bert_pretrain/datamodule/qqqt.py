"""qqqt finetune datamodule"""
import logging
from typing import List, Dict, Any

import torch
from cruise import CruiseDataModule
from cruise.data_module import DistributedCruiseDataLoader
from cruise.data_module.gpu_wrapper import GPUPrefetcher
from cruise.utilities.hdfs_io import hlist_files, hopen
from cruise.utilities import DIST_ENV


class QqqtProcessor:
    def __init__(self, batch_size):
        # bsz is required to ensure zero3 mode all batches have same shape
        self._bsz = batch_size

    def transform(self, data_dict):
        input_ids = data_dict['input_ids']
        input_mask = data_dict['input_mask']
        segment_ids = data_dict['segment_ids']
        pos_input_ids = data_dict['pos_input_ids']
        pos_segment_ids = data_dict['pos_segment_ids']
        pos_input_mask = data_dict['pos_input_mask']
        neg_input_ids = data_dict['neg_input_ids']
        neg_segment_ids = data_dict['neg_segment_ids']
        neg_input_mask = data_dict['neg_input_mask']
        pos_ids = list(range(len(data_dict['input_ids'])))
        instance_data = {
            'input_ids': torch.tensor(input_ids).long(),
            'input_mask': torch.tensor(input_mask),
            'segment_ids': torch.tensor(segment_ids).long(),
            'pos_input_ids': torch.tensor(pos_input_ids).long(),
            'pos_input_mask': torch.tensor(pos_input_mask),
            'pos_segment_ids': torch.tensor(pos_segment_ids).long(),
            'neg_input_ids': torch.tensor(neg_input_ids).long(),
            'neg_input_mask': torch.tensor(neg_input_mask),
            'neg_segment_ids': torch.tensor(neg_segment_ids).long(),
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


class QqqtFinetuneDatamodule(CruiseDataModule):
    """qqqt dataset module.

    The trainset creates the a dataloader from the previously qqqt
    data on hdfs://haruna/home/byte_search_nlp_lq/user/zhaoshenjian.01/data/bert/relevance_data/relevance_gaoxiang_oldcut_2019/train.txt or test.txt
    It was converted to parquet files to get rid of tensorflow dependency and the parquet file contains data that has been preprossed already.
    """
    def __init__(self,
                 train_path: str = 'hdfs://haruna/home/byte_data_aml_research/user/zhangzhi.joshua/data/qqqt_2019/train.parquet',
                 val_path: str = 'hdfs://haruna/home/byte_data_aml_research/user/zhangzhi.joshua/data/qqqt_2019/test.parquet',
                 label_path: str = 'hdfs://haruna/home/byte_data_aml_research/user/zhangzhi.joshua/data/qqqt_2019/human_ndcg_label.txt',
                 train_size: int = -1,
                 train_batch_size: int = 32,
                 train_num_workers: int = 2,
                 val_batch_size: int = 256,
                 val_num_workers: int = 1,
                 gpu_prefetch: bool = False,
                 ):
        super().__init__()
        self.save_hparams()
        self.labels = []
        with hopen(label_path, 'r') as label_f:
            for line in label_f:
                if line.strip():
                    self.labels.append([int(label) for label in line.strip().split()])
        num_actual_predict_examples = sum(len(label) for label in self.labels)
        logging.info(f"Num actual predict examples: {num_actual_predict_examples}")

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
            processor=QqqtProcessor(
                batch_size=self.hparams.train_batch_size),
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
            processor=QqqtProcessor(
                batch_size=self.hparams.val_batch_size),
        )
        if self.hparams.gpu_prefetch:
            loader = GPUPrefetcher(loader)
        return loader
