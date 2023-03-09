"""Demo legacy pretrain data datamodule wrapper"""
from typing import List, Dict, Any
import torch
from cruise.data_module import CruiseDataModule
from cruise.data_module import DistributedCruiseDataLoader
from cruise.data_module.gpu_wrapper import GPUPrefetcher
from cruise.utilities.distributed import DIST_ENV
from cruise.utilities.hdfs_io import hlist_files
from ...benchmark_dataloader import DelegateBenchmarkLoader


class DemoProcessor:
    def __init__(self, padding_index: int = 2):
        self._pad_id = padding_index

    def transform(self, data_dict):
        masked_lm_weights = data_dict['masked_lm_weights']  # != 0
        segment_ids = data_dict['segment_ids']
        input_ids = data_dict['input_ids']
        input_mask = data_dict['input_mask']
        input_ids = [i if j else self._pad_id for i, j in zip(input_ids, input_mask)]
        next_sentence_labels = data_dict['next_sentence_labels']  # noqa
        masked_lm_positions = data_dict['masked_lm_positions']
        masked_lm_ids = data_dict['masked_lm_ids']

        masked_labels = [0] * len(input_ids)
        for i, (mask, mpos, mtid) in enumerate(zip(masked_lm_weights, masked_lm_positions, masked_lm_ids)):
            if not mask:
                break
            masked_labels[mpos] = mtid

        instance_data = {
            'input_ids': torch.tensor(input_ids).long(),
            'input_mask': torch.from_numpy(input_mask).long(),
            'labels': torch.tensor(masked_labels).long(),
            'position_ids': torch.arange(len(input_ids)).long(),
            'segment_ids': torch.from_numpy(segment_ids).long(),
            'masked_lm_weights': torch.from_numpy(masked_lm_weights).long(),
            'masked_lm_positions': torch.from_numpy(masked_lm_positions).long(),
            'masked_lm_ids': torch.from_numpy(masked_lm_ids).long(),
            'sentence_label': torch.tensor(next_sentence_labels.item()),
        }
        return instance_data

    def batch_transform(self, batch_data):
        # stack tensors in dicts
        result_data = {}
        for key in batch_data[0].keys():
            result_data[key] = torch.stack([dd[key] for dd in batch_data], dim=0)
        return result_data


class MLMPretrainDatamodule(CruiseDataModule):
    """Demo LM pretrain dataset module.

    The trainset creates the a dataloader from the previously dumped chinese pretrain
    data on hdfs://haruna/home/byte_search_nlp_lq/user/zhaoshenjian.01/data/bert/zh_sdr_qa_site_web_20200518/*.tfrecord
    It was converted to parquet files to get rid of tensorflow dependency.

    TODO(Zhi): add validation.
    """
    def __init__(self,
                 train_path: str = 'hdfs://haruna/home/byte_data_aml_research/user/zhangzhi.joshua/data/bert_pretrain/zh_sdr_qa_site_web_20200518',
                 train_size: int = -1,
                 train_batch_size: int = 128,
                 train_num_workers: int = 4,
                 gpu_prefetch: bool = False,
                 padding_index: int = 2,
                 val_benchmarks: List[str] = ['cmnli_linearprob'],
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
            processor=DemoProcessor(padding_index=self.hparams.padding_index),
        )
        if self.hparams.gpu_prefetch:
            loader = GPUPrefetcher(loader)
        return loader

    def val_dataloader(self):
        return DelegateBenchmarkLoader(
            self.hparams.val_benchmarks,
            **self.hparams.val_kwargs)


if __name__ == '__main__':
    dm = MLMPretrainDatamodule()
    loader = dm.train_dataloader()
    for i, batch in enumerate(loader):
        print(i, type(batch), [(x, batch[x].shape) for x in batch.keys()])
        if i > 3:
            break
