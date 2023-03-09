from multiprocessing.sharedctypes import Value
import sys
from typing import Union, List, Tuple, Optional, Dict, Any
import tempfile

import os
import os.path as osp
import random
import torch
from queue import PriorityQueue

from torch.utils.data.dataset import IterableDataset
from torch.distributed import get_rank
from cruise.data_module.cruise_parquet_dataset import CruiseParquetDataset
from cruise.data_module.cruise_tfrecord_dataset import CruiseTFRecordDataset, BytedTFRecordDataset
from cruise.data_module.cruise_kv_dataset import CruiseKVDataset as BytedArnoldKVDataset
from cruise.data_module.cruise_tfidx_dataset import BytedTFIdxDataset
from cruise.data_module.utils import get_worker_info, tf_sample_sharding_is_enabled
from cruise.trainer.callback.base import Callback
from cruise.trainer.common_trainer import CruiseTrainer
from cruise.utilities.types import _METRIC, _PATH, STEP_OUTPUT
from cruise.utilities.distributed import DIST_ENV
from cruise.utilities.cloud_io import ensure_dir
from cruise.utilities.hdfs_io import hcopy, hexists
from cruise.utilities.rank_zero import rank_zero_info
from cruise import last_cli
import cruise as crs


def complement(index, batch_size):
    begin = len(index)
    index = index + (batch_size - begin) * [index[-1]]
    return index, begin


def chunk(iterable, chunk_size):
    ret = []
    for record in iterable:
        ret.append(record)
        if len(ret) == chunk_size:
            yield ret
            ret = []
    if ret:
        yield ret


def get_resume_step_per_worker(total_step, worker_id=None):
    local_worker_id, local_num_workers = get_worker_info()
    if worker_id is None:
        worker_id = local_worker_id
    resume_step = total_step // local_num_workers
    if worker_id < total_step - local_num_workers * resume_step:
        resume_step += 1
    return resume_step


def split_urls(urls, worker_id, worker_num):
    local_urls = urls[worker_id::worker_num]
    assert len(local_urls) > 0
    return local_urls


def rename_keys(sample, mappings):
    for key in mappings:
        sample[mappings[key]] = sample.pop(key)
    return sample


class TFRecordDataset(IterableDataset):
    def __init__(self,
                 urls: List[str],
                 batch_size: int,
                 rank: int = 0,
                 num_replicas: int = 1,
                 num_readers: int = 4,
                 decode_fn: callable = None,
                 return_keys: List[str] = None,
                 shuffle: bool = False,
                 seed: int = 0,
                 resume_step: int = 0,
                 task_id: int = 0,
                 remain_sample_idx: bool = False,
                 repeat: bool = True,
                 shuffle_buffer_size: int = 100,
                 synthetic_sample: bool = False):
        super(TFRecordDataset).__init__()
        use_tf_sample_sharding = tf_sample_sharding_is_enabled()
        if use_tf_sample_sharding:
            Dataset = CruiseTFRecordDataset
            deserializer = decode_fn
            ds_bz = min(4, batch_size)  # we use small bz for dataloading for better performance
        else:
            Dataset = BytedTFRecordDataset
            deserializer = None
            ds_bz = batch_size

        self.dataset = Dataset(
                urls=urls,
                rank=rank,
                num_replicas=num_replicas,
                num_readers=num_readers,
                shuffle=shuffle,
                deterministic=True,
                seed=seed,
                resume_step=resume_step,
                batch_size=ds_bz,
                repeat=repeat,
                deserializer=deserializer,
                shuffle_buffer_size=shuffle_buffer_size)

        self.use_tf_sample_sharding = use_tf_sample_sharding
        self.batch_size = batch_size
        self.return_keys = return_keys
        self.resume_step = resume_step
        self.seed = seed
        self.decode_fn = decode_fn
        self.task_id = task_id
        self.remain_sample_idx = remain_sample_idx
        self.repeat = repeat
        self.in_restart = False
        self.synthetic_sample = synthetic_sample
        self.shuffle_buffer_size = shuffle_buffer_size

    def _get_next_item(self, data_iter):
        try:
            if self.synthetic_sample and hasattr(self, "_first_sample"):
                item = self._first_sample
            else:
                item = next(data_iter)  # batch data
                if not hasattr(self, "_first_sample"):
                    self._first_sample = item
            if not self.use_tf_sample_sharding and self.decode_fn:
                item = [self.decode_fn(i) for i in item]
            else:
                item = [i for i in item]
            if self.return_keys:
                for i in item:
                    if isinstance(i, dict):
                        for k in i:
                            if k not in self.return_keys:
                                i.pop(k)
            if self.in_restart:
                self.in_restart = False
            return item, data_iter, False
        except StopIteration:
            # encase infinite loop
            # in_restart indicates that the current call of _get_next_item is just after a StopIteration
            # if it is true and the _get_next_item raised a StopIteration again, raise a runtime error
            if self.in_restart:
                raise RuntimeError("Something went wrong when repeating the dataset.")
            if not self.repeat:
                return None # tell dataset that the file has been read over
            self.in_restart = True
            data_iter = iter(self.dataset)
            return self._get_next_item(data_iter)

    def _data_generator(self):
        data_iter = iter(self.dataset)
        ds_idx = 0
        repeated = False
        res = []
        while True:
            data = self._get_next_item(data_iter)
            if data is None:
                break
            item, data_iter, tmp = data
            repeated = tmp if tmp else repeated
            for sample in item:
                if self.remain_sample_idx:
                    idx_str = f'{ds_idx}_repeat' if repeated else str(ds_idx)
                    sample['crs_sample_idx'] = idx_str
                    ds_idx += 1
                if self.task_id is not None:
                    sample['crs_task_id'] = self.task_id
            res += item
            if len(res) >= self.batch_size:
                yield res[:self.batch_size]
                res = res[self.batch_size:]
        if res:
            yield res

    def __iter__(self):
        return self._data_generator()


class DistIterableDataset(IterableDataset):
    def __init__(self,
                 urls: List[str],
                 batch_size: int,
                 url_format: str,
                 seed: int = 0,
                 shuffle: bool = False,
                 columns: List[str] = None,
                 shuffle_buffer_size: int = 100,
                 resume_step: int = 0,
                 task_id: int = 0,
                 remain_sample_idx: bool = False,
                 repeat: bool = True,
                 parquet_cache_on: bool = True,
                 synthetic_sample: bool = False,
                 fast_resume: bool = False,
                 dyn_bsz: bool = False,
                 max_seq_len: int = 2048
                 ):
        super(DistIterableDataset).__init__()
        self.use_cruise_parquet = url_format == "parquet"
        if self.use_cruise_parquet:
            self.dataset = CruiseParquetDataset(urls, columns=columns, repeat=repeat, cache_on=parquet_cache_on)
        else:
            from cruise.data_module.dist_iterable_dataset import BytedDistIterableDataset
            self.dataset = BytedDistIterableDataset(
                urls=urls, sort=False, repeat=repeat, url_format=url_format, columns=columns, parquet_iter_read=True)

        self.urls = urls
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_size = shuffle_buffer_size
        self.seed = seed
        self.resume_step = resume_step
        self.task_id = task_id
        self.remain_sample_idx = remain_sample_idx
        self.in_restart = False
        self.repeat = repeat
        self.synthetic_sample = synthetic_sample
        # fast_resume will speed up the data resume process, but the resumed result might be 
        # slightly different from the original one when shuffle = True
        self.fast_resume = fast_resume
        self.sample_idx_key = "crs_sample_idx"
        self.dyn_bsz = dyn_bsz
        self.max_seq_len = max_seq_len

        global_config = last_cli().hparams
        self.resume_ckpt_path = global_config['trainer']['resume_ckpt_path']

    def _get_next_item(self, data_iter):
        try:
            if self.synthetic_sample and hasattr(self, "_first_sample"):
                item = self._first_sample
            else:
                item = next(data_iter)  # batch data
                if not hasattr(self, "_first_sample"):
                    self._first_sample = item
            if self.in_restart:
                self.in_restart = False
            return item, data_iter
        except StopIteration:
            if self.in_restart:
                raise RuntimeError("Something went wrong when repeating the dataset.")
            if not self.repeat:
                return None # tell dataset that the file has been read over
            self.in_restart = True
            data_iter = iter(self.dataset)
            return self._get_next_item(data_iter)

    def _update_sample_idx(self, data, idx, sample_idx_base):
        if self.sample_idx_key not in data:
            data[self.sample_idx_key] = sample_idx_base + str(idx)

    def _data_generator(self):
        from cruise.data_module.dist_iterable_dataset import byted_split_and_shard
        local_worker_id, local_num_workers = get_worker_info()
        pid = os.getpid()
        data_id_base = "{}_".format(pid)
        if self.resume_step >= 0:
            if self.dyn_bsz and self.resume_ckpt_path:
                coeff = 2
                self.resume_step = max(0, self.resume_step - self.max_seq_len * local_num_workers * coeff)
            resume_idx = self.resume_step % local_num_workers
            resume_worker_id = (self.resume_step % local_num_workers + local_worker_id) % local_num_workers
            resume_step = get_resume_step_per_worker(self.resume_step, resume_worker_id)
            if self.use_cruise_parquet and not isinstance(self.urls[0], str):
                url_list = self.urls[:]
                if self.fast_resume:
                    self.dataset.resume_step = resume_step * self.batch_size
                    resume_step = 0
            else:
                url_list = []
                for i in range(local_num_workers):
                    url_list.append(byted_split_and_shard(self.urls,
                                                          num_replicas=local_num_workers,
                                                          rank=i,
                                                          sort=False))

            self.dataset.urls = url_list[resume_idx:] + url_list[:resume_idx]
        else:
            raise RuntimeError("resume_step must not be negative")

        data_iter = iter(self.dataset)
        shuffle_buffer = {}
        ds_idx = 0
        cur_buffer_size = 0
        if self.shuffle:
            for i in range(self.buffer_size):
                cur_data = self._get_next_item(data_iter)
                if cur_data is None:
                    break
                cur_buffer_size += 1
                item, data_iter = cur_data
                if self.remain_sample_idx:
                    self._update_sample_idx(item, ds_idx, data_id_base)
                shuffle_buffer[i] = item
                ds_idx += 1

        random.seed(self.seed)
        while True:
            data = []
            if not self.shuffle:
                for _ in range(self.batch_size):
                    cur_data = self._get_next_item(data_iter)
                    if cur_data is None:
                        break
                    item, data_iter = cur_data
                    if self.remain_sample_idx:
                        self._update_sample_idx(item, ds_idx, data_id_base)
                    data.append(item)
                    ds_idx += 1
            else:
                for _ in range(self.batch_size):
                    cur_data = self._get_next_item(data_iter)
                    if cur_data is not None:
                        item, data_iter = cur_data
                        if self.remain_sample_idx:
                            self._update_sample_idx(item, ds_idx, data_id_base)
                        idx = random.randint(0, cur_buffer_size - 1)
                        data.append(shuffle_buffer[idx])
                        shuffle_buffer[idx] = item
                        ds_idx += 1
                    else:
                        if cur_buffer_size == 0:
                            break
                        idx = random.randint(0, cur_buffer_size - 1)
                        if idx in shuffle_buffer:
                            data.append(shuffle_buffer[idx])
                            shuffle_buffer[idx] = shuffle_buffer[cur_buffer_size-1]
                            del shuffle_buffer[cur_buffer_size-1]
                            cur_buffer_size -= 1
            if not data:
                break
            if resume_step > 0:
                resume_step -= 1
                continue

            
            for i, sample in enumerate(data):
                if self.task_id is not None:
                    sample['crs_task_id'] = self.task_id

            yield data

    def __iter__(self):
        return self._data_generator()


class KVIterableDataset(IterableDataset):
    def __init__(self,
                 urls: List[str],
                 batch_size: int,
                 data_range: List[Tuple[int, int]] = [],
                 num_readers: int = 1,
                 repeat: bool = True,
                 shuffle: bool = False,
                 seed: int = 0,
                 transform: callable = None,
                 decode_fn: callable = None,
                 return_keys: List[str] = None,
                 resume_step: int = 0,
                 task_id: int = 0,
                 remain_sample_idx: bool = False,
                 drop_last: bool = True,
                 synthetic_sample: bool = False):
        super(KVIterableDataset).__init__()
        self.urls = urls
        self.shuffle = shuffle
        self.num_readers = num_readers
        self.transform = transform
        if len(data_range) == 0:
            data_range = [(-1, -1)] * len(urls)
        assert len(data_range) == len(urls)
        self.data_range = data_range
        self.seed = seed
        self.batch_size = batch_size
        self.epoch = 0
        self.repeat = repeat
        self.decode_fn = decode_fn
        self.return_keys = return_keys
        self.resume_step = resume_step
        self.task_id = task_id
        self.remain_sample_idx = remain_sample_idx
        self.drop_last = drop_last
        self.ds = [BytedArnoldKVDataset(url, self.num_readers, self.transform) for url in self.urls]
        self.synthetic_sample = synthetic_sample

    def _data_generator(self):
        local_worker_id, local_num_workers = get_worker_info()
        inner_iter = 0
        repeated = False
        while True:
            url_idx = list(range(len(self.urls)))
            # if self.shuffle:
            #     random.Random(self.seed).shuffle(url_idx)
            for idx in url_idx:
                url = self.urls[idx]
                rng_s, rng_e = self.data_range[idx]
                dataset = self.ds[idx]
                if rng_s >= 0:
                    ds_length = rng_e - rng_s
                else:
                    ds_length = len(dataset)
                ds_step = (ds_length + self.batch_size - 1) // self.batch_size
                if ds_step < self.resume_step:
                    self.resume_step -= ds_step
                    continue
                ds_index_list = list(range(ds_length))
                if rng_s >= 0:
                    ds_index_list = [i + rng_s for i in ds_index_list]
                if self.shuffle:
                    random.Random(self.seed + inner_iter).shuffle(ds_index_list)
                ds_index_list = ds_index_list[self.resume_step * self.batch_size:]
                self.resume_step = 0
                for index in chunk(ds_index_list, self.batch_size * local_num_workers):
                    cur_worker_num = (len(index) + self.batch_size - 1) // self.batch_size
                    if local_worker_id >= cur_worker_num:
                        break
                    remain = len(index) - self.batch_size * (cur_worker_num - 1)
                    s = self.batch_size * local_worker_id
                    e = self.batch_size * (local_worker_id + 1)
                    if local_worker_id == (cur_worker_num - 1):
                        e = s + remain
                    index = index[s: e]
                    repeat_id = self.batch_size
                    if len(index) < self.batch_size and not self.drop_last:
                        index, repeat_id = complement(index, self.batch_size)
                    ds_idx = ['0'] * len(index)
                    if self.synthetic_sample and hasattr(self, "_first_sample"):
                        data = self._first_sample
                    else:
                        data = dataset[index]
                        if not hasattr(self, "_first_sample"):
                            self._first_sample = data
                    if self.remain_sample_idx:
                        ds_idx = [dataset.keys[i] for i in index]

                    if self.decode_fn:
                        data = [self.decode_fn(item) for item in data]
                    if self.return_keys:
                        data = [{i: item[i] for i in self.return_keys if i in item} for item in data]

                    for j, sample in enumerate(data):
                        if self.remain_sample_idx:
                            # add _repeat for evaluation
                            sample_idx = ds_idx[j] + '_repeat' if repeated or (j>=repeat_id) else ds_idx[j]
                            sample['crs_sample_idx'] = sample_idx
                        if self.task_id is not None:
                            sample['crs_task_id'] = self.task_id
                    yield data
            if not self.repeat:
                break
            repeated = True
            inner_iter += 1
        self.epoch += 1

    def __iter__(self):
        return self._data_generator()


_src_type_dataset_dict = {
    "parquet": DistIterableDataset,
    "jsonl": DistIterableDataset,
    "kv": KVIterableDataset,
    "tfrecord": TFRecordDataset,
    "tfidx": BytedTFIdxDataset
}


class HybridDataset(IterableDataset):
    def __init__(self,
                 data_sources: List[List[str]],
                 source_types: List[str],
                 batch_sizes: List[int],
                 num_readers: List[int],
                 return_keys: List[List[str]] = [],
                 decode_fn: List[callable] = [],
                 trans_fn: Tuple[callable, callable, callable] = [None, None, None],
                 shuffle: bool = False,
                 seed: int = 0,
                 step: int = 0,
                 multiplex_weights: List[float] = [],
                 shard_rank_info: List[Tuple[int, int]] = [],
                 task_id_list: List = [],
                 remain_sample_idx: bool = False,
                 drop_last: bool = True,
                 repeat: bool = True,
                 stop_queue: object = None,
                 key_mapping_list: List[Dict] = None,
                 batch_shuffle: bool = False,
                 parquet_cache_on: bool = True,
                 shuffle_buffer_size: int = 100,
                 synthetic_sample: bool = False,
                 fast_resume: bool = False,
                 transform_many: bool = False,
                 dyn_bsz: bool = False,
                 dyn_bsz_margin: float = 0.0,
                 num_warmup_steps: int = -1):
        super(IterableDataset).__init__()
        
        local_worker_id, local_num_workers = get_worker_info()

        self.data_srcs = data_sources
        self.ds_num = len(self.data_srcs)
        assert self.ds_num == len(source_types)
        self.src_types = source_types
        self.batch_sizes = batch_sizes
        self.seed = seed
        self.shuffle = shuffle
        self.trans = trans_fn[0]
        self.batch_trans = trans_fn[1]
        self.post_trans = trans_fn[2]
        self.step = step
        self.multiplex_weights = multiplex_weights
        self.runtime_multiplex_weights = []
        self.multiplex_accumulate_weights = []
        self.remain_sample_idx = remain_sample_idx
        self.drop_last = drop_last
        self.repeat = repeat
        self.stop_queue = stop_queue
        self.key_mapping_list = key_mapping_list if key_mapping_list else [None] * len(data_sources)
        self.batch_shuffle = batch_shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.synthetic_sample = synthetic_sample
        self.transform_many = transform_many
        self.dyn_bsz = dyn_bsz
        self.dyn_bsz_margin = dyn_bsz_margin
        self.dyn_bsz_buffer = PriorityQueue()
        self.dyn_bsz_buffersize = 5000
        self.num_warmup_steps = num_warmup_steps//local_num_workers if num_warmup_steps>0 else num_warmup_steps
        self.ds_iters_counter = 0
        self.global_rank = get_rank()
        if self.global_rank == 0:
            print('batch size num_warmup_steps:', self.num_warmup_steps)
        if not task_id_list:
            task_id_list = [i for i in range(len(data_sources))]
        if multiplex_weights:
            assert len(multiplex_weights) == self.ds_num
            for bs in self.batch_sizes:
                assert self.batch_sizes[0] == bs, "For multiplex mode, each dataset should have same batch size"

        if decode_fn is None or len(decode_fn) == 0:
            decode_fn = [None] * self.ds_num
        if return_keys is None or len(return_keys) == 0:
            return_keys = [None] * self.ds_num

        if not shard_rank_info:
            shard_rank_info = [(0, 1) for _ in range(self.ds_num)]
        assert len(shard_rank_info) == self.ds_num

        self.datasets = []
        for i in range(self.ds_num):
            src_tp = self.src_types[i]
            assert src_tp in _src_type_dataset_dict
            Dataset = _src_type_dataset_dict[src_tp]
            src = self.data_srcs[i]
            if (src_tp == "kv" and isinstance(src[0], str)) or isinstance(src, str):
                src = [src]
            task_id = task_id_list[i]
            # create DistIterableDataset
            if src_tp in {"parquet", "jsonl"}:
                dataset = Dataset(
                    urls=src,
                    batch_size=batch_sizes[i],
                    url_format=src_tp,
                    seed=self.seed,
                    shuffle=self.shuffle,
                    columns=return_keys[i],
                    resume_step=step,
                    task_id=task_id,
                    remain_sample_idx=self.remain_sample_idx,
                    repeat=repeat,
                    parquet_cache_on=parquet_cache_on,
                    shuffle_buffer_size=shuffle_buffer_size,
                    synthetic_sample=self.synthetic_sample,
                    fast_resume=fast_resume,
                    dyn_bsz=dyn_bsz,
                )
            # create TFRecordDataset
            elif src_tp == "tfrecord":
                dataset = Dataset(
                    urls=src,
                    batch_size=batch_sizes[i],
                    decode_fn=decode_fn[i],
                    return_keys=return_keys[i],
                    shuffle=self.shuffle,
                    seed=self.seed,
                    resume_step=step,
                    num_replicas=shard_rank_info[i][1],
                    rank=shard_rank_info[i][0],
                    num_readers=num_readers[i],
                    task_id=task_id,
                    remain_sample_idx=self.remain_sample_idx,
                    repeat=repeat,
                    shuffle_buffer_size=shuffle_buffer_size,
                    synthetic_sample=self.synthetic_sample
                )
            elif src_tp == "tfidx":
                src_urls = [i[0] for i in src]
                data_range = [(i[1], i[2]) for i in src]
                dataset = Dataset(
                    urls=src_urls,
                    batch_size=batch_sizes[i],
                    data_range=data_range,
                    decode_fn=decode_fn[i],
                    seed=self.seed,
                    shuffle=self.shuffle,
                    resume_step=step,
                    task_id=task_id,
                    remain_sample_idx=self.remain_sample_idx,
                    drop_last=self.drop_last,
                    repeat=repeat,
                    batch_shuffle=batch_shuffle,
                    synthetic_sample=self.synthetic_sample
                )
            # create KVIterableDataset
            else:
                src_urls = [i[0] for i in src]
                data_range = [(i[1], i[2]) for i in src]
                dataset = Dataset(
                    urls=src_urls,
                    batch_size=batch_sizes[i],
                    data_range=data_range,
                    decode_fn=decode_fn[i],
                    return_keys=return_keys[i],
                    num_readers=num_readers[i],
                    seed=self.seed,
                    shuffle=self.shuffle,
                    resume_step=step,
                    task_id=task_id,
                    remain_sample_idx=self.remain_sample_idx,
                    drop_last=self.drop_last,
                    repeat=repeat,
                    synthetic_sample=self.synthetic_sample
                )
            self.datasets.append(dataset)

    def _get_raw_batch_data(self, ds_iters):
        if not self.multiplex_weights:
            data = []
            for i in range(self.ds_num):
                ds = ds_iters[i]
                try:
                    cur_ds_data = next(ds)
                except StopIteration:
                    # if we get stop iteration here, the repeat must be false for the sub dataset
                    cur_ds_data = None
                # skip current ds if the ds has finished
                if cur_ds_data is None:
                    continue
                if self.key_mapping_list[i]:
                    cur_ds_data = [rename_keys(sample, self.key_mapping_list[i]) for sample in cur_ds_data]
                data += cur_ds_data
            if not data:
                # all ds has finished, which indicates the end of the current epoch
                return None
            return data
        else:
            ds_idx = random.choices(list(range(self.ds_num)), weights=self.runtime_multiplex_weights, k=1)[0]
            ds_iter = ds_iters[ds_idx]
            try:
                data = next(ds_iter)
            except StopIteration:
                data = None
            if data is None:
                # the current dataset is finished, set the weight to be 0 
                self.runtime_multiplex_weights[ds_idx] = 0
                s = sum(self.runtime_multiplex_weights)
                # all ds is finished, return None to indicate the epoch finish
                if s == 0:
                    return None
                # update the weights for the left datasets
                for i in range(self.ds_num):
                    self.runtime_multiplex_weights[i] /= s
                # get data again
                data = self._get_raw_batch_data(ds_iters)
            if self.key_mapping_list[ds_idx]:
                cur_ds_data = [rename_keys(sample, self.key_mapping_list[ds_idx]) for sample in cur_ds_data]
            return data

    def _data_generator(self, ds_iters):
        if self.multiplex_weights:
            random.seed(self.seed)
        self.runtime_multiplex_weights = self.multiplex_weights
        trans_data = []
        counter = 0
        while True:
            if self.stop_queue is not None and not self.stop_queue.empty():
                break
            data = self._get_raw_batch_data(ds_iters)
            self.ds_iters_counter += 1
            if data is None:
                break
            
            bs = len(data)

            if self.trans:
                for item in data:
                    processed = self.trans(item)
                    if processed is not None and self.dyn_bsz:
                        if not self.transform_many:
                            processed = [processed]
                        for x in processed:
                            #self.step的目的是让长seq也能被采样到
                            #Ref:https://stackoverflow.com/questions/70015262/how-python-priorityqueue-works-with-irregular-tuple-object
                            self.dyn_bsz_buffer.put([self.step + x['attention_mask'].sum().item(), counter, x]) 
                            counter += 1
                        cur_buffersize = self.dyn_bsz_buffer.qsize()
                        if cur_buffersize < self.dyn_bsz_buffersize:
                            continue
                        max_seq_len = processed[0]['input_ids'].shape[0] 
                        # we make sure every step return a full batch in repeat mode 
                        if self.num_warmup_steps>0 and self.step<self.num_warmup_steps:
                            # eff tokens need to start from max_seq_len*1*#gpu.
                            # If need more small #eff tokens, we can dyn adjust max_seq_len.
                            ith_bs =  0.2 + (self.step+1) / self.num_warmup_steps * (bs-0.2)
                            dyn_bsz_thres = max_seq_len*(ith_bs-self.dyn_bsz_margin)
                        else:
                            dyn_bsz_thres = max_seq_len*(bs-self.dyn_bsz_margin)
                        cum_seq_len = self.dyn_bsz_buffer.queue[0][-1]['attention_mask'].sum().item()
                        trans_data = [self.dyn_bsz_buffer.get()[-1]]
                        for i in range(cur_buffersize-1):
                            seq_len = self.dyn_bsz_buffer.queue[0][-1]['attention_mask'].sum().item()
                            if seq_len+cum_seq_len<=dyn_bsz_thres:
                                trans_data.append(self.dyn_bsz_buffer.get()[-1])
                                cum_seq_len = cum_seq_len + seq_len
                            else:
                                break
                        yield self.process_batch(trans_data)
                        self.step += 1
                    elif processed:
                        if self.transform_many:
                            trans_data.extend(processed)
                        else:
                            trans_data.append(processed)
                        # we make sure every step return a full batch in repeat mode
                        if len(trans_data) >= bs and self.repeat:
                            remain_data = trans_data[bs:]
                            yield self.process_batch(trans_data[:bs])
                            self.step += 1
                            trans_data = remain_data      
            else:
                for item in data:
                    if item:
                        if self.transform_many:
                            trans_data.extend(item)
                        else:
                            trans_data.append(item)
                        if len(trans_data) >= bs and self.repeat:
                            remain_data = trans_data[bs:]
                            yield self.process_batch(trans_data[:bs])
                            self.step += 1
                            trans_data = remain_data
            
            if not self.repeat:
                # we return the batch no matter it's full or not
                yield self.process_batch(trans_data)
                trans_data = []

            if self.stop_queue is not None and not self.stop_queue.empty():
                break

    def process_batch(self, trans_data):
        if self.batch_trans:
            batch_trans_data = self.batch_trans(trans_data)
        else:
            batch_trans_data = trans_data
        if self.post_trans:
            post_trans_data = self.post_trans(batch_trans_data)
        else:
            post_trans_data = batch_trans_data
        
        return post_trans_data

    def __iter__(self):
        ds_iters = []
        for ds in self.datasets:
            ds_iters.append(iter(ds))
        return self._data_generator(ds_iters)
        


if __name__ == "__main__":
    # kv_src = [[("hdfs://haruna/home/byte_arnold_lq/user/yibairen.byron/imagenet/train", 0, 10000)]]
    kv_src = [('hdfs://haruna/home/byte_arnold_hl_vc/arnold_dataset/imagenet1k/train', 0, 640582)]
    parquet_base = "hdfs://haruna/home/byte_arnold_lq_vc/zhouboyan/datasets/multimodal/imagenet1k/parquets/train/imagenet1k_train_{:03}.parquet"
    parquet_src = [parquet_base.format(i) for i in range(10)]
    tfrecord_base = "hdfs://robertbackend/user/yibairen.byron/dataset/imagenet/tfrecords/train-{:05}-of-01024"
    tfrecord_src = [tfrecord_base.format(i) for i in range(10)]
    ds = HybridDataset(
        data_sources=[kv_src, parquet_src, tfrecord_src],
        source_types=["kv", "parquet", "tfrecord"],
        batch_sizes=[100, 100, 100],
        num_readers=[4, None, 4],
    )
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, num_workers=4, collate_fn=lambda x: x)
    for data in loader:
        print(len(data[0]))
