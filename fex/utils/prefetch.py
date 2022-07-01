"""
prefetch, 对dataloader 包了一层，提前做prefetch，并且将数据move to cuda
"""
from typing import Dict, List, Union, Tuple

import threading
import queue as Queue

import torch
from torch.utils.data import DataLoader


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):

    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            self.batch = self._move_to_cuda(self.batch)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch

    def _move_to_cuda(self, batch_dict: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """
        TODO: 后续补充
        """
        if isinstance(batch_dict, list):
            batch_dict = batch_dict[0]
        for k in batch_dict:
            if isinstance(batch_dict[k], torch.Tensor):
                if not batch_dict[k].is_cuda:
                    batch_dict[k] = batch_dict[k].cuda(non_blocking=True)
        return batch_dict
