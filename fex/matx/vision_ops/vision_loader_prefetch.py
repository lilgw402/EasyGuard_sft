# -*- coding: utf-8 -*-
from typing import List
import os
import sys
import time
from abc import ABC, abstractclassmethod

import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils import MP_STATUS_CHECK_INTERVAL
import torch.multiprocessing as multiprocessing


if sys.version_info[0] < 3:
    from Queue import Empty
else:
    from queue import Empty


class ExceptionItem(object):
    def __init__(self, exception):
        self.exception = exception


class VisionLoaderPrefetchBase(ABC):
    def __init__(self,
                 data_iter: DataLoader,
                 mode: str,
                 output_map: List[str],
                 max_prefetch_num: int = 1):
        self.max_prefetch_num = max_prefetch_num
        self.mode = mode
        self.output_map = output_map

        assert len(
            self.output_map) == 1, "The output_map params lengths should equals to 1. "
        self._workers = []
        if mode == 'train':
            multiprocessing_context = multiprocessing.get_context("spawn")
            self.data_queue = multiprocessing_context.Queue(
                self.max_prefetch_num)
            w = multiprocessing_context.Process(
                target=self.worker_loop,
                args=(self.data_queue, data_iter, self.output_map))
            # w.daemon = True
            w.start()
            self._workers.append(w)
        else:
            self.data_loader_iter = iter(data_iter)
            self.vision_pipe = self.init_vision_pipeline()
            self._first_batch = None
            try:
                self._first_batch = VisionLoaderPrefetchBase.__next__(self)
            except StopIteration:
                assert False, "It seems that there is no data in the pipeline."

    @abstractclassmethod
    def init_vision_pipeline(self):
        pass

    def worker_loop(self,
                    data_queue,
                    data_iter,
                    output_map,
                    *args,
                    **kwargs):
        # Set decivce for sub-process.
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
        data_iter = iter(data_iter)
        vision_pipeline = self.init_vision_pipeline()
        try:
            for item in data_iter:
                image_tensor = vision_pipeline(item[output_map[0]])
                item[output_map[0]] = image_tensor

                while True:
                    data_queue.put(item)
                    time.sleep(0.01)
                    break

            raise StopIteration()

        except KeyboardInterrupt:
            # Main process will raise KeyboardInterrupt anyways.
            data_queue.close()
        except Exception as e:
            data_queue.put(ExceptionItem(e))

    def _shutdown_workers(self):
        """
        We can only terminate the child process from the parent process
        """
        for w in self._workers:
            w.join(timeout=MP_STATUS_CHECK_INTERVAL)
            if w.is_alive():
                # Existing mechanisms try to make the workers exit
                # peacefully, but in case that we unfortunately reach
                # here, which we shouldn't, (e.g., pytorch/pytorch#39570),
                # we kill the worker.
                w.terminate()

    def __del__(self):
        if self.mode == "train":
            self._shutdown_workers()

    def __iter__(self):
        return self

    def __next__(self):
        if self.mode == "train":
            try:
                item_received = False
                while not item_received:
                    try:
                        item = self.data_queue.get(
                            timeout=MP_STATUS_CHECK_INTERVAL)
                        item_received = True
                    except Empty:
                        # check that the process is still alive
                        if not self._workers[0].is_alive():
                            raise ValueError(
                                "The generator died unexpectedly.")

                if isinstance(item, ExceptionItem):
                    raise item.exception
                return item

            except Exception:
                self._shutdown_workers()
                raise
        else:
            if self._first_batch is not None:
                batch = self._first_batch
                self._first_batch = None
                return batch
            data_dict = self.data_loader_iter.__next__()
            image_tensor = self.vision_pipe(data_dict[self.output_map[0]])
            data_dict[self.output_map[0]] = image_tensor
            return data_dict
