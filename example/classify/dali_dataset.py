""" dataset """


import random
import os
import traceback
import json
import base64
import numpy as np
import torch
import multiprocessing.dummy as mp

from fex.data import DistLineReadingDataset
from fex import _logger as log
from fex.utils.hdfs_io import hopen
from fex.data import PytorchDaliIter

try:
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
except ImportError:
    raise ImportError(
        "Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")


class ValFrameDecoderQueuePipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, external_data, min_size, max_size, seed=100, dali_cpu=False,
                 need_text=False, need_title=False,
                 has_mlm=False):
        super(ValFrameDecoderQueuePipeline, self).__init__(
            batch_size, num_threads, device_id, seed=seed)
        self.external_data = external_data
        self.data_iter = iter(self.external_data)
        frame_num = 8
        self.input_raw_images = [ops.ExternalSource(device="cpu") for _ in range(frame_num)]

        self.queue = mp.Queue()

        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'

        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu",
                              resize_shorter=min_size,
                              interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop_w=min_size,
                                            crop_h=max_size,
                                            crop_pos_x=0.5,
                                            crop_pos_y=0.5,
                                            mean=[0.485 * 255, 0.456 *
                                                  255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                                            out_of_bounds_policy="pad")

    def define_graph(self):
        self.raw_images = [i() for i in self.input_raw_images]
        images = self.decode(self.raw_images)
        images = self.res(images)
        images = self.cmnp(images)
        images = fn.stack(*images)
        return images

    def iter_setup(self):
        try:
            data_dict = self.data_iter.next()
            for i, inp in enumerate(self.raw_images):
                self.feed_input(inp, data_dict.pop('image%s' % i))
            self.queue.put(data_dict)

        except StopIteration:
            self.data_iter = iter(self.external_data)
            raise StopIteration


class DaliLabelDataset(DistLineReadingDataset):
    """
    dataset，继承的dataset是 DistLineReadingDataset，是一个逐行读hdfs数据的IterableDataset。
    """

    def __init__(self, config, data_path, rank=0, world_size=1, shuffle=True, repeat=False,
                 transform=None, debug=False):
        super().__init__(data_path, rank, world_size, shuffle, repeat)

        self.frame_len = config.DATASET.FRAME_LEN
        print(self.frame_len, 'frame len')

        with hopen('hdfs://haruna/home/byte_search_nlp_lq/user/huangwenguan/fex/black_frame.jpg', 'rb') as f:
            self.black_frame = np.array(np.frombuffer(f.read(), dtype=np.uint8))
        print(self.black_frame, 'self.black_frame')

    def __iter__(self):
        """

        """
        for example in self.generate():
            try:
                data_item = json.loads(example)
                print(data_item.keys())
                frames = []
                frames_raw = data_item.pop('frames')[:self.frame_len]
                for frame in frames_raw:
                    image_str = self.b64_decode(frame['b64_binary'])
                    image = np.array(np.frombuffer(image_str, dtype=np.uint8))
                    frames.append(image)
                input_dict = {'frames': frames, 'doc': data_item}
                yield input_dict
            except Exception as e:
                log.error(traceback.format_exc())
                log.error('encounter broken data: %s' % e)

    def collect_fn(self, data):
        frames = [[] for _ in range(self.frame_len)]
        frames_mask = []

        docs = []
        res = {}

        for ib, ibatch in enumerate(data):
            docs.append(ibatch['doc'])
            img_np = ibatch['frames']

            frames_mask_cur = []

            # 如果不够8帧，要补帧
            if len(img_np) < self.frame_len:
                print('encouter not 8 gid ', len(img_np))
                for i, img in enumerate(img_np):
                    frames[i].append(img)
                    frames_mask_cur.append(1)
                for i in range(len(img_np), self.frame_len):
                    frames[i].append(self.black_frame)  # 如果这个视频没有帧，就用黑帧来替代
                    frames_mask_cur.append(0)
            else:
                # 够的话就冲啊
                for i, img in enumerate(img_np):
                    frames[i].append(img)
                    frames_mask_cur.append(1)

            frames_mask.append(frames_mask_cur)

        for i, image_pac in enumerate(frames):
            res['image%s' % i] = image_pac
        res['frames_mask'] = frames_mask
        res['docs'] = docs

        return res

    def b64_decode(self, string):
        if isinstance(string, str):
            string = string.encode()
        return base64.decodebytes(string)
