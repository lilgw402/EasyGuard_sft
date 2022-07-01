# -*- coding: utf-8 -*-
'''
video dataset json format, from hdfa
'''
from typing import Callable
import json
import numpy as np
import traceback
import torch
import base64
import io
import random
from PIL import Image

from fex.config import CfgNode
from fex import _logger as log
from fex.data.datasets.dist_dataset import DistLineReadingDataset
from fex.data import BertTokenizer
from fex.utils.hdfs_io import hopen


class VideoDataset(DistLineReadingDataset):
    """
    video dataset
    """

    def __init__(self,
                 config: CfgNode,
                 data_path: str,
                 rank: int = 0,
                 world_size: int = 1,
                 shuffle: bool = True,
                 repeat: bool = False,
                 transform: Callable = None,
                 pick_query: bool = True):
        super().__init__(data_path, rank, world_size, shuffle, repeat)

        self.transform = transform
        self.is_torchvision = self.transform is not None

        vocab_file = config.get('network.vocab_file')
        self.max_len = config.get('dataset.max_len', 32)
        self.frame_len = config.get('dataset.max_frame_len', 8)
        self.add_title = config.get('dataset.add_title', False)
        self.tokenizer = BertTokenizer(vocab_file,
                                       do_lower_case=config.get(
                                           'network.do_lower_case', True),
                                       tokenize_emoji=config.get(
                                           'network.tokenize_emoji', False),
                                       greedy_sharp=config.get(
                                           'network.greedy_sharp', False)
                                       )
        self.PAD = self.tokenizer.vocab['[PAD]']
        self.use_pick_query = pick_query

        # 读一张黑色的帧，作为padding 用
        with hopen('hdfs://haruna/home/byte_search_nlp_lq/multimodal/black_frame.jpg', 'rb') as f:
            self.black_frame = np.array(
                np.frombuffer(f.read(), dtype=np.uint8))

    def __iter__(self):
        for example in self.generate():
            try:
                data_item = json.loads(example)

                # 1. 图片的处理
                frames = []
                frames_mask = []
                select_frames = data_item.pop('frames')[:self.frame_len]
                for frame in select_frames:
                    image_str = self.b64_decode(frame['b64_content'])
                    # if self.is_torchvision: # 如果有 torchvision 的transform，则在这做预处理
                    #     image = self.load_image(image_str)
                    #     image = self.transform(image)
                    # else: # 否则只是解析，预处理留到后面的dali pipeline 做
                    image = np.array(np.frombuffer(image_str, dtype=np.uint8))
                    frames.append(image)
                    frames_mask.append(1)
                while len(frames) < self.frame_len:  # padding 图片
                    frames.append(self.black_frame)
                    frames_mask.append(0)

                # 2. 文本的处理
                query = self.pick_query(data_item['queries'], self.use_pick_query)
                query_ids = self.get_tokens(query)
                res = {'frames': frames,
                       'frames_mask': frames_mask,
                       'input_ids': query_ids}
                if self.add_title:
                    title_ids = self.get_tokens(data_item['title'])
                    res['title_ids'] = title_ids
                yield res

            except Exception as e:
                log.error(traceback.format_exc())
                log.error('encounter broken data: %s' % e)
                log.error(data_item.keys())

    def collect_fn(self, data):
        images = []
        images_mask = []
        input_ids = []
        input_mask = []
        input_segment_ids = []
        max_len = max([len(b['input_ids']) for b in data])
        if 'title_input_ids' in data[0]:
            title_max_len = max([len(b['title_input_ids']) for b in data])
            max_len = max(max_len, title_max_len)
        # max_len = self.max_len
        title_ids = []
        title_input_mask = []
        title_input_segment_ids = []

        for i, ibatch in enumerate(data):
            # 排序方式 [b1f1, b1f2, .. b1f8, b2f1, ..., b2f8, .. bnf8]
            images.extend(ibatch["frames"])
            images_mask.extend(ibatch['frames_mask'])
            input_ids.append(ibatch['input_ids'][:max_len] +
                             [self.PAD] * (max_len - len(ibatch['input_ids'])))
            input_mask.append([1] * len(ibatch['input_ids'][:max_len]) +
                              [0] * (max_len - len(ibatch['input_ids'])))
            input_segment_ids.append([0] * max_len)
            if 'title_input_ids' in ibatch:
                input_ids.append(ibatch['title_input_ids'][:max_len] +
                                 [self.PAD] * (max_len - len(ibatch['title_input_ids'])))
                input_mask.append([1] * len(ibatch['title_input_ids'][:max_len]) +
                                  [0] * (max_len - len(ibatch['title_input_ids'])))
                input_segment_ids.append([0] * max_len)

        res = {"image": torch.stack(images, dim=0) if self.is_torchvision else images,  # torchvision 的话，需要stack一下
               "images_mask": torch.tensor(images_mask, dtype=torch.bool),
               "input_ids": torch.tensor(input_ids),
               "input_mask": torch.tensor(input_mask),
               "input_segment_ids": torch.tensor(input_segment_ids)}
        if title_ids:
            res['title_input_ids'] = torch.tensor(title_ids)
            res['title_input_mask'] = torch.tensor(title_input_mask)
            res['title_input_segment_ids'] = torch.tensor(
                title_input_segment_ids)

        return res

    def get_tokens(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) == 0:
            # log.error('empty input: %s, will skip, %s/%s=%s' % (text, self.emp, self.tot, round(self.emp/self.tot, 4)))
            return []
        tokens = ['[CLS]'] + tokens[:self.max_len - 2] + ['[SEP]']
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return token_ids

    def b64_decode(self, string):
        if isinstance(string, str):
            string = string.encode()
        return base64.decodebytes(string)

    def load_image(self, buffer):
        # return cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_COLOR)
        return Image.open(io.BytesIO(buffer)).convert("RGB")

    def pick_query(self, queries, use_pick_query):
        """ 选择一个query """
        if not use_pick_query:
            return queries[0]['query']
        for _ in range(5):
            pick_query = random.choice(queries)
            if pick_query['action'].get('loose_action_cnt', 0) >= 1:
                break
            if pick_query.get('rank', 10) <= 3:
                break
            if any([v['score'] > 0.6 for v in pick_query['sims'].values()]):
                break
        return pick_query['query']
