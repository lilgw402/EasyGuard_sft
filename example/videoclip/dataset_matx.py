# -*- coding: utf-8 -*-
'''
video dataset json format, from hdfa
'''
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
from example.videoclip.matx_text_pipeline import MatxTextPipe
from fex.utils.hdfs_io import hopen


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.matx_text_pipe = MatxTextPipe(vocab_file=dataset.vocab_file,
                                          max_seq_len=dataset.max_len,
                                          add_title=dataset.add_title,
                                          do_lower_case=dataset.do_lower_case,
                                          tokenize_emoji=dataset.tokenize_emoji,
                                          greedy_sharp=dataset.greedy_sharp)


class VideoDatasetMatx(DistLineReadingDataset):
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
                 pick_query: bool = True):
        super().__init__(data_path, rank, world_size, shuffle, repeat)

        self.vocab_file = config.get('network.vocab_file')
        self.max_len = config.get('dataset.max_len', 32)
        self.frame_len = config.get('dataset.max_frame_len', 8)
        self.add_title = config.get('dataset.add_title', False)
        self.do_lower_case = config.get('network.do_lower_case', True)
        self.tokenize_emoji = config.get('network.tokenize_emoji', False)
        self.greedy_sharp = config.get('network.greedy_sharp', False)
        self.use_pick_query = pick_query

        self.matx_text_pipe = None

        # 读一张黑色的帧，作为padding 用
        with hopen('hdfs://haruna/home/byte_search_nlp_lq/multimodal/black_frame.jpg', 'rb') as f:
            self.black_frame = np.array(
                np.frombuffer(f.read(), dtype=np.uint8))

    def __iter__(self):
        for example in self.generate():
            try:
                frames = []
                frames_mask = []
                res = {}

                data_item = json.loads(example)

                # 1. 图片的处理
                select_frames = data_item.pop('frames')[:self.frame_len]
                for frame in select_frames:
                    image_str = self.b64_decode(frame['b64_content'])
                    frames.append(image_str)
                    frames_mask.append(1)
                while len(frames) < self.frame_len:  # padding 图片
                    frames.append(self.black_frame)
                    frames_mask.append(0)
                res["frames"] = frames
                res["frames_mask"] = frames_mask

                # 2. 文本的处理
                raw_data_input = {}
                raw_data_input["query"] = self.pick_query(
                    data_item['queries'], self.use_pick_query)
                if self.add_title:
                    raw_data_input["title"] = data_item["title"]

                res.update(raw_data_input)
                yield res

            except Exception as e:
                log.error(traceback.format_exc())
                log.error('encounter broken data: %s' % e)
                log.error(data_item.keys())

    def collect_fn(self, data):
        images = []
        images_mask = []
        queries = []
        titles = []

        for i, ibatch in enumerate(data):
            # 排序方式 [b1f1, b1f2, .. b1f8, b2f1, ..., b2f8, .. bnf8]
            images.extend(ibatch["frames"])
            images_mask.extend(ibatch['frames_mask'])
            queries.append(ibatch['query'])
            if 'title' in ibatch and self.add_title:
                titles.append(ibatch['title'])

        matx_ret = self.matx_text_pipe(queries=queries, titles=titles)

        res = {"images": images,
               "images_mask": torch.tensor(images_mask, dtype=torch.bool),
               "input_ids": matx_ret["query_input_ids"],
               "input_mask": matx_ret["query_input_mask"],
               "input_segment_ids": matx_ret["query_segment_ids"]}

        if len(titles) > 0 and self.add_title:
            res['title_input_ids'] = matx_ret["title_input_ids"]
            res['title_input_mask'] = matx_ret["title_input_mask"]
            res['title_input_segment_ids'] = matx_ret["title_segment_ids"]

        return res

    def b64_decode(self, string):
        if isinstance(string, str):
            string = string.encode()
        return base64.decodebytes(string)

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
