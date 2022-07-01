# -*- coding: utf-8 -*-

import json
import traceback
from base64 import b64decode
import torch

from fex import _logger as log
from fex.utils.hdfs_io import hopen
from fex.data.datasets.dist_dataset import DistLineReadingDataset


class ImageNetJsonMatxVisionDataset(DistLineReadingDataset):
    """
    ImageNet Json Dataset
    """

    def __init__(self, data_path, rank=0, world_size=1, shuffle=True, repeat=False):
        super().__init__(data_path, rank, world_size, shuffle, repeat)
        self.label2idx, self.idx2label = load_label2idx()

    def __iter__(self):
        for example in self.generate():
            try:
                data_item = json.loads(example)
                image_str = b64decode(data_item["b64_resized_binary"])
                if isinstance(image_str, str):
                    image_str = image_str.decode()

                label_idx = self.label2idx[data_item["english_name"]]
                res = {'image': image_str, 'label': label_idx}
                yield res

            except Exception as e:
                log.error(traceback.format_exc())
                log.error('encounter broken data: %s' % e)
                log.error(data_item.keys())

    def collect_fn(self, data):
        images = []
        labels = []

        for i, ibatch in enumerate(data):
            images.append(ibatch["image"])
            labels.append(ibatch["label"])

        labels = torch.tensor(labels)
        res = {"image": images, "label": labels}
        return res


def load_label2idx():
    with hopen('hdfs://haruna/home/byte_search_nlp_lq/user/huangwenguan/imagenet_label/idx2label.json') as f:
        idx2label = json.loads(f.read())
        label2idx = {l: int(i) for i, l in idx2label.items()}
    return label2idx, idx2label
