"""Douyin frame dataset"""
import io
import json
import logging
import traceback

import base64
from PIL import Image

import torch
from torchvision import transforms
from ..dist_dataset import DistLineReadingDataset


class TusouImageDataset(DistLineReadingDataset):  # pylint: disable=abstract-method
    """
    图片集的 image dataset，will output image

    """
    def __init__(self, data_path, rank=0, world_size=1, shuffle=False, repeat=False,
                 data_size=-1, **kwargs):
        super().__init__(data_path, rank, world_size, shuffle, repeat, data_size=data_size)
        self.debug = kwargs.get('debug', False)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self._transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    def __iter__(self):
        for example in self.generate():
            try:
                if example:
                    example = self.transform(example)
                    if example:
                        yield example
            except Exception as e:  # pylint: disable=broad-except
                logging.warn('encounter broken data: %s; %s', e,
                              traceback.format_exc())

    def transform(self, example):
        example = json.loads(example)
        image_str = base64.b64decode(example['image_content'])
        image = Image.open(io.BytesIO(image_str)).convert("RGB")
        image_input = self._transform(image)
        model_input = {'image': image_input}
        if self.debug:
            model_input['doc'] = {
                'query': example['query'],
                'gid': example['doc_image_id']
            }

        return model_input

    def collect_fn(self, batch):
        """ 实现batching的操作，主要是一些pad的逻辑 """
        image = []
        image_mask = []
        docs = []
        for ibatch in batch:
            image.append(ibatch['image'])
            if self.debug:
                docs.append(ibatch['doc'])
        image = torch.stack(image, dim=0)
        result = {'image': image}
        if self.debug:
            result['docs'] = docs

        return result
