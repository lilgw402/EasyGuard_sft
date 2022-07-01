""" image captioning dataset """


import math
from PIL import Image
import io
import base64
import json
import traceback

import torchvision.transforms as transforms

import torch

from fex.data.datasets.dist_dataset import DistLineReadingDataset
from fex.data import BertTokenizer
from fex import _logger as log


class ImageCaptionDataset(DistLineReadingDataset):

    def __init__(self, config, data_path, transform, rank=0, world_size=1, shuffle=True, repeat=False):
        super().__init__(data_path, rank, world_size, shuffle, repeat)

        self.max_len = config.NETWORK.seq_length
    
        vocab_file = config.DATASET.VOCAB_FILE
        self.tokenizer = BertTokenizer(vocab_file, do_lower_case=False)
        self.PAD = '[PAD]'
        self.EOS = '[SEP]'

        self.preprocess = transform
    

    def __iter__(self):
        for example in self.generate():
            try:
                data_item = json.loads(example)
                text = data_item['text'].strip()
                tokens = self.tokenizer.tokenize(text)
                if len(tokens) == 0:
                    continue
                tokens = self.truncate_and_pad(tokens)
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

                image_tensor = self.image_preprocess(data_item['b64_resized_binary'])
                
                # x 比 y 少一个位置，是因为留给了图像
                x = torch.tensor(token_ids[:-1], dtype=torch.long) # 把最后一个token去掉
                y = torch.tensor(token_ids, dtype=torch.long)
                yield {'image': image_tensor, 'x': x, 'y': y}

            except Exception as e:
                log.error(traceback.format_exc())
                log.error('encounter broken data: %s' % e)

    def collect_fn(self, data):
        images = []
        x = []
        y = []
        for i, ibatch in enumerate(data):
            images.append(ibatch['image'])
            x.append(ibatch['x'])
            y.append(ibatch['y'])
        
        images = torch.stack(images, dim=0)
        x = torch.stack(x, dim=0)
        y = torch.stack(y, dim=0)

        return {'image': images, 'x': x, 'y': y}


    def truncate_and_pad(self, tks):
        tks = tks[:self.max_len-1]
        tks = tks + [self.EOS]  # 加sep代表句子结尾。 #TODO: 后面试试要不要
        return tks + [self.PAD] * (self.max_len - len(tks))
        
    
    def image_preprocess(self, image_str):
        image = self._load_image(self.b64_decode(image_str))
        image_tensor = self.preprocess(image)
        return image_tensor
    
    @staticmethod
    def _load_image(buffer):
        return Image.open(io.BytesIO(buffer))

    @staticmethod
    def b64_decode(string):
        if isinstance(string, str):
            string = string.encode()
        return base64.decodebytes(string)
    

def get_transform(mode: str = "train"):
    """
    根据不同的data，返回不同的transform
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if mode == "train":
        com_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    elif mode == 'val':
        com_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])
    else:
        raise ValueError('mode [%s] is not in [train, val]' % mode)
    return com_transforms