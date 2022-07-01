
from PIL import Image
import io
import os
import torch
import base64
import traceback
import json
import numpy as np

import torchvision.transforms as transforms


from fex.data import BertTokenizer
from fex.data.datasets.dist_dataset import DistLineReadingDataset
from fex.utils.hdfs_io import hopen

from fex import _logger as log


class TorchvisionLabelDataset(DistLineReadingDataset):
    """
    dataset，继承的dataset是 DistLineReadingDataset，是一个逐行读hdfs数据的IterableDataset。
    """

    def __init__(self, config, data_path, rank=0, world_size=1, shuffle=True, repeat=False,
                 is_training=False):
        super().__init__(data_path, rank, world_size, shuffle, repeat)

        vocab_file = config.DATASET.VOCAB_FILE
        self.max_len = 32
        self.tokenizer = BertTokenizer(vocab_file, do_lower_case=False)
        self.PAD = self.tokenizer.vocab['[PAD]']

        self.preprocess = get_transform(mode='train' if is_training else 'val')

        self.frame_len = config.DATASET.FRAME_LEN
        with hopen('hdfs://haruna/home/byte_search_nlp_lq/user/huangwenguan/fex/black_frame.jpg', 'rb') as f:
            self.black_frame = self.preprocess(self._load_image(f.read()))
        print(self.black_frame.shape, 'self.black_frame')

    def __iter__(self):
        """

        """
        for example in self.generate():
            try:
                data_item = json.loads(example)
                label_idx = data_item['label']

                # 文本
                text = data_item['name']
                tokens = self.tokenizer.tokenize(text)
                tokens = tokens[:self.max_len - 2]
                tokens = ['[CLS]'] + tokens + ['[SEP]']
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

                # 多帧
                frames = []
                frames_raw = data_item.pop('frames')[:self.frame_len]
                for frame_str in frames_raw:
                    image_tensor = self.image_preprocess(frame_str)
                    frames.append(image_tensor)
                input_dict = {'frames': frames,
                              'label': label_idx,
                              'input_ids': token_ids}

                yield input_dict
            except Exception as e:
                log.error(traceback.format_exc())
                log.error('encounter broken data: %s' % e)

    def collect_fn(self, data):
        frames = []
        frames_mask = []
        labels = []
        input_ids = []
        input_mask = []
        input_segment_ids = []
        max_len = self.max_len

        for ib, ibatch in enumerate(data):

            labels.append(ibatch["label"])
            input_ids.append(ibatch['input_ids'][:max_len] + [self.PAD] * (max_len - len(ibatch['input_ids'])))
            input_mask.append([1] * len(ibatch['input_ids'][:max_len]) + [0] * (max_len - len(ibatch['input_ids'])))
            input_segment_ids.append([0] * max_len)

            img_np = ibatch['frames']
            frames_mask_cur = []
            # 如果不够8帧，要补帧
            if len(img_np) < self.frame_len:
                print('encouter not %s frames: %s ' % (self.frame_len, len(img_np)))
                for i, img in enumerate(img_np):
                    frames.append(img)
                    frames_mask_cur.append(1)
                for i in range(len(img_np), self.frame_len):
                    frames.append(self.black_frame)  # 如果这个视频没有帧，就用黑帧来替代
                    frames_mask_cur.append(0)
            else:
                # 够的话就冲啊
                for i, img in enumerate(img_np):
                    frames.append(img)
                    frames_mask_cur.append(1)

            frames_mask.append(frames_mask_cur)

        frames_mask = torch.tensor(frames_mask)  # [bsz, frame_num]
        frames = torch.stack(frames, dim=0)  # [bsz * frame_num, c, h, w]
        _, c, h, w = frames.shape
        bsz, frame_num = frames_mask.shape
        frames = frames.reshape([bsz, frame_num, c, h, w])
        labels = torch.tensor(labels)
        input_ids = torch.tensor(input_ids)
        input_mask = torch.tensor(input_mask)
        input_segment_ids = torch.tensor(input_segment_ids)

        res = {"frames": frames, "frames_mask": frames_mask,
               "label": labels,
               "input_ids": input_ids, "input_mask": input_mask,
               "input_segment_ids": input_segment_ids,
               }
        return res

    def image_preprocess(self, image_str):
        image = self._load_image(self.b64_decode(image_str))
        image_tensor = self.preprocess(image)
        return image_tensor

    @staticmethod
    def b64_decode(string):
        if isinstance(string, str):
            string = string.encode()
        return base64.decodebytes(string)

    @staticmethod
    def _load_image(buffer):
        return Image.open(io.BytesIO(buffer))


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


def creat_torchvision_loader(cfg):
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)

    train_dataset = TorchvisionLabelDataset(
        cfg,
        cfg.DATASET.TRAIN_PATH,
        rank=int(os.environ.get('RANK') or 0),
        world_size=int(os.environ.get('WORLD_SIZE') or 1),
        shuffle=True,
        repeat=True,
        is_training=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.TRAINER.TRAIN_BATCH_SIZE,
                                               num_workers=cfg.TRAIN.NUM_WORKERS,
                                               pin_memory=True,
                                               drop_last=True,
                                               collate_fn=train_dataset.collect_fn)

    val_dataset = TorchvisionLabelDataset(
        cfg,
        cfg.DATASET.VAL_PATH,
        rank=int(os.environ.get('RANK') or 0),
        world_size=int(os.environ.get('WORLD_SIZE') or 1),
        shuffle=False,
        repeat=True,
        is_training=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.TRAINER.VAL_BATCH_SIZE,
                                             num_workers=cfg.VAL.NUM_WORKERS,
                                             pin_memory=True,
                                             drop_last=False,
                                             collate_fn=val_dataset.collect_fn)
    return train_loader, val_loader
